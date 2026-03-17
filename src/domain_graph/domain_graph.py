"""
domain_graph.py — Layer 1: Domain Graph (from Structured DB)

Pipeline:
  1. Introspect SQLite schema (tables, columns, FKs, row counts)
  2. Enrich each table with LLM-generated descriptions (Azure OpenAI GPT-4.1)
  3. Build Domain Graph in Neo4j (local Docker container)
  4. Provide a simple query interface for agents

Prerequisites:
  Start Neo4j locally with Docker:
    docker run -d --name neo4j-kg \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/password123 \
      -e NEO4J_PLUGINS='[]' \
      neo4j:5-community

  Neo4j Browser: http://localhost:7474
  Bolt endpoint:  bolt://localhost:7687

The Domain Graph is a MAP of the data landscape — it stores semantics about
what tables mean and how they relate, NOT the actual row-level data.
"""

import argparse
import json
import os
import sqlite3
import sys

from openai import AzureOpenAI

# ─── Ensure src/ is on sys.path for utils imports ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # src/domain_graph/
SRC_DIR = os.path.dirname(BASE_DIR)                         # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)                     # repo root
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.llm import get_llm_client, call_llm, parse_llm_json
from utils.cosmos_helpers import (
    get_gremlin_client, run_gremlin, run_gremlin_write,
    esc, make_vertex_id, gval,
)

# ─── Configuration ───────────────────────────────────────────────────────────
DB_PATH = os.path.join(PROJECT_ROOT, "source_data", "airlines.db")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 0 — LLM Client & Cosmos DB helpers imported from utils/
#  get_llm_client(), call_llm(), parse_llm_json()  → utils.llm
#  get_gremlin_client(), run_gremlin(), run_gremlin_write() → utils.cosmos_helpers
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Schema Introspection (SQLite)
# ═══════════════════════════════════════════════════════════════════════════════

def introspect_sqlite(db_path: str) -> dict:
    """
    Introspect a SQLite database and return schema metadata.

    Returns:
        {
            "table_name": {
                "columns": [{"name": str, "type": str, "notnull": bool, "pk": bool}],
                "foreign_keys": [{"from_col": str, "to_table": str, "to_col": str}],
                "row_count": int,
                "primary_keys": [str]
            }
        }
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get all user tables (skip sqlite_ internal tables)
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

    schema = {}
    for (table_name,) in tables:
        # Column info
        cols_raw = cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = []
        primary_keys = []
        for col in cols_raw:
            # col = (cid, name, type, notnull, default, pk)
            col_info = {
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "pk": bool(col[5]),
            }
            columns.append(col_info)
            if col[5]:
                primary_keys.append(col[1])

        # Foreign keys
        fks_raw = cur.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
        foreign_keys = []
        for fk in fks_raw:
            # fk = (id, seq, table, from, to, on_update, on_delete, match)
            foreign_keys.append({
                "from_col": fk[3],
                "to_table": fk[2],
                "to_col": fk[4],
            })

        # Row count
        row_count = cur.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]

        schema[table_name] = {
            "columns": columns,
            "foreign_keys": foreign_keys,
            "row_count": row_count,
            "primary_keys": primary_keys,
        }

    conn.close()
    return schema


def print_schema(schema: dict):
    """Pretty-print introspected schema info."""
    print("\n" + "=" * 70)
    print("  INTROSPECTED SCHEMA")
    print("=" * 70)
    for table_name, info in schema.items():
        print(f"\n  Table: {table_name} ({info['row_count']} rows)")
        print(f"  PK: {info['primary_keys']}")
        for col in info["columns"]:
            pk_marker = " [PK]" if col["pk"] else ""
            nn_marker = " NOT NULL" if col["notnull"] else ""
            print(f"    - {col['name']} ({col['type']}{nn_marker}{pk_marker})")
        if info["foreign_keys"]:
            for fk in info["foreign_keys"]:
                print(f"    FK: {fk['from_col']} → {fk['to_table']}({fk['to_col']})")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2 — LLM Enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_with_llm(schema: dict, client: AzureOpenAI) -> dict:
    """
    For each table, call GPT-4.1 to generate:
      - description: 1-2 sentence summary of what the table represents
      - domain: category/domain label
      - semantic_relationships: list of {target_table, relationship_type, reason}
    """
    all_table_names = list(schema.keys())
    enriched = {}

    for table_name, info in schema.items():
        col_summary = ", ".join(
            f"{c['name']} ({c['type']})" for c in info["columns"]
        )
        fk_summary = "; ".join(
            f"{fk['from_col']} → {fk['to_table']}({fk['to_col']})"
            for fk in info["foreign_keys"]
        ) or "None"

        prompt = f"""You are a data architect analyzing a database schema for an airlines company.

Given the following table, provide a JSON response with exactly these fields:
- "description": A 1-2 sentence summary of what this table represents and its business purpose.
- "domain": A single domain category label (e.g., "fleet_management", "flight_operations", "crew_management", "bookings", "revenue", "maintenance", "route_network", "passenger_services").
- "semantic_relationships": An array of objects, each with:
    - "target_table": name of a related table from this list: {all_table_names}
    - "relationship_type": a semantic edge label in UPPER_SNAKE_CASE (e.g., OPERATES_ON, ASSIGNED_TO, BOOKED_FOR, BELONGS_TO)
    - "reason": brief explanation of why this relationship exists
- "concepts": An array of 2-5 abstract, high-level concepts that this table encapsulates. Each with:
    - "name": A concise concept label (e.g., "Fleet Composition", "Route Network", "Crew Scheduling", "Revenue Management")
    - "description": 1 sentence explaining what this concept represents in the business domain
    - "derived_from": array of column names from this table that inform this concept
  IMPORTANT: Concepts should be ABSTRACT business ideas, NOT raw column values or column names.
  Think about what business knowledge this table captures at a higher level.
  Examples: from a 'flights' table with columns (flight_number, origin, destination, departure_time, aircraft_id) you might derive
  concepts like "Flight Schedule", "Route Coverage", "Aircraft Utilization".

Table: {table_name}
Columns: {col_summary}
Foreign Keys: {fk_summary}
Row Count: {info['row_count']}

Other tables in this database: {[t for t in all_table_names if t != table_name]}

Respond ONLY with valid JSON, no markdown fences, no extra text."""

        print(f"  Enriching: {table_name}...", end=" ", flush=True)
        response = call_llm(client, prompt)

        # Parse JSON — handle potential markdown fences from LLM
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]  # remove first line
            cleaned = cleaned.rsplit("```", 1)[0]  # remove last fence
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            enriched[table_name] = parsed
            n_concepts = len(parsed.get('concepts', []))
            print(f"OK — {parsed.get('domain', 'unknown')} ({n_concepts} concepts)")
        except json.JSONDecodeError as e:
            print(f"WARN: JSON parse failed ({e}), using fallback")
            enriched[table_name] = {
                "description": f"Database table '{table_name}' with {info['row_count']} rows.",
                "domain": "unknown",
                "semantic_relationships": [],
                "concepts": [],
            }

    return enriched


def print_enrichment(enriched: dict):
    """Pretty-print LLM enrichment results."""
    print("\n" + "=" * 70)
    print("  LLM-ENRICHED TABLE DESCRIPTIONS")
    print("=" * 70)
    for table_name, info in enriched.items():
        print(f"\n  {table_name}")
        print(f"    Description: {info['description']}")
        print(f"    Domain: {info['domain']}")
        if info.get("semantic_relationships"):
            for rel in info["semantic_relationships"]:
                print(
                    f"    → {rel['relationship_type']} → {rel['target_table']}"
                    f"  ({rel.get('reason', '')})"
                )
        if info.get("concepts"):
            print("    Concepts:")
            for c in info["concepts"]:
                cols = ", ".join(c.get("derived_from", []))
                print(f"      • {c['name']} — {c.get('description', '')}")
                if cols:
                    print(f"        (from: {cols})")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2b — Concept Normalization & Cross-Table Linking
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_concepts(enriched: dict, client: AzureOpenAI, verbose: bool = True) -> tuple:
    """
    Normalize and deduplicate concepts across all tables, then identify
    cross-table concept relationships.

    Similar concepts from different tables (e.g., "Route Coverage" from
    flights and "Network Planning" from routes) are merged into a single
    canonical concept that retains provenance from all contributing tables.

    This mirrors the entity resolution pattern from the Lexical Graph layer.

    Args:
        enriched: Dict of table enrichments, each containing a "concepts" list.
        client:   AzureOpenAI client for the normalization LLM call.
        verbose:  Print progress.

    Returns:
        (normalized_concepts, cross_links) where:
          normalized_concepts: list of {
              "name": str (canonical),
              "description": str (merged),
              "source_tables": [str],
              "derived_from": {"table_name": ["col1", "col2"], ...}
          }
          cross_links: list of {
              "concept_a": str,
              "concept_b": str,
              "relationship_type": str,
              "reason": str
          }
    """
    if verbose:
        print("\n  [Concept Normalization] Collecting concepts across tables...")

    # ── Step 1: Collect all concepts with source attribution ──────────
    all_concepts = []
    for table_name, enr in enriched.items():
        for c in enr.get("concepts", []):
            all_concepts.append({
                "name": c.get("name", "Unknown"),
                "description": c.get("description", ""),
                "derived_from": c.get("derived_from", []),
                "source_table": table_name,
            })

    if not all_concepts:
        if verbose:
            print("  [Concept Normalization] No concepts found — skipping.")
        return [], []

    if verbose:
        print(f"  [Concept Normalization] Found {len(all_concepts)} raw concepts across {len(enriched)} tables.")

    # ── Step 2: LLM-based normalization ──────────────────────────────
    concept_list_json = json.dumps(all_concepts, indent=2)

    prompt = f"""You are a senior data architect reviewing abstract concepts extracted from database tables for an airlines company.

Below is a list of concepts, each attributed to a source table. Your tasks:

TASK 1 — MERGE GROUPS: Identify concepts from DIFFERENT tables that represent the same
abstract idea (even if named differently). Group them and pick the best canonical name.
Only merge if they truly represent the same concept — do not over-merge.

TASK 2 — CROSS-LINKS: Identify pairs of concepts that are DISTINCT but semantically
related (e.g., "Flight Schedule" is related to "Crew Assignment" via DEPENDS_ON).
Only include meaningful, non-trivial relationships.

Concepts:
{concept_list_json}

Respond with a JSON object (no markdown fences, no extra text):
{{
  "merge_groups": [
    {{
      "canonical_name": "<best name to use for this concept>",
      "canonical_description": "<merged 1-sentence description covering all variants>",
      "variants": [
        {{"name": "<original concept name>", "source_table": "<table>"}}
      ],
      "reason": "<why these are the same concept>"
    }}
  ],
  "cross_links": [
    {{
      "concept_a": "<canonical concept name>",
      "concept_b": "<canonical concept name>",
      "relationship_type": "UPPER_SNAKE_CASE (e.g., COMPOSED_OF, DEPENDS_ON, ENABLES)",
      "reason": "<1 sentence explaining the relationship>"
    }}
  ]
}}

RULES:
- Only merge concepts from DIFFERENT tables. A concept appearing once stays as-is.
- canonical_name should be concise, abstract, and domain-appropriate.
- If no merges are needed, return an empty merge_groups array.
- If no cross-links exist, return an empty cross_links array.
- cross_links should reference final canonical names (post-merge).
"""

    if verbose:
        print("  [Concept Normalization] Calling LLM for cross-table concept resolution...")

    response = call_llm(client, prompt, temperature=0.1)

    # Parse the response
    parsed = parse_llm_json(response)
    if not isinstance(parsed, dict):
        if verbose:
            print("  [Concept Normalization] WARN: Could not parse LLM response, skipping normalization.")
        # Return concepts as-is (no merging)
        normalized = _build_unmerged_concepts(all_concepts)
        return normalized, []

    merge_groups = parsed.get("merge_groups", [])
    cross_links = parsed.get("cross_links", [])

    if verbose:
        print(f"  [Concept Normalization] LLM found {len(merge_groups)} merge group(s), {len(cross_links)} cross-link(s).")

    # ── Step 3: Build merge map ──────────────────────────────────────
    # Maps (concept_name_lower, source_table) → canonical_name
    merge_map = {}
    canonical_descriptions = {}  # canonical_name_lower → description

    for group in merge_groups:
        canonical = group.get("canonical_name", "").strip()
        canonical_desc = group.get("canonical_description", "")
        if not canonical:
            continue

        canonical_descriptions[canonical.lower()] = canonical_desc

        for variant in group.get("variants", []):
            vname = variant.get("name", "").strip()
            vtable = variant.get("source_table", "").strip()
            if vname and vtable:
                merge_map[(vname.lower(), vtable)] = canonical

    if verbose and merge_map:
        print("  [Concept Normalization] Merge map:")
        for (vname, vtable), canon in merge_map.items():
            print(f"    \"{vname}\" (from {vtable}) → \"{canon}\"")

    # ── Step 4: Apply merges and deduplicate ─────────────────────────
    # Accumulate by canonical name
    concept_accum = {}  # canonical_name_lower → concept_data

    for c in all_concepts:
        key = (c["name"].lower(), c["source_table"])
        canonical_name = merge_map.get(key, c["name"])
        canon_lower = canonical_name.lower()

        if canon_lower in concept_accum:
            # Merge: add this table's provenance
            existing = concept_accum[canon_lower]
            if c["source_table"] not in existing["source_tables"]:
                existing["source_tables"].append(c["source_table"])
            if c["source_table"] not in existing["derived_from"]:
                existing["derived_from"][c["source_table"]] = c.get("derived_from", [])
            else:
                # Extend column list for this table
                for col in c.get("derived_from", []):
                    if col not in existing["derived_from"][c["source_table"]]:
                        existing["derived_from"][c["source_table"]].append(col)
        else:
            # First occurrence
            desc = canonical_descriptions.get(canon_lower, c.get("description", ""))
            concept_accum[canon_lower] = {
                "name": canonical_name,
                "description": desc or c.get("description", ""),
                "source_tables": [c["source_table"]],
                "derived_from": {c["source_table"]: c.get("derived_from", [])},
            }

    normalized_concepts = list(concept_accum.values())

    # ── Step 5: Validate cross-links reference existing concepts ─────
    valid_names = {nc["name"].lower() for nc in normalized_concepts}
    validated_links = []
    for link in cross_links:
        a = link.get("concept_a", "").strip()
        b = link.get("concept_b", "").strip()
        if a.lower() in valid_names and b.lower() in valid_names and a.lower() != b.lower():
            validated_links.append({
                "concept_a": a,
                "concept_b": b,
                "relationship_type": link.get("relationship_type", "RELATED_TO"),
                "reason": link.get("reason", ""),
            })

    if verbose:
        n_shared = sum(1 for nc in normalized_concepts if len(nc["source_tables"]) > 1)
        print(f"  [Concept Normalization] Result: {len(normalized_concepts)} canonical concepts "
              f"({n_shared} shared across tables), {len(validated_links)} cross-links.")
        for nc in normalized_concepts:
            tables_str = ", ".join(nc["source_tables"])
            shared_marker = " *SHARED*" if len(nc["source_tables"]) > 1 else ""
            print(f"    • {nc['name']} [{tables_str}]{shared_marker}")

    return normalized_concepts, validated_links


def _build_unmerged_concepts(all_concepts: list[dict]) -> list[dict]:
    """Fallback: convert raw concept list to normalized format without merging."""
    accum = {}
    for c in all_concepts:
        key = c["name"].lower()
        if key in accum:
            existing = accum[key]
            if c["source_table"] not in existing["source_tables"]:
                existing["source_tables"].append(c["source_table"])
            if c["source_table"] not in existing["derived_from"]:
                existing["derived_from"][c["source_table"]] = c.get("derived_from", [])
        else:
            accum[key] = {
                "name": c["name"],
                "description": c.get("description", ""),
                "source_tables": [c["source_table"]],
                "derived_from": {c["source_table"]: c.get("derived_from", [])},
            }
    return list(accum.values())


def print_normalized_concepts(concepts: list[dict], cross_links: list[dict]):
    """Pretty-print normalized concepts and cross-links."""
    print("\n" + "=" * 70)
    print("  NORMALIZED CONCEPTS (Cross-Table)")
    print("=" * 70)
    for nc in concepts:
        tables_str = ", ".join(nc["source_tables"])
        shared = " [SHARED]" if len(nc["source_tables"]) > 1 else ""
        print(f"\n  {nc['name']}{shared}")
        print(f"    Description: {nc['description']}")
        print(f"    Source tables: {tables_str}")
        for table, cols in nc.get("derived_from", {}).items():
            if cols:
                print(f"    Derived from ({table}): {', '.join(cols)}")
    if cross_links:
        print(f"\n  Cross-links:")
        for link in cross_links:
            print(f"    {link['concept_a']} —[{link['relationship_type']}]→ {link['concept_b']}")
            print(f"      Reason: {link['reason']}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Build Domain Graph in Neo4j
# ═══════════════════════════════════════════════════════════════════════════════

def build_domain_graph(schema: dict, enriched: dict, client=None,
                       normalized_concepts: list[dict] = None,
                       cross_links: list[dict] = None):
    """
    Build the Domain Graph in Cosmos DB (Gremlin API):
      - DomainEntity vertices (one per DB table — stores metadata, NOT row data)
      - HAS_FK edges for foreign key relationships
      - Semantic edges (PART_OF, SUPPLIES, etc.) from LLM enrichment
      - Concept vertices (abstract business concepts derived from table columns)
      - HAS_CONCEPT edges (DomainEntity → Concept)
      - RELATED_CONCEPT edges (Concept → Concept cross-links)
    """
    if client is None:
        client = get_gremlin_client()

    normalized_concepts = normalized_concepts or []
    cross_links = cross_links or []

    # ── Clean slate — drop all existing Domain Graph vertices ─────────
    print("  Clearing existing Domain Graph data...")
    try:
        run_gremlin_write(client, "g.V().hasLabel('DomainEntity').drop()")
    except Exception as e:
        print(f"  WARN: Could not drop DomainEntity vertices: {e}")
    try:
        run_gremlin_write(client, "g.V().hasLabel('Concept').drop()")
    except Exception as e:
        print(f"  WARN: Could not drop Concept vertices: {e}")

    # ── Insert DomainEntity vertices ──────────────────────────────────
    print("\n  Inserting DomainEntity vertices...")
    for table_name, info in schema.items():
        enr = enriched.get(table_name, {})
        description = enr.get("description", f"Table: {table_name}")
        domain = enr.get("domain", "unknown")
        key_cols = json.dumps(info["primary_keys"])
        col_info = json.dumps(
            [{"name": c["name"], "type": c["type"]} for c in info["columns"]]
        )
        vid = make_vertex_id("DomainEntity", table_name)
        try:
            run_gremlin_write(client, (
                f"g.addV('DomainEntity')"
                f".property('id', '{esc(vid)}')"
                f".property('category', 'domain')"
                f".property('name', '{esc(table_name)}')"
                f".property('description', '{esc(description)}')"
                f".property('key_columns', '{esc(key_cols)}')"
                f".property('column_info', '{esc(col_info)}')"
                f".property('row_count', {info['row_count']})"
                f".property('domain', '{esc(domain)}')"
                f".property('source_type', 'structured_db')"
            ))
            print(f"    + {table_name} (domain: {domain})")
        except Exception as e:
            print(f"    WARN: DomainEntity '{table_name}' failed: {e}")

    # ── Insert FK-based edges (HAS_FK) ────────────────────────────────
    print("\n  Inserting FK edges...")
    for table_name, info in schema.items():
        for fk in info["foreign_keys"]:
            target = fk["to_table"]
            reason = f"FK: {table_name}.{fk['from_col']} -> {target}.{fk['to_col']}"
            from_id = make_vertex_id("DomainEntity", table_name)
            to_id = make_vertex_id("DomainEntity", target)
            try:
                run_gremlin_write(client, (
                    f"g.V('{esc(from_id)}')"
                    f".addE('HAS_FK')"
                    f".to(g.V('{esc(to_id)}'))"
                    f".property('reason', '{esc(reason)}')"
                ))
                print(f"    + {table_name} —[HAS_FK]→ {target}")
            except Exception as e:
                print(f"    WARN: FK edge {table_name}→{target} failed: {e}")

    # ── Insert semantic edges (from LLM enrichment) ───────────────────
    print("\n  Inserting semantic edges...")
    for table_name, enr in enriched.items():
        for rel in enr.get("semantic_relationships", []):
            rel_type = rel["relationship_type"]
            target = rel["target_table"]
            reason = rel.get("reason", "")

            if target not in schema:
                print(f"    SKIP: {table_name} —[{rel_type}]→ {target} (target not in schema)")
                continue
            if rel_type == "HAS_FK":
                continue

            safe_rel_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel_type)
            from_id = make_vertex_id("DomainEntity", table_name)
            to_id = make_vertex_id("DomainEntity", target)
            try:
                run_gremlin_write(client, (
                    f"g.V('{esc(from_id)}')"
                    f".addE('{esc(safe_rel_type)}')"
                    f".to(g.V('{esc(to_id)}'))"
                    f".property('reason', '{esc(reason)}')"
                ))
                print(f"    + {table_name} —[{safe_rel_type}]→ {target}")
            except Exception as e:
                print(f"    WARN: Semantic edge {table_name}—[{safe_rel_type}]→{target} failed: {e}")

    # ── Insert Concept vertices ───────────────────────────────────────
    if normalized_concepts:
        print(f"\n  Inserting {len(normalized_concepts)} Concept vertices...")
        for nc in normalized_concepts:
            source_tables_json = json.dumps(nc["source_tables"])
            derived_from_json = json.dumps(nc.get("derived_from", {}))
            is_shared = len(nc["source_tables"]) > 1
            vid = make_vertex_id("Concept", nc["name"])
            nc_name = nc["name"]
            nc_desc = nc["description"]
            try:
                run_gremlin_write(client, (
                    f"g.addV('Concept')"
                    f".property('id', '{esc(vid)}')"
                    f".property('category', 'domain')"
                    f".property('name', '{esc(nc_name)}')"
                    f".property('description', '{esc(nc_desc)}')"
                    f".property('source_tables', '{esc(source_tables_json)}')"
                    f".property('derived_from', '{esc(derived_from_json)}')"
                    f".property('shared', {str(is_shared).lower()})"
                ))
                shared_tag = " [SHARED]" if is_shared else ""
                print(f"    + {nc['name']}{shared_tag}")
            except Exception as e:
                print(f"    WARN: Concept vertex '{nc['name']}' failed: {e}")

        # ── Insert HAS_CONCEPT edges (DomainEntity → Concept) ─────────
        print("\n  Inserting HAS_CONCEPT edges...")
        for nc in normalized_concepts:
            for table in nc["source_tables"]:
                cols = nc.get("derived_from", {}).get(table, [])
                derived_str = ", ".join(cols) if cols else "inferred"
                de_id = make_vertex_id("DomainEntity", table)
                concept_id = make_vertex_id("Concept", nc["name"])
                try:
                    run_gremlin_write(client, (
                        f"g.V('{esc(de_id)}')"
                        f".addE('HAS_CONCEPT')"
                        f".to(g.V('{esc(concept_id)}'))"
                        f".property('derived_from', '{esc(derived_str)}')"
                    ))
                    print(f"    + {table} —[HAS_CONCEPT]→ {nc['name']}")
                except Exception as e:
                    print(f"    WARN: HAS_CONCEPT {table}→{nc['name']} failed: {e}")

    # ── Insert RELATED_CONCEPT edges (cross-links) ────────────────────
    if cross_links:
        print(f"\n  Inserting {len(cross_links)} RELATED_CONCEPT edges...")
        for link in cross_links:
            concept_a_id = make_vertex_id("Concept", link["concept_a"])
            concept_b_id = make_vertex_id("Concept", link["concept_b"])
            rel_type = link.get("relationship_type", "RELATED_TO")
            reason = link.get("reason", "")
            try:
                run_gremlin_write(client, (
                    f"g.V('{esc(concept_a_id)}')"
                    f".addE('RELATED_CONCEPT')"
                    f".to(g.V('{esc(concept_b_id)}'))"
                    f".property('relationship_type', '{esc(rel_type)}')"
                    f".property('reason', '{esc(reason)}')"
                ))
                print(f"    + {link['concept_a']} —[RELATED_CONCEPT/{rel_type}]→ {link['concept_b']}")
            except Exception as e:
                print(f"    WARN: RELATED_CONCEPT edge failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    n_concepts = len(normalized_concepts)
    n_shared = sum(1 for nc in normalized_concepts if len(nc["source_tables"]) > 1)
    print(f"\n  Domain Graph built in Cosmos DB (indigokg/knowledgegraph)!")
    print(f"  • {len(schema)} DomainEntity vertices")
    print(f"  • {n_concepts} Concept vertices ({n_shared} shared across tables)")
    print(f"  • {len(cross_links)} concept cross-links")
    return client


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Query Interface
# ═══════════════════════════════════════════════════════════════════════════════

def query_domain_graph(question: str, client=None) -> list[dict]:
    """
    Simple agent-facing query: find DomainEntity vertices whose name or description
    matches keywords from the question. Returns matching vertices + their relationships.
    Uses Cosmos DB Gremlin text predicates for contains matching.
    """
    if client is None:
        client = get_gremlin_client()

    stop_words = {"which", "what", "about", "have", "does", "that", "this",
                  "with", "from", "tell", "show", "find", "data", "tables", "related"}
    keywords = [
        w.strip("?.,!\"'").lower()
        for w in question.split()
        if len(w.strip("?.,!\"'")) > 3 and w.strip("?.,!\"'").lower() not in stop_words
    ]

    results = []

    # Fetch all DomainEntity vertices and filter in Python for keyword match
    all_entities = run_gremlin(client, (
        "g.V().hasLabel('DomainEntity')"
        ".valueMap('name','description','domain','key_columns','row_count')"
    ))

    for rec in all_entities:
        name = gval(rec, "name", "")
        description = gval(rec, "description", "")
        for kw in keywords:
            if kw in name.lower() or kw in description.lower():
                if not any(r["name"] == name for r in results):
                    results.append({
                        "name": name,
                        "description": description,
                        "domain": gval(rec, "domain", "unknown"),
                        "key_columns": gval(rec, "key_columns", "[]"),
                        "row_count": gval(rec, "row_count", 0),
                        "matched_keyword": kw,
                    })
                break

    # Also search Concept vertices
    concept_results = []
    all_concepts = run_gremlin(client, (
        "g.V().hasLabel('Concept')"
        ".valueMap('name','description','source_tables','shared')"
    ))
    for rec in all_concepts:
        c_name = gval(rec, "name", "")
        c_desc = gval(rec, "description", "")
        for kw in keywords:
            if kw in c_name.lower() or kw in c_desc.lower():
                if not any(cr["concept_name"] == c_name for cr in concept_results):
                    # Find parent DomainEntities via HAS_CONCEPT edges
                    cid = make_vertex_id("Concept", c_name)
                    parent_recs = run_gremlin(client, (
                        f"g.V('{esc(cid)}').in('HAS_CONCEPT').values('name')"
                    ))
                    parent_entities = [p for p in parent_recs if p]
                    concept_results.append({
                        "concept_name": c_name,
                        "concept_desc": c_desc,
                        "source_tables": gval(rec, "source_tables", "[]"),
                        "shared": gval(rec, "shared", False),
                        "parent_entities": parent_entities,
                        "matched_keyword": kw,
                    })
                    # Add parent DomainEntities to results if not already there
                    for parent in parent_entities:
                        if parent and not any(r["name"] == parent for r in results):
                            parent_recs2 = run_gremlin(client, (
                                f"g.V().hasLabel('DomainEntity').has('name', '{esc(parent)}')"
                                ".valueMap('name','description','domain','key_columns','row_count')"
                            ))
                            for pr in parent_recs2:
                                results.append({
                                    "name": gval(pr, "name", ""),
                                    "description": gval(pr, "description", ""),
                                    "domain": gval(pr, "domain", "unknown"),
                                    "key_columns": gval(pr, "key_columns", "[]"),
                                    "row_count": gval(pr, "row_count", 0),
                                    "matched_keyword": f"{kw} (via concept '{c_name}')",
                                })
                break

    # For each matched node, fetch outgoing + incoming relationships + concepts
    for node in results:
        node["relationships"] = []
        node["concepts"] = []
        vid = make_vertex_id("DomainEntity", node["name"])

        out_edges = run_gremlin(client, (
            f"g.V('{esc(vid)}').outE().hasLabel(neq('HAS_CONCEPT'))"
            ".where(inV().hasLabel('DomainEntity'))"
            ".project('rel_type','target','reason')"
            ".by(label())"
            ".by(inV().values('name'))"
            ".by(coalesce(values('reason'), constant('')))"
        ))
        for rec in out_edges:
            node["relationships"].append({
                "direction": "outgoing",
                "type": rec.get("rel_type", ""),
                "target": rec.get("target", ""),
                "reason": rec.get("reason", ""),
            })

        in_edges = run_gremlin(client, (
            f"g.V('{esc(vid)}').inE()"
            ".where(outV().hasLabel('DomainEntity'))"
            ".project('rel_type','source','reason')"
            ".by(label())"
            ".by(outV().values('name'))"
            ".by(coalesce(values('reason'), constant('')))"
        ))
        for rec in in_edges:
            node["relationships"].append({
                "direction": "incoming",
                "type": rec.get("rel_type", ""),
                "source": rec.get("source", ""),
                "reason": rec.get("reason", ""),
            })

        concept_recs = run_gremlin(client, (
            f"g.V('{esc(vid)}').out('HAS_CONCEPT')"
            ".valueMap('name','description','shared')"
        ))
        for rec in concept_recs:
            c_name = gval(rec, "name", "")
            node["concepts"].append({
                "name": c_name,
                "description": gval(rec, "description", ""),
                "shared": gval(rec, "shared", False),
                "related_concepts": [],
            })

    return results, concept_results


def print_query_results(question: str, results: list[dict],
                        concept_results: list[dict] = None):
    """Pretty-print query results."""
    print(f"\n{'=' * 70}")
    print(f"  QUERY: \"{question}\"")
    print(f"{'=' * 70}")
    if not results and not concept_results:
        print("  No matching nodes found.")
    for node in results:
        print(f"\n  Node: {node['name']}")
        print(f"    Description: {node['description']}")
        print(f"    Domain: {node['domain']}")
        print(f"    Key Columns: {node['key_columns']}")
        print(f"    Row Count: {node['row_count']}")
        print(f"    Matched on: '{node['matched_keyword']}'")
        if node.get("relationships"):
            print("    Relationships:")
            for rel in node["relationships"]:
                if rel["direction"] == "outgoing":
                    print(f"      —[{rel['type']}]→ {rel['target']}")
                else:
                    print(f"      {rel['source']} —[{rel['type']}]→ THIS")
        if node.get("concepts"):
            print("    Concepts:")
            for c in node["concepts"]:
                shared_tag = " [SHARED]" if c.get("shared") else ""
                print(f"      • {c['name']}{shared_tag}: {c['description']}")
                for rc in c.get("related_concepts", []):
                    print(f"        ↔ {rc['related']} ({rc['type']})")
    if concept_results:
        print("\n  Matched Concepts:")
        for cr in concept_results:
            shared_tag = " [SHARED]" if cr.get("shared") else ""
            parents = ", ".join(cr.get("parent_entities", []))
            print(f"    • {cr['concept_name']}{shared_tag}: {cr['concept_desc']}")
            if parents:
                print(f"      Tables: {parents}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Visualize the Full Graph
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_graph(client=None):
    """Print all vertices and edges in the Domain Graph (Cosmos DB Gremlin)."""
    if client is None:
        client = get_gremlin_client()

    print(f"\n{'=' * 70}")
    print("  DOMAIN GRAPH — ALL VERTICES & EDGES (Cosmos DB)")
    print(f"{'=' * 70}")

    # DomainEntity vertices
    print("\n  DomainEntity Vertices:")
    records = run_gremlin(client, (
        "g.V().hasLabel('DomainEntity')"
        ".valueMap('name','description','domain','row_count')"
    ))
    for rec in records:
        name = gval(rec, "name", "")
        domain = gval(rec, "domain", "")
        desc = gval(rec, "description", "")
        row_count = gval(rec, "row_count", 0)
        print(f"    [{domain}] {name} — {desc} ({row_count} rows)")

    # Concept vertices
    print("\n  Concept Vertices:")
    concept_records = run_gremlin(client, (
        "g.V().hasLabel('Concept')"
        ".valueMap('name','description','source_tables','shared')"
    ))
    if not concept_records:
        print("    (none)")
    for rec in concept_records:
        name = gval(rec, "name", "")
        shared = gval(rec, "shared", False)
        shared_tag = " [SHARED]" if shared else ""
        desc = gval(rec, "description", "")
        src_tables = gval(rec, "source_tables", "[]")
        print(f"    {name}{shared_tag} — {desc}")
        print(f"      Source tables: {src_tables}")

    # DomainEntity → DomainEntity edges
    print("\n  DomainEntity Edges:")
    de_edges = run_gremlin(client, (
        "g.V().hasLabel('DomainEntity').outE()"
        ".where(inV().hasLabel('DomainEntity'))"
        ".project('from_node','rel_type','to_node','reason')"
        ".by(outV().values('name'))"
        ".by(label())"
        ".by(inV().values('name'))"
        ".by(coalesce(values('reason'), constant('')))"
    ))
    for rec in de_edges:
        print(f"    {rec.get('from_node','')} —[{rec.get('rel_type','')}]→ "
              f"{rec.get('to_node','')}  ({rec.get('reason','')})")

    # HAS_CONCEPT edges
    print("\n  HAS_CONCEPT Edges:")
    hc_records = run_gremlin(client, (
        "g.V().hasLabel('DomainEntity').outE('HAS_CONCEPT')"
        ".project('entity','concept','derived_from')"
        ".by(outV().values('name'))"
        ".by(inV().values('name'))"
        ".by(coalesce(values('derived_from'), constant('')))"
    ))
    if not hc_records:
        print("    (none)")
    for rec in hc_records:
        print(f"    {rec.get('entity','')} —[HAS_CONCEPT]→ "
              f"{rec.get('concept','')}  (derived from: {rec.get('derived_from','')})")

    # RELATED_CONCEPT edges
    print("\n  RELATED_CONCEPT Edges:")
    rc_records = run_gremlin(client, (
        "g.V().hasLabel('Concept').outE('RELATED_CONCEPT')"
        ".project('from_concept','to_concept','rel_type','reason')"
        ".by(outV().values('name'))"
        ".by(inV().values('name'))"
        ".by(coalesce(values('relationship_type'), constant('')))"
        ".by(coalesce(values('reason'), constant('')))"
    ))
    if not rc_records:
        print("    (none)")
    for rec in rc_records:
        print(f"    {rec.get('from_concept','')} —[{rec.get('rel_type','')}]→ "
              f"{rec.get('to_concept','')}  ({rec.get('reason','')})")

    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 1 — Domain Graph Builder")
    parser.add_argument(
        "--advanced", action="store_true",
        help="Use ReAct agent-based enrichment (iterative exploration) instead of single-shot LLM calls",
    )
    args = parser.parse_args()

    mode = "ReAct Agent" if args.advanced else "Single-shot LLM"
    print("=" * 70)
    print("  LAYER 1 — DOMAIN GRAPH BUILDER")
    print("  Airlines Database Ontology")
    print(f"  Enrichment mode: {mode}")
    print("=" * 70)

    # Step 1: Introspect
    print("\n[Step 1] Introspecting SQLite schema...")
    if not os.path.exists(DB_PATH):
        print(f"  ERROR: Database not found at {DB_PATH}")
        print("  Run 'python source_data/setup_new_db.py' first to create the sample database.")
        return
    schema = introspect_sqlite(DB_PATH)
    print_schema(schema)

    # Step 2: LLM Enrichment
    client = get_llm_client()
    if args.advanced:
        print("\n[Step 2] Enriching with ReAct Agent (iterative exploration)...")
        from agents.domain_agent import enrich_with_llm_advanced
        enriched = enrich_with_llm_advanced(schema, client, db_path=DB_PATH)
    else:
        print("\n[Step 2] Enriching with LLM (GPT-4.1, single-shot)...")
        enriched = enrich_with_llm(schema, client)
    print_enrichment(enriched)

    # Step 2b: Concept Normalization
    print("\n[Step 2b] Normalizing concepts across tables...")
    normalized_concepts, cross_links = normalize_concepts(enriched, client)
    print_normalized_concepts(normalized_concepts, cross_links)

    # Step 3: Build Graph
    print("\n[Step 3] Building Domain Graph in Cosmos DB (indigokg/knowledgegraph)...")
    gremlin = get_gremlin_client()
    build_domain_graph(schema, enriched, gremlin,
                       normalized_concepts=normalized_concepts,
                       cross_links=cross_links)

    # Step 4: Visualize
    visualize_graph(gremlin)

    # Step 5: Demo Queries
    demo_questions = [
        "Which aircraft operate on specific routes?",
        "What data do we have about flight operations?",
        "Tell me about crew assignments",
        "Which tables are related to bookings?",
    ]
    print("\n\n[Step 5] Running demo queries...")
    for q in demo_questions:
        results, concept_results = query_domain_graph(q, gremlin)
        print_query_results(q, results, concept_results)

    gremlin.close()
    print("\nDone! Domain Graph stored in Cosmos DB (indigokg / knowledgegraph).")
    print("Query with Gremlin: g.V().hasLabel('DomainEntity').valueMap(true)")


if __name__ == "__main__":
    main()
