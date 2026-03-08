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

# ─── Ensure project root is on sys.path for utils imports ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # one level up from domain_graph/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.llm import get_llm_client, call_llm, parse_llm_json
from utils.neo4j_helpers import get_neo4j_driver, run_cypher, run_cypher_write

# ─── Configuration ───────────────────────────────────────────────────────────
DB_PATH = os.path.join(PROJECT_ROOT, "data", "manufacturing.db")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 0 — LLM Client & Neo4j helpers imported from utils/
#  get_llm_client(), call_llm(), parse_llm_json()  → utils.llm
#  get_neo4j_driver(), run_cypher(), run_cypher_write() → utils.neo4j_helpers
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

        prompt = f"""You are a data architect analyzing a database schema for IndiGo Airlines.

Given the following table, provide a JSON response with exactly these fields:
- "description": A 1-2 sentence summary of what this table represents and its business purpose.
- "domain": A single domain category label (e.g., "fleet_management", "supply_chain", "maintenance").
- "semantic_relationships": An array of objects, each with:
    - "target_table": name of a related table from this list: {all_table_names}
    - "relationship_type": a semantic edge label in UPPER_SNAKE_CASE (e.g., PART_OF, SUPPLIES, BELONGS_TO)
    - "reason": brief explanation of why this relationship exists
- "concepts": An array of 2-5 abstract, high-level concepts that this table encapsulates. Each with:
    - "name": A concise concept label (e.g., "Aircraft Fleet Composition", "Supplier Relationship", "Component Lifecycle")
    - "description": 1 sentence explaining what this concept represents in the business domain
    - "derived_from": array of column names from this table that inform this concept
  IMPORTANT: Concepts should be ABSTRACT business ideas, NOT raw column values or column names.
  Think about what business knowledge this table captures at a higher level.
  Examples: from a 'parts' table with columns (part_name, weight_kg, material, supplier_id) you might derive
  concepts like "Component Specification", "Material Classification", "Supplier Relationship".

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

    Similar concepts from different tables (e.g., "Supplier Relationship" from
    parts and "Vendor Management" from suppliers) are merged into a single
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

    prompt = f"""You are a senior data architect reviewing abstract concepts extracted from database tables for IndiGo Airlines.

Below is a list of concepts, each attributed to a source table. Your tasks:

TASK 1 — MERGE GROUPS: Identify concepts from DIFFERENT tables that represent the same
abstract idea (even if named differently). Group them and pick the best canonical name.
Only merge if they truly represent the same concept — do not over-merge.

TASK 2 — CROSS-LINKS: Identify pairs of concepts that are DISTINCT but semantically
related (e.g., "Aircraft Component" is related to "Assembly Structure" via COMPOSED_OF).
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

def build_domain_graph(schema: dict, enriched: dict, driver=None,
                       normalized_concepts: list[dict] = None,
                       cross_links: list[dict] = None):
    """
    Build the Domain Graph in Neo4j:
      - DomainEntity nodes (one per DB table — stores metadata, NOT row data)
      - HAS_FK relationships for foreign key edges
      - Semantic relationships (PART_OF, SUPPLIES, etc.) from LLM enrichment
      - Concept nodes (abstract business concepts derived from table columns)
      - HAS_CONCEPT edges (DomainEntity → Concept)
      - RELATED_CONCEPT edges (Concept → Concept cross-links)
    """
    if driver is None:
        driver = get_neo4j_driver()

    normalized_concepts = normalized_concepts or []
    cross_links = cross_links or []

    # ── Clean slate — drop all existing Domain Graph data ────────────
    print("  Clearing existing Domain Graph data...")
    run_cypher_write(driver, "MATCH (n:DomainEntity) DETACH DELETE n")
    run_cypher_write(driver, "MATCH (n:Concept) DETACH DELETE n")

    # ── Create constraints & indexes ─────────────────────────────────
    try:
        run_cypher(driver,
            "CREATE CONSTRAINT domain_entity_name IF NOT EXISTS "
            "FOR (d:DomainEntity) REQUIRE d.name IS UNIQUE"
        )
        print("  Created uniqueness constraint on DomainEntity.name")
    except Exception as e:
        print(f"  Constraint already exists or skipped: {e}")

    try:
        run_cypher(driver,
            "CREATE CONSTRAINT concept_name IF NOT EXISTS "
            "FOR (c:Concept) REQUIRE c.name IS UNIQUE"
        )
        print("  Created uniqueness constraint on Concept.name")
    except Exception as e:
        print(f"  Constraint already exists or skipped: {e}")

    # ── Insert nodes ─────────────────────────────────────────────────
    print("\n  Inserting nodes...")
    for table_name, info in schema.items():
        enr = enriched.get(table_name, {})
        description = enr.get("description", f"Table: {table_name}")
        domain = enr.get("domain", "unknown")
        key_cols = json.dumps(info["primary_keys"])
        col_info = json.dumps(
            [{"name": c["name"], "type": c["type"]} for c in info["columns"]]
        )

        run_cypher_write(driver, """
            CREATE (d:DomainEntity {
                name: $name,
                description: $description,
                key_columns: $key_columns,
                column_info: $column_info,
                row_count: $row_count,
                domain: $domain,
                source_type: 'structured_db'
            })
        """, {
            "name": table_name,
            "description": description,
            "key_columns": key_cols,
            "column_info": col_info,
            "row_count": info["row_count"],
            "domain": domain,
        })
        print(f"    + {table_name} (domain: {domain})")

    # ── Insert FK-based edges (HAS_FK) ──────────────────────────────
    print("\n  Inserting FK edges...")
    for table_name, info in schema.items():
        for fk in info["foreign_keys"]:
            target = fk["to_table"]
            reason = f"FK: {table_name}.{fk['from_col']} → {target}.{fk['to_col']}"
            try:
                run_cypher_write(driver, """
                    MATCH (a:DomainEntity {name: $from_name})
                    MATCH (b:DomainEntity {name: $to_name})
                    CREATE (a)-[:HAS_FK {reason: $reason}]->(b)
                """, {
                    "from_name": table_name,
                    "to_name": target,
                    "reason": reason,
                })
                print(f"    + {table_name} —[HAS_FK]→ {target}")
            except Exception as e:
                print(f"    WARN: FK edge {table_name}→{target} failed: {e}")

    # ── Insert semantic edges (from LLM enrichment) ──────────────────
    print("\n  Inserting semantic edges...")
    for table_name, enr in enriched.items():
        for rel in enr.get("semantic_relationships", []):
            rel_type = rel["relationship_type"]
            target = rel["target_table"]
            reason = rel.get("reason", "")

            # Skip if target table doesn't exist in schema
            if target not in schema:
                print(f"    SKIP: {table_name} —[{rel_type}]→ {target} (target not in schema)")
                continue

            # Skip HAS_FK duplicates (already inserted above)
            if rel_type == "HAS_FK":
                continue

            # Sanitize rel_type for Cypher (only alphanumeric + underscore)
            safe_rel_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel_type)

            try:
                # Neo4j doesn't allow parameterized relationship types,
                # so we use f-string for the type but parameters for data
                run_cypher_write(driver, f"""
                    MATCH (a:DomainEntity {{name: $from_name}})
                    MATCH (b:DomainEntity {{name: $to_name}})
                    CREATE (a)-[:{safe_rel_type} {{reason: $reason}}]->(b)
                """, {
                    "from_name": table_name,
                    "to_name": target,
                    "reason": reason,
                })
                print(f"    + {table_name} —[{safe_rel_type}]→ {target}")
            except Exception as e:
                print(f"    WARN: Semantic edge {table_name}—[{safe_rel_type}]→{target} failed: {e}")

    # ── Insert Concept nodes ────────────────────────────────────────
    if normalized_concepts:
        print(f"\n  Inserting {len(normalized_concepts)} Concept nodes...")
        for nc in normalized_concepts:
            source_tables_json = json.dumps(nc["source_tables"])
            derived_from_json = json.dumps(nc.get("derived_from", {}))
            try:
                run_cypher_write(driver, """
                    CREATE (c:Concept {
                        name: $name,
                        description: $description,
                        source_tables: $source_tables,
                        derived_from: $derived_from,
                        shared: $shared
                    })
                """, {
                    "name": nc["name"],
                    "description": nc["description"],
                    "source_tables": source_tables_json,
                    "derived_from": derived_from_json,
                    "shared": len(nc["source_tables"]) > 1,
                })
                shared_tag = " [SHARED]" if len(nc["source_tables"]) > 1 else ""
                print(f"    + {nc['name']}{shared_tag}")
            except Exception as e:
                print(f"    WARN: Concept node '{nc['name']}' failed: {e}")

        # ── Insert HAS_CONCEPT edges (DomainEntity → Concept) ────────
        print("\n  Inserting HAS_CONCEPT edges...")
        for nc in normalized_concepts:
            for table in nc["source_tables"]:
                cols = nc.get("derived_from", {}).get(table, [])
                derived_str = ", ".join(cols) if cols else "inferred"
                try:
                    run_cypher_write(driver, """
                        MATCH (d:DomainEntity {name: $table_name})
                        MATCH (c:Concept {name: $concept_name})
                        CREATE (d)-[:HAS_CONCEPT {derived_from: $derived_from}]->(c)
                    """, {
                        "table_name": table,
                        "concept_name": nc["name"],
                        "derived_from": derived_str,
                    })
                    print(f"    + {table} —[HAS_CONCEPT]→ {nc['name']}")
                except Exception as e:
                    print(f"    WARN: HAS_CONCEPT {table}→{nc['name']} failed: {e}")

    # ── Insert RELATED_CONCEPT edges (cross-links) ───────────────────
    if cross_links:
        print(f"\n  Inserting {len(cross_links)} RELATED_CONCEPT edges...")
        for link in cross_links:
            safe_type = "".join(
                c if c.isalnum() or c == "_" else "_"
                for c in link.get("relationship_type", "RELATED_TO")
            )
            try:
                run_cypher_write(driver, f"""
                    MATCH (a:Concept {{name: $concept_a}})
                    MATCH (b:Concept {{name: $concept_b}})
                    CREATE (a)-[:RELATED_CONCEPT {{
                        relationship_type: $rel_type,
                        reason: $reason
                    }}]->(b)
                """, {
                    "concept_a": link["concept_a"],
                    "concept_b": link["concept_b"],
                    "rel_type": link["relationship_type"],
                    "reason": link.get("reason", ""),
                })
                print(f"    + {link['concept_a']} —[RELATED_CONCEPT/{link['relationship_type']}]→ {link['concept_b']}")
            except Exception as e:
                print(f"    WARN: RELATED_CONCEPT edge failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    n_concepts = len(normalized_concepts)
    n_shared = sum(1 for nc in normalized_concepts if len(nc["source_tables"]) > 1)
    print(f"\n  Domain Graph built in Neo4j!")
    print(f"  • {len(schema)} DomainEntity nodes")
    print(f"  • {n_concepts} Concept nodes ({n_shared} shared across tables)")
    print(f"  • {len(cross_links)} concept cross-links")
    print(f"  View at: http://localhost:7474")
    print(f"  Try: MATCH (n:DomainEntity)-[:HAS_CONCEPT]->(c:Concept) RETURN n, c")
    return driver


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Query Interface
# ═══════════════════════════════════════════════════════════════════════════════

def query_domain_graph(question: str, driver=None) -> list[dict]:
    """
    Simple agent-facing query: find DomainEntity nodes whose name or description
    matches keywords from the question. Returns matching nodes + their relationships.

    For POC: uses keyword CONTAINS matching on name and description fields.
    For production: would use embedding-based vector similarity search.
    """
    if driver is None:
        driver = get_neo4j_driver()

    # Extract simple keywords (lowercase, skip short/common words)
    stop_words = {"which", "what", "about", "have", "does", "that", "this",
                  "with", "from", "tell", "show", "find", "data", "tables", "related"}
    keywords = [
        w.strip("?.,!\"'").lower()
        for w in question.split()
        if len(w.strip("?.,!\"'")) > 3 and w.strip("?.,!\"'").lower() not in stop_words
    ]

    results = []

    # Search DomainEntity nodes by keyword match on name or description
    for kw in keywords:
        records = run_cypher(driver, """
            MATCH (n:DomainEntity)
            WHERE toLower(n.name) CONTAINS $kw OR toLower(n.description) CONTAINS $kw
            RETURN n.name AS name, n.description AS description,
                   n.domain AS domain, n.key_columns AS key_columns,
                   n.row_count AS row_count
        """, {"kw": kw})

        for rec in records:
            if not any(r["name"] == rec["name"] for r in results):
                rec["matched_keyword"] = kw
                results.append(rec)

    # Also search Concept nodes and pull in their parent DomainEntities
    concept_results = []
    for kw in keywords:
        records = run_cypher(driver, """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS $kw OR toLower(c.description) CONTAINS $kw
            OPTIONAL MATCH (d:DomainEntity)-[:HAS_CONCEPT]->(c)
            RETURN c.name AS concept_name, c.description AS concept_desc,
                   c.source_tables AS source_tables, c.shared AS shared,
                   collect(d.name) AS parent_entities
        """, {"kw": kw})
        for rec in records:
            if not any(cr["concept_name"] == rec["concept_name"] for cr in concept_results):
                rec["matched_keyword"] = kw
                concept_results.append(rec)
            # Also ensure the parent DomainEntities appear in results
            for parent in rec.get("parent_entities", []):
                if parent and not any(r["name"] == parent for r in results):
                    parent_records = run_cypher(driver, """
                        MATCH (n:DomainEntity {name: $name})
                        RETURN n.name AS name, n.description AS description,
                               n.domain AS domain, n.key_columns AS key_columns,
                               n.row_count AS row_count
                    """, {"name": parent})
                    for pr in parent_records:
                        pr["matched_keyword"] = f"{kw} (via concept '{rec['concept_name']}')"
                        results.append(pr)

    # For each matched node, fetch outgoing + incoming relationships + concepts
    for node in results:
        node["relationships"] = []
        node["concepts"] = []

        # Outgoing edges to DomainEntity
        out_records = run_cypher(driver, """
            MATCH (a:DomainEntity {name: $name})-[r]->(b:DomainEntity)
            RETURN type(r) AS rel_type, b.name AS target, r.reason AS reason
        """, {"name": node["name"]})
        for rec in out_records:
            node["relationships"].append({
                "direction": "outgoing",
                "type": rec["rel_type"],
                "target": rec["target"],
                "reason": rec["reason"],
            })

        # Incoming edges from DomainEntity
        in_records = run_cypher(driver, """
            MATCH (a:DomainEntity)-[r]->(b:DomainEntity {name: $name})
            RETURN type(r) AS rel_type, a.name AS source, r.reason AS reason
        """, {"name": node["name"]})
        for rec in in_records:
            node["relationships"].append({
                "direction": "incoming",
                "type": rec["rel_type"],
                "source": rec["source"],
                "reason": rec["reason"],
            })

        # Concepts linked via HAS_CONCEPT
        concept_records = run_cypher(driver, """
            MATCH (d:DomainEntity {name: $name})-[:HAS_CONCEPT]->(c:Concept)
            OPTIONAL MATCH (c)-[rc:RELATED_CONCEPT]->(c2:Concept)
            RETURN c.name AS concept_name, c.description AS concept_desc,
                   c.shared AS shared,
                   collect(DISTINCT {related: c2.name, type: rc.relationship_type}) AS related_concepts
        """, {"name": node["name"]})
        for rec in concept_records:
            related = [r for r in rec.get("related_concepts", []) if r.get("related")]
            node["concepts"].append({
                "name": rec["concept_name"],
                "description": rec["concept_desc"],
                "shared": rec.get("shared", False),
                "related_concepts": related,
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

def visualize_graph(driver=None):
    """Print all nodes and edges in the Domain Graph."""
    if driver is None:
        driver = get_neo4j_driver()

    print(f"\n{'=' * 70}")
    print("  DOMAIN GRAPH — ALL NODES & EDGES")
    print(f"{'=' * 70}")

    # DomainEntity nodes
    print("\n  DomainEntity Nodes:")
    records = run_cypher(driver, """
        MATCH (n:DomainEntity)
        RETURN n.name AS name, n.description AS description,
               n.domain AS domain, n.row_count AS row_count
        ORDER BY n.name
    """)
    for rec in records:
        print(f"    [{rec['domain']}] {rec['name']} — {rec['description']} ({rec['row_count']} rows)")

    # Concept nodes
    print("\n  Concept Nodes:")
    concept_records = run_cypher(driver, """
        MATCH (c:Concept)
        RETURN c.name AS name, c.description AS description,
               c.source_tables AS source_tables, c.shared AS shared
        ORDER BY c.name
    """)
    if not concept_records:
        print("    (none)")
    for rec in concept_records:
        shared_tag = " [SHARED]" if rec.get("shared") else ""
        print(f"    {rec['name']}{shared_tag} — {rec['description']}")
        print(f"      Source tables: {rec['source_tables']}")

    # DomainEntity edges
    print("\n  DomainEntity Edges:")
    records = run_cypher(driver, """
        MATCH (a:DomainEntity)-[r]->(b:DomainEntity)
        RETURN a.name AS from_node, type(r) AS rel_type,
               b.name AS to_node, r.reason AS reason
        ORDER BY a.name
    """)
    for rec in records:
        print(f"    {rec['from_node']} —[{rec['rel_type']}]→ {rec['to_node']}  ({rec['reason']})")

    # HAS_CONCEPT edges
    print("\n  HAS_CONCEPT Edges:")
    hc_records = run_cypher(driver, """
        MATCH (d:DomainEntity)-[r:HAS_CONCEPT]->(c:Concept)
        RETURN d.name AS entity, c.name AS concept, r.derived_from AS derived_from
        ORDER BY d.name, c.name
    """)
    if not hc_records:
        print("    (none)")
    for rec in hc_records:
        print(f"    {rec['entity']} —[HAS_CONCEPT]→ {rec['concept']}  (derived from: {rec['derived_from']})")

    # RELATED_CONCEPT edges
    print("\n  RELATED_CONCEPT Edges:")
    rc_records = run_cypher(driver, """
        MATCH (a:Concept)-[r:RELATED_CONCEPT]->(b:Concept)
        RETURN a.name AS from_concept, b.name AS to_concept,
               r.relationship_type AS rel_type, r.reason AS reason
        ORDER BY a.name
    """)
    if not rc_records:
        print("    (none)")
    for rec in rc_records:
        print(f"    {rec['from_concept']} —[{rec['rel_type']}]→ {rec['to_concept']}  ({rec['reason']})")

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
    print("  IndiGo Airlines Fleet / Supply Chain Ontology")
    print(f"  Enrichment mode: {mode}")
    print("=" * 70)

    # Step 1: Introspect
    print("\n[Step 1] Introspecting SQLite schema...")
    if not os.path.exists(DB_PATH):
        print(f"  ERROR: Database not found at {DB_PATH}")
        print("  Run 'python data/setup_db.py' first to create the sample database.")
        return
    schema = introspect_sqlite(DB_PATH)
    print_schema(schema)

    # Step 2: LLM Enrichment
    client = get_llm_client()
    if args.advanced:
        print("\n[Step 2] Enriching with ReAct Agent (iterative exploration)...")
        from enrich_advanced import enrich_with_llm_advanced
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
    print("\n[Step 3] Building Domain Graph in Neo4j...")
    driver = get_neo4j_driver()
    build_domain_graph(schema, enriched, driver,
                       normalized_concepts=normalized_concepts,
                       cross_links=cross_links)

    # Step 4: Visualize
    visualize_graph(driver)

    # Step 5: Demo Queries
    demo_questions = [
        "Which suppliers provide parts?",
        "What data do we have about engines?",
        "Tell me about the landing gear",
        "Which tables are related to assemblies?",
    ]
    print("\n\n[Step 5] Running demo queries...")
    for q in demo_questions:
        results, concept_results = query_domain_graph(q, driver)
        print_query_results(q, results, concept_results)

    driver.close()
    print("\nDone! View your graph at http://localhost:7474")
    print("Try: MATCH (n:DomainEntity)-[r]->(m) RETURN n, r, m")


if __name__ == "__main__":
    main()
