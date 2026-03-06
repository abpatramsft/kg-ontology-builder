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
            print(f"OK — {parsed.get('domain', 'unknown')}")
        except json.JSONDecodeError as e:
            print(f"WARN: JSON parse failed ({e}), using fallback")
            enriched[table_name] = {
                "description": f"Database table '{table_name}' with {info['row_count']} rows.",
                "domain": "unknown",
                "semantic_relationships": [],
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
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Build Domain Graph in Neo4j
# ═══════════════════════════════════════════════════════════════════════════════

def build_domain_graph(schema: dict, enriched: dict, driver=None):
    """
    Build the Domain Graph in Neo4j:
      - DomainEntity nodes (one per DB table — stores metadata, NOT row data)
      - HAS_FK relationships for foreign key edges
      - Semantic relationships (PART_OF, SUPPLIES, etc.) from LLM enrichment
    """
    if driver is None:
        driver = get_neo4j_driver()

    # ── Clean slate — drop all existing Domain Graph data ────────────
    print("  Clearing existing Domain Graph data...")
    run_cypher_write(driver, "MATCH (n:DomainEntity) DETACH DELETE n")

    # ── Create constraints & indexes ─────────────────────────────────
    try:
        run_cypher(driver,
            "CREATE CONSTRAINT domain_entity_name IF NOT EXISTS "
            "FOR (d:DomainEntity) REQUIRE d.name IS UNIQUE"
        )
        print("  Created uniqueness constraint on DomainEntity.name")
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

    print("\n  Domain Graph built in Neo4j!")
    print(f"  View at: http://localhost:7474  (run: MATCH (n:DomainEntity) RETURN n)")
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

    # Search nodes by keyword match on name or description
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

    # For each matched node, fetch outgoing + incoming relationships
    for node in results:
        node["relationships"] = []

        # Outgoing
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

        # Incoming
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

    return results


def print_query_results(question: str, results: list[dict]):
    """Pretty-print query results."""
    print(f"\n{'=' * 70}")
    print(f"  QUERY: \"{question}\"")
    print(f"{'=' * 70}")
    if not results:
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

    # All nodes
    print("\n  Nodes:")
    records = run_cypher(driver, """
        MATCH (n:DomainEntity)
        RETURN n.name AS name, n.description AS description,
               n.domain AS domain, n.row_count AS row_count
        ORDER BY n.name
    """)
    for rec in records:
        print(f"    [{rec['domain']}] {rec['name']} — {rec['description']} ({rec['row_count']} rows)")

    # All edges
    print("\n  Edges:")
    records = run_cypher(driver, """
        MATCH (a:DomainEntity)-[r]->(b:DomainEntity)
        RETURN a.name AS from_node, type(r) AS rel_type,
               b.name AS to_node, r.reason AS reason
        ORDER BY a.name
    """)
    for rec in records:
        print(f"    {rec['from_node']} —[{rec['rel_type']}]→ {rec['to_node']}  ({rec['reason']})")

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

    # Step 3: Build Graph
    print("\n[Step 3] Building Domain Graph in Neo4j...")
    driver = get_neo4j_driver()
    build_domain_graph(schema, enriched, driver)

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
        results = query_domain_graph(q, driver)
        print_query_results(q, results)

    driver.close()
    print("\nDone! View your graph at http://localhost:7474")
    print("Try: MATCH (n:DomainEntity)-[r]->(m) RETURN n, r, m")


if __name__ == "__main__":
    main()
