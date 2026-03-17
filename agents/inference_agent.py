"""
agent_inference.py — ReAct Inference Agent over the Unified Knowledge Graph

A conversational ReAct agent that answers user questions by navigating the
unified 3-layer knowledge graph (Domain Graph, Lexical Graph, Subject Graph),
running vector similarity searches against the LanceDB document store, and
executing SQL queries against the structured database — all guided by the
ontology it discovers from the graph.

Architecture:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                       InferenceAgent                                     │
  │                                                                          │
  │  THOUGHT ──► ACTION (pick one of 3 tools) ──► OBSERVE                    │
  │     ▲                                              │                     │
  │     └──────────────────────────────────────────────┘                     │
  │                                                                          │
  │  Tools:                                                                  │
  │    1. graph_ontology_tool  — Navigate the Neo4j KG ontology              │
  │    2. vector_search_tool   — Semantic / filtered search in LanceDB       │
  │    3. sql_query_tool       — Run SQL against the structured database     │
  │                                                                          │
  │  The agent's workflow:                                                   │
  │    a) Explore the ontology to understand what data exists & how it       │
  │       connects  (graph_ontology_tool)                                    │
  │    b) Use that knowledge to search relevant documents                    │
  │       (vector_search_tool) and/or query structured tables                │
  │       (sql_query_tool)                                                   │
  │    c) Synthesize findings into a final answer                            │
  └──────────────────────────────────────────────────────────────────────────┘

Usage:
    python agent_inference.py
    # Opens an interactive REPL. Ask questions about IndiGo Airlines data.

    # Or programmatic usage:
    from agents.inference_agent import InferenceAgent, build_tools
    tools = build_tools(gremlin_client, lance_table, embedding_client, db_path)
    agent = InferenceAgent(llm_client, tools)
    answer = agent.run("Which suppliers provide parts for the A320neo landing gear?")
"""

import json
import os
import re
import sqlite3
import sys
import textwrap

from openai import AzureOpenAI

# Ensure src/ is on sys.path for utils imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))    # agents/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                  # project root
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.llm import get_llm_client, get_embedding_client, embed_texts
from utils.cosmos_helpers import (
    get_gremlin_client, run_gremlin,
    esc, make_vertex_id, gval,
)

import lancedb


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 1: Graph Ontology Tool (Cosmos DB Gremlin)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphOntologyTool:
    """
    Read-only tool to explore the unified 3-layer knowledge graph in Cosmos DB
    (Gremlin API, database: indigokg, container: knowledgegraph).

    Gives the agent a map of the data landscape: what domain entities
    (structured tables) exist, what subjects were extracted from documents,
    how they correspond, and what documents mention which subjects.

    Supported actions:
      - list_node_labels()                → all vertex labels in the graph
      - list_relationship_types()         → all edge labels
      - list_domain_entities()            → all DomainEntity vertices
      - list_subjects()                   → all Subject vertices + SPO triplets
      - list_documents()                  → all Document vertices
      - get_domain_entity_detail(name)    → entity + relationships + column info
      - get_subject_context(name)         → subject + documents + SPO triplets
      - get_correspondences(name)         → CORRESPONDS_TO links for a subject
      - find_path(from_name, to_name)     → shortest path between two nodes
      - query_graph(gremlin)              → arbitrary read-only Gremlin traversal
    """

    NAME = "graph_ontology_tool"
    TOOL_DESCRIPTION = textwrap.dedent("""\
    graph_ontology_tool — Read-only access to the unified Cosmos DB knowledge graph
    (Domain Graph + Lexical Graph + Subject Graph, Gremlin API).

    Use this tool FIRST to understand the data landscape before querying data.

    Graph model (all vertices in database 'indigokg', container 'knowledgegraph'):
    - DomainEntity vertices (structured DB tables) connected by HAS_FK/SEMANTIC edges
    - Concept vertices connected to DomainEntity via HAS_CONCEPT; cross-linked via RELATED_CONCEPT
    - Document vertices → Subject vertices (via MENTIONS edges)
    - Subject vertices → Object vertices (via RELATES_TO {predicate} edges)
    - Subject → DomainEntity (via CORRESPONDS_TO) bridges unstructured ↔ structured data

    Available actions (pass as JSON):

    1. {"action": "list_node_labels"}
       Returns: all vertex labels (DomainEntity, Concept, Document, Subject, Object).

    2. {"action": "list_relationship_types"}
       Returns: all edge labels (HAS_FK, HAS_CONCEPT, RELATED_CONCEPT, MENTIONS, RELATES_TO, CORRESPONDS_TO, ...).

    3. {"action": "list_domain_entities"}
       Returns: all DomainEntity vertices — structured DB tables.
       Each has: name, description, domain, key_columns, row_count.

    4. {"action": "list_subjects"}
       Returns: all Subject vertices with SPO triplets.
       Each has: name, type, description, mention_count, spo_triplets.

    5. {"action": "list_documents"}
       Returns: all Document vertices.
       Each has: name, topic_summary, subject_count.

    6. {"action": "get_domain_entity_detail", "name": "<entity_name>"}
       Returns: full detail for a DomainEntity — description, column info,
       relationships to other entities, linked Concept vertices, corresponding subjects.

    7. {"action": "get_subject_context", "name": "<subject_name>"}
       Returns: Subject vertex + documents that mention it + SPO triplets + corresponding entities.

    8. {"action": "get_correspondences", "name": "<subject_name>"}
       Returns: all CORRESPONDS_TO edges from a Subject to DomainEntity vertices.

    9. {"action": "find_path", "from_name": "<vertex_name>", "to_name": "<vertex_name>"}
       Returns: path between two named vertices in the graph.

    10. {"action": "query_graph", "gremlin": "<Gremlin traversal>"}
        Runs a read-only Gremlin traversal. Only g.V() / g.E() reads are allowed.
        Example: g.V().hasLabel('DomainEntity').values('name')
        Example: g.V().hasLabel('Subject').out('CORRESPONDS_TO').values('name')
    """)

    def __init__(self, client):
        self.client = client

    def execute(self, action_input: dict) -> str:
        action = action_input.get("action", "").strip().lower()
        try:
            dispatch = {
                "list_node_labels":        self._list_node_labels,
                "list_relationship_types": self._list_relationship_types,
                "list_domain_entities":    self._list_domain_entities,
                "list_subjects":           self._list_subjects,
                "list_documents":          self._list_documents,
                "get_domain_entity_detail": lambda: self._get_domain_entity_detail(action_input["name"]),
                "get_subject_context":      lambda: self._get_subject_context(action_input["name"]),
                "get_correspondences":      lambda: self._get_correspondences(action_input["name"]),
                "find_path":               lambda: self._find_path(
                    action_input["from_name"], action_input["to_name"]
                ),
                "query_graph":             lambda: self._query_graph(action_input["gremlin"]),
            }
            fn = dispatch.get(action)
            if fn is None:
                return (
                    f"ERROR: Unknown action '{action}'. Available: "
                    + ", ".join(dispatch.keys())
                )
            return fn() if callable(fn) else fn
        except KeyError as e:
            return f"ERROR: Missing required parameter: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    # ── Action Implementations ────────────────────────────────────────

    def _list_node_labels(self) -> str:
        labels = run_gremlin(self.client, "g.V().label().dedup()")
        return json.dumps({"vertex_labels": sorted(labels)}, indent=2)

    def _list_relationship_types(self) -> str:
        types = run_gremlin(self.client, "g.E().label().dedup()")
        return json.dumps({"edge_labels": sorted(types)}, indent=2)

    def _list_domain_entities(self) -> str:
        records = run_gremlin(self.client,
            "g.V().hasLabel('DomainEntity')"
            ".valueMap('name','description','domain','key_columns','row_count')"
        )
        entities = [
            {
                "name": gval(r, "name", ""),
                "description": gval(r, "description", ""),
                "domain": gval(r, "domain", ""),
                "key_columns": gval(r, "key_columns", "[]"),
                "row_count": gval(r, "row_count", 0),
            }
            for r in records
        ]
        return json.dumps({"domain_entities": entities, "count": len(entities)}, indent=2)

    def _list_subjects(self) -> str:
        subj_records = run_gremlin(self.client,
            "g.V().hasLabel('Subject')"
            ".valueMap('name','type','description','mention_count')"
        )
        subjects = []
        for r in subj_records:
            name = gval(r, "name", "")
            sid = make_vertex_id("Subject", name)
            spo = run_gremlin(self.client,
                f"g.V('{esc(sid)}').outE('RELATES_TO')"
                ".project('predicate','object','object_type')"
                ".by(values('predicate'))"
                ".by(inV().values('name'))"
                ".by(inV().coalesce(values('type'), constant('')))"
            )
            subjects.append({
                "name": name,
                "type": gval(r, "type", ""),
                "description": gval(r, "description", ""),
                "mention_count": gval(r, "mention_count", 0),
                "spo_triplets": spo,
            })
        return json.dumps({"subjects": subjects, "count": len(subjects)}, indent=2)

    def _list_documents(self) -> str:
        records = run_gremlin(self.client,
            "g.V().hasLabel('Document').valueMap('name','topic_summary','chunk_count')"
        )
        docs = []
        for r in records:
            name = gval(r, "name", "")
            did = make_vertex_id("Document", name)
            subject_count_result = run_gremlin(self.client,
                f"g.V('{esc(did)}').out('MENTIONS').count()"
            )
            subject_count = subject_count_result[0] if subject_count_result else 0
            docs.append({
                "name": name,
                "topic_summary": gval(r, "topic_summary", ""),
                "subject_count": subject_count,
            })
        return json.dumps({"documents": docs, "count": len(docs)}, indent=2)

    def _get_domain_entity_detail(self, name: str) -> str:
        # Try exact ID lookup first, then scan
        vid = make_vertex_id("DomainEntity", name)
        de_recs = run_gremlin(self.client,
            f"g.V('{esc(vid)}').valueMap('name','description','domain','key_columns','column_info','row_count')"
        )
        if not de_recs:
            # Fallback: scan all DomainEntity vertices for name match
            all_de = run_gremlin(self.client,
                "g.V().hasLabel('DomainEntity').valueMap('name','description','domain','key_columns','column_info','row_count')"
            )
            de_recs = [r for r in all_de if gval(r, "name", "").lower() == name.lower()]

        if not de_recs:
            return f"ERROR: DomainEntity '{name}' not found."

        r = de_recs[0]
        resolved_name = gval(r, "name", name)
        resolved_id = make_vertex_id("DomainEntity", resolved_name)

        entity = {
            "name": resolved_name,
            "description": gval(r, "description", ""),
            "domain": gval(r, "domain", ""),
            "key_columns": gval(r, "key_columns", "[]"),
            "column_info": gval(r, "column_info", "[]"),
            "row_count": gval(r, "row_count", 0),
        }

        out_rels = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').outE()"
            ".project('rel_type','target','target_label','reason')"
            ".by(label())"
            ".by(inV().values('name'))"
            ".by(inV().label())"
            ".by(coalesce(values('reason'), constant('')))"
        )

        in_rels = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').inE()"
            ".project('rel_type','source','source_label','reason')"
            ".by(label())"
            ".by(outV().values('name'))"
            ".by(outV().label())"
            ".by(coalesce(values('reason'), constant('')))"
        )

        correspondences = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').inE('CORRESPONDS_TO')"
            ".project('subject','subject_type','confidence','reason')"
            ".by(outV().values('name'))"
            ".by(outV().coalesce(values('type'), constant('')))"
            ".by(values('confidence'))"
            ".by(coalesce(values('reason'), constant('')))"
        )

        concept_recs = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').out('HAS_CONCEPT')"
            ".valueMap('name','description','shared')"
        )
        concepts = []
        for cr in concept_recs:
            c_name = gval(cr, "name", "")
            cid = make_vertex_id("Concept", c_name)
            related = run_gremlin(self.client,
                f"g.V('{esc(cid)}').outE('RELATED_CONCEPT')"
                ".project('related','type','reason')"
                ".by(inV().values('name'))"
                ".by(coalesce(values('relationship_type'), constant('')))"
                ".by(coalesce(values('reason'), constant('')))"
            )
            concepts.append({
                "concept_name": c_name,
                "concept_desc": gval(cr, "description", ""),
                "shared": gval(cr, "shared", False),
                "related_concepts": related,
            })

        return json.dumps({
            "entity": entity,
            "outgoing_relationships": out_rels,
            "incoming_relationships": in_rels,
            "corresponding_subjects": correspondences,
            "concepts": concepts,
        }, indent=2)

    def _get_subject_context(self, name: str) -> str:
        sid = make_vertex_id("Subject", name)
        subj_recs = run_gremlin(self.client,
            f"g.V('{esc(sid)}').valueMap('name','type','description','mention_count')"
        )
        if not subj_recs:
            # Fallback: scan
            all_s = run_gremlin(self.client,
                "g.V().hasLabel('Subject').valueMap('name','type','description','mention_count')"
            )
            subj_recs = [r for r in all_s if gval(r, "name", "").lower() == name.lower()]

        if not subj_recs:
            return f"ERROR: Subject '{name}' not found."

        r = subj_recs[0]
        resolved_name = gval(r, "name", name)
        resolved_id = make_vertex_id("Subject", resolved_name)

        subject = {
            "name": resolved_name,
            "type": gval(r, "type", ""),
            "description": gval(r, "description", ""),
            "mention_count": gval(r, "mention_count", 0),
        }

        docs = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').inE('MENTIONS')"
            ".project('document','topic_summary','mention_context')"
            ".by(outV().values('name'))"
            ".by(outV().coalesce(values('topic_summary'), constant('')))"
            ".by(coalesce(values('context'), constant('')))"
        )

        spo = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').outE('RELATES_TO')"
            ".project('predicate','object_name','object_type')"
            ".by(values('predicate'))"
            ".by(inV().values('name'))"
            ".by(inV().coalesce(values('type'), constant('')))"
        )

        correspondences = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').outE('CORRESPONDS_TO')"
            ".project('domain_entity','entity_description','confidence','reason')"
            ".by(inV().values('name'))"
            ".by(inV().coalesce(values('description'), constant('')))"
            ".by(values('confidence'))"
            ".by(coalesce(values('reason'), constant('')))"
        )

        return json.dumps({
            "subject": subject,
            "mentioned_in_documents": docs,
            "spo_triplets": spo,
            "corresponds_to_entities": correspondences,
        }, indent=2)

    def _get_correspondences(self, name: str) -> str:
        sid = make_vertex_id("Subject", name)
        records = run_gremlin(self.client,
            f"g.V('{esc(sid)}').outE('CORRESPONDS_TO')"
            ".project('subject','domain_entity','entity_description','key_columns','confidence','reason')"
            ".by(outV().values('name'))"
            ".by(inV().values('name'))"
            ".by(inV().coalesce(values('description'), constant('')))"
            ".by(inV().coalesce(values('key_columns'), constant('[]')))"
            ".by(values('confidence'))"
            ".by(coalesce(values('reason'), constant('')))"
        )
        if not records:
            return f"No CORRESPONDS_TO links found for subject '{name}'."
        return json.dumps({"correspondences": records, "count": len(records)}, indent=2)

    def _find_path(self, from_name: str, to_name: str) -> str:
        try:
            results = run_gremlin(self.client,
                f"g.V().has('name', '{esc(from_name)}')"
                f".repeat(bothE().otherV().simplePath())"
                f".until(has('name', '{esc(to_name)}')).limit(1)"
                f".path().by(project('name','label').by('name').by(label()))"
                f".by(project('type').by(label()))"
            )
            if not results:
                return f"No path found between '{from_name}' and '{to_name}'."
            return json.dumps({"paths": results}, indent=2)
        except Exception as e:
            return f"Path search failed: {e}"

    def _query_graph(self, gremlin_query: str) -> str:
        stripped = gremlin_query.strip().lower()
        write_keywords = ["addv(", "adde(", ".drop(", ".property(", "g.v().drop", "g.e().drop"]
        if any(kw in stripped for kw in write_keywords):
            return "ERROR: Only read traversals (g.V()..., g.E()...) are allowed."

        try:
            records = run_gremlin(self.client, gremlin_query)
            if len(records) > 30:
                records = records[:30]
                return json.dumps(records, indent=2) + "\n... (truncated to 30 results)"
            return json.dumps(records, indent=2)
        except Exception as e:
            return f"ERROR: Gremlin query failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 2: Vector Search Tool (LanceDB)
# ═══════════════════════════════════════════════════════════════════════════════

class VectorSearchTool:
    """
    Read-only tool for semantic and filtered searches over the LanceDB
    document chunk store.

    Supported actions:
      - search_similar(query, n)                → semantic similarity search
      - search_filtered(query, doc_name, n)     → similarity search within a document
      - get_chunk(chunk_id)                     → retrieve full chunk text
      - get_chunks_by_doc(doc_name)             → all chunks from a document
      - get_collection_stats()                  → table stats
    """

    NAME = "vector_search_tool"
    TOOL_DESCRIPTION = textwrap.dedent("""\
    vector_search_tool — Semantic & filtered search over the LanceDB document store.

    Use this tool to find relevant document passages after you've used
    graph_ontology_tool to identify which documents/subjects to look at.

    Available actions (pass as JSON):

    1. {"action": "search_similar", "query": "<natural language query>", "n": 5}
       Returns: top-N chunks most semantically similar to the query.
       Includes chunk_id, text, similarity score, and metadata.

    2. {"action": "search_filtered", "query": "<query>", "doc_name": "<document>", "n": 5}
       Returns: top-N similar chunks filtered to a specific document.
       Use this when the ontology told you which document to look at.

    3. {"action": "get_chunk", "chunk_id": "<chunk_id>"}
       Returns: full text and metadata for a specific chunk.

    4. {"action": "get_chunks_by_doc", "doc_name": "<document_name>"}
       Returns: all chunks from a specific document, ordered by index.

    5. {"action": "get_collection_stats"}
       Returns: total chunk count, document count, and table metadata.
    """)

    def __init__(self, table: lancedb.table.Table, embedding_client):
        self.table = table
        self.embedding_client = embedding_client

    def execute(self, action_input: dict) -> str:
        action = action_input.get("action", "").strip().lower()
        try:
            dispatch = {
                "search_similar":    lambda: self._search_similar(
                    action_input.get("query", ""), action_input.get("n", 5)
                ),
                "search_filtered":   lambda: self._search_filtered(
                    action_input.get("query", ""),
                    action_input.get("doc_name", ""),
                    action_input.get("n", 5),
                ),
                "get_chunk":         lambda: self._get_chunk(action_input["chunk_id"]),
                "get_chunks_by_doc": lambda: self._get_chunks_by_doc(action_input["doc_name"]),
                "get_collection_stats": self._get_collection_stats,
            }
            fn = dispatch.get(action)
            if fn is None:
                return (
                    f"ERROR: Unknown action '{action}'. Available: "
                    + ", ".join(dispatch.keys())
                )
            return fn() if callable(fn) else fn
        except KeyError as e:
            return f"ERROR: Missing required parameter: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    # ── Implementations ──────────────────────────────────────────────

    def _search_similar(self, query: str, n: int) -> str:
        if not query:
            return "ERROR: 'query' parameter is required."

        q_embedding = embed_texts(self.embedding_client, [query])[0]
        total = len(self.table)
        n = min(n, total)
        if n == 0:
            return json.dumps({"results": [], "message": "Table is empty"})

        results = (
            self.table.search(q_embedding)
            .metric("cosine")
            .limit(n)
            .to_list()
        )

        output = []
        for row in results:
            score = round(1 - row["_distance"], 4)
            output.append({
                "chunk_id": row["chunk_id"],
                "similarity_score": score,
                "text": row["text"],
                "metadata": {
                    "doc_name": row["doc_name"],
                    "chunk_index": row["chunk_index"],
                    "text_preview": row["text_preview"],
                    "char_count": row["char_count"],
                },
            })
        return json.dumps({"results": output}, indent=2)

    def _search_filtered(self, query: str, doc_name: str, n: int) -> str:
        if not query:
            return "ERROR: 'query' parameter is required."
        if not doc_name:
            return "ERROR: 'doc_name' parameter is required for filtered search."

        q_embedding = embed_texts(self.embedding_client, [query])[0]

        results = (
            self.table.search(q_embedding)
            .metric("cosine")
            .where(f"doc_name = '{doc_name}'", prefilter=True)
            .limit(n)
            .to_list()
        )

        output = []
        for row in results:
            score = round(1 - row["_distance"], 4)
            output.append({
                "chunk_id": row["chunk_id"],
                "similarity_score": score,
                "text": row["text"],
                "metadata": {
                    "doc_name": row["doc_name"],
                    "chunk_index": row["chunk_index"],
                },
            })
        return json.dumps({"results": output, "filter": f"doc_name='{doc_name}'"}, indent=2)

    def _get_chunk(self, chunk_id: str) -> str:
        df = self.table.to_pandas()
        match = df[df["chunk_id"] == chunk_id]
        if match.empty:
            return f"ERROR: Chunk '{chunk_id}' not found."

        row = match.iloc[0]
        return json.dumps({
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "metadata": {
                "doc_name": row["doc_name"],
                "chunk_index": int(row["chunk_index"]),
                "text_preview": row["text_preview"],
                "char_count": int(row["char_count"]),
            },
        }, indent=2)

    def _get_chunks_by_doc(self, doc_name: str) -> str:
        df = self.table.to_pandas()
        doc_df = df[df["doc_name"] == doc_name].sort_values("chunk_index")
        if doc_df.empty:
            return f"ERROR: No chunks found for document '{doc_name}'."

        chunks = []
        for _, row in doc_df.iterrows():
            chunks.append({
                "chunk_id": row["chunk_id"],
                "chunk_index": int(row["chunk_index"]),
                "text_preview": row["text"][:300],
                "char_count": int(row["char_count"]),
            })
        return json.dumps({"doc_name": doc_name, "chunks": chunks, "count": len(chunks)}, indent=2)

    def _get_collection_stats(self) -> str:
        df = self.table.to_pandas()
        doc_names = sorted(df["doc_name"].unique().tolist())
        return json.dumps({
            "total_chunks": len(df),
            "documents": len(doc_names),
            "document_names": doc_names,
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 3: SQL Query Tool (SQLite)
# ═══════════════════════════════════════════════════════════════════════════════

class SQLQueryTool:
    """
    Read-only tool for querying the structured SQLite database.

    Supported actions:
      - list_tables()                         → all table names
      - describe_table(table)                 → columns, types, PKs, FKs
      - sample_rows(table, limit)             → first N rows
      - query(sql)                            → arbitrary SELECT query
      - distinct_values(table, column, limit) → unique values in a column
    """

    NAME = "sql_query_tool"
    TOOL_DESCRIPTION = textwrap.dedent("""\
    sql_query_tool — Read-only access to the structured SQLite database.

    Use this tool AFTER the ontology has told you which tables are relevant.
    Run SQL queries to get actual data values that help answer the question.

    Available actions (pass as JSON):

    1. {"action": "list_tables"}
       Returns: all table names in the database.

    2. {"action": "describe_table", "table": "<table_name>"}
       Returns: columns, types, primary keys, foreign keys, row count.

    3. {"action": "sample_rows", "table": "<table_name>", "limit": 5}
       Returns: first N rows from the table (default 5).

    4. {"action": "query", "sql": "<SELECT ...>"}
       Runs an arbitrary read-only SQL query. ONLY SELECT statements are allowed.
       Use JOINs, WHERE, GROUP BY, ORDER BY, etc. as needed.

    5. {"action": "distinct_values", "table": "<table_name>", "column": "<col>", "limit": 20}
       Returns: distinct values in a column (up to limit).
    """)

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, action_input: dict) -> str:
        action = action_input.get("action", "").strip().lower()
        try:
            dispatch = {
                "list_tables":     self._list_tables,
                "describe_table":  lambda: self._describe_table(action_input["table"]),
                "sample_rows":     lambda: self._sample_rows(
                    action_input["table"], action_input.get("limit", 5)
                ),
                "query":           lambda: self._query(action_input["sql"]),
                "distinct_values": lambda: self._distinct_values(
                    action_input["table"], action_input["column"],
                    action_input.get("limit", 20),
                ),
            }
            fn = dispatch.get(action)
            if fn is None:
                return (
                    f"ERROR: Unknown action '{action}'. Available: "
                    + ", ".join(dispatch.keys())
                )
            return fn() if callable(fn) else fn
        except KeyError as e:
            return f"ERROR: Missing required parameter: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _list_tables(self) -> str:
        conn = self._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        conn.close()
        return json.dumps({"tables": [t["name"] for t in tables]}, indent=2)

    def _describe_table(self, table: str) -> str:
        conn = self._get_conn()
        cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        fks = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
        row_count = conn.execute(f"SELECT COUNT(*) as cnt FROM [{table}]").fetchone()["cnt"]
        conn.close()

        columns = [{"name": c["name"], "type": c["type"], "notnull": bool(c["notnull"]),
                     "primary_key": bool(c["pk"])} for c in cols]
        foreign_keys = [{"from_column": fk["from"], "to_table": fk["table"],
                         "to_column": fk["to"]} for fk in fks]
        return json.dumps({"table": table, "row_count": row_count,
                           "columns": columns, "foreign_keys": foreign_keys}, indent=2)

    def _sample_rows(self, table: str, limit: int) -> str:
        conn = self._get_conn()
        rows = conn.execute(f"SELECT * FROM [{table}] LIMIT {int(limit)}").fetchall()
        conn.close()
        return json.dumps([dict(r) for r in rows], indent=2)

    def _query(self, sql: str) -> str:
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return "ERROR: Only SELECT queries are allowed (read-only access)."

        conn = self._get_conn()
        try:
            rows = conn.execute(sql).fetchall()
            result = [dict(r) for r in rows]
            if len(result) > 50:
                result = result[:50]
                return json.dumps(result, indent=2) + "\n... (truncated to 50 rows)"
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"ERROR: SQL execution failed: {e}"
        finally:
            conn.close()

    def _distinct_values(self, table: str, column: str, limit: int) -> str:
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT DISTINCT [{column}] FROM [{table}] LIMIT {int(limit)}"
        ).fetchall()
        conn.close()
        values = [dict(r)[column] for r in rows]
        return json.dumps({"table": table, "column": column, "distinct_values": values}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  ReAct Inference Agent
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceAgent:
    """
    ReAct (Reason → Act → Observe) agent that answers user questions by
    navigating the knowledge graph ontology, searching document embeddings,
    and querying the structured database.

    This is a custom agent — no frameworks, no LangChain, no CrewAI.
    Just structured prompting + a parse-execute loop with 3 tools.
    """

    MAX_ITERATIONS = 15  # more generous than enrichment — inference may need more hops

    def __init__(
        self,
        llm_client: AzureOpenAI,
        tools: dict,
        verbose: bool = True,
    ):
        """
        Args:
            llm_client: AzureOpenAI client for chat completions.
            tools: dict mapping tool name → tool instance (each with .execute()).
                   Expected keys: "graph_ontology_tool", "vector_search_tool", "sql_query_tool"
            verbose: Print reasoning trace to stdout.
        """
        self.client = llm_client
        self.tools = tools
        self.verbose = verbose
        self.messages: list[dict] = []
        self.trace: list[dict] = []

    # ── Public API ───────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """
        Answer a user question using the ReAct loop.

        Args:
            question: The natural-language question to answer.

        Returns:
            str — the final answer text.
        """
        self.messages = []
        self.trace = []
        self._build_system_prompt()
        self._build_user_prompt(question)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  QUESTION: {question}")
            print(f"{'='*70}")

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            if self.verbose:
                print(f"\n  [Iteration {iteration}/{self.MAX_ITERATIONS}]")

            response_text = self._call_llm()
            parsed = self._parse_response(response_text)

            if parsed["type"] == "final_answer":
                if self.verbose:
                    print(f"  → FINAL ANSWER after {iteration} iteration(s)\n")
                return parsed["answer"]

            elif parsed["type"] == "action":
                thought = parsed.get("thought", "")
                tool_name = parsed["tool_name"]
                action_input = parsed["action_input"]

                if self.verbose:
                    print(f"  THOUGHT: {thought[:150]}{'...' if len(thought) > 150 else ''}")
                    print(f"  ACTION:  {tool_name} → {json.dumps(action_input)}")

                # Execute the tool
                tool = self.tools.get(tool_name)
                if tool is None:
                    observation = (
                        f"ERROR: Unknown tool '{tool_name}'. Available tools: "
                        + ", ".join(self.tools.keys())
                    )
                else:
                    observation = tool.execute(action_input)

                if self.verbose:
                    obs_preview = observation[:250] + ("..." if len(observation) > 250 else "")
                    print(f"  OBSERVE: {obs_preview}")

                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "tool": tool_name,
                    "action": action_input,
                    "observation": observation,
                })

                # Feed back to LLM
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"OBSERVATION:\n{observation}\n\n"
                        "Continue your analysis. You may take another ACTION "
                        "(using any of the 3 tools) or provide your FINAL_ANSWER."
                    ),
                })

            elif parsed["type"] == "error":
                if self.verbose:
                    print(f"  PARSE ERROR: {parsed.get('error', 'unknown')}")
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        "I could not parse your response. Please follow the exact format:\n\n"
                        "For an action:\n"
                        "THOUGHT: <your reasoning>\n"
                        "ACTION: <tool_name>\n"
                        "ACTION_INPUT: <valid JSON>\n\n"
                        "Or for your final answer:\n"
                        "THOUGHT: <your final reasoning>\n"
                        "FINAL_ANSWER: <your complete answer to the user's question>"
                    ),
                })

        # Max iterations — force an answer
        if self.verbose:
            print(f"\n  → Max iterations ({self.MAX_ITERATIONS}) reached, forcing final answer...")
        return self._force_final_answer(question)

    # ── Prompt Construction ──────────────────────────────────────────

    def _build_system_prompt(self):
        tool_docs = "\n\n".join(
            f"── TOOL: {name} ──────────────────────────────────────\n{tool.TOOL_DESCRIPTION}"
            for name, tool in self.tools.items()
        )

        system = textwrap.dedent(f"""\
        You are an expert data analyst agent for IndiGo Airlines. You answer questions
        by exploring a unified knowledge graph ontology, searching document embeddings,
        and querying a structured database.

        You work in a ReAct (Reason → Act → Observe) loop:
          1. THINK about what information you need
          2. ACT by calling one of your tools
          3. OBSERVE the result
          4. Repeat until you can answer confidently

        ── YOUR TOOLS ─────────────────────────────────────────────────────────
        {tool_docs}

        ── RECOMMENDED WORKFLOW ───────────────────────────────────────────────

        1. START with graph_ontology_tool to understand the data landscape:
           - List domain entities to see what structured tables exist
           - List subjects to see what concepts were extracted from documents
           - Check correspondences to find bridges between documents and tables

        2. USE vector_search_tool to find relevant document passages:
           - Semantic search for the user's question
           - Filter by document if the ontology told you which doc to look at

        3. USE sql_query_tool to get actual data from structured tables:
           - Use the ontology's knowledge of table schemas and relationships
           - Write JOINs across related tables as needed

        4. SYNTHESIZE your findings into a comprehensive FINAL_ANSWER.

        ── RESPONSE FORMAT ────────────────────────────────────────────────────

        On each turn, respond in EXACTLY one of these two formats:

        FORMAT A — Take an action:
        THOUGHT: <your reasoning about what to explore next and why>
        ACTION: <tool_name — one of: {', '.join(self.tools.keys())}>
        ACTION_INPUT: <valid JSON object for the tool — on a SINGLE line>

        FORMAT B — Final answer:
        THOUGHT: <your final reasoning summarizing what you found>
        FINAL_ANSWER: <your complete, well-structured answer to the user's question>

        ── IMPORTANT RULES ────────────────────────────────────────────────────

        - Respond with EXACTLY one format per turn (A or B). Never both.
        - ACTION_INPUT must be valid JSON on a SINGLE line.
        - Always start by exploring the ontology before diving into data queries.
        - Cite your sources: mention which tables, documents, or graph paths
          informed your answer.
        - If the data doesn't contain the answer, say so clearly.
        - Aim for 3-8 exploration steps. Be thorough but efficient.
        """)

        self.messages.append({"role": "system", "content": system})

    def _build_user_prompt(self, question: str):
        self.messages.append({
            "role": "user",
            "content": (
                f"Please answer the following question:\n\n"
                f"{question}\n\n"
                f"Start by exploring the knowledge graph ontology to understand "
                f"what data is available, then gather the information you need."
            ),
        })

    # ── LLM Interaction ──────────────────────────────────────────────

    def _call_llm(self) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=self.messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content

    # ── Response Parsing ─────────────────────────────────────────────

    def _parse_response(self, text: str) -> dict:
        """
        Parse LLM response into:
          - {"type": "action", "thought": str, "tool_name": str, "action_input": dict}
          - {"type": "final_answer", "answer": str}
          - {"type": "error", "error": str}
        """
        text = text.strip()

        # Check for FINAL_ANSWER
        if "FINAL_ANSWER:" in text:
            return self._parse_final_answer(text)

        # Check for ACTION
        if "ACTION_INPUT:" in text:
            return self._parse_action(text)

        # Fallback: if response is long and doesn't match either format,
        # treat the whole thing as a final answer (graceful degradation)
        if len(text) > 200 and "THOUGHT:" not in text:
            return {"type": "final_answer", "answer": text}

        return {"type": "error", "error": "Could not parse THOUGHT/ACTION or FINAL_ANSWER"}

    def _parse_action(self, text: str) -> dict:
        thought = ""
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\nACTION:)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract tool name
        tool_match = re.search(r"ACTION:\s*(\S+)", text)
        if not tool_match:
            return {"type": "error", "error": "Found ACTION_INPUT but no ACTION tool name"}
        tool_name = tool_match.group(1).strip()

        # Extract ACTION_INPUT JSON
        input_match = re.search(r"ACTION_INPUT:\s*(.+)", text, re.DOTALL)
        if not input_match:
            return {"type": "error", "error": "No ACTION_INPUT content found"}

        raw_input = input_match.group(1).strip()

        for candidate in self._json_candidates(raw_input):
            try:
                action_input = json.loads(candidate)
                if isinstance(action_input, dict):
                    return {
                        "type": "action",
                        "thought": thought,
                        "tool_name": tool_name,
                        "action_input": action_input,
                    }
            except json.JSONDecodeError:
                continue

        return {"type": "error", "error": f"Could not parse ACTION_INPUT JSON: {raw_input[:200]}"}

    def _parse_final_answer(self, text: str) -> dict:
        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL)
        if not fa_match:
            return {"type": "error", "error": "Found FINAL_ANSWER marker but no content"}

        answer = fa_match.group(1).strip()

        # Strip markdown fences if the LLM wrapped the answer
        if answer.startswith("```"):
            answer = re.sub(r"^```\w*\n?", "", answer)
            answer = re.sub(r"\n?```$", "", answer)
            answer = answer.strip()

        return {"type": "final_answer", "answer": answer}

    def _json_candidates(self, text: str) -> list[str]:
        """Generate candidate JSON strings from text."""
        candidates = []

        # Strategy 1: entire text
        candidates.append(text.strip())

        # Strategy 2: first line
        first_line = text.split("\n")[0].strip()
        candidates.append(first_line)

        # Strategy 3: brace-matched block
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : i + 1])
                        break

        # Strategy 4: greedy regex
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group())

        return candidates

    # ── Fallback ─────────────────────────────────────────────────────

    def _force_final_answer(self, question: str) -> str:
        """Force the agent to produce an answer from accumulated observations."""
        observations_summary = "\n\n".join(
            f"Step {t['iteration']} ({t['tool']}): {t['thought'][:200]}\n"
            f"  Result preview: {t['observation'][:200]}"
            for t in self.trace
        )

        self.messages.append({
            "role": "user",
            "content": textwrap.dedent(f"""\
            You have run out of exploration steps. Based on everything you have
            observed so far, produce your FINAL_ANSWER now.

            Original question: {question}

            Summary of your exploration:
            {observations_summary}

            Respond with ONLY:
            FINAL_ANSWER: <your complete answer>
            """),
        })

        response = self._call_llm()
        parsed = self._parse_response(response)

        if parsed["type"] == "final_answer":
            return parsed["answer"]

        # Absolute fallback
        return (
            "I was unable to fully answer your question after exploring the "
            "available data sources. Here is what I found:\n\n"
            + observations_summary
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: Build all tools
# ═══════════════════════════════════════════════════════════════════════════════

def build_tools(gremlin_client, lance_table, embedding_client, db_path: str) -> dict:
    """Build and return the tool dict for the InferenceAgent."""
    return {
        "graph_ontology_tool": GraphOntologyTool(gremlin_client),
        "vector_search_tool":  VectorSearchTool(lance_table, embedding_client),
        "sql_query_tool":      SQLQueryTool(db_path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive REPL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive question-answering loop."""
    print("=" * 70)
    print("  IndiGo Airlines — Knowledge Graph Inference Agent")
    print("  Unified ontology over structured DB + document corpus")
    print("=" * 70)

    # ── Connect to all data sources ────────────────────────────────────
    print("\n[1/4] Connecting to Cosmos DB (Gremlin)...")
    gremlin = get_gremlin_client()
    # Quick connectivity check
    try:
        run_gremlin(gremlin, "g.V().limit(1).count()")
        print("  ✓ Cosmos DB Gremlin connected (indigokg/knowledgegraph)")
    except Exception as e:
        print(f"  ✗ Cosmos DB connection failed: {e}")
        print("    Check COSMOS_DB_KEY in your .env file.")
        return

    print("[2/4] Connecting to LanceDB...")
    lance_db_path = os.path.join(BASE_DIR, "source_data", "lancedb_store")
    db = lancedb.connect(lance_db_path)
    try:
        lance_table = db.open_table("lexical_chunks")
        print(f"  ✓ LanceDB table 'lexical_chunks' ({len(lance_table)} rows)")
    except Exception as e:
        print(f"  ✗ LanceDB open failed: {e}")
        print("    Run the lexical_graph pipeline first to create the table.")
        return

    print("[3/4] Initializing Azure OpenAI clients...")
    llm_client = get_llm_client()
    embedding_client = get_embedding_client()
    print("  ✓ LLM + Embedding clients ready")

    print("[4/4] Locating SQLite database...")
    db_path = os.path.join(BASE_DIR, "source_data", "airlines.db")
    if not os.path.exists(db_path):
        print(f"  ✗ Database not found at {db_path}")
        print("    Run: python source_data/setup_new_db.py")
        return
    print(f"  ✓ SQLite database: {db_path}")

    # ── Build tools and agent ──────────────────────────────────────────
    tools = build_tools(gremlin, lance_table, embedding_client, db_path)
    print(f"\n  Tools loaded: {', '.join(tools.keys())}")
    print("-" * 70)
    print("  Type your question and press Enter. Type 'quit' or 'exit' to stop.")
    print("  Type 'trace' to see the last agent trace.")
    print("-" * 70)

    agent = InferenceAgent(llm_client=llm_client, tools=tools, verbose=True)
    last_trace = []

    while True:
        try:
            question = input("\n❓ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "trace":
            if last_trace:
                print(json.dumps(last_trace, indent=2, default=str))
            else:
                print("No trace available yet.")
            continue

        # Run the agent
        answer = agent.run(question)
        last_trace = agent.trace

        print(f"\n{'─'*70}")
        print(f"💡 Answer:\n")
        print(answer)
        print(f"{'─'*70}")
        print(f"  ({len(agent.trace)} tool calls made)")


if __name__ == "__main__":
    main()
