# Knowledge Graph Ontology — Semantic Layer for IndiGo Airlines

A 3-layer Knowledge Graph (KG) ontology that acts as a **semantic layer** over structured (SQL) and unstructured (text) data sources. The KG is not a replica of the data — it is a *map of the data landscape* that helps agents understand what exists, where it lives, and how entities relate to each other.

> **Analogy:** The KG is a map of a city. The map doesn't contain the buildings — it tells you where they are, what category they belong to, and how to navigate between them.

---

## Architecture

```
Unstructured Sources          Structured Sources
       │                              │
       ▼                              ▼
 [Lexical Graph]              [Domain Graph]          ← Layer 1 ✅
       │                              ▲
       │   MENTIONS                   │ CORRESPONDS_TO
       ▼                              │
 [Subject Graph] ─────────────────────
```

| Layer | Source | What it captures | Status |
|---|---|---|---|
| **Layer 1 — Domain Graph** | SQLite DB | Schema semantics: table descriptions, domains, FK + semantic relationships | ✅ Complete |
| **Layer 2 — Lexical Graph** | Text files | Document landscape: Document → Chunk → Subject hierarchy + vector embeddings | ✅ Complete |
| **Layer 3 — Subject Graph** | Cross-layer | Bridge: `CORRESPONDS_TO` edges linking unstructured subjects to structured entities | ⬜ Planned |

---

## Project Structure

```
KG_ontology_generation/
│
├── data/                           # Shared data (structured + unstructured)
│   ├── setup_db.py                 # Creates sample SQLite DB with IndiGo fleet data
│   ├── manufacturing.db            # SQLite DB: products, assemblies, parts, suppliers
│   ├── quality_reviews.txt         # Unstructured: Q3 2025 maintenance review notes
│   ├── customer_feedback.txt       # Unstructured: Q3 2025 customer feedback & cabin quality
│   └── lancedb_store/              # LanceDB vector DB (auto-created by Layer 2)
│
├── utils/                          # Shared utilities (DRY)
│   ├── __init__.py
│   ├── llm.py                      # Azure OpenAI LLM + embedding client helpers
│   └── neo4j_helpers.py            # Neo4j driver, Cypher read/write helpers
│
├── domain_graph/                   # Layer 1 — Domain Graph (structured data)
│   ├── __init__.py
│   ├── domain_graph.py             # Full pipeline: introspect → enrich → build → query
│   └── enrich_advanced.py          # ReAct agent with SQLDBQueryTool
│
├── lexical_graph/                  # Layer 2 — Lexical Graph (unstructured data)
│   ├── __init__.py
│   ├── lexical_graph.py            # Full pipeline: load → chunk → embed → extract → build → query
│   └── enrich_advanced.py          # ReAct agent with VectorDBQueryTool
│
├── kg_ontology_semantic_layer.md   # Design document (3-layer architecture)
├── llm_test.py                     # Reference script for Azure OpenAI LLM calls
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Prerequisites

### 1. Python Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Neo4j (Docker)

```bash
docker run -d --name neo4j-kg \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5-community
```

- **Browser:** http://localhost:7474
- **Bolt endpoint:** `bolt://localhost:7687`
- **Auth:** `neo4j` / `password123`

### 3. Azure OpenAI

The project uses **Azure OpenAI GPT-4.1** via `DefaultAzureCredential` (your `az login` session). No API keys in code.

```bash
az login
```

### 4. Sample Data

```bash
python data/setup_db.py
```

Creates `data/manufacturing.db` with 4 tables:

| Table | Rows | Description |
|---|---|---|
| `products` | 5 | Aircraft fleet models (A320neo, A321neo, ATR 72-600, A320ceo, A321XLR) |
| `assemblies` | 12 | Major aircraft subsystems (Landing Gear, Engine, Avionics, etc.) |
| `parts` | 20 | Component parts (Fan Blade, Brake Disc, FADEC Controller, etc.) |
| `suppliers` | 15 | Aerospace suppliers (Safran, Pratt & Whitney, Honeywell, etc.) |

FK chain: `products ← assemblies ← parts ← suppliers`

---

## Usage

### Layer 1 — Domain Graph

**Single-shot enrichment** (one LLM call per table):
```bash
python domain_graph/domain_graph.py
```

**ReAct agent enrichment** (iterative DB exploration per table):
```bash
python domain_graph/domain_graph.py --advanced
```

**Standalone agent test:**
```bash
python domain_graph/enrich_advanced.py
```

#### What the pipeline does:

1. **Introspect** — reads SQLite schema (tables, columns, PKs, FKs, row counts)
2. **Enrich** — LLM generates descriptions, domain labels, and semantic relationships for each table
3. **Build** — creates `DomainEntity` nodes and relationship edges in Neo4j
4. **Query** — keyword-based search over the graph for agent routing
5. **Visualize** — prints the full graph (nodes + edges)

---

### Layer 2 — Lexical Graph

**Single-shot extraction** (one LLM call per chunk):
```bash
python lexical_graph/lexical_graph.py
```

**ReAct agent extraction** (iterative vector DB exploration per document):
```bash
python lexical_graph/lexical_graph.py --advanced
```

#### What the pipeline does:

1. **Load** — scans `data/` for `.txt` files
2. **Chunk** — splits documents into sections using header/underline detection (fallback: paragraph splitting)
3. **Embed & Store** — embeds all chunks via Azure OpenAI `text-embedding-3-small`, stores vectors + metadata in LanceDB
4. **Extract Subjects** — LLM extracts named entities (aircraft, parts, suppliers, metrics, etc.) from each chunk
5. **Summarize** — generates 1-2 sentence summaries for each document and chunk
6. **Build Graph** — creates `:Document`, `:Chunk`, `:Subject` nodes and `:CONTAINS`, `:MENTIONS` edges in Neo4j
7. **Query** — dual query interface: keyword graph traversal + vector similarity search via LanceDB

#### Neo4j Schema (Layer 2):

```
(:Document {name, source_path, topic_summary, chunk_count})
    -[:CONTAINS {chunk_index}]->
(:Chunk {chunk_id, doc_name, summary, text_preview, char_count})
    -[:MENTIONS {context}]->
(:Subject {name, type, description, mention_count})
```

#### Sample Cypher queries:

```cypher
-- Full Lexical Graph
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(s:Subject) RETURN d, c, s

-- All subjects from a document
MATCH (d:Document {name: "quality_reviews.txt"})-[:CONTAINS]->(c)-[:MENTIONS]->(s)
RETURN DISTINCT s.name, s.type, s.mention_count ORDER BY s.mention_count DESC

-- Cross-document entity overlap (subjects mentioned in both files)
MATCH (d1:Document)-[:CONTAINS]->()-[:MENTIONS]->(s:Subject)<-[:MENTIONS]-()<-[:CONTAINS]-(d2:Document)
WHERE d1.name < d2.name
RETURN s.name, s.type, d1.name, d2.name

-- Supplier entities across all documents
MATCH (s:Subject {type: "supplier"}) RETURN s.name, s.mention_count ORDER BY s.mention_count DESC
```

---

## Enrichment Modes Compared

Both layers follow the same dual-mode pattern: a fast single-shot mode and a deeper ReAct agent mode.

### Layer 1 — Domain Graph Modes

#### Single-shot (`enrich_with_llm`)

One LLM call per table. Sends schema metadata (column names, types, FKs) and asks for a JSON description. Fast but shallow — the LLM never sees actual data values.

#### ReAct Agent (`enrich_with_llm_advanced`)

A custom Reason-Act agent that iteratively explores the database:

```
THOUGHT → ACTION (sql_db_query_tool) → OBSERVATION → repeat → FINAL_ANSWER
```

The agent has a single tool — `SQLDBQueryTool` — with 6 actions:
- `list_tables` — enumerate all tables
- `describe_table` — schema + PKs + FKs + row count
- `sample_rows` — see actual data values
- `query` — arbitrary SELECT (JOINs, aggregations, filters)
- `get_foreign_keys` — FK chain navigation
- `distinct_values` — understand categorical columns

Typical agent run per table: **5-7 iterations** of tool use before producing a final answer.

A cross-table **validation checkpoint** runs after all tables are enriched, catching inconsistencies and missing relationships.

| Aspect | Single-shot | ReAct Agent |
|---|---|---|
| LLM calls per table | 1 | 5-8 |
| Sees actual data | No | Yes (sample rows, JOINs) |
| FK chain depth | Direct only | Multi-hop traversal |
| Cross-table validation | No | Yes |
| Semantic edges (our DB) | ~7 | ~12 |

---

### Layer 2 — Lexical Graph Modes

#### Single-shot (`extract_subjects_simple`)

One LLM call per chunk. Sends the chunk text and asks for a JSON array of entities. Fast — each chunk is processed independently with no cross-chunk awareness.

#### ReAct Agent (`extract_subjects_advanced`)

A custom Reason-Act agent that iteratively explores the vector database to build richer, cross-chunk-aware entity extractions:

```
THOUGHT → ACTION (vector_db_query_tool) → OBSERVATION → repeat → FINAL_ANSWER
```

The agent runs **one instance per document** (not per chunk), so it sees all chunks from a document and can discover cross-references. It has a single tool — `VectorDBQueryTool` — with 5 actions:

- `search_similar(query, n)` — semantic similarity search across all embedded chunks
- `list_documents` — enumerate all document names in the vector store
- `get_chunk(chunk_id)` — retrieve full text + metadata for a specific chunk
- `get_chunks_by_doc(doc_name)` — all chunks from a document, ordered by index
- `get_collection_stats` — total chunks, document count, metadata

**How the agent works step by step:**

1. **Reads all chunks** — starts by calling `get_chunks_by_doc` to understand the full document
2. **Semantic exploration** — runs `search_similar` queries for key topics mentioned in the text (e.g., "brake disc issues", "supplier performance") to find cross-chunk and cross-document references
3. **Deep-dives** — uses `get_chunk` to read the full text of related chunks it discovered
4. **Builds entity map** — accumulates entities across iterations, with richer context from multiple chunks
5. **Produces FINAL_ANSWER** — a JSON object keyed by chunk_id, with entities for each chunk

After all documents are processed, a **cross-document validation** step reviews the full extraction for consistency, deduplication, and completeness.

Typical agent run per document: **3-6 iterations** of tool use before producing a final answer.

| Aspect | Single-shot | ReAct Agent |
|---|---|---|
| LLM calls per chunk | 1 | — |
| LLM calls per document | N (one per chunk) | 5-8 (iterative exploration) |
| Cross-chunk awareness | No | Yes (semantic search) |
| Cross-document awareness | No | Yes (vector similarity) |
| Post-extraction validation | No | Yes |
| Entity context quality | Chunk-local | Multi-chunk grounded |

---

## Querying the Graph

After building, explore in the **Neo4j Browser** at http://localhost:7474:

```cypher
-- All Domain Graph nodes and edges (Layer 1)
MATCH (n:DomainEntity)-[r]->(m) RETURN n, r, m

-- All Lexical Graph nodes and edges (Layer 2)
MATCH (d:Document)-[r1:CONTAINS]->(c:Chunk)-[r2:MENTIONS]->(s:Subject) RETURN d, r1, c, r2, s

-- Both layers together
MATCH (n) WHERE n:DomainEntity OR n:Document OR n:Chunk OR n:Subject
OPTIONAL MATCH (n)-[r]-(m)
RETURN n, r, m

-- Find a specific table (Layer 1)
MATCH (n:DomainEntity {name: "suppliers"}) RETURN n

-- Find a specific subject (Layer 2)
MATCH (s:Subject {name: "A320neo"}) RETURN s

-- All relationships for a table
MATCH (n:DomainEntity {name: "parts"})-[r]-(m) RETURN n, r, m

-- Tables in a domain
MATCH (n:DomainEntity) WHERE n.domain = "fleet_management" RETURN n.name, n.description

-- Multi-hop: products → assemblies → parts
MATCH path = (p:DomainEntity {name: "products"})-[*1..3]->(target)
RETURN path
```

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| Graph DB | Neo4j 5 Community (Docker) | Stores the KG ontology (both layers) |
| Vector DB | LanceDB (local, on-disk) | Chunk embeddings + similarity search (Layer 2) |
| Structured DB | SQLite (stdlib) | Source data for Layer 1 |
| LLM | Azure OpenAI GPT-4.1 | Schema enrichment + entity extraction |
| Embeddings | Azure OpenAI text-embedding-3-small | Chunk vectorization for Layer 2 |
| Auth | DefaultAzureCredential | Token-based, no API keys |
| Python | 3.11+ | Runtime |

---

## Design Document

See [kg_ontology_semantic_layer.md](kg_ontology_semantic_layer.md) for the full architecture design, including the Lexical Graph (Layer 2) and Subject Graph (Layer 3) specs.
