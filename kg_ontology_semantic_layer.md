# Knowledge Graph Ontology as a Semantic Layer — POC Design Guide

## Overview

This document outlines the approach for building a **Knowledge Graph (KG) ontology** that acts as a semantic layer over both structured (relational DB) and unstructured (documents, files) data sources. The KG is **not** a replica of the data — it is a **map of the data landscape** that helps agents understand what exists, where it lives, and how different data entities relate to each other.

> **Analogy:** Think of the KG as a map of a city. The map doesn't contain the buildings — it tells you where they are, what category they belong to, and how to navigate between them.

---

## Architecture: The 3-Layer Graph Model

The architecture is based on three interconnected graph layers, each serving a distinct purpose.

```
Unstructured Sources          Structured Sources
       │                              │
       ▼                              ▼
 [Lexical Graph]              [Domain Graph]
       │                              ▲
       │   MENTIONS                   │ CORRESPONDS_TO
       ▼                              │
 [Subject Graph] ─────────────────────
```

### Layer 1 — Domain Graph (from Structured DB)

Represents the **schema semantics** of your structured data. It does not store rows or values — it stores knowledge about what the tables mean and how they relate.

**What goes in:**
- Node types = your table entities (`:Product`, `:Customer`, `:Order`)
- Edge types = foreign key relationships + semantic relationships (`BELONGS_TO`, `PLACED_BY`)
- Properties on nodes = metadata about each table (description, key columns, row count, data types)

**Example node:**
```cypher
(:Table {
  name: "orders",
  description: "Captures customer purchase transactions",
  key_columns: ["order_id", "customer_id"],
  row_count: 50000,
  domain: "sales"
})
```

**Design Pattern — Schema Introspection Pipeline:**
```
Structured DB
    → Introspect schema (tables, columns, foreign keys)
    → LLM enrichment: "describe what this table represents"
    → Generate ontology nodes + edges
    → Write to graph DB (Neo4j / Kuzu)
```

---

### Layer 2 — Lexical Graph (from Unstructured Data)

Represents your **document landscape** — what documents exist, what chunks they contain, and what concepts are mentioned in those chunks.

**What goes in:**
- `:Document` nodes — file metadata, source path, topic summary
- `:Chunk` nodes — chunk ID, embedding summary, key themes
- `:Subject` nodes — entities/concepts extracted from chunks (e.g., product names, events, people)

**Design Pattern — Chunk Enrichment Pipeline:**
```
Unstructured files (PDFs, Markdown, Word docs, etc.)
    → Chunk + embed (LangChain / LlamaIndex)
    → NER / entity extraction per chunk (LLM structured output)
    → Build :Document → :Chunk → :Subject graph
    → Store in graph DB
```

**Example graph fragment:**
```
(:Document {name: "Q3_reviews.md"})
    -[:CONTAINS]→ (:Chunk {id: "chunk_42", summary: "brake assembly failure reports"})
    -[:MENTIONS]→ (:Subject {name: "brake assembly"})
```

---

### Layer 3 — Subject Graph & The Bridge (CORRESPONDS_TO)

The **Subject Graph** is the middle semantic layer. It holds entities extracted from unstructured data and links them back to the Domain Graph via `CORRESPONDS_TO` relationships.

This is the most valuable part of the architecture — it allows agents to bridge a concept mentioned in a document to the actual structured table where data about that concept lives.

**Design Pattern — Entity Resolution:**
```
For each :Subject extracted from chunks:
    → Embed the subject label
    → Similarity search against Domain Graph node descriptions
    → If similarity > threshold → create CORRESPONDS_TO edge
    → Optionally: LLM confirmation call to validate the match
```

**Example:**
> A review says *"the brake assembly failed"* → `:Subject {name: "brake assembly"}` gets a `CORRESPONDS_TO` edge pointing to `:Table {name: "assemblies"}` — so agents know to query both the reviews vector store AND the assemblies table.

---

## Concrete Example: Manufacturing / Supply Chain

This maps directly to the reference data model:

### Structured Side

| Source File | Graph Node | Relationships |
|---|---|---|
| `products.csv` | `:Product` | — |
| `assemblies.csv` | `:Assembly` | `PART_OF` → `:Product` |
| `parts.csv` | `:Part` | `PART_OF` → `:Assembly` |
| `suppliers.csv` | `:Supplier` | `SUPPLIES` → `:Part` |

### Unstructured Side

| Source File | Graph Nodes | Relationships |
|---|---|---|
| `reviews/*.md` | `:Document` → `:Chunk` | `CONTAINS` |
| (extracted) | `:Chunk` → `:Subject` | `MENTIONS` |
| (resolved) | `:Subject` → `:Product` | `CORRESPONDS_TO` |

---

## How Agents Use the KG

The KG acts as a **router and planner** for agent queries — not a data store.

### Agent Query Flow

```
User Question: "Which suppliers have quality issues?"

Step 1: Agent queries the KG ontology
    → Find nodes semantically related to "quality" and "suppliers"
    → KG returns:
        - :Supplier (structured — query suppliers table)
        - :Chunk nodes mentioning quality issues (unstructured — fetch from vector store)

Step 2: Agent knows WHERE to fetch data
    → Query suppliers table from structured DB (SQL)
    → Fetch relevant chunks from vector store (semantic search)

Step 3: Agent synthesizes response
    → Combine structured supplier records + unstructured review context
    → Generate final answer
```

The KG answers **"what exists and where"** — the actual databases answer **"what is the value"**.

---

## Tech Stack Recommendations

| Component | Recommended Options |
|---|---|
| **Graph DB** | Neo4j (production), Kuzu (embedded, great for POC) |
| **Schema Introspection** | SQLAlchemy + LLM enrichment (GPT-4o / Claude) |
| **Chunking & Embedding** | LangChain, LlamaIndex |
| **Entity Extraction** | LLM with structured output (Instructor + Pydantic) |
| **Entity Resolution** | pgvector / FAISS similarity + LLM confirmation |
| **Agent Framework** | LangGraph, LlamaIndex Workflows |

---

## POC Implementation Phases

### Phase 1 — Domain Graph (Structured DB Only)

**Goal:** Prove that agents can use the KG to figure out which table to query.

- Introspect your structured DB schema
- Enrich each table/column with LLM-generated descriptions
- Store as nodes + edges in Kuzu or Neo4j
- Build a simple agent tool: `query_kg_for_table(user_question)` → returns relevant table names + column hints

**Success criterion:** Agent correctly identifies the right table(s) to query for 80%+ of test questions.

---

### Phase 2 — Lexical Graph (Unstructured Data)

**Goal:** Represent the document landscape in the KG.

- Chunk your unstructured files
- Extract subjects/entities per chunk using LLM structured output
- Build `:Document → :Chunk → :Subject` hierarchy in the graph
- Validate the graph reflects the themes and concepts in your documents

**Success criterion:** Querying the KG for "documents about X" returns relevant chunks.

---

### Phase 3 — The Bridge (CORRESPONDS_TO Links)

**Goal:** Connect unstructured mentions to structured entities.

- Embed all `:Subject` node labels
- Embed all `:Table` / `:Domain` node descriptions
- Run cosine similarity — create `CORRESPONDS_TO` edges above threshold
- Optionally run an LLM pass to confirm ambiguous matches

**Success criterion:** A subject like "brake assembly" correctly resolves to the `:Assembly` table node.

---

### Phase 4 — Agent Integration

**Goal:** Full end-to-end query using the KG as the semantic routing layer.

- Give the agent two tools:
  1. `query_ontology(question)` — queries the KG, returns relevant nodes (tables, chunks, subjects)
  2. `fetch_data(source, query)` — fetches actual data from the identified source
- Agent uses Tool 1 to plan, then Tool 2 to execute
- Synthesize the final response from combined structured + unstructured data

**Success criterion:** Agent answers multi-hop questions that span both structured and unstructured data sources accurately.

---

## Key Design Principles

1. **The KG is a map, not a database** — store semantics and relationships, not raw data values.
2. **LLM enrichment is essential** — raw schema introspection gives you structure; LLMs give you meaning.
3. **Entity resolution is the hardest step** — invest time here; it's what makes the bridge work.
4. **Keep the ontology coarse-grained** — node per table, not node per row. Node per document topic, not node per sentence.
5. **The KG should be queryable by agents in natural language** — use graph traversal + semantic search together.
6. **Iterative refinement** — the ontology will improve as you add more data sources and observe agent failures.

---

## Summary

| Layer | Source | Represents | Key Relationships |
|---|---|---|---|
| Domain Graph | Structured DB | Table/entity semantics | `PART_OF`, `BELONGS_TO`, `SUPPLIES` |
| Lexical Graph | Unstructured files | Document & chunk landscape | `CONTAINS`, `MENTIONS` |
| Subject Graph | Extracted from Lexical | Concepts & entities | `PREDICATE`, `CORRESPONDS_TO` |

The full power of this architecture is unlocked at **Phase 3** — when a concept mentioned in an unstructured document is automatically linked to the structured table where real data about that concept lives. This is what enables agents to answer complex, cross-source questions with precision.
