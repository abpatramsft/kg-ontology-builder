"""
lexical_graph.py — Layer 2: Lexical Graph (from Unstructured Data)

Pipeline:
  1. Load .txt documents from data/ directory
  2. Chunk documents internally (for LLM context window management)
  3. Embed chunks & store in LanceDB (local persistent vector DB for search)
  4. Extract subjects/entities from chunks via LLM, mapped to parent documents
  5. Build Lexical Graph in Neo4j (:Document → :Subject — no Chunk nodes in KG)
  6. Provide query interface (graph traversal + vector similarity)
  7. Visualize the graph

Note: Chunks are used internally for LLM extraction and vector search but are
      NOT stored as nodes in the knowledge graph. Subjects link directly to
      their parent Documents.

Prerequisites:
  - Neo4j running locally (same instance as Layer 1):
      docker run -d --name neo4j-kg \\
        -p 7474:7474 -p 7687:7687 \\
        -e NEO4J_AUTH=neo4j/password123 \\
        neo4j:5-community

  - LanceDB installed: pip install lancedb
  - Azure OpenAI access for embeddings + LLM

Usage:
  python lexical_graph/lexical_graph.py              # simple mode
  python lexical_graph/lexical_graph.py --advanced    # ReAct agent mode
"""

import argparse
import json
import os
import re
import sys

# ─── Ensure project root is on sys.path ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import lancedb
import pyarrow as pa

from utils.llm import (
    get_llm_client,
    call_llm,
    parse_llm_json,
    get_embedding_client,
    embed_texts,
)
from utils.neo4j_helpers import get_neo4j_driver, run_cypher, run_cypher_write

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LANCEDB_DIR = os.path.join(DATA_DIR, "lancedb_store")
TABLE_NAME = "lexical_chunks"


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Load Documents
# ═══════════════════════════════════════════════════════════════════════════════

def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Scan data directory for .txt files and load them.

    Returns list of:
        {"name": filename, "source_path": abs_path, "content": full_text}
    """
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({
            "name": fname,
            "source_path": fpath,
            "content": content,
        })
    return docs


def print_documents(docs: list[dict]):
    """Pretty-print loaded documents."""
    print(f"\n{'=' * 70}")
    print("  LOADED DOCUMENTS")
    print(f"{'=' * 70}")
    for doc in docs:
        lines = doc["content"].count("\n") + 1
        chars = len(doc["content"])
        print(f"  {doc['name']}  ({lines} lines, {chars} chars)")
        print(f"    Path: {doc['source_path']}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Chunk Documents (Paragraph-based)
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_document(doc: dict) -> list[dict]:
    """
    Split a document into chunks based on section boundaries.

    Strategy:
      - Detect section headers (lines followed by --- or === underlines)
      - Group header + body text as one chunk
      - Fall back to double-newline splits for documents without headers

    Returns list of:
        {"chunk_id": "doc_name::chunk_N", "doc_name": str, "text": str, "index": int}
    """
    content = doc["content"]
    lines = content.split("\n")

    # Detect header+underline patterns (e.g., "Title\n-----" or "Title\n=====")
    sections = []
    current_section = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if next line is an underline
        if (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and all(c in "-=" for c in lines[i + 1].strip())
            and len(lines[i + 1].strip()) >= 3
        ):
            # This is a header line — save current section, start new one
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    sections.append(section_text)
            current_section = [line, lines[i + 1]]
            i += 2
            continue
        current_section.append(line)
        i += 1

    # Don't forget the last section
    if current_section:
        section_text = "\n".join(current_section).strip()
        if section_text:
            sections.append(section_text)

    # If no sections found (no headers detected), fall back to double-newline splitting
    if len(sections) <= 1:
        raw_chunks = re.split(r"\n\s*\n", content)
        sections = [c.strip() for c in raw_chunks if c.strip()]

    # Filter out very short chunks (less than 50 chars = likely just a header or blank)
    sections = [s for s in sections if len(s) >= 50]

    chunks = []
    for idx, text in enumerate(sections):
        chunks.append({
            "chunk_id": f"{doc['name']}::chunk_{idx}",
            "doc_name": doc["name"],
            "text": text,
            "index": idx,
        })
    return chunks


def print_chunks(all_chunks: list[dict]):
    """Pretty-print chunks."""
    print(f"\n{'=' * 70}")
    print("  DOCUMENT CHUNKS")
    print(f"{'=' * 70}")
    current_doc = None
    for chunk in all_chunks:
        if chunk["doc_name"] != current_doc:
            current_doc = chunk["doc_name"]
            print(f"\n  Document: {current_doc}")
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"    [{chunk['index']}] {chunk['chunk_id']}")
        print(f"        {preview}...")
        print(f"        ({len(chunk['text'])} chars)")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Embed & Store in LanceDB
# ═══════════════════════════════════════════════════════════════════════════════

def init_lancedb(db_dir: str = LANCEDB_DIR) -> lancedb.DBConnection:
    """Create or open a persistent LanceDB database."""
    os.makedirs(db_dir, exist_ok=True)
    return lancedb.connect(db_dir)


def store_chunks_in_vectordb(
    lance_db: lancedb.DBConnection,
    embedding_client,
    chunks: list[dict],
) -> lancedb.table.Table:
    """
    Embed all chunks and store in LanceDB.

    Returns the LanceDB table.
    """
    # Drop existing table if it exists (clean rebuild)
    try:
        lance_db.drop_table(TABLE_NAME)
        print("  Cleared existing LanceDB table.")
    except Exception:
        pass

    # Embed all chunk texts
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks via Azure OpenAI...")
    embeddings = embed_texts(embedding_client, texts)

    # Build records as list-of-dicts (LanceDB native format)
    records = []
    for i, c in enumerate(chunks):
        records.append({
            "chunk_id": c["chunk_id"],
            "doc_name": c["doc_name"],
            "chunk_index": c["index"],
            "text": c["text"],
            "text_preview": c["text"][:200],
            "char_count": len(c["text"]),
            "vector": embeddings[i],
        })

    table = lance_db.create_table(TABLE_NAME, data=records)
    print(f"  Stored {len(records)} chunks in LanceDB (table: {TABLE_NAME})")
    return table


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Extract Subjects / Entities per Chunk (Simple Mode)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_subjects_simple(chunks: list[dict], llm_client) -> dict:
    """
    For each chunk, call LLM to extract named entities and key concepts.

    Returns:
        {
            "chunk_id": [
                {"name": str, "type": str, "context": str},
                ...
            ],
            ...
        }
    """
    subjects_by_chunk = {}

    for chunk in chunks:
        prompt = f"""You are an NER / entity extraction system analyzing IndiGo Airlines
maintenance and quality review documents.

Given the text chunk below, extract ALL named entities, technical components, and key concepts.

Return a JSON array of objects with these fields:
- "name": the entity name (use the most specific form, e.g., "PW1100G" not just "engine")
- "type": one of: "aircraft", "assembly", "part", "supplier", "person", "event", "metric", "system", "location"
- "context": a brief phrase (10-20 words) from the text explaining why this entity is mentioned

Rules:
- Extract specific product models, part numbers, supplier names, aircraft registrations
- Include technical systems and subsystems (e.g., "landing gear assembly", "FADEC controller")
- Include performance metrics and ratings if mentioned
- Do NOT include generic words — each entity should be a proper noun or specific technical term
- Deduplicate: if an entity appears multiple times, include it once with the most informative context

Text chunk:
---
{chunk['text']}
---

Respond ONLY with a valid JSON array, no markdown fences, no extra text."""

        print(f"  Extracting entities: {chunk['chunk_id']}...", end=" ", flush=True)
        response = call_llm(llm_client, prompt)
        parsed = parse_llm_json(response)

        if parsed and isinstance(parsed, list):
            subjects_by_chunk[chunk["chunk_id"]] = parsed
            print(f"OK — {len(parsed)} entities")
        else:
            print("WARN: parse failed, using empty list")
            subjects_by_chunk[chunk["chunk_id"]] = []

    return subjects_by_chunk


def generate_document_summary(doc: dict, llm_client) -> str:
    """Generate a short topic summary for a document."""
    prompt = f"""Summarize the following document in 1-2 sentences. Focus on the main topics,
key themes, and what domain/department it relates to.

Document: {doc['name']}
Content (first 2000 chars):
---
{doc['content'][:2000]}
---

Respond with ONLY the summary text, no extra formatting."""

    return call_llm(llm_client, prompt).strip()


def generate_chunk_summary(chunk: dict, llm_client) -> str:
    """Generate a short summary for a chunk."""
    prompt = f"""Summarize the following text section in 1 sentence (max 30 words).
Focus on the main topic and key finding/issue mentioned.

Text:
---
{chunk['text'][:1000]}
---

Respond with ONLY the summary text."""

    return call_llm(llm_client, prompt).strip()


def deduplicate_subjects(subjects_by_chunk: dict) -> list[dict]:
    """
    Merge subjects across all chunks into a deduplicated list.
    Tracks which *documents* each subject appears in (derived from chunk_id).

    Returns list of:
        {"name": str, "type": str, "description": str, "mention_count": int,
         "mentioned_in_docs": [doc_name, ...],
         "contexts_by_doc": {doc_name: [context_strings]}}
    """
    merged = {}  # normalized_name -> subject_data

    for chunk_id, subjects in subjects_by_chunk.items():
        # Extract document name from chunk_id (format: "doc_name::chunk_N")
        doc_name = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id

        for subj in subjects:
            name = subj.get("name", "").strip()
            if not name:
                continue
            norm_name = name.lower().strip()

            context = subj.get("context", "")

            if norm_name in merged:
                merged[norm_name]["mention_count"] += 1
                if doc_name not in merged[norm_name]["mentioned_in_docs"]:
                    merged[norm_name]["mentioned_in_docs"].append(doc_name)
                # Accumulate contexts per document
                merged[norm_name]["contexts_by_doc"].setdefault(doc_name, [])
                if context and context not in merged[norm_name]["contexts_by_doc"][doc_name]:
                    merged[norm_name]["contexts_by_doc"][doc_name].append(context)
                # Keep the longer context as main description
                existing_ctx = merged[norm_name].get("description", "")
                if len(context) > len(existing_ctx):
                    merged[norm_name]["description"] = context
            else:
                merged[norm_name] = {
                    "name": name,
                    "type": subj.get("type", "unknown"),
                    "description": context,
                    "mention_count": 1,
                    "mentioned_in_docs": [doc_name],
                    "contexts_by_doc": {doc_name: [context]} if context else {doc_name: []},
                }

    return list(merged.values())


def print_subjects(subjects_by_chunk: dict, deduped: list[dict]):
    """Pretty-print extracted subjects."""
    print(f"\n{'=' * 70}")
    print("  EXTRACTED SUBJECTS / ENTITIES")
    print(f"{'=' * 70}")
    total_raw = sum(len(v) for v in subjects_by_chunk.values())
    print(f"  Raw extractions: {total_raw} across {len(subjects_by_chunk)} chunks")
    print(f"  Deduplicated: {len(deduped)} unique subjects\n")
    for subj in sorted(deduped, key=lambda s: s["mention_count"], reverse=True):
        print(f"  [{subj['type']:10s}] {subj['name']}")
        print(f"              Mentions: {subj['mention_count']}  |  "
              f"Documents: {', '.join(subj['mentioned_in_docs'])}")
        if subj.get("description"):
            print(f"              Context: {subj['description'][:100]}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Build Lexical Graph in Neo4j
# ═══════════════════════════════════════════════════════════════════════════════

def build_lexical_graph(
    documents: list[dict],
    chunks: list[dict],
    subjects_by_chunk: dict,
    deduped_subjects: list[dict],
    doc_summaries: dict,
    chunk_summaries: dict,
    driver=None,
):
    """
    Build the Lexical Graph in Neo4j:
      - :Document nodes (one per file)
      - :Subject nodes (deduplicated entities)
      - :Document -[:MENTIONS]-> :Subject

    Note: Chunks are used internally for extraction and vector search
    but are NOT stored as nodes in the knowledge graph.
    """
    if driver is None:
        driver = get_neo4j_driver()

    # ── Clean existing Lexical Graph data (keep DomainEntity from Layer 1) ──
    print("  Clearing existing Lexical Graph data...")
    run_cypher_write(driver, "MATCH (n:Document) DETACH DELETE n")
    run_cypher_write(driver, "MATCH (n:Chunk) DETACH DELETE n")  # clean up any legacy Chunk nodes
    run_cypher_write(driver, "MATCH (n:Subject) DETACH DELETE n")

    # ── Create constraints ────────────────────────────────────────────────
    for label, prop in [("Document", "name"), ("Subject", "name")]:
        try:
            run_cypher(
                driver,
                f"CREATE CONSTRAINT {label.lower()}_{prop} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE",
            )
            print(f"  Created constraint on {label}.{prop}")
        except Exception as e:
            print(f"  Constraint {label}.{prop} exists or skipped: {e}")

    # ── Insert Document nodes ─────────────────────────────────────────────
    print("\n  Inserting Document nodes...")
    for doc in documents:
        doc_chunks = [c for c in chunks if c["doc_name"] == doc["name"]]
        summary = doc_summaries.get(doc["name"], "")
        run_cypher_write(
            driver,
            """
            CREATE (d:Document {
                name: $name,
                source_path: $source_path,
                chunk_count: $chunk_count,
                topic_summary: $topic_summary,
                source_type: 'unstructured'
            })
            """,
            {
                "name": doc["name"],
                "source_path": doc["source_path"],
                "chunk_count": len(doc_chunks),
                "topic_summary": summary,
            },
        )
        print(f"    + Document: {doc['name']} ({len(doc_chunks)} chunks used for extraction)")

    # ── Insert Subject nodes ──────────────────────────────────────────
    print("\n  Inserting Subject nodes...")
    for subj in deduped_subjects:
        run_cypher_write(
            driver,
            """
            CREATE (s:Subject {
                name: $name,
                type: $type,
                description: $description,
                mention_count: $mention_count
            })
            """,
            {
                "name": subj["name"],
                "type": subj["type"],
                "description": subj.get("description", ""),
                "mention_count": subj["mention_count"],
            },
        )
        print(f"    + Subject: {subj['name']} ({subj['type']})")

    # ── Insert MENTIONS edges (Document → Subject) ────────────────────
    #    Aggregate contexts from all chunks of each document per subject.
    print("\n  Inserting MENTIONS edges (Document → Subject)...")
    for subj in deduped_subjects:
        for doc_name in subj.get("mentioned_in_docs", []):
            # Combine contexts from this document for this subject
            contexts = subj.get("contexts_by_doc", {}).get(doc_name, [])
            combined_context = "; ".join(c for c in contexts if c)[:500]

            try:
                run_cypher_write(
                    driver,
                    """
                    MATCH (d:Document {name: $doc_name})
                    MATCH (s:Subject {name: $subject_name})
                    CREATE (d)-[:MENTIONS {context: $context}]->(s)
                    """,
                    {
                        "doc_name": doc_name,
                        "subject_name": subj["name"],
                        "context": combined_context,
                    },
                )
                print(f"    + {doc_name} —[MENTIONS]→ {subj['name']}")
            except Exception:
                # Try case-insensitive fallback
                try:
                    run_cypher_write(
                        driver,
                        """
                        MATCH (d:Document {name: $doc_name})
                        MATCH (s:Subject) WHERE toLower(s.name) = toLower($subject_name)
                        CREATE (d)-[:MENTIONS {context: $context}]->(s)
                        """,
                        {
                            "doc_name": doc_name,
                            "subject_name": subj["name"],
                            "context": combined_context,
                        },
                    )
                    print(f"    + {doc_name} —[MENTIONS]→ {subj['name']} (case-insensitive)")
                except Exception as e:
                    print(f"    WARN: MENTIONS edge failed: {doc_name} → {subj['name']}: {e}")

    print("\n  Lexical Graph built in Neo4j!")
    return driver


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6 — Query Interface
# ═══════════════════════════════════════════════════════════════════════════════

def query_lexical_graph(
    question: str,
    driver=None,
    lance_table=None,
    embedding_client=None,
    top_k: int = 5,
) -> dict:
    """
    Two-pronged query over the Lexical Graph:
      1. Graph traversal — keyword match on Subject names + Document summaries
      2. Vector similarity — embed question, search LanceDB for top-K chunks

    Returns:
        {
            "graph_results": {
                "subjects": [...],
                "documents": [...]
            },
            "vector_results": [
                {"chunk_id": str, "text": str, "score": float, "metadata": dict}
            ]
        }
    """
    if driver is None:
        driver = get_neo4j_driver()

    result = {"graph_results": {"subjects": [], "documents": []}, "vector_results": []}

    # ── 1. Graph traversal (keyword CONTAINS) ─────────────────────────
    stop_words = {
        "which", "what", "about", "have", "does", "that", "this",
        "with", "from", "tell", "show", "find", "data", "document",
        "documents", "mention", "mentions", "related", "are", "the",
    }
    keywords = [
        w.strip("?.,!\"'").lower()
        for w in question.split()
        if len(w.strip("?.,!\"'")) > 2 and w.strip("?.,!\"'").lower() not in stop_words
    ]

    seen_subjects = set()
    seen_docs = set()

    for kw in keywords:
        # Search Subjects
        records = run_cypher(
            driver,
            """
            MATCH (s:Subject)
            WHERE toLower(s.name) CONTAINS $kw
            RETURN s.name AS name, s.type AS type, s.description AS description,
                   s.mention_count AS mention_count
            """,
            {"kw": kw},
        )
        for rec in records:
            if rec["name"] not in seen_subjects:
                seen_subjects.add(rec["name"])
                rec["matched_keyword"] = kw
                result["graph_results"]["subjects"].append(rec)

        # Search Documents (by summary)
        records = run_cypher(
            driver,
            """
            MATCH (d:Document)
            WHERE toLower(d.topic_summary) CONTAINS $kw OR toLower(d.name) CONTAINS $kw
            RETURN d.name AS name, d.topic_summary AS topic_summary,
                   d.chunk_count AS chunk_count, d.source_path AS source_path
            """,
            {"kw": kw},
        )
        for rec in records:
            if rec["name"] not in seen_docs:
                seen_docs.add(rec["name"])
                rec["matched_keyword"] = kw
                result["graph_results"]["documents"].append(rec)

    # Also collect parent documents for matching subjects
    for subj in result["graph_results"]["subjects"]:
        records = run_cypher(
            driver,
            """
            MATCH (d:Document)-[:MENTIONS]->(s:Subject {name: $name})
            RETURN d.name AS name, d.topic_summary AS topic_summary,
                   d.chunk_count AS chunk_count, d.source_path AS source_path
            """,
            {"name": subj["name"]},
        )
        for rec in records:
            if rec["name"] not in seen_docs:
                seen_docs.add(rec["name"])
                result["graph_results"]["documents"].append(rec)

    # ── 2. Vector similarity (LanceDB) ───────────────────────────────
    if lance_table is not None and embedding_client:
        q_embedding = embed_texts(embedding_client, [question])[0]
        lance_results = (
            lance_table.search(q_embedding)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

        for row in lance_results:
            score = round(1 - row["_distance"], 4)  # cosine similarity
            result["vector_results"].append({
                "chunk_id": row["chunk_id"],
                "text": row["text"][:300],
                "score": score,
                "metadata": {
                    "doc_name": row["doc_name"],
                    "chunk_index": row["chunk_index"],
                    "text_preview": row["text_preview"],
                    "char_count": row["char_count"],
                },
            })

    return result


def print_query_results(question: str, results: dict):
    """Pretty-print query results."""
    print(f"\n{'=' * 70}")
    print(f"  QUERY: \"{question}\"")
    print(f"{'=' * 70}")

    gr = results["graph_results"]
    vr = results["vector_results"]

    # Graph results
    if gr["subjects"]:
        print(f"\n  Graph — Matching Subjects ({len(gr['subjects'])}):")
        for s in gr["subjects"]:
            print(f"    [{s['type']:10s}] {s['name']}  (mentions: {s['mention_count']}, "
                  f"keyword: '{s['matched_keyword']}')")
            if s.get("description"):
                print(f"               {s['description'][:80]}")

    if gr["documents"]:
        print(f"\n  Graph — Matching Documents ({len(gr['documents'])}):")
        for d in gr["documents"]:
            print(f"    {d['name']} — {d.get('topic_summary', '')[:80]}")

    # Vector results
    if vr:
        print(f"\n  Vector — Top {len(vr)} Similar Chunks (from LanceDB):")
        for v in vr:
            print(f"    {v['chunk_id']}  (similarity: {v['score']})")
            print(f"      {v['text'][:120]}...")

    if not gr["subjects"] and not gr["documents"] and not vr:
        print("  No matching results found.")

    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 7 — Visualize the Full Lexical Graph
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_lexical_graph(driver=None):
    """Print all nodes and edges in the Lexical Graph."""
    if driver is None:
        driver = get_neo4j_driver()

    print(f"\n{'=' * 70}")
    print("  LEXICAL GRAPH — ALL NODES & EDGES")
    print(f"{'=' * 70}")

    # Documents
    print("\n  Documents:")
    records = run_cypher(driver, """
        MATCH (d:Document)
        RETURN d.name AS name, d.topic_summary AS summary, d.chunk_count AS chunks
        ORDER BY d.name
    """)
    for rec in records:
        print(f"    :Document {rec['name']} — {rec.get('summary', '')[:60]} "
              f"({rec['chunks']} chunks used for extraction)")

    # Subjects
    print("\n  Subjects:")
    records = run_cypher(driver, """
        MATCH (s:Subject)
        RETURN s.name AS name, s.type AS type, s.mention_count AS mentions
        ORDER BY s.mention_count DESC
    """)
    for rec in records:
        print(f"    :Subject [{rec['type']:10s}] {rec['name']} (mentioned {rec['mentions']}x)")

    # Edges: MENTIONS (Document → Subject)
    print("\n  Edges (MENTIONS — Document → Subject):")
    records = run_cypher(driver, """
        MATCH (d:Document)-[r:MENTIONS]->(s:Subject)
        RETURN d.name AS doc, s.name AS subject, r.context AS context
        ORDER BY d.name, s.name
    """)
    for rec in records:
        ctx = (rec.get("context") or "")[:50]
        print(f"    {rec['doc']} —[MENTIONS]→ {rec['subject']}  ({ctx})")

    # Count summary
    print(f"\n  Summary:")
    for label in ["Document", "Subject"]:
        count = run_cypher(driver, f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]
        print(f"    {label} nodes: {count}")
    edge_count = run_cypher(
        driver,
        "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c",
    )[0]["c"]
    print(f"    MENTIONS edges: {edge_count}")

    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 2 — Lexical Graph Builder")
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Use ReAct agent-based entity extraction (iterative LanceDB exploration) "
        "instead of single-shot LLM calls",
    )
    args = parser.parse_args()

    mode = "ReAct Agent" if args.advanced else "Single-shot LLM"
    print("=" * 70)
    print("  LAYER 2 — LEXICAL GRAPH BUILDER")
    print("  IndiGo Airlines Unstructured Document Ontology")
    print(f"  Entity extraction mode: {mode}")
    print("=" * 70)

    # Step 1: Load documents
    print("\n[Step 1] Loading documents from data/...")
    documents = load_documents()
    if not documents:
        print("  ERROR: No .txt files found in data/ directory.")
        return
    print_documents(documents)

    # Step 2: Chunk documents
    print("\n[Step 2] Chunking documents (paragraph-based)...")
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
    print_chunks(all_chunks)

    # Step 3: Embed & store in LanceDB
    print("\n[Step 3] Embedding chunks & storing in LanceDB...")
    embedding_client = get_embedding_client()
    lance_db = init_lancedb()
    lance_table = store_chunks_in_vectordb(lance_db, embedding_client, all_chunks)

    # Step 4: Extract subjects
    llm_client = get_llm_client()
    if args.advanced:
        print("\n[Step 4] Extracting subjects with ReAct Agent (iterative exploration)...")
        from lexical_graph.enrich_advanced import extract_subjects_advanced
        subjects_by_chunk = extract_subjects_advanced(
            all_chunks, llm_client, lance_table, embedding_client
        )
    else:
        print("\n[Step 4] Extracting subjects with LLM (single-shot per chunk)...")
        subjects_by_chunk = extract_subjects_simple(all_chunks, llm_client)

    deduped_subjects = deduplicate_subjects(subjects_by_chunk)
    print_subjects(subjects_by_chunk, deduped_subjects)

    # Generate summaries
    print("\n[Step 4b] Generating document summaries...")
    doc_summaries = {}
    for doc in documents:
        print(f"  Summarizing document: {doc['name']}...", end=" ", flush=True)
        doc_summaries[doc["name"]] = generate_document_summary(doc, llm_client)
        print("OK")

    # chunk_summaries kept as empty dict (no longer stored in graph, but
    # build_lexical_graph signature accepts it for compatibility)
    chunk_summaries = {}

    # Step 5: Build graph
    print("\n[Step 5] Building Lexical Graph in Neo4j (Document → Subject, no Chunk nodes)...")
    driver = get_neo4j_driver()
    build_lexical_graph(
        documents, all_chunks, subjects_by_chunk, deduped_subjects,
        doc_summaries, chunk_summaries, driver,
    )

    # Step 6: Visualize
    visualize_lexical_graph(driver)

    # Step 7: Demo queries
    demo_questions = [
        "What documents mention brake issues?",
        "Which suppliers are discussed in reviews?",
        "Tell me about engine performance problems",
        "What quality issues are reported for the ATR fleet?",
        "Landing gear maintenance concerns",
    ]
    print("\n\n[Step 7] Running demo queries...")
    for q in demo_questions:
        results = query_lexical_graph(
            q, driver, lance_table, embedding_client, top_k=3
        )
        print_query_results(q, results)

    driver.close()
    print("\nDone! View your Lexical Graph at http://localhost:7474")
    print("Try: MATCH (d:Document)-[:MENTIONS]->(s:Subject) RETURN d, s")


if __name__ == "__main__":
    main()
