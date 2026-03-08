"""
lexical_graph.py — Layer 2: Lexical Graph (from Unstructured Data)

Pipeline:
  1. Load .txt documents from data/ directory
  2. Chunk documents internally (for LLM context window management)
  3. Embed chunks & store in LanceDB (local persistent vector DB for search)
  4. Extract ONE SPO (Subject–Predicate–Object) triplet per chunk via LLM
  5. Build Lexical Graph in Neo4j:
        :Document -[:MENTIONS]-> :Subject -[:RELATES_TO {predicate}]-> :Object
     (no Chunk nodes in KG)
  6. Provide query interface (graph traversal + vector similarity)
  7. Visualize the graph

Note: Chunks are used internally for LLM extraction and vector search but are
      NOT stored as nodes in the knowledge graph. Each chunk produces exactly
      one SPO triplet that captures its core meaning.

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
#  Step 4 — Extract SPO Triplets per Chunk (Simple Mode)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_spo_triplets_simple(chunks: list[dict], llm_client) -> dict:
    """
    For each chunk, call LLM to extract a single SPO (Subject–Predicate–Object)
    triplet that captures the core meaning of the chunk.

    Returns:
        {
            "chunk_id": {
                "subject": str,
                "subject_type": str,
                "predicate": str,
                "object": str,
                "object_type": str
            },
            ...
        }
    """
    spo_by_chunk = {}

    for chunk in chunks:
        prompt = f"""You are an information extraction system analyzing IndiGo Airlines
maintenance and quality review documents.

Given the text chunk below, extract ONE SPO (Subject–Predicate–Object) triplet that
best captures the core meaning of the entire chunk.

Return a JSON object with these fields:
- "subject": the main entity/concept (specific: e.g., "PW1100G engine" not just "engine")
- "subject_type": one of: aircraft, assembly, part, supplier, person, event, metric, system, location
- "predicate": the relationship/action connecting subject to object (e.g., "shows performance degradation in", "requires inspection of", "was supplied by")
- "object": the target entity/concept the subject relates to
- "object_type": one of: aircraft, assembly, part, supplier, person, event, metric, system, location

Rules:
- Extract the SINGLE most important SPO triplet that summarizes what this chunk is about
- Subject and object should be specific proper nouns or technical terms, not generic words
- The predicate should be a concise verbal phrase that connects subject to object
- Together, the triplet should capture the chunk's core message
- Example: {{"subject": "PW1100G engine", "subject_type": "assembly", "predicate": "shows performance degradation in", "object": "high-altitude operations", "object_type": "event"}}

Text chunk:
---
{chunk['text']}
---

Respond ONLY with a valid JSON object, no markdown fences, no extra text."""

        print(f"  Extracting SPO triplet: {chunk['chunk_id']}...", end=" ", flush=True)
        response = call_llm(llm_client, prompt)
        parsed = parse_llm_json(response)

        if parsed and isinstance(parsed, dict) and "subject" in parsed and "object" in parsed:
            spo_by_chunk[chunk["chunk_id"]] = parsed
            print(f"OK — ({parsed['subject']} → {parsed.get('predicate', '?')} → {parsed['object']})")
        else:
            print("WARN: parse failed, using empty triplet")
            spo_by_chunk[chunk["chunk_id"]] = {
                "subject": "", "subject_type": "unknown",
                "predicate": "", "object": "", "object_type": "unknown",
            }

    return spo_by_chunk


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


def deduplicate_spo_triplets(spo_by_chunk: dict) -> dict:
    """
    Merge SPO triplets across all chunks into deduplicated subjects, objects,
    and triplets. Tracks which documents each entity appears in.

    Returns:
        {
            "subjects": [
                {"name": str, "type": str, "mention_count": int,
                 "mentioned_in_docs": [doc_name, ...],
                 "spo_contexts": [{"predicate": str, "object": str, "doc_name": str}]}
            ],
            "objects": [
                {"name": str, "type": str, "mention_count": int,
                 "mentioned_in_docs": [doc_name, ...]}
            ],
            "triplets": [
                {"subject": str, "predicate": str, "object": str,
                 "chunk_id": str, "doc_name": str}
            ]
        }
    """
    subjects_merged = {}  # normalized_name -> subject_data
    objects_merged = {}   # normalized_name -> object_data
    triplets = []

    for chunk_id, spo in spo_by_chunk.items():
        doc_name = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id

        subj_name = spo.get("subject", "").strip()
        obj_name = spo.get("object", "").strip()
        predicate = spo.get("predicate", "").strip()

        if not subj_name or not obj_name:
            continue

        # Track triplet
        triplets.append({
            "subject": subj_name,
            "predicate": predicate,
            "object": obj_name,
            "chunk_id": chunk_id,
            "doc_name": doc_name,
        })

        # Merge subjects
        norm_subj = subj_name.lower().strip()
        if norm_subj in subjects_merged:
            subjects_merged[norm_subj]["mention_count"] += 1
            if doc_name not in subjects_merged[norm_subj]["mentioned_in_docs"]:
                subjects_merged[norm_subj]["mentioned_in_docs"].append(doc_name)
            subjects_merged[norm_subj]["spo_contexts"].append({
                "predicate": predicate, "object": obj_name, "doc_name": doc_name,
            })
        else:
            subjects_merged[norm_subj] = {
                "name": subj_name,
                "type": spo.get("subject_type", "unknown"),
                "mention_count": 1,
                "mentioned_in_docs": [doc_name],
                "spo_contexts": [{"predicate": predicate, "object": obj_name, "doc_name": doc_name}],
            }

        # Merge objects
        norm_obj = obj_name.lower().strip()
        if norm_obj in objects_merged:
            objects_merged[norm_obj]["mention_count"] += 1
            if doc_name not in objects_merged[norm_obj]["mentioned_in_docs"]:
                objects_merged[norm_obj]["mentioned_in_docs"].append(doc_name)
        else:
            objects_merged[norm_obj] = {
                "name": obj_name,
                "type": spo.get("object_type", "unknown"),
                "mention_count": 1,
                "mentioned_in_docs": [doc_name],
            }

    return {
        "subjects": list(subjects_merged.values()),
        "objects": list(objects_merged.values()),
        "triplets": triplets,
    }


def resolve_entities_across_documents(
    spo_by_chunk: dict,
    llm_client,
    verbose: bool = True,
) -> dict:
    """
    LLM-based entity resolution: find subjects/objects that refer to the same
    real-world entity across documents and normalize them to a canonical name.

    For example:
      - "Collins Aerospace smoke detector" and "Collins Aerospace" → keep both,
        but the parent entity "Collins Aerospace" should also be connected to
        documents that mention its products.
      - "Passenger Experience Division, IndiGo Airlines" and "IndiGo Airlines"
        → normalize to the more general form where appropriate.

    This function rewrites spo_by_chunk IN PLACE with normalized entity names
    so that deduplicate_spo_triplets will merge them into shared nodes.

    Returns:
        Updated spo_by_chunk dict with normalized entity names.
    """
    if verbose:
        print("\n  [Entity Resolution] Resolving entities across documents...")

    # Collect all unique subjects and objects with their source documents
    all_subjects = {}
    all_objects = {}
    for chunk_id, spo in spo_by_chunk.items():
        doc_name = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id
        subj = spo.get("subject", "").strip()
        obj = spo.get("object", "").strip()
        if subj:
            all_subjects.setdefault(subj, []).append(doc_name)
        if obj:
            all_objects.setdefault(obj, []).append(doc_name)

    # Build entity list for LLM
    entity_list = []
    for name, docs in all_subjects.items():
        entity_list.append({"name": name, "role": "subject", "documents": list(set(docs))})
    for name, docs in all_objects.items():
        entity_list.append({"name": name, "role": "object", "documents": list(set(docs))})

    if len(entity_list) < 3:
        if verbose:
            print("    Too few entities for resolution, skipping.")
        return spo_by_chunk

    prompt = f"""You are an entity resolution expert for IndiGo Airlines aviation maintenance documents.

Below is a list of entity names extracted from multiple documents, along with which
documents they appear in.

TASK: Identify groups of entities that refer to the SAME real-world entity or concept
(just named differently across documents). For each group, choose the best canonical
name — prefer the shorter, more general form that people would commonly use.

IMPORTANT RULES:
- Only merge entities that truly refer to the same thing
- "Collins Aerospace smoke detector" is a PRODUCT made by "Collins Aerospace" (the company).
  These are different entities — do NOT merge them. But DO note the parent-child relationship.
- When a specific product/part name contains a company name, the company itself should be
  recognized as an entity mentioned in that document too.
- If an entity is unique and has no matches, leave it as-is (do not include it in groups)

Entities:
{json.dumps(entity_list, indent=2)}

Respond with a JSON object:
{{
  "merge_groups": [
    {{
      "canonical_name": "<best name to use>",
      "canonical_type": "<entity type>",
      "variants": ["<name1>", "<name2>", ...],
      "reason": "<why these are the same entity>"
    }}
  ],
  "implicit_mentions": [
    {{
      "parent_entity": "<company/org name that is implicitly mentioned>",
      "parent_type": "<entity type>",
      "because_of": "<the product/part name that implies this parent>",
      "in_document": "<document name>"
    }}
  ]
}}

Return ONLY valid JSON. If no merges or implicit mentions are found, return
{{"merge_groups": [], "implicit_mentions": []}}.
"""

    response = call_llm(llm_client, prompt)
    parsed = parse_llm_json(response)

    if not parsed or not isinstance(parsed, dict):
        if verbose:
            print("    WARN: Could not parse entity resolution response, skipping.")
        return spo_by_chunk

    # Apply merge groups — rename entities to canonical names
    merge_map = {}  # old_name_lower → canonical_name
    type_map = {}   # canonical_name_lower → canonical_type
    merge_groups = parsed.get("merge_groups", [])
    for group in merge_groups:
        canonical = group.get("canonical_name", "").strip()
        canonical_type = group.get("canonical_type", "unknown")
        variants = group.get("variants", [])
        if not canonical or not variants:
            continue
        type_map[canonical.lower()] = canonical_type
        for variant in variants:
            if variant.strip().lower() != canonical.lower():
                merge_map[variant.strip().lower()] = canonical
                if verbose:
                    print(f"    Merge: \"{variant}\" → \"{canonical}\" ({group.get('reason', '')})")

    # Apply implicit mentions — add synthetic SPO triplets for parent entities
    implicit_mentions = parsed.get("implicit_mentions", [])
    synthetic_count = 0
    for mention in implicit_mentions:
        parent = mention.get("parent_entity", "").strip()
        parent_type = mention.get("parent_type", "unknown")
        child = mention.get("because_of", "").strip()
        doc = mention.get("in_document", "").strip()
        if not parent or not doc:
            continue

        # Check if this parent is already a subject in a chunk from this document
        already_present = False
        for chunk_id, spo in spo_by_chunk.items():
            chunk_doc = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id
            subj = spo.get("subject", "").strip()
            if chunk_doc == doc and subj.lower() == parent.lower():
                already_present = True
                break

        if not already_present:
            # Find a chunk from this doc that mentions the child entity
            target_chunk_id = None
            for chunk_id, spo in spo_by_chunk.items():
                chunk_doc = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id
                subj = spo.get("subject", "").strip()
                obj = spo.get("object", "").strip()
                if chunk_doc == doc and (child.lower() in subj.lower() or child.lower() in obj.lower()):
                    target_chunk_id = chunk_id
                    break

            if target_chunk_id:
                # Create a synthetic chunk ID for the implicit mention
                synth_id = f"{doc}::implicit_{parent.lower().replace(' ', '_')}_{synthetic_count}"
                spo_by_chunk[synth_id] = {
                    "subject": parent,
                    "subject_type": parent_type,
                    "predicate": "is referenced through",
                    "object": child,
                    "object_type": spo_by_chunk[target_chunk_id].get("subject_type", "unknown"),
                }
                synthetic_count += 1
                if verbose:
                    print(f"    Implicit: \"{parent}\" added to {doc} (because of \"{child}\")")

    # Now apply merge_map to rename entities in spo_by_chunk
    renamed_count = 0
    for chunk_id, spo in spo_by_chunk.items():
        subj = spo.get("subject", "").strip()
        obj = spo.get("object", "").strip()

        if subj.lower() in merge_map:
            new_name = merge_map[subj.lower()]
            spo["subject"] = new_name
            if new_name.lower() in type_map:
                spo["subject_type"] = type_map[new_name.lower()]
            renamed_count += 1

        if obj.lower() in merge_map:
            new_name = merge_map[obj.lower()]
            spo["object"] = new_name
            if new_name.lower() in type_map:
                spo["object_type"] = type_map[new_name.lower()]
            renamed_count += 1

    if verbose:
        print(f"    Entity resolution complete: {len(merge_groups)} merge group(s), "
              f"{renamed_count} rename(s), {synthetic_count} implicit mention(s) added.")

    return spo_by_chunk


def print_spo_triplets(spo_by_chunk: dict, deduped: dict):
    """Pretty-print extracted SPO triplets."""
    print(f"\n{'=' * 70}")
    print("  EXTRACTED SPO TRIPLETS")
    print(f"{'=' * 70}")
    total_raw = len(spo_by_chunk)
    subjects = deduped["subjects"]
    objects = deduped["objects"]
    triplets = deduped["triplets"]
    print(f"  Triplets extracted: {total_raw} (one per chunk)")
    print(f"  Unique subjects: {len(subjects)}")
    print(f"  Unique objects: {len(objects)}")
    print(f"\n  Triplets:")
    for t in triplets:
        print(f"    [{t['doc_name']:25s}] {t['subject']} —[{t['predicate']}]→ {t['object']}")
    print(f"\n  Deduplicated Subjects:")
    for subj in sorted(subjects, key=lambda s: s["mention_count"], reverse=True):
        print(f"    [{subj['type']:10s}] {subj['name']} (mentions: {subj['mention_count']}, "
              f"docs: {', '.join(subj['mentioned_in_docs'])})")
    print(f"\n  Deduplicated Objects:")
    for obj in sorted(objects, key=lambda o: o["mention_count"], reverse=True):
        print(f"    [{obj['type']:10s}] {obj['name']} (mentions: {obj['mention_count']})")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Build Lexical Graph in Neo4j
# ═══════════════════════════════════════════════════════════════════════════════

def build_lexical_graph(
    documents: list[dict],
    chunks: list[dict],
    spo_by_chunk: dict,
    deduped: dict,
    doc_summaries: dict,
    chunk_summaries: dict,
    driver=None,
):
    """
    Build the Lexical Graph in Neo4j:
      - :Document nodes (one per file)
      - :Subject nodes (deduplicated subjects from SPO triplets)
      - :Object nodes (deduplicated objects from SPO triplets)
      - :Document -[:MENTIONS]-> :Subject
      - :Subject -[:RELATES_TO {predicate}]-> :Object

    Note: Chunks are used internally for extraction and vector search
    but are NOT stored as nodes in the knowledge graph.
    Each chunk contributes one SPO triplet.
    """
    if driver is None:
        driver = get_neo4j_driver()

    # ── Clean existing Lexical Graph data (keep DomainEntity from Layer 1) ──
    print("  Clearing existing Lexical Graph data...")
    run_cypher_write(driver, "MATCH (n:Document) DETACH DELETE n")
    run_cypher_write(driver, "MATCH (n:Chunk) DETACH DELETE n")  # clean up any legacy Chunk nodes
    run_cypher_write(driver, "MATCH (n:Subject) DETACH DELETE n")
    run_cypher_write(driver, "MATCH (n:Object) DETACH DELETE n")

    # ── Create constraints ────────────────────────────────────────────────
    for label, prop in [("Document", "name"), ("Subject", "name"), ("Object", "name")]:
        try:
            run_cypher(
                driver,
                f"CREATE CONSTRAINT {label.lower()}_{prop} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE",
            )
            print(f"  Created constraint on {label}.{prop}")
        except Exception as e:
            print(f"  Constraint {label}.{prop} exists or skipped: {e}")

    subjects = deduped["subjects"]
    objects = deduped["objects"]
    triplets = deduped["triplets"]

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
    for subj in subjects:
        # Build description from SPO contexts
        spo_desc = "; ".join(
            f"{subj['name']} {ctx['predicate']} {ctx['object']}"
            for ctx in subj.get("spo_contexts", [])[:3]
        )[:500]
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
                "description": spo_desc,
                "mention_count": subj["mention_count"],
            },
        )
        print(f"    + Subject: {subj['name']} ({subj['type']})")

    # ── Insert Object nodes ───────────────────────────────────────────
    print("\n  Inserting Object nodes...")
    for obj in objects:
        run_cypher_write(
            driver,
            """
            CREATE (o:Object {
                name: $name,
                type: $type,
                mention_count: $mention_count
            })
            """,
            {
                "name": obj["name"],
                "type": obj["type"],
                "mention_count": obj["mention_count"],
            },
        )
        print(f"    + Object: {obj['name']} ({obj['type']})")

    # ── Insert MENTIONS edges (Document → Subject) ────────────────────
    print("\n  Inserting MENTIONS edges (Document → Subject)...")
    for subj in subjects:
        for doc_name in subj.get("mentioned_in_docs", []):
            # Combine SPO contexts from this document for this subject
            doc_spo_contexts = [
                f"{ctx['predicate']} {ctx['object']}"
                for ctx in subj.get("spo_contexts", [])
                if ctx.get("doc_name") == doc_name
            ]
            combined_context = "; ".join(doc_spo_contexts)[:500]

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

    # ── Insert RELATES_TO edges (Subject → Object) ───────────────────
    print("\n  Inserting RELATES_TO edges (Subject → Object)...")
    seen_edges = set()
    for triplet in triplets:
        edge_key = (triplet["subject"].lower(), triplet["predicate"].lower(), triplet["object"].lower())
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        try:
            run_cypher_write(
                driver,
                """
                MATCH (s:Subject {name: $subject_name})
                MATCH (o:Object {name: $object_name})
                CREATE (s)-[:RELATES_TO {predicate: $predicate}]->(o)
                """,
                {
                    "subject_name": triplet["subject"],
                    "object_name": triplet["object"],
                    "predicate": triplet["predicate"],
                },
            )
            print(f"    + {triplet['subject']} —[{triplet['predicate']}]→ {triplet['object']}")
        except Exception:
            # Case-insensitive fallback
            try:
                run_cypher_write(
                    driver,
                    """
                    MATCH (s:Subject) WHERE toLower(s.name) = toLower($subject_name)
                    MATCH (o:Object) WHERE toLower(o.name) = toLower($object_name)
                    CREATE (s)-[:RELATES_TO {predicate: $predicate}]->(o)
                    """,
                    {
                        "subject_name": triplet["subject"],
                        "object_name": triplet["object"],
                        "predicate": triplet["predicate"],
                    },
                )
                print(f"    + {triplet['subject']} —[{triplet['predicate']}]→ {triplet['object']} (case-insensitive)")
            except Exception as e:
                print(f"    WARN: RELATES_TO edge failed: {triplet['subject']} → {triplet['object']}: {e}")

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

    result = {"graph_results": {"subjects": [], "documents": [], "spo_triplets": []}, "vector_results": []}

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

        # Search Objects (by name) — may lead to Subject via RELATES_TO
        obj_records = run_cypher(
            driver,
            """
            MATCH (s:Subject)-[r:RELATES_TO]->(o:Object)
            WHERE toLower(o.name) CONTAINS $kw OR toLower(s.name) CONTAINS $kw
            RETURN s.name AS subject, r.predicate AS predicate, o.name AS object
            """,
            {"kw": kw},
        )
        for rec in obj_records:
            result["graph_results"]["spo_triplets"].append(rec)
            # Also add the subject if not already seen
            if rec["subject"] not in seen_subjects:
                seen_subjects.add(rec["subject"])
                subj_detail = run_cypher(driver, """
                    MATCH (s:Subject {name: $name})
                    RETURN s.name AS name, s.type AS type, s.description AS description,
                           s.mention_count AS mention_count
                """, {"name": rec["subject"]})
                if subj_detail:
                    subj_detail[0]["matched_keyword"] = kw
                    result["graph_results"]["subjects"].append(subj_detail[0])

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

    if gr.get("spo_triplets"):
        print(f"\n  Graph — Matching SPO Triplets ({len(gr['spo_triplets'])}):")
        for t in gr["spo_triplets"]:
            print(f"    {t['subject']} —[{t['predicate']}]→ {t['object']}")

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

    # Objects
    print("\n  Objects:")
    records = run_cypher(driver, """
        MATCH (o:Object)
        RETURN o.name AS name, o.type AS type, o.mention_count AS mentions
        ORDER BY o.mention_count DESC
    """)
    for rec in records:
        print(f"    :Object  [{rec['type']:10s}] {rec['name']} (mentioned {rec['mentions']}x)")

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

    # Edges: RELATES_TO (Subject → Object)
    print("\n  Edges (RELATES_TO — Subject → Object):")
    records = run_cypher(driver, """
        MATCH (s:Subject)-[r:RELATES_TO]->(o:Object)
        RETURN s.name AS subject, r.predicate AS predicate, o.name AS object
        ORDER BY s.name
    """)
    for rec in records:
        print(f"    {rec['subject']} —[{rec['predicate']}]→ {rec['object']}")

    # Count summary
    print(f"\n  Summary:")
    for label in ["Document", "Subject", "Object"]:
        count = run_cypher(driver, f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]
        print(f"    {label} nodes: {count}")
    for rel_type in ["MENTIONS", "RELATES_TO"]:
        edge_count = run_cypher(
            driver,
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c",
        )[0]["c"]
        print(f"    {rel_type} edges: {edge_count}")

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

    # Step 4: Extract SPO triplets
    llm_client = get_llm_client()
    if args.advanced:
        print("\n[Step 4] Extracting SPO triplets with ReAct Agent (iterative exploration)...")
        from lexical_graph.enrich_advanced import extract_spo_triplets_advanced
        spo_by_chunk = extract_spo_triplets_advanced(
            all_chunks, llm_client, lance_table, embedding_client
        )
    else:
        print("\n[Step 4] Extracting SPO triplets with LLM (single-shot per chunk)...")
        spo_by_chunk = extract_spo_triplets_simple(all_chunks, llm_client)

    deduped = deduplicate_spo_triplets(spo_by_chunk)
    print_spo_triplets(spo_by_chunk, deduped)

    # Step 4a: Entity resolution across documents
    print("\n[Step 4a] Resolving entities across documents (LLM-based)...")
    spo_by_chunk = resolve_entities_across_documents(spo_by_chunk, llm_client)

    # Re-deduplicate after entity resolution
    deduped = deduplicate_spo_triplets(spo_by_chunk)
    print_spo_triplets(spo_by_chunk, deduped)

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
    print("\n[Step 5] Building Lexical Graph in Neo4j (Document → Subject → Object, no Chunk nodes)...")
    driver = get_neo4j_driver()
    build_lexical_graph(
        documents, all_chunks, spo_by_chunk, deduped,
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
    print("Try: MATCH (d:Document)-[:MENTIONS]->(s:Subject)-[:RELATES_TO]->(o:Object) RETURN d, s, o")


if __name__ == "__main__":
    main()
