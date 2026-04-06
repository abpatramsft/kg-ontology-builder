"""
lexical_graph.py — Layer 2: Lexical Graph (from Unstructured Data)

Pipeline:
  1. Load .txt documents from source_data/ directory
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
  python src/lexical_graph/lexical_graph.py              # simple mode
  python src/lexical_graph/lexical_graph.py --advanced    # ReAct agent mode
"""

import argparse
import json
import os
import re
import sys

# ─── Ensure src/ is on sys.path ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # src/lexical_graph/
SRC_DIR = os.path.dirname(BASE_DIR)                         # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)                     # repo root
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.llm import (
    get_llm_client,
    call_llm,
    parse_llm_json,
    get_embedding_client,
    embed_texts,
)
from utils.cosmos_helpers import (
    get_gremlin_client, run_gremlin, run_gremlin_write,
    esc, make_vertex_id, gval,
)
from utils.cosmos_vector_helpers import (
    get_vector_container,
    vector_search as cosmos_vector_search,
)

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "source_data")


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

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Vector DB is now pre-populated during setup
#           (see source_data/setup_vector_db.py)
#           Chunks are stored in Cosmos DB NoSQL with vector index.
#           No LanceDB initialization needed here.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Extract SPO Triplets per Chunk (Simple Mode)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_spo_triplets_simple(chunks: list[dict], llm_client) -> dict:
    """
    For each chunk, call LLM to extract a single abstract, ontology-level SPO
    (Subject–Predicate–Object) triplet that captures the conceptual/thematic
    meaning of the chunk. Subjects and objects are abstract categories, roles,
    or concept classes — never specific proper nouns or instance identifiers.

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
        prompt = f"""You are an information extraction system analyzing airlines and aviation
industry documents. Your goal is to build an ABSTRACT ONTOLOGY — a conceptual schema
of the aviation domain — NOT a factual knowledge graph of specific events.

Given the text chunk below, extract ONE abstract, ontology-level SPO
(Subject–Predicate–Object) triplet that captures the CONCEPTUAL / THEMATIC meaning
of the entire chunk.

Return a JSON object with these fields:
- "subject": an ABSTRACT CATEGORY, ROLE, or CONCEPT CLASS — NEVER a specific proper noun,
  individual name, flight number, date, or instance-level identifier.
  Abstraction examples:
    Person names ("Anil Kumar") → their ROLE ("Pilot", "Maintenance Engineer")
    Specific aircraft ("VT-ANQ", "Boeing 737-800 MSN 29019") → class ("Narrow-Body Aircraft")
    Specific flights ("AI-302") → category ("Domestic Flight", "Long-Haul Flight")
    Specific airports ("JFK", "DEL") → role ("Hub Airport", "International Airport")
    Specific dates/incidents → pattern ("Recurring Maintenance Issue")
    Specific metrics ("87.3% OTP") → concept ("On-Time Performance Metric")
- "subject_type": one of: aircraft, flight, route, airport, crew, passenger, booking, fare_class, maintenance, incident, organization, person, event, metric, system, location, equipment, service, process, policy, regulation
- "predicate": a conceptual relationship connecting subject to object (e.g., "undergoes",
  "impacts", "requires", "is governed by", "contributes to", "triggers")
- "object": an ABSTRACT CATEGORY, ROLE, or CONCEPT CLASS (same rules as subject)
- "object_type": one of: aircraft, flight, route, airport, crew, passenger, booking, fare_class, maintenance, incident, organization, person, event, metric, system, location, equipment, service, process, policy, regulation

Rules:
- Extract the SINGLE most important abstract SPO triplet that summarizes what
  general pattern or conceptual relationship this chunk illustrates
- Subject and object must be ABSTRACT — never specific names, IDs, or instance data
- The predicate should be a concise, reusable conceptual verb phrase
- Together, the triplet should describe a GENERAL PATTERN, not specific facts
- WRONG example: {{"subject": "Flight AI-402", "predicate": "experienced diversion due to", "object": "engine vibration warning"}}
- RIGHT example: {{"subject": "Domestic Flight", "subject_type": "flight", "predicate": "experiences diversion due to", "object": "Engine Monitoring Alert", "object_type": "incident"}}

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
    LLM-based concept resolution: find abstract subjects/objects that refer to
    the same conceptual category across documents and normalize them to a
    canonical name.

    For example:
      - "Aircraft Maintenance" and "Plane Servicing" → unify to one term
      - "Engine Maintenance" implies parent concept "Aircraft Maintenance"
      - Any residual instance-level names are elevated to abstract categories

    This function rewrites spo_by_chunk IN PLACE with normalized concept names
    so that deduplicate_spo_triplets will merge them into shared nodes.

    Returns:
        Updated spo_by_chunk dict with normalized concept names.
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

    prompt = f"""You are a concept resolution expert for an abstract aviation ontology.

Below is a list of ABSTRACT CONCEPT names (categories, roles, concept classes) extracted
from multiple aviation industry documents, along with which documents they appear in.

These are ontology-level concepts (e.g., "Fleet Operations", "Crew Scheduling",
"Narrow-Body Aircraft"), NOT specific instance-level entities.

TASK: Identify groups of concepts that refer to the SAME abstract category or concept
(just named differently across documents). For each group, choose the best canonical
name — prefer the clearer, more standard domain term.

IMPORTANT RULES:
- Only merge concepts that truly refer to the same abstract category
- A broader concept and a narrower sub-concept are DIFFERENT — do NOT merge them.
  But DO note the parent-child (broader/narrower) relationship.
  (e.g., "Aircraft Maintenance" and "Engine Maintenance" are related but distinct)
- When a specific sub-concept implies a broader parent concept, the parent concept
  should be recognized as implicitly present in that document too.
- If a concept is unique and has no matches, leave it as-is (do not include it in groups)
- If you see any SPECIFIC proper nouns or instance-level names that slipped through
  (e.g., "Anil Kumar", "VT-ANQ", "Flight AI-302"), replace them with their abstract
  category in the canonical_name (e.g., "Pilot", "Narrow-Body Aircraft", "Domestic Flight")

Concepts:
{json.dumps(entity_list, indent=2)}

Respond with a JSON object:
{{
  "merge_groups": [
    {{
      "canonical_name": "<best abstract concept name to use>",
      "canonical_type": "<concept type>",
      "variants": ["<name1>", "<name2>", ...],
      "reason": "<why these are the same concept>"
    }}
  ],
  "implicit_mentions": [
    {{
      "parent_entity": "<broader concept that is implicitly present>",
      "parent_type": "<concept type>",
      "because_of": "<the narrower concept that implies this parent>",
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
    client=None,
):
    """
    Build the Lexical Graph in Cosmos DB (Gremlin API):
      - Document vertices (one per file)
      - Subject vertices (deduplicated subjects from SPO triplets)
      - Object vertices (deduplicated objects from SPO triplets)
      - Document -[MENTIONS]-> Subject edges
      - Subject -[RELATES_TO {predicate}]-> Object edges

    Note: Chunks are used internally for extraction and vector search
    but are NOT stored as vertices in the knowledge graph.
    Each chunk contributes one SPO triplet.
    """
    if client is None:
        client = get_gremlin_client()

    # ── Clean existing Lexical Graph data (keep DomainEntity/Concept from Layer 1) ──
    print("  Clearing existing Lexical Graph data...")
    for label in ["Document", "Chunk", "Subject", "Object"]:
        try:
            run_gremlin_write(client, f"g.V().hasLabel('{label}').drop()")
        except Exception as e:
            print(f"  WARN: Could not drop {label} vertices: {e}")

    subjects = deduped["subjects"]
    objects = deduped["objects"]
    triplets = deduped["triplets"]

    # ── Insert Document vertices ──────────────────────────────────────
    print("\n  Inserting Document vertices...")
    for doc in documents:
        doc_chunks = [c for c in chunks if c["doc_name"] == doc["name"]]
        summary = doc_summaries.get(doc["name"], "")
        vid = make_vertex_id("Document", doc["name"])
        doc_name = doc["name"]
        doc_source_path = doc["source_path"]
        try:
            run_gremlin_write(client, (
                f"g.addV('Document')"
                f".property('id', '{esc(vid)}')"
                f".property('category', 'lexical')"
                f".property('name', '{esc(doc_name)}')"
                f".property('source_path', '{esc(doc_source_path)}')"
                f".property('chunk_count', {len(doc_chunks)})"
                f".property('topic_summary', '{esc(summary)}')"
                f".property('source_type', 'unstructured')"
            ))
            print(f"    + Document: {doc['name']} ({len(doc_chunks)} chunks used for extraction)")
        except Exception as e:
            print(f"    WARN: Document '{doc['name']}' failed: {e}")

    # ── Insert Subject vertices ───────────────────────────────────────
    print("\n  Inserting Subject vertices...")
    for subj in subjects:
        spo_desc = "; ".join(
            f"{subj['name']} {ctx['predicate']} {ctx['object']}"
            for ctx in subj.get("spo_contexts", [])[:3]
        )[:500]
        vid = make_vertex_id("Subject", subj["name"])
        subj_name = subj["name"]
        subj_type = subj["type"]
        try:
            run_gremlin_write(client, (
                f"g.addV('Subject')"
                f".property('id', '{esc(vid)}')"
                f".property('category', 'lexical')"
                f".property('name', '{esc(subj_name)}')"
                f".property('type', '{esc(subj_type)}')"
                f".property('description', '{esc(spo_desc)}')"
                f".property('mention_count', {subj['mention_count']})"
            ))
            print(f"    + Subject: {subj['name']} ({subj['type']})")
        except Exception as e:
            print(f"    WARN: Subject '{subj['name']}' failed: {e}")

    # ── Insert Object vertices ────────────────────────────────────────
    print("\n  Inserting Object vertices...")
    for obj in objects:
        vid = make_vertex_id("Object", obj["name"])
        obj_name = obj["name"]
        obj_type = obj["type"]
        try:
            run_gremlin_write(client, (
                f"g.addV('Object')"
                f".property('id', '{esc(vid)}')"
                f".property('category', 'lexical')"
                f".property('name', '{esc(obj_name)}')"
                f".property('type', '{esc(obj_type)}')"
                f".property('mention_count', {obj['mention_count']})"
            ))
            print(f"    + Object: {obj['name']} ({obj['type']})")
        except Exception as e:
            print(f"    WARN: Object '{obj['name']}' failed: {e}")

    # ── Insert MENTIONS edges (Document → Subject) ────────────────────
    print("\n  Inserting MENTIONS edges (Document → Subject)...")
    for subj in subjects:
        subj_id = make_vertex_id("Subject", subj["name"])
        for doc_name in subj.get("mentioned_in_docs", []):
            doc_spo_contexts = [
                f"{ctx['predicate']} {ctx['object']}"
                for ctx in subj.get("spo_contexts", [])
                if ctx.get("doc_name") == doc_name
            ]
            combined_context = "; ".join(doc_spo_contexts)[:500]
            doc_id = make_vertex_id("Document", doc_name)
            try:
                run_gremlin_write(client,
                    f"g.V('{esc(doc_id)}')"
                    f".addE('MENTIONS')"
                    f".to(g.V('{esc(subj_id)}'))"
                    f".property('context', '{esc(combined_context)}')"
                )
                print(f"    + {doc_name} —[MENTIONS]→ {subj['name']}")
            except Exception as e:
                print(f"    WARN: MENTIONS edge failed: {doc_name} → {subj['name']}: {e}")

    # ── Insert RELATES_TO edges (Subject → Object) ────────────────────
    print("\n  Inserting RELATES_TO edges (Subject → Object)...")
    seen_edges = set()
    for triplet in triplets:
        edge_key = (triplet["subject"].lower(), triplet["predicate"].lower(), triplet["object"].lower())
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        subj_id = make_vertex_id("Subject", triplet["subject"])
        obj_id = make_vertex_id("Object", triplet["object"])
        triplet_predicate = triplet["predicate"]
        try:
            run_gremlin_write(client,
                f"g.V('{esc(subj_id)}')"
                f".addE('RELATES_TO')"
                f".to(g.V('{esc(obj_id)}'))"
                f".property('predicate', '{esc(triplet_predicate)}')"
            )
            print(f"    + {triplet['subject']} —[{triplet['predicate']}]→ {triplet['object']}")
        except Exception as e:
            print(f"    WARN: RELATES_TO edge failed: {triplet['subject']} → {triplet['object']}: {e}")

    print("\n  Lexical Graph built in Cosmos DB (indigokg/knowledgegraph)!")
    return client


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6 — Query Interface
# ═══════════════════════════════════════════════════════════════════════════════

def query_lexical_graph(
    question: str,
    client=None,
    vector_container=None,
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
    if client is None:
        client = get_gremlin_client()

    result = {"graph_results": {"subjects": [], "documents": [], "spo_triplets": []}, "vector_results": []}

    # ── 1. Graph traversal (keyword match in Python) ──────────────────
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

    # Fetch all subjects and filter in Python
    all_subjects = run_gremlin(client,
        "g.V().hasLabel('Subject').valueMap('name','type','description','mention_count')"
    )
    for rec in all_subjects:
        s_name = gval(rec, "name", "")
        for kw in keywords:
            if kw in s_name.lower():
                if s_name not in seen_subjects:
                    seen_subjects.add(s_name)
                    result["graph_results"]["subjects"].append({
                        "name": s_name,
                        "type": gval(rec, "type", ""),
                        "description": gval(rec, "description", ""),
                        "mention_count": gval(rec, "mention_count", 0),
                        "matched_keyword": kw,
                    })
                break

    # SPO triplets for matching subjects
    for kw in keywords:
        all_triplets = run_gremlin(client,
            "g.V().hasLabel('Subject').outE('RELATES_TO')"
            ".project('subject','predicate','object')"
            ".by(outV().values('name'))"
            ".by(values('predicate'))"
            ".by(inV().values('name'))"
        )
        for rec in all_triplets:
            subj_name = rec.get("subject", "")
            obj_name = rec.get("object", "")
            if kw in subj_name.lower() or kw in obj_name.lower():
                result["graph_results"]["spo_triplets"].append(rec)
                if subj_name not in seen_subjects:
                    seen_subjects.add(subj_name)
                    # find subject details
                    sid = make_vertex_id("Subject", subj_name)
                    s_recs = run_gremlin(client,
                        f"g.V('{esc(sid)}').valueMap('name','type','description','mention_count')"
                    )
                    if s_recs:
                        r = s_recs[0]
                        result["graph_results"]["subjects"].append({
                            "name": gval(r, "name", subj_name),
                            "type": gval(r, "type", ""),
                            "description": gval(r, "description", ""),
                            "mention_count": gval(r, "mention_count", 0),
                            "matched_keyword": kw,
                        })

    # Fetch all documents and filter in Python
    all_docs = run_gremlin(client,
        "g.V().hasLabel('Document').valueMap('name','topic_summary','chunk_count','source_path')"
    )
    for rec in all_docs:
        d_name = gval(rec, "name", "")
        summary = gval(rec, "topic_summary", "")
        for kw in keywords:
            if kw in d_name.lower() or kw in summary.lower():
                if d_name not in seen_docs:
                    seen_docs.add(d_name)
                    result["graph_results"]["documents"].append({
                        "name": d_name,
                        "topic_summary": summary,
                        "chunk_count": gval(rec, "chunk_count", 0),
                        "source_path": gval(rec, "source_path", ""),
                        "matched_keyword": kw,
                    })
                break

    # Also collect parent documents for matching subjects
    for subj in result["graph_results"]["subjects"]:
        sid = make_vertex_id("Subject", subj["name"])
        records = run_gremlin(client,
            f"g.V('{esc(sid)}').in('MENTIONS').valueMap('name','topic_summary','chunk_count','source_path')"
        )
        for rec in records:
            d_name = gval(rec, "name", "")
            if d_name not in seen_docs:
                seen_docs.add(d_name)
                result["graph_results"]["documents"].append({
                    "name": d_name,
                    "topic_summary": gval(rec, "topic_summary", ""),
                    "chunk_count": gval(rec, "chunk_count", 0),
                    "source_path": gval(rec, "source_path", ""),
                })

    # ── 2. Vector similarity (Cosmos DB) ───────────────────────────────
    if vector_container is not None and embedding_client:
        q_embedding = embed_texts(embedding_client, [question])[0]
        cosmos_results = cosmos_vector_search(vector_container, q_embedding, top_k=top_k)

        for row in cosmos_results:
            score = round(row.get("similarity_score", 0), 4)
            result["vector_results"].append({
                "chunk_id": row["chunk_id"],
                "text": row["text"][:300],
                "score": score,
                "metadata": {
                    "doc_name": row["doc_name"],
                    "chunk_index": row["chunk_index"],
                    "text_preview": row.get("text_preview", row["text"][:200]),
                    "char_count": row.get("char_count", len(row["text"])),
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

def visualize_lexical_graph(client=None):
    """Print all vertices and edges in the Lexical Graph (Cosmos DB Gremlin)."""
    if client is None:
        client = get_gremlin_client()

    print(f"\n{'=' * 70}")
    print("  LEXICAL GRAPH — ALL VERTICES & EDGES (Cosmos DB)")
    print(f"{'=' * 70}")

    # Documents
    print("\n  Documents:")
    records = run_gremlin(client,
        "g.V().hasLabel('Document').valueMap('name','topic_summary','chunk_count')"
    )
    for rec in records:
        name = gval(rec, "name", "")
        summary = gval(rec, "topic_summary", "")
        chunks = gval(rec, "chunk_count", 0)
        print(f"    :Document {name} — {summary[:60]} ({chunks} chunks used for extraction)")

    # Subjects
    print("\n  Subjects:")
    records = run_gremlin(client,
        "g.V().hasLabel('Subject').valueMap('name','type','mention_count')"
    )
    for rec in records:
        name = gval(rec, "name", "")
        stype = gval(rec, "type", "")
        mentions = gval(rec, "mention_count", 0)
        print(f"    :Subject [{stype:10s}] {name} (mentioned {mentions}x)")

    # Objects
    print("\n  Objects:")
    records = run_gremlin(client,
        "g.V().hasLabel('Object').valueMap('name','type','mention_count')"
    )
    for rec in records:
        name = gval(rec, "name", "")
        otype = gval(rec, "type", "")
        mentions = gval(rec, "mention_count", 0)
        print(f"    :Object  [{otype:10s}] {name} (mentioned {mentions}x)")

    # Edges: MENTIONS
    print("\n  Edges (MENTIONS — Document → Subject):")
    records = run_gremlin(client,
        "g.V().hasLabel('Document').outE('MENTIONS')"
        ".project('doc','subject','context')"
        ".by(outV().values('name'))"
        ".by(inV().values('name'))"
        ".by(coalesce(values('context'), constant('')))"
    )
    for rec in records:
        ctx = (rec.get("context") or "")[:50]
        print(f"    {rec.get('doc','')} —[MENTIONS]→ {rec.get('subject','')}  ({ctx})")

    # Edges: RELATES_TO
    print("\n  Edges (RELATES_TO — Subject → Object):")
    records = run_gremlin(client,
        "g.V().hasLabel('Subject').outE('RELATES_TO')"
        ".project('subject','predicate','object')"
        ".by(outV().values('name'))"
        ".by(values('predicate'))"
        ".by(inV().values('name'))"
    )
    for rec in records:
        print(f"    {rec.get('subject','')} —[{rec.get('predicate','')}]→ {rec.get('object','')}")

    # Count summary
    print(f"\n  Summary:")
    for label in ["Document", "Subject", "Object"]:
        count_result = run_gremlin(client, f"g.V().hasLabel('{label}').count()")
        count = count_result[0] if count_result else 0
        print(f"    {label} vertices: {count}")
    for rel_type in ["MENTIONS", "RELATES_TO"]:
        count_result = run_gremlin(client, f"g.E().hasLabel('{rel_type}').count()")
        count = count_result[0] if count_result else 0
        print(f"    {rel_type} edges: {count}")

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
    print("  Airlines Unstructured Document Ontology")
    print(f"  Entity extraction mode: {mode}")
    print("=" * 70)

    # Step 1: Load documents
    print("\n[Step 1] Loading documents from source_data/...")
    documents = load_documents()
    if not documents:
        print("  ERROR: No .txt files found in source_data/ directory.")
        return
    print_documents(documents)

    # Step 2: Chunk documents
    print("\n[Step 2] Chunking documents (paragraph-based)...")
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
    print_chunks(all_chunks)

    # Step 3: Connect to Cosmos DB vector store (pre-populated by setup_vector_db.py)
    print("\n[Step 3] Connecting to Cosmos DB vector store...")
    embedding_client = get_embedding_client()
    vector_container = get_vector_container()
    print("  Connected to Cosmos DB vector store")

    # Step 4: Extract SPO triplets
    llm_client = get_llm_client()
    if args.advanced:
        print("\n[Step 4] Extracting SPO triplets with ReAct Agent (iterative exploration)...")
        from agents.lexical_agent import extract_spo_triplets_advanced
        spo_by_chunk = extract_spo_triplets_advanced(
            all_chunks, llm_client, vector_container, embedding_client
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
    print("\n[Step 5] Building Lexical Graph in Cosmos DB (indigokg/knowledgegraph)...")
    gremlin = get_gremlin_client()
    build_lexical_graph(
        documents, all_chunks, spo_by_chunk, deduped,
        doc_summaries, chunk_summaries, gremlin,
    )

    # Step 6: Visualize
    visualize_lexical_graph(gremlin)

    # Step 7: Demo queries
    demo_questions = [
        "What documents mention maintenance issues?",
        "Which aircraft are discussed in reports?",
        "Tell me about flight delay problems",
        "What incidents are reported for recent operations?",
        "Crew scheduling concerns",
    ]
    print("\n\n[Step 7] Running demo queries...")
    for q in demo_questions:
        results = query_lexical_graph(
            q, gremlin, vector_container, embedding_client, top_k=3
        )
        print_query_results(q, results)

    gremlin.close()
    print("\nDone! Lexical Graph stored in Cosmos DB (indigokg / knowledgegraph).")
    print("Query with Gremlin: g.V().hasLabel('Document').out('MENTIONS').out('RELATES_TO').valueMap(true)")


if __name__ == "__main__":
    main()
