"""
subject_graph.py — Layer 3: Subject Graph Bridge (CORRESPONDS_TO)

Pipeline:
  1. Fetch existing :Subject nodes from Neo4j (created by Layer 2)
  2. Fetch existing :DomainEntity nodes from Neo4j (created by Layer 1)
  3. Embed both sets using Azure OpenAI embeddings
  4. Compute cosine similarity — match subjects to domain entities
  5. Create CORRESPONDS_TO edges in Neo4j (with confidence + reasoning)
  6. Provide a cross-source query interface for agents
  7. Visualize the full 3-layer bridge

Prerequisites:
  - Layer 1 (domain_graph.py) and Layer 2 (lexical_graph.py) must be run first.
  - Neo4j running locally with DomainEntity, Document, Chunk, Subject nodes.
  - LanceDB store with embedded chunks (for advanced mode).

Usage:
  python src/subject_graph/subject_graph.py              # simple mode (embedding similarity)
  python src/subject_graph/subject_graph.py --advanced    # ReAct agent mode
"""

import argparse
import json
import math
import os
import sys

# ─── Ensure src/ is on sys.path ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # src/subject_graph/
SRC_DIR = os.path.dirname(BASE_DIR)                         # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)                     # repo root
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.llm import get_llm_client, call_llm, parse_llm_json, get_embedding_client, embed_texts
from utils.cosmos_helpers import (
    get_gremlin_client, run_gremlin, run_gremlin_write,
    esc, make_vertex_id, gval,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Fetch Subject Nodes (from Layer 2)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_subjects(client=None) -> list[dict]:
    """
    Read all Subject vertices from Cosmos DB (Gremlin), including document context
    via MENTIONS edges and SPO triplet context via RELATES_TO edges.

    Returns list of:
        {
            "name": str,
            "type": str,
            "description": str,
            "mention_count": int,
            "doc_contexts": [{"doc_name": str, "topic_summary": str, "mention_context": str}],
            "spo_contexts": [{"predicate": str, "object_name": str, "object_type": str}]
        }
    """
    if client is None:
        client = get_gremlin_client()

    # Get all Subject vertices
    records = run_gremlin(client,
        "g.V().hasLabel('Subject').valueMap('name','type','description','mention_count')"
    )

    subjects = []
    for rec in records:
        name = gval(rec, "name", "")
        sid = make_vertex_id("Subject", name)

        # Fetch document contexts via MENTIONS edges
        doc_recs = run_gremlin(client,
            f"g.V('{esc(sid)}').inE('MENTIONS')"
            ".project('doc_name','topic_summary','context')"
            ".by(outV().values('name'))"
            ".by(outV().coalesce(values('topic_summary'), constant('')))"
            ".by(coalesce(values('context'), constant('')))"
        )

        # Fetch SPO triplet contexts via RELATES_TO edges
        spo_recs = run_gremlin(client,
            f"g.V('{esc(sid)}').outE('RELATES_TO')"
            ".project('predicate','object_name','object_type')"
            ".by(values('predicate'))"
            ".by(inV().values('name'))"
            ".by(inV().coalesce(values('type'), constant('')))"
        )

        subjects.append({
            "name": name,
            "type": gval(rec, "type") or "unknown",
            "description": gval(rec, "description") or "",
            "mention_count": gval(rec, "mention_count") or 1,
            "doc_contexts": [
                {
                    "doc_name": d.get("doc_name", ""),
                    "topic_summary": d.get("topic_summary", ""),
                    "mention_context": d.get("context", ""),
                }
                for d in doc_recs
            ],
            "spo_contexts": [
                {
                    "predicate": sr.get("predicate", ""),
                    "object_name": sr.get("object_name", ""),
                    "object_type": sr.get("object_type", ""),
                }
                for sr in spo_recs
            ],
        })

    return subjects


def print_subjects(subjects: list[dict]):
    """Pretty-print fetched subjects."""
    print(f"\n{'=' * 70}")
    print("  SUBJECTS FROM LAYER 2 (Lexical Graph)")
    print(f"{'=' * 70}")
    print(f"  Total: {len(subjects)} unique subjects\n")
    for subj in subjects:
        print(f"  [{subj['type']:10s}] {subj['name']}")
        print(f"              Mentions: {subj['mention_count']}  |  "
              f"Documents: {len(subj['doc_contexts'])}")
        if subj.get("description"):
            print(f"              Context: {subj['description'][:100]}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Fetch Domain Entity Nodes (from Layer 1)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_domain_entities(client=None) -> list[dict]:
    """
    Read all DomainEntity vertices from Cosmos DB (Gremlin), including relationships.

    Returns list of:
        {
            "name": str,
            "description": str,
            "domain": str,
            "key_columns": str (JSON array),
            "column_info": str (JSON array),
            "row_count": int,
            "relationships": [{"type": str, "direction": str, "target": str}],
            "concepts": [{"name": str, "description": str, "shared": bool}]
        }
    """
    if client is None:
        client = get_gremlin_client()

    records = run_gremlin(client,
        "g.V().hasLabel('DomainEntity')"
        ".valueMap('name','description','domain','key_columns','column_info','row_count')"
    )

    entities = []
    for rec in records:
        name = gval(rec, "name", "")
        vid = make_vertex_id("DomainEntity", name)

        out_rels = run_gremlin(client,
            f"g.V('{esc(vid)}').outE()"
            ".where(inV().hasLabel('DomainEntity'))"
            ".project('rel_type','target','reason')"
            ".by(label())"
            ".by(inV().values('name'))"
            ".by(coalesce(values('reason'), constant('')))"
        )

        in_rels = run_gremlin(client,
            f"g.V('{esc(vid)}').inE()"
            ".where(outV().hasLabel('DomainEntity'))"
            ".project('rel_type','source','reason')"
            ".by(label())"
            ".by(outV().values('name'))"
            ".by(coalesce(values('reason'), constant('')))"
        )

        relationships = []
        for r in out_rels:
            relationships.append({
                "type": r.get("rel_type", ""),
                "direction": "outgoing",
                "target": r.get("target", ""),
                "reason": r.get("reason", ""),
            })
        for r in in_rels:
            relationships.append({
                "type": r.get("rel_type", ""),
                "direction": "incoming",
                "target": r.get("source", ""),
                "reason": r.get("reason", ""),
            })

        concept_recs = run_gremlin(client,
            f"g.V('{esc(vid)}').out('HAS_CONCEPT')"
            ".valueMap('name','description','shared')"
        )
        concepts = [
            {
                "name": gval(cr, "name", ""),
                "description": gval(cr, "description", ""),
                "shared": gval(cr, "shared", False),
            }
            for cr in concept_recs
        ]

        entities.append({
            "name": name,
            "description": gval(rec, "description") or "",
            "domain": gval(rec, "domain") or "unknown",
            "key_columns": gval(rec, "key_columns") or "[]",
            "column_info": gval(rec, "column_info") or "[]",
            "row_count": gval(rec, "row_count") or 0,
            "relationships": relationships,
            "concepts": concepts,
        })

    return entities


def print_domain_entities(entities: list[dict]):
    """Pretty-print fetched domain entities."""
    print(f"\n{'=' * 70}")
    print("  DOMAIN ENTITIES FROM LAYER 1 (Domain Graph)")
    print(f"{'=' * 70}")
    print(f"  Total: {len(entities)} domain entities (tables)\n")
    for ent in entities:
        print(f"  [{ent['domain']:15s}] {ent['name']} ({ent['row_count']} rows)")
        print(f"                    {ent['description'][:80]}")
        if ent["relationships"]:
            for rel in ent["relationships"]:
                direction = "→" if rel["direction"] == "outgoing" else "←"
                print(f"                    {direction} [{rel['type']}] {rel['target']}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Embed Entities for Similarity Matching
# ═══════════════════════════════════════════════════════════════════════════════

def build_subject_text(subject: dict) -> str:
    """
    Build a rich text representation of a subject for embedding.
    Uses the full SPO triplet context (Subject → Predicate → Object).
    Falls back to document mention context if no SPO data is available.
    """
    parts = [subject["name"]]
    if subject.get("type") and subject["type"] != "unknown":
        parts.append(f"(type: {subject['type']})")

    # Use SPO triplet contexts for richer semantics
    spo_parts = []
    for spo in subject.get("spo_contexts", [])[:3]:  # limit to 3 triplets
        spo_parts.append(
            f"{subject['name']} {spo['predicate']} {spo['object_name']}"
        )
    if spo_parts:
        parts.append("| Triplets: " + "; ".join(spo_parts))

    # Fall back to document mention contexts if no SPO triplets
    if not spo_parts:
        if subject.get("description"):
            parts.append(f"— {subject['description']}")
        contexts = []
        for dc in subject.get("doc_contexts", [])[:3]:
            if dc.get("mention_context"):
                contexts.append(dc["mention_context"])
            elif dc.get("topic_summary"):
                contexts.append(dc["topic_summary"])
        if contexts:
            parts.append("| Context: " + "; ".join(contexts))

    return " ".join(parts)


def build_domain_entity_text(entity: dict) -> str:
    """
    Build a rich text representation of a domain entity for embedding.
    Combines the entity name, description, domain, and column info.
    """
    parts = [entity["name"]]
    if entity.get("description"):
        parts.append(f"— {entity['description']}")
    if entity.get("domain") and entity["domain"] != "unknown":
        parts.append(f"(domain: {entity['domain']})")

    # Add column names for richer matching
    try:
        columns = json.loads(entity.get("column_info", "[]"))
        col_names = [c["name"] for c in columns if isinstance(c, dict)]
        if col_names:
            parts.append(f"| Columns: {', '.join(col_names)}")
    except (json.JSONDecodeError, TypeError):
        pass

    # Add relationship context
    rel_labels = [
        f"{r['type']} {r['target']}" for r in entity.get("relationships", [])[:4]
    ]
    if rel_labels:
        parts.append(f"| Related: {'; '.join(rel_labels)}")

    # Add concept names for richer semantic matching
    concept_names = [c["name"] for c in entity.get("concepts", []) if c.get("name")]
    if concept_names:
        parts.append(f"| Concepts: {', '.join(concept_names)}")

    return " ".join(parts)


def embed_entities(subjects: list[dict], domain_entities: list[dict], embedding_client) -> tuple:
    """
    Embed subjects and domain entities for cosine similarity matching.

    Returns:
        (subject_embeddings: list[list[float]], domain_embeddings: list[list[float]])
    """
    subject_texts = [build_subject_text(s) for s in subjects]
    domain_texts = [build_domain_entity_text(e) for e in domain_entities]

    print(f"  Embedding {len(subject_texts)} subjects...")
    subject_embeddings = embed_texts(embedding_client, subject_texts)

    print(f"  Embedding {len(domain_texts)} domain entities...")
    domain_embeddings = embed_texts(embedding_client, domain_texts)

    return subject_embeddings, domain_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Resolve Correspondences (Simple Mode)
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def resolve_correspondences_simple(
    subjects: list[dict],
    domain_entities: list[dict],
    subject_embeddings: list[list[float]],
    domain_embeddings: list[list[float]],
    llm_client=None,
    threshold: float = 0.45,
    high_confidence: float = 0.65,
    direction: str = "subject",
) -> list[dict]:
    """
    Match subjects to domain entities using cosine similarity.

    direction controls the outer loop:
      - "subject" (default): For each subject, find best matching table(s).
        Ensures every subject is evaluated. Some tables may end up unlinked.
      - "domain_entity": For each table, find best matching subject(s).
        Ensures every table is evaluated. Some subjects may end up unlinked.

    For each entity in the outer loop:
      - Compute similarity against all candidates
      - If best score > high_confidence: create match directly
      - If best score between threshold and high_confidence: LLM confirmation
      - If best score < threshold: skip

    Returns list of:
        {
            "subject_name": str,
            "domain_entity_name": str,
            "confidence": float,
            "method": "embedding_similarity" | "llm_confirmed" | "llm_rejected",
            "reason": str,
        }
    """
    correspondences = []

    # ── Per-domain-entity direction ──────────────────────────────────
    if direction == "domain_entity":
        for j, de in enumerate(domain_entities):
            scores = []
            for i, subj in enumerate(subjects):
                sim = cosine_similarity(subject_embeddings[i], domain_embeddings[j])
                scores.append((i, sim))
            scores.sort(key=lambda x: x[1], reverse=True)

            for rank, (subj_idx, sim) in enumerate(scores):
                if sim < threshold:
                    break

                subj = subjects[subj_idx]
                sim_rounded = round(sim, 4)

                if sim >= high_confidence:
                    correspondences.append({
                        "subject_name": subj["name"],
                        "domain_entity_name": de["name"],
                        "confidence": sim_rounded,
                        "method": "embedding_similarity",
                        "reason": f"High cosine similarity ({sim_rounded}) between "
                                  f"table '{de['name']}' and subject '{subj['name']}'",
                    })
                    print(f"    + {de['name']:15s} ← {subj['name']:30s}  "
                          f"(sim: {sim_rounded}, HIGH confidence)")

                elif llm_client is not None:
                    print(f"    ? {de['name']:15s} ← {subj['name']:30s}  "
                          f"(sim: {sim_rounded}, checking with LLM...)", end=" ", flush=True)

                    confirmation = _llm_confirm_match(subj, de, sim_rounded, llm_client)

                    if confirmation["match"]:
                        correspondences.append({
                            "subject_name": subj["name"],
                            "domain_entity_name": de["name"],
                            "confidence": confirmation["confidence"],
                            "method": "llm_confirmed",
                            "reason": confirmation["reason"],
                        })
                        print(f"CONFIRMED ({confirmation['confidence']})")
                    else:
                        print(f"REJECTED — {confirmation['reason'][:50]}")
                else:
                    correspondences.append({
                        "subject_name": subj["name"],
                        "domain_entity_name": de["name"],
                        "confidence": sim_rounded,
                        "method": "embedding_similarity",
                        "reason": f"Cosine similarity ({sim_rounded}) between "
                                  f"table '{de['name']}' and subject '{subj['name']}'",
                    })
                    print(f"    + {de['name']:15s} ← {subj['name']:30s}  "
                          f"(sim: {sim_rounded}, AMBIGUOUS — no LLM confirmation)")

                # Keep top-5 subjects per domain entity
                if rank >= 4:
                    break

        return correspondences

    # ── Per-subject direction (default) ──────────────────────────────
    for i, subj in enumerate(subjects):
        # Compute similarity against all domain entities
        scores = []
        for j, de in enumerate(domain_entities):
            sim = cosine_similarity(subject_embeddings[i], domain_embeddings[j])
            scores.append((j, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Consider top matches
        for rank, (de_idx, sim) in enumerate(scores):
            if sim < threshold:
                break  # all remaining are below threshold

            de = domain_entities[de_idx]
            sim_rounded = round(sim, 4)

            if sim >= high_confidence:
                # High confidence — create edge directly
                correspondences.append({
                    "subject_name": subj["name"],
                    "domain_entity_name": de["name"],
                    "confidence": sim_rounded,
                    "method": "embedding_similarity",
                    "reason": f"High cosine similarity ({sim_rounded}) between "
                              f"subject '{subj['name']}' and table '{de['name']}'",
                })
                print(f"    + {subj['name']:30s} → {de['name']:15s}  "
                      f"(sim: {sim_rounded}, HIGH confidence)")

            elif llm_client is not None:
                # Ambiguous zone — ask LLM to confirm
                print(f"    ? {subj['name']:30s} → {de['name']:15s}  "
                      f"(sim: {sim_rounded}, checking with LLM...)", end=" ", flush=True)

                confirmation = _llm_confirm_match(
                    subj, de, sim_rounded, llm_client
                )

                if confirmation["match"]:
                    correspondences.append({
                        "subject_name": subj["name"],
                        "domain_entity_name": de["name"],
                        "confidence": confirmation["confidence"],
                        "method": "llm_confirmed",
                        "reason": confirmation["reason"],
                    })
                    print(f"CONFIRMED ({confirmation['confidence']})")
                else:
                    print(f"REJECTED — {confirmation['reason'][:50]}")
            else:
                # No LLM client — use embedding score directly in ambiguous zone
                correspondences.append({
                    "subject_name": subj["name"],
                    "domain_entity_name": de["name"],
                    "confidence": sim_rounded,
                    "method": "embedding_similarity",
                    "reason": f"Cosine similarity ({sim_rounded}) between "
                              f"subject '{subj['name']}' and table '{de['name']}'",
                })
                print(f"    + {subj['name']:30s} → {de['name']:15s}  "
                      f"(sim: {sim_rounded}, AMBIGUOUS — no LLM confirmation)")

            # Only link to top match (don't create multiple CORRESPONDS_TO per subject
            # unless scores are very close)
            if rank == 0:
                # If next score is significantly lower, stop
                if len(scores) > 1 and scores[1][1] < sim - 0.1:
                    break
            else:
                break  # only keep top-2 at most

    return correspondences


def _llm_confirm_match(
    subject: dict, domain_entity: dict, similarity: float, llm_client
) -> dict:
    """
    Ask LLM to confirm whether a subject corresponds to a domain entity.

    Returns: {"match": bool, "confidence": float, "reason": str}
    """
    # Build context from subject documents
    context_snippets = []
    for dc in subject.get("doc_contexts", [])[:3]:
        if dc.get("mention_context"):
            context_snippets.append(dc["mention_context"])
        elif dc.get("topic_summary"):
            context_snippets.append(dc["topic_summary"][:150])
    context_text = "; ".join(context_snippets) if context_snippets else subject.get("description", "")

    prompt = f"""You are an entity resolution system for an airlines knowledge graph.

Determine if the concept/entity from unstructured documents corresponds to the
structured database table. The "CORRESPONDS_TO" relationship means: data about
this concept can be found in or is related to this database table.

Subject (from unstructured documents):
  Name: {subject['name']}
  Type: {subject.get('type', 'unknown')}
  Context: {context_text}

Database Table (structured data):
  Table Name: {domain_entity['name']}
  Description: {domain_entity['description']}
  Domain: {domain_entity.get('domain', 'unknown')}
  Columns: {domain_entity.get('column_info', '[]')}

Embedding similarity score: {similarity}

Respond ONLY with valid JSON:
{{
  "match": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "1-2 sentence explanation"
}}"""

    response = call_llm(llm_client, prompt, temperature=0.1)
    parsed = parse_llm_json(response)

    if parsed and isinstance(parsed, dict):
        return {
            "match": bool(parsed.get("match", False)),
            "confidence": round(float(parsed.get("confidence", similarity)), 4),
            "reason": parsed.get("reason", "LLM confirmation"),
        }

    # Fallback — if LLM response can't be parsed, use moderate confidence
    return {
        "match": similarity >= 0.50,
        "confidence": round(similarity, 4),
        "reason": "LLM confirmation parse failed; using embedding similarity as fallback",
    }


def print_correspondences(correspondences: list[dict]):
    """Pretty-print resolved correspondences."""
    print(f"\n{'=' * 70}")
    print("  RESOLVED CORRESPONDENCES (Subject → Domain Entity)")
    print(f"{'=' * 70}")
    if not correspondences:
        print("  No correspondences found above threshold.")
    else:
        print(f"  Total: {len(correspondences)} correspondence(s)\n")
        for corr in sorted(correspondences, key=lambda c: c["confidence"], reverse=True):
            print(f"  {corr['subject_name']:30s} → {corr['domain_entity_name']:15s}"
                  f"  (confidence: {corr['confidence']}, method: {corr['method']})")
            if corr.get("reason"):
                print(f"    {' ' * 30}   {corr['reason'][:80]}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Build Subject Graph (CORRESPONDS_TO edges) in Neo4j
# ═══════════════════════════════════════════════════════════════════════════════

def build_subject_graph(correspondences: list[dict], client=None):
    """
    Write CORRESPONDS_TO edges to Cosmos DB (Gremlin).

    Only deletes existing CORRESPONDS_TO edges — preserves Layer 1 and Layer 2 data.
    Each edge has properties: confidence, method, reason.
    """
    if client is None:
        client = get_gremlin_client()

    # ── Clean existing CORRESPONDS_TO edges ───────────────────────────
    print("  Clearing existing CORRESPONDS_TO edges...")
    try:
        run_gremlin_write(client, "g.E().hasLabel('CORRESPONDS_TO').drop()")
    except Exception as e:
        print(f"  WARN: Could not drop CORRESPONDS_TO edges: {e}")

    # ── Insert CORRESPONDS_TO edges ───────────────────────────────────
    print("\n  Inserting CORRESPONDS_TO edges...")
    created = 0
    for corr in correspondences:
        subj_id = make_vertex_id("Subject", corr["subject_name"])
        de_id = make_vertex_id("DomainEntity", corr["domain_entity_name"])
        corr_method = corr["method"]
        corr_reason = corr["reason"]
        try:
            run_gremlin_write(client,
                f"g.V('{esc(subj_id)}')"
                f".addE('CORRESPONDS_TO')"
                f".to(g.V('{esc(de_id)}'))"
                f".property('confidence', {corr['confidence']})"
                f".property('method', '{esc(corr_method)}')"
                f".property('reason', '{esc(corr_reason)}')"
            )
            created += 1
            print(f"    + {corr['subject_name']} —[CORRESPONDS_TO]→ {corr['domain_entity_name']}  "
                  f"(confidence: {corr['confidence']})")
        except Exception as e:
            print(f"    WARN: CORRESPONDS_TO edge failed: "
                  f"{corr['subject_name']} → {corr['domain_entity_name']}: {e}")

    print(f"\n  Subject Graph built! {created} CORRESPONDS_TO edge(s) created in Cosmos DB.")
    return client


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6 — Cross-Source Query Interface
# ═══════════════════════════════════════════════════════════════════════════════

def query_subject_graph(question: str, client=None) -> dict:
    """
    Full cross-layer query: traverses Subject → CORRESPONDS_TO → DomainEntity
    and Subject ← MENTIONS ← Document.

    This is the core "agent router" function — given a question, it returns:
      - structured_sources: DomainEntity tables to query via SQL
      - unstructured_sources: Documents to fetch from vector store
      - bridge_subjects: The subjects that bridge both worlds

    Returns:
        {
            "structured_sources": [
                {"table_name": str, "description": str, "domain": str,
                 "key_columns": str, "confidence": float, "bridged_by": [subject_names]}
            ],
            "unstructured_sources": [
                {"document": str, "summary": str, "subjects": [str]}
            ],
            "bridge_subjects": [
                {"name": str, "type": str, "linked_table": str, "confidence": float}
            ]
        }
    """
    result = {
        "structured_sources": [],
        "unstructured_sources": [],
        "bridge_subjects": [],
    }

    # ── Keyword extraction ──────────────────────────────────────────
    stop_words = {
        "which", "what", "about", "have", "does", "that", "this",
        "with", "from", "tell", "show", "find", "data", "tables",
        "related", "are", "the", "document", "documents", "mention",
        "mentions", "where", "who", "how", "many", "any", "issues",
    }
    keywords = [
        w.strip("?.,!\"'").lower()
        for w in question.split()
        if len(w.strip("?.,!\"'")) > 2 and w.strip("?.,!\"'").lower() not in stop_words
    ]

    seen_tables = set()
    seen_docs = set()
    seen_subjects = set()

    # ── 1. Find subjects matching the question ────────────────────────
    all_subjects = run_gremlin(client,
        "g.V().hasLabel('Subject').valueMap('name','type','description')"
    )
    for rec in all_subjects:
        subj_name = gval(rec, "name", "")
        subj_type = gval(rec, "type", "")
        if not any(kw in subj_name.lower() for kw in keywords):
            continue
        if subj_name in seen_subjects:
            continue
        seen_subjects.add(subj_name)

        sid = make_vertex_id("Subject", subj_name)

        # Follow CORRESPONDS_TO to structured side
        bridges = run_gremlin(client,
            f"g.V('{esc(sid)}').outE('CORRESPONDS_TO')"
            ".project('table_name','description','domain','key_columns','confidence','reason')"
            ".by(inV().values('name'))"
            ".by(inV().coalesce(values('description'), constant('')))"
            ".by(inV().coalesce(values('domain'), constant('')))"
            ".by(inV().coalesce(values('key_columns'), constant('[]')))"
            ".by(values('confidence'))"
            ".by(coalesce(values('reason'), constant('')))"
        )
        for br in bridges:
            table_name = br.get("table_name", "")
            if table_name not in seen_tables:
                seen_tables.add(table_name)
                result["structured_sources"].append({
                    "table_name": table_name,
                    "description": br.get("description", ""),
                    "domain": br.get("domain", ""),
                    "key_columns": br.get("key_columns", "[]"),
                    "confidence": br.get("confidence", 0.0),
                    "bridged_by": [subj_name],
                })
            else:
                for src in result["structured_sources"]:
                    if src["table_name"] == table_name and subj_name not in src["bridged_by"]:
                        src["bridged_by"].append(subj_name)

            result["bridge_subjects"].append({
                "name": subj_name,
                "type": subj_type,
                "linked_table": table_name,
                "confidence": br.get("confidence", 0.0),
            })

        # Follow MENTIONS back to documents
        doc_records = run_gremlin(client,
            f"g.V('{esc(sid)}').in('MENTIONS')"
            ".project('doc_name','doc_summary')"
            ".by(values('name'))"
            ".by(coalesce(values('topic_summary'), constant('')))"
        )
        for dr in doc_records:
            doc_name = dr.get("doc_name", "")
            if doc_name not in seen_docs:
                seen_docs.add(doc_name)
                result["unstructured_sources"].append({
                    "document": doc_name,
                    "summary": dr.get("doc_summary", ""),
                    "subjects": [],
                })
            for unsrc in result["unstructured_sources"]:
                if unsrc["document"] == doc_name and subj_name not in unsrc["subjects"]:
                    unsrc["subjects"].append(subj_name)

    # ── 2. Also match DomainEntity directly by keyword ────────────────
    all_entities = run_gremlin(client,
        "g.V().hasLabel('DomainEntity')"
        ".valueMap('name','description','domain','key_columns')"
    )
    for rec in all_entities:
        name = gval(rec, "name", "")
        desc = gval(rec, "description", "")
        if any(kw in name.lower() or kw in desc.lower() for kw in keywords):
            if name not in seen_tables:
                seen_tables.add(name)
                result["structured_sources"].append({
                    "table_name": name,
                    "description": desc,
                    "domain": gval(rec, "domain", ""),
                    "key_columns": gval(rec, "key_columns", "[]"),
                    "confidence": 1.0,
                    "bridged_by": ["direct_keyword_match"],
                })

    return result


def print_query_results(question: str, results: dict):
    """Pretty-print cross-source query results."""
    print(f"\n{'=' * 70}")
    print(f"  CROSS-SOURCE QUERY: \"{question}\"")
    print(f"{'=' * 70}")

    ss = results["structured_sources"]
    us = results["unstructured_sources"]
    bs = results["bridge_subjects"]

    if ss:
        print(f"\n  Structured Sources ({len(ss)} table(s)):")
        for src in ss:
            print(f"    Table: {src['table_name']}  (domain: {src['domain']}, "
                  f"confidence: {src['confidence']})")
            print(f"      Description: {src['description'][:80]}")
            print(f"      Bridged by: {', '.join(src['bridged_by'])}")

    if us:
        print(f"\n  Unstructured Sources ({len(us)} document(s)):")
        for unsrc in us:
            print(f"    Document: {unsrc['document']}")
            if unsrc.get("summary"):
                print(f"      Summary: {unsrc['summary'][:80]}")
            print(f"      Subjects: {', '.join(unsrc['subjects'])}")

    if bs:
        print(f"\n  Bridge Subjects ({len(bs)}):")
        for b in bs:
            print(f"    [{b['type']:10s}] {b['name']} → {b['linked_table']}  "
                  f"(confidence: {b['confidence']})")

    if not ss and not us and not bs:
        print("  No matching results found across any layer.")

    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 7 — Visualize the Full 3-Layer Bridge
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_subject_graph(client=None):
    """Print all CORRESPONDS_TO edges and full 3-layer graph summary (Cosmos DB)."""
    if client is None:
        client = get_gremlin_client()

    print(f"\n{'=' * 70}")
    print("  SUBJECT GRAPH — CORRESPONDS_TO BRIDGE (Cosmos DB)")
    print(f"{'=' * 70}")

    # CORRESPONDS_TO edges
    print("\n  CORRESPONDS_TO Edges (Subject → DomainEntity):")
    records = run_gremlin(client,
        "g.V().hasLabel('Subject').outE('CORRESPONDS_TO')"
        ".project('subject','subject_type','table_name','domain','confidence','method','reason')"
        ".by(outV().values('name'))"
        ".by(outV().coalesce(values('type'), constant('')))"
        ".by(inV().values('name'))"
        ".by(inV().coalesce(values('domain'), constant('')))"
        ".by(values('confidence'))"
        ".by(coalesce(values('method'), constant('')))"
        ".by(coalesce(values('reason'), constant('')))"
    )
    if not records:
        print("    (no CORRESPONDS_TO edges found)")
    for rec in records:
        stype = rec.get("subject_type", "")
        subj = rec.get("subject", "")
        table = rec.get("table_name", "")
        conf = rec.get("confidence", 0)
        method = rec.get("method", "")
        reason = rec.get("reason", "")
        print(f"    [{stype:10s}] {subj:30s} —[CORRESPONDS_TO]→ "
              f"{table:15s}  (confidence: {conf}, method: {method})")
        if reason:
            print(f"      {' ' * 42} {reason[:70]}")

    # Full paths: Document → Subject → DomainEntity
    print(f"\n  Full Paths (Document → Subject → DomainEntity):")
    path_recs = run_gremlin(client,
        "g.V().hasLabel('Document').as('doc')"
        ".out('MENTIONS').as('s')"
        ".out('CORRESPONDS_TO').as('de')"
        ".select('doc','s','de')"
        ".by(values('name'))"
        ".by(values('name'))"
        ".by(values('name'))"
    )
    if not path_recs:
        print("    (no complete Document → Subject → DomainEntity paths found)")
    for rec in path_recs:
        print(f"    {rec.get('doc',''):25s} → {rec.get('s',''):25s} → {rec.get('de','')}")

    # SPO triplets
    print(f"\n  SPO Triplets (Subject → Object):")
    spo_recs = run_gremlin(client,
        "g.V().hasLabel('Subject').outE('RELATES_TO')"
        ".project('subject','predicate','object')"
        ".by(outV().values('name'))"
        ".by(values('predicate'))"
        ".by(inV().values('name'))"
    )
    if not spo_recs:
        print("    (no RELATES_TO edges found)")
    for rec in spo_recs:
        print(f"    {rec.get('subject',''):25s} —[{rec.get('predicate','')}]→ {rec.get('object','')}")

    # Summary counts
    print(f"\n  Summary:")
    for label in ["DomainEntity", "Document", "Subject", "Object"]:
        count_result = run_gremlin(client, f"g.V().hasLabel('{label}').count()")
        count = count_result[0] if count_result else 0
        print(f"    {label} vertices: {count}")
    for rel_type in ["HAS_FK", "MENTIONS", "RELATES_TO", "CORRESPONDS_TO"]:
        count_result = run_gremlin(client, f"g.E().hasLabel('{rel_type}').count()")
        count = count_result[0] if count_result else 0
        print(f"    {rel_type} edges: {count}")

    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 3 — Subject Graph Bridge Builder")
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Use ReAct agent-based entity resolution (iterative graph exploration) "
        "instead of embedding similarity",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Cosine similarity threshold for matching (default: 0.45)",
    )
    parser.add_argument(
        "--direction",
        choices=["subject", "domain_entity"],
        default="subject",
        help="Resolution direction: 'subject' (default) loops per subject to find "
        "matching tables; 'domain_entity' loops per table to find matching subjects",
    )
    args = parser.parse_args()

    mode = "ReAct Agent" if args.advanced else "Embedding Similarity"
    direction_label = "per subject → tables" if args.direction == "subject" else "per table → subjects"
    print("=" * 70)
    print("  LAYER 3 — SUBJECT GRAPH BRIDGE BUILDER")
    print("  Airlines: Bridging Unstructured ↔ Structured")
    print(f"  Resolution mode: {mode}")
    print(f"  Resolution direction: {direction_label}")
    if not args.advanced:
        print(f"  Similarity threshold: {args.threshold}")
    print("=" * 70)

    # Step 1: Fetch Subjects
    print("\n[Step 1] Fetching Subject vertices from Cosmos DB (Layer 2)...")
    gremlin = get_gremlin_client()
    subjects = fetch_subjects(gremlin)
    if not subjects:
        print("  ERROR: No Subject vertices found. Run Layer 2 (lexical_graph.py) first.")
        gremlin.close()
        return
    print_subjects(subjects)

    # Step 2: Fetch Domain Entities
    print("\n[Step 2] Fetching DomainEntity vertices from Cosmos DB (Layer 1)...")
    domain_entities = fetch_domain_entities(gremlin)
    if not domain_entities:
        print("  ERROR: No DomainEntity vertices found. Run Layer 1 (domain_graph.py) first.")
        gremlin.close()
        return
    print_domain_entities(domain_entities)

    # Step 3-4: Resolve correspondences
    llm_client = get_llm_client()
    embedding_client = get_embedding_client()

    if args.advanced:
        print("\n[Step 3] Resolving correspondences with ReAct Agent (iterative exploration)...")
        from agents.subject_agent import resolve_correspondences_advanced

        # Open LanceDB for the agent's vector search tool
        import lancedb
        lance_db = lancedb.connect(os.path.join(PROJECT_ROOT, "source_data", "lancedb_store"))
        try:
            lance_table = lance_db.open_table("lexical_chunks")
        except Exception as e:
            print(f"  WARN: Could not open LanceDB table: {e}")
            print("  The advanced agent will work without vector search.")
            lance_table = None

        correspondences = resolve_correspondences_advanced(
            subjects, domain_entities, llm_client, gremlin,
            lance_table=lance_table, embedding_client=embedding_client,
            direction=args.direction,
        )
    else:
        print("\n[Step 3] Embedding subjects and domain entities...")
        subject_embeddings, domain_embeddings = embed_entities(
            subjects, domain_entities, embedding_client
        )

        print("\n[Step 4] Resolving correspondences (cosine similarity + LLM confirmation)...")
        correspondences = resolve_correspondences_simple(
            subjects, domain_entities,
            subject_embeddings, domain_embeddings,
            llm_client=llm_client,
            threshold=args.threshold,
            direction=args.direction,
        )

    print_correspondences(correspondences)

    # Step 5: Build graph
    print("\n[Step 5] Building Subject Graph (CORRESPONDS_TO edges) in Cosmos DB...")
    build_subject_graph(correspondences, gremlin)

    # Step 6: Visualize
    visualize_subject_graph(gremlin)

    # Step 7: Demo queries
    demo_questions = [
        "Which aircraft have had maintenance issues?",
        "What do we know about flight delays?",
        "Tell me about crew scheduling problems",
        "Which routes have the highest passenger load?",
        "What incident data exists for recent flights?",
    ]
    print("\n\n[Step 7] Running demo cross-source queries...")
    for q in demo_questions:
        results = query_subject_graph(q, gremlin)
        print_query_results(q, results)

    gremlin.close()
    print("\nDone! Subject Graph stored in Cosmos DB (indigokg / knowledgegraph).")
    print("Query with Gremlin: g.V().hasLabel('Subject').out('CORRESPONDS_TO').valueMap(true)")


if __name__ == "__main__":
    main()
