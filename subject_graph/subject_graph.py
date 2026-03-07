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
  python subject_graph/subject_graph.py              # simple mode (embedding similarity)
  python subject_graph/subject_graph.py --advanced    # ReAct agent mode
"""

import argparse
import json
import math
import os
import sys

# ─── Ensure project root is on sys.path ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.llm import get_llm_client, call_llm, parse_llm_json, get_embedding_client, embed_texts
from utils.neo4j_helpers import get_neo4j_driver, run_cypher, run_cypher_write


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Fetch Subject Nodes (from Layer 2)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_subjects(driver=None) -> list[dict]:
    """
    Read all :Subject nodes from Neo4j, including their document context
    via MENTIONS edges (Document → Subject).

    Returns list of:
        {
            "name": str,
            "type": str,
            "description": str,
            "mention_count": int,
            "doc_contexts": [{"doc_name": str, "topic_summary": str, "mention_context": str}]
        }
    """
    if driver is None:
        driver = get_neo4j_driver()

    # Get all Subject nodes
    records = run_cypher(driver, """
        MATCH (s:Subject)
        RETURN s.name AS name, s.type AS type,
               s.description AS description, s.mention_count AS mention_count
        ORDER BY s.mention_count DESC
    """)

    subjects = []
    for rec in records:
        # For each subject, fetch document contexts via MENTIONS edges
        docs = run_cypher(driver, """
            MATCH (d:Document)-[r:MENTIONS]->(s:Subject {name: $name})
            RETURN d.name AS doc_name, d.topic_summary AS topic_summary,
                   r.context AS context
            ORDER BY d.name
        """, {"name": rec["name"]})

        subjects.append({
            "name": rec["name"],
            "type": rec["type"] or "unknown",
            "description": rec["description"] or "",
            "mention_count": rec["mention_count"] or 1,
            "doc_contexts": [
                {
                    "doc_name": d["doc_name"],
                    "topic_summary": d.get("topic_summary", ""),
                    "mention_context": d.get("context", ""),
                }
                for d in docs
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

def fetch_domain_entities(driver=None) -> list[dict]:
    """
    Read all :DomainEntity nodes from Neo4j, including their relationships.

    Returns list of:
        {
            "name": str,
            "description": str,
            "domain": str,
            "key_columns": str (JSON array),
            "column_info": str (JSON array),
            "row_count": int,
            "relationships": [{"type": str, "direction": str, "target": str}]
        }
    """
    if driver is None:
        driver = get_neo4j_driver()

    records = run_cypher(driver, """
        MATCH (d:DomainEntity)
        RETURN d.name AS name, d.description AS description,
               d.domain AS domain, d.key_columns AS key_columns,
               d.column_info AS column_info, d.row_count AS row_count
        ORDER BY d.name
    """)

    entities = []
    for rec in records:
        # Fetch relationships
        out_rels = run_cypher(driver, """
            MATCH (a:DomainEntity {name: $name})-[r]->(b:DomainEntity)
            RETURN type(r) AS rel_type, b.name AS target, r.reason AS reason
        """, {"name": rec["name"]})

        in_rels = run_cypher(driver, """
            MATCH (a:DomainEntity)-[r]->(b:DomainEntity {name: $name})
            RETURN type(r) AS rel_type, a.name AS source, r.reason AS reason
        """, {"name": rec["name"]})

        relationships = []
        for r in out_rels:
            relationships.append({
                "type": r["rel_type"],
                "direction": "outgoing",
                "target": r["target"],
                "reason": r.get("reason", ""),
            })
        for r in in_rels:
            relationships.append({
                "type": r["rel_type"],
                "direction": "incoming",
                "target": r["source"],
                "reason": r.get("reason", ""),
            })

        entities.append({
            "name": rec["name"],
            "description": rec["description"] or "",
            "domain": rec["domain"] or "unknown",
            "key_columns": rec["key_columns"] or "[]",
            "column_info": rec["column_info"] or "[]",
            "row_count": rec["row_count"] or 0,
            "relationships": relationships,
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
    Combines the subject name, type, description, and document context.
    """
    parts = [subject["name"]]
    if subject.get("type") and subject["type"] != "unknown":
        parts.append(f"(type: {subject['type']})")
    if subject.get("description"):
        parts.append(f"— {subject['description']}")

    # Add document contexts for richer semantics
    contexts = []
    for dc in subject.get("doc_contexts", [])[:3]:  # limit to 3 contexts
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

    prompt = f"""You are an entity resolution system for IndiGo Airlines knowledge graph.

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

def build_subject_graph(correspondences: list[dict], driver=None):
    """
    Write CORRESPONDS_TO edges to Neo4j.

    Only deletes existing CORRESPONDS_TO edges — preserves Layer 1 and Layer 2 data.
    Each edge has properties: confidence, method, reason.
    """
    if driver is None:
        driver = get_neo4j_driver()

    # ── Clean existing CORRESPONDS_TO edges only ─────────────────────
    print("  Clearing existing CORRESPONDS_TO edges...")
    run_cypher_write(driver, "MATCH ()-[r:CORRESPONDS_TO]->() DELETE r")

    # ── Insert CORRESPONDS_TO edges ──────────────────────────────────
    print("\n  Inserting CORRESPONDS_TO edges...")
    created = 0
    for corr in correspondences:
        try:
            run_cypher_write(driver, """
                MATCH (s:Subject {name: $subject_name})
                MATCH (d:DomainEntity {name: $domain_entity_name})
                CREATE (s)-[:CORRESPONDS_TO {
                    confidence: $confidence,
                    method: $method,
                    reason: $reason
                }]->(d)
            """, {
                "subject_name": corr["subject_name"],
                "domain_entity_name": corr["domain_entity_name"],
                "confidence": corr["confidence"],
                "method": corr["method"],
                "reason": corr["reason"],
            })
            created += 1
            print(f"    + {corr['subject_name']} —[CORRESPONDS_TO]→ {corr['domain_entity_name']}  "
                  f"(confidence: {corr['confidence']})")
        except Exception as e:
            # Try case-insensitive fallback for Subject match
            try:
                run_cypher_write(driver, """
                    MATCH (s:Subject) WHERE toLower(s.name) = toLower($subject_name)
                    MATCH (d:DomainEntity {name: $domain_entity_name})
                    CREATE (s)-[:CORRESPONDS_TO {
                        confidence: $confidence,
                        method: $method,
                        reason: $reason
                    }]->(d)
                """, {
                    "subject_name": corr["subject_name"],
                    "domain_entity_name": corr["domain_entity_name"],
                    "confidence": corr["confidence"],
                    "method": corr["method"],
                    "reason": corr["reason"],
                })
                created += 1
                print(f"    + {corr['subject_name']} —[CORRESPONDS_TO]→ {corr['domain_entity_name']}  "
                      f"(confidence: {corr['confidence']}, case-insensitive match)")
            except Exception as e2:
                print(f"    WARN: CORRESPONDS_TO edge failed: "
                      f"{corr['subject_name']} → {corr['domain_entity_name']}: {e2}")

    print(f"\n  Subject Graph built! {created} CORRESPONDS_TO edge(s) created.")
    print(f"  View at: http://localhost:7474")
    print(f"  Try: MATCH (s:Subject)-[r:CORRESPONDS_TO]->(d:DomainEntity) RETURN s, r, d")
    return driver


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6 — Cross-Source Query Interface
# ═══════════════════════════════════════════════════════════════════════════════

def query_subject_graph(question: str, driver=None) -> dict:
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
    if driver is None:
        driver = get_neo4j_driver()

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

    # ── 1. Find subjects matching the question ───────────────────────
    for kw in keywords:
        records = run_cypher(driver, """
            MATCH (s:Subject)
            WHERE toLower(s.name) CONTAINS $kw
            RETURN s.name AS name, s.type AS type, s.description AS description
        """, {"kw": kw})

        for rec in records:
            subj_name = rec["name"]
            if subj_name in seen_subjects:
                continue
            seen_subjects.add(subj_name)

            # ── Follow CORRESPONDS_TO to structured side ─────────
            bridges = run_cypher(driver, """
                MATCH (s:Subject {name: $name})-[r:CORRESPONDS_TO]->(d:DomainEntity)
                RETURN d.name AS table_name, d.description AS description,
                       d.domain AS domain, d.key_columns AS key_columns,
                       r.confidence AS confidence, r.reason AS reason
            """, {"name": subj_name})

            for br in bridges:
                table_name = br["table_name"]
                if table_name not in seen_tables:
                    seen_tables.add(table_name)
                    result["structured_sources"].append({
                        "table_name": table_name,
                        "description": br["description"],
                        "domain": br["domain"],
                        "key_columns": br["key_columns"],
                        "confidence": br["confidence"],
                        "bridged_by": [subj_name],
                    })
                else:
                    # Add this subject as additional bridge
                    for src in result["structured_sources"]:
                        if src["table_name"] == table_name:
                            if subj_name not in src["bridged_by"]:
                                src["bridged_by"].append(subj_name)

                result["bridge_subjects"].append({
                    "name": subj_name,
                    "type": rec["type"],
                    "linked_table": table_name,
                    "confidence": br["confidence"],
                })

            # ── Follow MENTIONS back to documents ─────────────────
            doc_records = run_cypher(driver, """
                MATCH (d:Document)-[:MENTIONS]->(s:Subject {name: $name})
                RETURN d.name AS doc_name, d.topic_summary AS doc_summary
            """, {"name": subj_name})

            for dr in doc_records:
                doc_name = dr["doc_name"]
                if doc_name not in seen_docs:
                    seen_docs.add(doc_name)
                    result["unstructured_sources"].append({
                        "document": doc_name,
                        "summary": dr.get("doc_summary", ""),
                        "subjects": [],
                    })

                # Add subject to document entry
                for unsrc in result["unstructured_sources"]:
                    if unsrc["document"] == doc_name:
                        if subj_name not in unsrc["subjects"]:
                            unsrc["subjects"].append(subj_name)

    # ── 2. Also match DomainEntity directly (for keywords like table names) ──
    for kw in keywords:
        records = run_cypher(driver, """
            MATCH (d:DomainEntity)
            WHERE toLower(d.name) CONTAINS $kw OR toLower(d.description) CONTAINS $kw
            RETURN d.name AS name, d.description AS description,
                   d.domain AS domain, d.key_columns AS key_columns
        """, {"kw": kw})

        for rec in records:
            if rec["name"] not in seen_tables:
                seen_tables.add(rec["name"])
                result["structured_sources"].append({
                    "table_name": rec["name"],
                    "description": rec["description"],
                    "domain": rec["domain"],
                    "key_columns": rec["key_columns"],
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

def visualize_subject_graph(driver=None):
    """Print all CORRESPONDS_TO edges and full 3-layer graph summary."""
    if driver is None:
        driver = get_neo4j_driver()

    print(f"\n{'=' * 70}")
    print("  SUBJECT GRAPH — CORRESPONDS_TO BRIDGE")
    print(f"{'=' * 70}")

    # ── CORRESPONDS_TO edges ──────────────────────────────────────────
    print("\n  CORRESPONDS_TO Edges (Subject → DomainEntity):")
    records = run_cypher(driver, """
        MATCH (s:Subject)-[r:CORRESPONDS_TO]->(d:DomainEntity)
        RETURN s.name AS subject, s.type AS subject_type,
               d.name AS table_name, d.domain AS domain,
               r.confidence AS confidence, r.method AS method,
               r.reason AS reason
        ORDER BY r.confidence DESC
    """)
    if not records:
        print("    (no CORRESPONDS_TO edges found)")
    for rec in records:
        print(f"    [{rec['subject_type']:10s}] {rec['subject']:30s} —[CORRESPONDS_TO]→ "
              f"{rec['table_name']:15s}  (confidence: {rec['confidence']}, "
              f"method: {rec['method']})")
        if rec.get("reason"):
            print(f"      {' ' * 42} {rec['reason'][:70]}")

    # ── Full 2-layer traversal (no Chunk nodes in KG) ───────────────
    print(f"\n  Full Paths (Document → Subject → DomainEntity):")
    records = run_cypher(driver, """
        MATCH (doc:Document)-[:MENTIONS]->(s:Subject)
              -[r:CORRESPONDS_TO]->(de:DomainEntity)
        RETURN doc.name AS document,
               s.name AS subject, de.name AS table_name,
               r.confidence AS confidence
        ORDER BY doc.name, s.name
    """)
    if not records:
        print("    (no complete Document → Subject → DomainEntity paths found)")
    for rec in records:
        print(f"    {rec['document']:25s} → "
              f"{rec['subject']:25s} → {rec['table_name']}"
              f"  ({rec['confidence']})")

    # ── Summary counts ────────────────────────────────────────────────
    print(f"\n  Summary:")
    for label in ["DomainEntity", "Document", "Subject"]:
        count = run_cypher(driver, f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]
        print(f"    {label} nodes: {count}")

    for rel_type in ["HAS_FK", "MENTIONS", "CORRESPONDS_TO"]:
        count = run_cypher(
            driver,
            f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c"
        )[0]["c"]
        print(f"    {rel_type} edges: {count}")

    # Semantic edges (dynamic types from Layer 1)
    sem_count = run_cypher(driver, """
        MATCH (a:DomainEntity)-[r]->(b:DomainEntity)
        WHERE type(r) <> 'HAS_FK'
        RETURN count(r) AS c
    """)[0]["c"]
    print(f"    Semantic edges (Layer 1): {sem_count}")

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
    print("  IndiGo Airlines: Bridging Unstructured ↔ Structured")
    print(f"  Resolution mode: {mode}")
    print(f"  Resolution direction: {direction_label}")
    if not args.advanced:
        print(f"  Similarity threshold: {args.threshold}")
    print("=" * 70)

    # Step 1: Fetch Subjects
    print("\n[Step 1] Fetching Subject nodes from Neo4j (Layer 2)...")
    driver = get_neo4j_driver()
    subjects = fetch_subjects(driver)
    if not subjects:
        print("  ERROR: No Subject nodes found. Run Layer 2 (lexical_graph.py) first.")
        driver.close()
        return
    print_subjects(subjects)

    # Step 2: Fetch Domain Entities
    print("\n[Step 2] Fetching DomainEntity nodes from Neo4j (Layer 1)...")
    domain_entities = fetch_domain_entities(driver)
    if not domain_entities:
        print("  ERROR: No DomainEntity nodes found. Run Layer 1 (domain_graph.py) first.")
        driver.close()
        return
    print_domain_entities(domain_entities)

    # Step 3-4: Resolve correspondences
    llm_client = get_llm_client()
    embedding_client = get_embedding_client()

    if args.advanced:
        print("\n[Step 3] Resolving correspondences with ReAct Agent (iterative exploration)...")
        from subject_graph.enrich_advanced import resolve_correspondences_advanced

        # Open LanceDB for the agent's vector search tool
        import lancedb
        lance_db = lancedb.connect(os.path.join(PROJECT_ROOT, "data", "lancedb_store"))
        try:
            lance_table = lance_db.open_table("lexical_chunks")
        except Exception as e:
            print(f"  WARN: Could not open LanceDB table: {e}")
            print("  The advanced agent will work without vector search.")
            lance_table = None

        correspondences = resolve_correspondences_advanced(
            subjects, domain_entities, llm_client, driver,
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
    print("\n[Step 5] Building Subject Graph (CORRESPONDS_TO edges) in Neo4j...")
    build_subject_graph(correspondences, driver)

    # Step 6: Visualize
    visualize_subject_graph(driver)

    # Step 7: Demo queries
    demo_questions = [
        "Which suppliers have quality issues?",
        "What do we know about brake assemblies?",
        "Tell me about engine maintenance problems",
        "Which parts come from Pratt & Whitney?",
        "What quality data exists for the A320neo fleet?",
    ]
    print("\n\n[Step 7] Running demo cross-source queries...")
    for q in demo_questions:
        results = query_subject_graph(q, driver)
        print_query_results(q, results)

    driver.close()
    print("\nDone! View the full graph at http://localhost:7474")
    print("Try: MATCH (d:Document)-[:MENTIONS]->(s:Subject)"
          "-[:CORRESPONDS_TO]->(de:DomainEntity) RETURN d, s, de")


if __name__ == "__main__":
    main()
