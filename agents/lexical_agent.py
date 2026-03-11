"""
enrich_advanced.py — Advanced ReAct Agent-based Abstract SPO Triplet Extraction for Layer 2

Instead of a single LLM call per chunk (lexical_graph.extract_spo_triplets_simple),
this module uses a custom ReAct agent that iteratively explores the document
landscape via a VectorDB query tool (LanceDB) to build richer, more accurate
ABSTRACT / ONTOLOGY-LEVEL SPO triplet extractions with cross-chunk awareness.

Key design principle: Subjects and objects are ABSTRACT CATEGORIES, ROLES, or
CONCEPT CLASSES — never specific proper nouns or instance-level identifiers.
For example, "Anil Kumar" becomes "Pilot", "VT-ANQ" becomes "Narrow-Body Aircraft",
"JFK" becomes "Hub Airport". This produces a conceptual ontology rather than
a factual knowledge graph of specific events.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │             LexicalEnrichmentAgent                           │
  │                                                              │
  │  THOUGHT ──► ACTION (vector_db_query_tool) ──► OBSERVE       │
  │     ▲                                            │           │
  │     └────────────────────────────────────────────┘           │
  │                                                              │
  │  Repeats until FINAL_ANSWER or max iterations reached        │
  └──────────────────────────────────────────────────────────────┘

The agent has one tool — VectorDBQueryTool — which gives it access to the
LanceDB table of embedded document chunks: semantic search, chunk
retrieval, metadata filtering, and collection stats.

Usage:
    from agents.lexical_agent import extract_spo_triplets_advanced

    spo_by_chunk = extract_spo_triplets_advanced(
        chunks, llm_client, lance_table, embedding_client
    )
    # Returns same dict format as lexical_graph.extract_spo_triplets_simple
"""

import json
import os
import re
import sys
import textwrap

from openai import AzureOpenAI

# Ensure src/ is on sys.path for utils imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))    # agents/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                  # project root
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.llm import call_llm, parse_llm_json, embed_texts

import lancedb


# ═══════════════════════════════════════════════════════════════════════════════
#  Vector DB Query Tool (LanceDB)
# ═══════════════════════════════════════════════════════════════════════════════

class VectorDBQueryTool:
    """
    Read-only tool that gives an agent access to explore a LanceDB table
    of embedded document chunks.

    Supported actions:
      - search_similar(query, n)    → semantic similarity search
      - list_documents()            → list all unique document names
      - get_chunk(chunk_id)         → retrieve full text + metadata for a chunk
      - get_chunks_by_doc(doc_name) → all chunks from a specific document
      - get_collection_stats()      → table size, document count, etc.
    """

    TOOL_DESCRIPTION = textwrap.dedent("""\
    vector_db_query_tool — Read-only access to the LanceDB document chunk table.

    Available actions (pass as JSON):

    1. {"action": "search_similar", "query": "<natural language query>", "n": 5}
       Returns: top-N chunks most semantically similar to the query.
       Includes chunk_id, text, similarity score, and metadata.
       Great for finding chunks about specific topics, entities, or events.

    2. {"action": "list_documents"}
       Returns: list of all unique document names in the table.

    3. {"action": "get_chunk", "chunk_id": "<chunk_id>"}
       Returns: full text and metadata for a specific chunk by ID.
       Use this to read a chunk you found via search_similar.

    4. {"action": "get_chunks_by_doc", "doc_name": "<document_name>"}
       Returns: all chunks belonging to a specific document, ordered by index.
       Use this to understand the full structure of a document.

    5. {"action": "get_collection_stats"}
       Returns: total chunk count, document count, and table metadata.
    """)

    def __init__(self, table: lancedb.table.Table, embedding_client):
        self.table = table
        self.embedding_client = embedding_client

    def execute(self, action_input: dict) -> str:
        """Execute a tool action and return a string result."""
        action = action_input.get("action", "").strip().lower()

        try:
            if action == "search_similar":
                return self._search_similar(
                    action_input.get("query", ""),
                    action_input.get("n", 5),
                )
            elif action == "list_documents":
                return self._list_documents()
            elif action == "get_chunk":
                return self._get_chunk(action_input["chunk_id"])
            elif action == "get_chunks_by_doc":
                return self._get_chunks_by_doc(action_input["doc_name"])
            elif action == "get_collection_stats":
                return self._get_collection_stats()
            else:
                return (
                    f"ERROR: Unknown action '{action}'. "
                    "Available: search_similar, list_documents, get_chunk, "
                    "get_chunks_by_doc, get_collection_stats"
                )
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    # ── Action implementations ──────────────────────────────────────────

    def _search_similar(self, query: str, n: int) -> str:
        if not query:
            return "ERROR: 'query' parameter is required for search_similar."

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
            score = round(1 - row["_distance"], 4)  # cosine similarity
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

    def _list_documents(self) -> str:
        df = self.table.to_pandas()
        doc_names = sorted(df["doc_name"].unique().tolist())
        return json.dumps({"documents": doc_names, "count": len(doc_names)}, indent=2)

    def _get_chunk(self, chunk_id: str) -> str:
        df = self.table.to_pandas()
        match = df[df["chunk_id"] == chunk_id]
        if match.empty:
            return f"ERROR: Chunk '{chunk_id}' not found."

        row = match.iloc[0]
        return json.dumps(
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": {
                    "doc_name": row["doc_name"],
                    "chunk_index": int(row["chunk_index"]),
                    "text_preview": row["text_preview"],
                    "char_count": int(row["char_count"]),
                },
            },
            indent=2,
        )

    def _get_chunks_by_doc(self, doc_name: str) -> str:
        df = self.table.to_pandas()
        doc_df = df[df["doc_name"] == doc_name].sort_values("chunk_index")
        chunks = []
        for _, row in doc_df.iterrows():
            chunks.append({
                "chunk_id": row["chunk_id"],
                "chunk_index": int(row["chunk_index"]),
                "text_preview": row["text"][:300],
                "char_count": int(row["char_count"]),
            })
        return json.dumps({"doc_name": doc_name, "chunks": chunks}, indent=2)

    def _get_collection_stats(self) -> str:
        df = self.table.to_pandas()
        total = len(df)
        doc_names = sorted(df["doc_name"].unique().tolist())
        return json.dumps(
            {
                "total_chunks": total,
                "documents": len(doc_names),
                "document_names": doc_names,
            },
            indent=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ReAct Lexical Enrichment Agent
# ═══════════════════════════════════════════════════════════════════════════════

class LexicalEnrichmentAgent:
    """
    Custom ReAct agent that iteratively explores embedded document chunks
    via LanceDB to produce rich entity extractions for a batch of chunks
    belonging to a single document.

    Unlike the simple single-call extraction, the agent can:
      - Run similarity searches to find cross-chunk references
      - Read related chunks for context
      - Build a comprehensive entity list with relationships between entities
      - Validate extracted entities against the broader document landscape
    """

    MAX_ITERATIONS = 10

    def __init__(
        self,
        llm_client: AzureOpenAI,
        tool: VectorDBQueryTool,
        target_chunks: list[dict],
        doc_name: str,
        verbose: bool = True,
    ):
        self.client = llm_client
        self.tool = tool
        self.target_chunks = target_chunks
        self.doc_name = doc_name
        self.verbose = verbose
        self.messages: list[dict] = []
        self.trace: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the ReAct loop and return extracted entities per chunk.

        Returns:
            {
                "chunk_id": [
                    {"name": str, "type": str, "context": str},
                    ...
                ],
                ...
            }
        """
        self._build_system_prompt()
        self._build_initial_user_prompt()

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            if self.verbose:
                print(f"      [Iteration {iteration}/{self.MAX_ITERATIONS}]")

            response_text = self._call_llm()
            parsed = self._parse_response(response_text)

            if parsed["type"] == "final_answer":
                if self.verbose:
                    print(f"      -> FINAL_ANSWER after {iteration} iteration(s)")
                return parsed["result"]

            elif parsed["type"] == "action":
                thought = parsed.get("thought", "")
                action_input = parsed["action_input"]

                if self.verbose:
                    print(f"      THOUGHT: {thought[:120]}{'...' if len(thought) > 120 else ''}")
                    print(f"      ACTION:  {json.dumps(action_input)}")

                observation = self.tool.execute(action_input)

                if self.verbose:
                    obs_preview = observation[:200] + ("..." if len(observation) > 200 else "")
                    print(f"      OBSERVE: {obs_preview}")

                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action_input,
                    "observation": observation,
                })

                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"OBSERVATION:\n{observation}\n\n"
                        "Continue your analysis. Either take another ACTION "
                        "or provide your FINAL_ANSWER."
                    ),
                })

            elif parsed["type"] == "error":
                if self.verbose:
                    print(f"      PARSE ERROR: {parsed.get('error', 'unknown')}")
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        "I could not parse your response. Please follow the exact format:\n\n"
                        "For an action:\n"
                        "THOUGHT: <your reasoning>\n"
                        "ACTION: vector_db_query_tool\n"
                        "ACTION_INPUT: <valid JSON>\n\n"
                        "Or for your final answer:\n"
                        "THOUGHT: <your final reasoning>\n"
                        "FINAL_ANSWER:\n<valid JSON — see required schema>"
                    ),
                })

        # Max iterations exhausted
        if self.verbose:
            print(f"      -> Max iterations ({self.MAX_ITERATIONS}) reached, forcing final answer...")
        return self._force_final_answer()

    # ── Prompt Construction ──────────────────────────────────────────

    def _build_system_prompt(self):
        chunk_ids = [c["chunk_id"] for c in self.target_chunks]
        chunk_previews = "\n".join(
            f"  - {c['chunk_id']}: \"{c['text'][:150]}...\""
            for c in self.target_chunks
        )

        system = textwrap.dedent(f"""\
        You are a senior information extraction agent analyzing airlines and aviation
        industry documents.

        Your task: Extract ONE **abstract, ontology-level** SPO (Subject–Predicate–Object)
        triplet per chunk from the document "{self.doc_name}" using the document chunks
        stored in a vector database. Each triplet should capture the **conceptual meaning**
        of a chunk — NOT specific instances or proper nouns.

        CRITICAL DISTINCTION:
        - WRONG (instance-level): "Anil Kumar" → reported → "redeye flight 302"
        - RIGHT (concept-level):  "Pilot" → reports → "Night Flight Operations"

        - WRONG: "Boeing 737-800 VT-ANQ" → experienced → "engine failure on 2024-03-15"
        - RIGHT: "Narrow-Body Aircraft" → experiences → "Engine Failure Incidents"

        - WRONG: "JFK Airport" → delayed → "Flight AA-1023"
        - RIGHT: "Hub Airport" → causes delays for → "Domestic Flights"

        Think of this as building an **ontology / conceptual schema** for the aviation
        domain, NOT a factual knowledge graph of specific events. Subjects and objects
        should be abstract categories, roles, or concept classes.

        You work in a ReAct loop — you THINK, then ACT (use a tool to explore
        the document chunks), then OBSERVE the result, and repeat until you have
        an abstract conceptual SPO triplet for every chunk.

        ── YOUR TOOL ──────────────────────────────────────────────────────────
        {self.tool.TOOL_DESCRIPTION}
        ── TARGET DOCUMENT ────────────────────────────────────────────────────

        Document: {self.doc_name}
        Chunks ({len(self.target_chunks)}):
        {chunk_previews}

        ── RESPONSE FORMAT ────────────────────────────────────────────────────

        On each turn, respond in EXACTLY one of these two formats:

        FORMAT A — Take an action:
        THOUGHT: <your reasoning about what to explore next>
        ACTION: vector_db_query_tool
        ACTION_INPUT: <valid JSON object for the tool>

        FORMAT B — Provide final answer:
        THOUGHT: <your final reasoning about the SPO triplets>
        FINAL_ANSWER:
        <valid JSON — see schema below>

        ── REQUIRED OUTPUT SCHEMA (for FINAL_ANSWER) ──────────────────────────

        {{
          "<chunk_id>": {{
            "subject": "<abstract category, role, or concept class — NOT a specific name/identifier>",
            "subject_type": "<one of: aircraft, flight, route, airport, crew, passenger, booking, fare_class, maintenance, incident, organization, person, event, metric, system, location, equipment, service, process, policy, regulation>",
            "predicate": "<conceptual relationship connecting subject concept to object concept>",
            "object": "<abstract category, role, or concept class — NOT a specific name/identifier>",
            "object_type": "<one of: aircraft, flight, route, airport, crew, passenger, booking, fare_class, maintenance, incident, organization, person, event, metric, system, location, equipment, service, process, policy, regulation>"
          }},
          ... (one key per chunk_id from: {chunk_ids})
        }}

        ── EXTRACTION RULES ───────────────────────────────────────────────────

        - Extract ONE abstract/conceptual SPO triplet per chunk that captures the
          chunk's thematic meaning at an **ontology level**
        - Subject and object must be ABSTRACT CATEGORIES, ROLES, or CONCEPT CLASSES
          — NEVER specific proper nouns, individual names, flight numbers, dates,
          serial numbers, or instance-level identifiers
        - Abstraction examples:
            • Person names ("Anil Kumar", "Jane Smith") → their ROLE ("Pilot", "Maintenance Engineer", "Passenger")
            • Specific aircraft ("VT-ANQ", "Boeing 737-800 MSN 29019") → class ("Narrow-Body Aircraft", "Commercial Aircraft")
            • Specific flights ("AI-302", "Flight 1023") → category ("Domestic Flight", "Long-Haul Flight", "Night Flight")
            • Specific airports ("JFK", "DEL") → role ("Hub Airport", "International Airport", "Origin Airport")
            • Specific dates/incidents → pattern ("Recurring Maintenance Issue", "Weather Disruption Event")
            • Specific metrics ("87.3% OTP") → concept ("On-Time Performance Metric")
        - The predicate should be a concise, reusable verbal phrase describing the
          conceptual relationship (e.g., "undergoes", "impacts", "requires",
          "is governed by", "contributes to", "triggers")
        - Together, the triplet should describe a GENERAL PATTERN or CONCEPTUAL
          RELATIONSHIP that the chunk illustrates — not the specific facts in it
        - Use cross-chunk context to identify recurring themes and abstract them
        - Ground entity types in the airlines and aviation domain
        - IMPORTANT: Use CONSISTENT abstract concept names across chunks and documents!
          If a concept appears in other documents, use the same abstract label.
          The goal is to create a shared conceptual ontology that links documents together.

        ── STRATEGY ───────────────────────────────────────────────────────────

        1. Start by reading each chunk's full text (use get_chunk or get_chunks_by_doc)
        2. CRITICAL: Use search_similar to find related chunks in OTHER documents!
           Search for key THEMES (maintenance patterns, operational issues, crew roles,
           fleet management) to understand the conceptual landscape across the corpus.
           This ensures you pick the right abstract categories.
        3. For each chunk, identify the ABSTRACT THEME — what general concept or
           pattern does this chunk illustrate? Elevate specific facts to their
           conceptual category.
        4. Determine the single best abstract/conceptual SPO triplet for each chunk
        5. When confident, produce FINAL_ANSWER with one conceptual SPO per chunk_id
        6. Aim for 4-8 exploration steps — at least 1-2 should be search_similar calls.
        """)

        self.messages.append({"role": "system", "content": system})

    def _build_initial_user_prompt(self):
        prompt = textwrap.dedent(f"""\
        Begin your analysis of the document "{self.doc_name}".

        Start by reading the full text of each chunk to understand the content.
        Then IMPORTANT: use search_similar to search for key THEMES and CONCEPTS
        (like maintenance patterns, crew roles, operational metrics, fleet management,
        safety incidents, route planning) across ALL documents in the vector DB.
        This helps you identify the right abstract categories and ensures conceptual
        consistency across documents.

        Remember: extract ABSTRACT, ONTOLOGY-LEVEL triplets — not instance-level facts.
        Replace specific names, numbers, and identifiers with their conceptual roles
        or categories.

        Produce ONE abstract SPO (Subject–Predicate–Object) triplet per chunk.
        """)
        self.messages.append({"role": "user", "content": prompt})

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
        """Parse LLM response into action, final_answer, or error."""
        text = text.strip()

        if "FINAL_ANSWER" in text:
            return self._parse_final_answer(text)

        if "ACTION_INPUT" in text:
            return self._parse_action(text)

        # Fallback: check for embedded JSON that looks like chunk-keyed output
        json_obj = self._try_extract_json(text)
        if json_obj and isinstance(json_obj, dict):
            # Check if it has chunk_id keys
            chunk_ids = {c["chunk_id"] for c in self.target_chunks}
            if any(k in chunk_ids for k in json_obj.keys()):
                return {"type": "final_answer", "result": self._normalize_result(json_obj)}

        return {"type": "error", "error": "Could not parse response"}

    def _parse_action(self, text: str) -> dict:
        thought = ""
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\nACTION)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        input_match = re.search(r"ACTION_INPUT:\s*(.+)", text, re.DOTALL)
        if not input_match:
            return {"type": "error", "error": "Found ACTION but no ACTION_INPUT"}

        raw_input = input_match.group(1).strip()

        for candidate in self._json_candidates(raw_input):
            try:
                action_input = json.loads(candidate)
                if isinstance(action_input, dict):
                    return {"type": "action", "thought": thought, "action_input": action_input}
            except json.JSONDecodeError:
                continue

        return {"type": "error", "error": f"Could not parse ACTION_INPUT: {raw_input[:200]}"}

    def _parse_final_answer(self, text: str) -> dict:
        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL)
        if not fa_match:
            return {"type": "error", "error": "Found FINAL_ANSWER marker but no content"}

        raw = fa_match.group(1).strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = raw.strip()

        for candidate in self._json_candidates(raw):
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return {"type": "final_answer", "result": self._normalize_result(result)}
            except json.JSONDecodeError:
                continue

        return {"type": "error", "error": f"Could not parse FINAL_ANSWER JSON: {raw[:300]}"}

    def _json_candidates(self, text: str) -> list[str]:
        """Generate candidate JSON strings from text."""
        candidates = []
        candidates.append(text.strip())
        first_line = text.split("\n")[0].strip()
        candidates.append(first_line)

        # Find the first { ... } block
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

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group())

        return candidates

    def _try_extract_json(self, text: str) -> dict | list | None:
        """Try to extract JSON from text."""
        for candidate in self._json_candidates(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _normalize_result(self, result: dict) -> dict:
        """Ensure result has correct structure: {chunk_id: {subject, subject_type, predicate, object, object_type}}."""
        chunk_ids = {c["chunk_id"] for c in self.target_chunks}
        empty_spo = {
            "subject": "", "subject_type": "unknown",
            "predicate": "", "object": "", "object_type": "unknown",
        }
        normalized = {}

        for key, spo in result.items():
            if key not in chunk_ids:
                continue
            if isinstance(spo, dict) and "subject" in spo and "object" in spo:
                normalized[key] = {
                    "subject": spo.get("subject", "").strip(),
                    "subject_type": spo.get("subject_type", "unknown"),
                    "predicate": spo.get("predicate", "").strip(),
                    "object": spo.get("object", "").strip(),
                    "object_type": spo.get("object_type", "unknown"),
                }
            else:
                normalized[key] = dict(empty_spo)

        # Ensure all chunk_ids are present
        for cid in chunk_ids:
            if cid not in normalized:
                normalized[cid] = dict(empty_spo)

        return normalized

    # ── Fallback ─────────────────────────────────────────────────────

    def _force_final_answer(self) -> dict:
        chunk_ids = [c["chunk_id"] for c in self.target_chunks]
        observations_summary = "\n\n".join(
            f"Step {t['iteration']}: {t['thought'][:150]}" for t in self.trace
        )

        self.messages.append({
            "role": "user",
            "content": textwrap.dedent(f"""\
            You have run out of exploration steps. Based on everything you observed,
            produce your FINAL_ANSWER now.

            Summary of your exploration:
            {observations_summary}

            Required chunk_ids in output: {chunk_ids}

            Respond with ONLY:
            FINAL_ANSWER:
            <valid JSON with one SPO triplet object per chunk_id>
            """),
        })

        response = self._call_llm()
        parsed = self._parse_response(response)

        if parsed["type"] == "final_answer":
            return parsed["result"]

        # Absolute fallback — empty SPO triplets
        empty_spo = {
            "subject": "", "subject_type": "unknown",
            "predicate": "", "object": "", "object_type": "unknown",
        }
        return {cid: dict(empty_spo) for cid in chunk_ids}


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-Document Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_extraction(
    spo_by_chunk: dict,
    all_chunks: list[dict],
    client: AzureOpenAI,
    verbose: bool = True,
) -> dict:
    """
    Post-extraction validation: review all extracted SPO triplets across all chunks
    for consistency, specificity, and completeness.
    """
    if verbose:
        print("\n    [Validation] Cross-chunk SPO triplet consistency check...")

    # Build a summary of what was extracted
    extraction_summary = {}
    for chunk_id, spo in spo_by_chunk.items():
        extraction_summary[chunk_id] = {
            "subject": spo.get("subject", ""),
            "subject_type": spo.get("subject_type", ""),
            "predicate": spo.get("predicate", ""),
            "object": spo.get("object", ""),
            "object_type": spo.get("object_type", ""),
        }

    # Get chunk text previews for context
    chunk_previews = {}
    for c in all_chunks:
        chunk_previews[c["chunk_id"]] = c["text"][:200]

    prompt = textwrap.dedent(f"""\
    You are a senior reviewer validating ABSTRACT / ONTOLOGY-LEVEL SPO
    (Subject-Predicate-Object) triplet extractions from airlines and aviation
    industry documents.

    Below are the extracted SPO triplets per chunk, along with chunk text previews.
    Review for:
    1. Abstraction level: Are subjects and objects ABSTRACT CATEGORIES, ROLES, or
       CONCEPT CLASSES — NOT specific proper nouns, individual names, flight numbers,
       dates, or instance-level identifiers? If you see specific names like
       "Anil Kumar", "VT-ANQ", "Flight AI-302", "JFK" etc., replace them with their
       abstract role ("Pilot", "Narrow-Body Aircraft", "Domestic Flight", "Hub Airport").
    2. Conceptual accuracy: Does each triplet capture the THEMATIC / CONCEPTUAL
       meaning of its chunk (the general pattern), not just the literal facts?
    3. Consistency: Are the same abstract concepts named consistently across chunks?
       (e.g., "Aircraft Maintenance" and "Plane Servicing" should be unified)
    4. Type accuracy: Are entity types correct?
       (e.g., a role like "Pilot" should be "crew" not "person")

    Chunk previews:
    {json.dumps(chunk_previews, indent=2)}

    Extracted SPO triplets:
    {json.dumps(extraction_summary, indent=2)}

    If the extraction is good, respond with EXACTLY:
    VALIDATED

    If you want to fix issues, respond with a corrected JSON object (same schema:
    {{chunk_id: {{subject, subject_type, predicate, object, object_type}}}} for ALL
    chunks). Include ONLY the JSON, no extra text.
    """)

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    response = completion.choices[0].message.content.strip()

    if "VALIDATED" in response.upper() and len(response) < 50:
        if verbose:
            print("    [Validation] SPO triplets validated — no changes needed.")
        return spo_by_chunk

    # Try to parse corrected JSON
    parsed = parse_llm_json(response)
    if parsed and isinstance(parsed, dict):
        corrected = dict(spo_by_chunk)
        changes = 0
        for chunk_id, spo in parsed.items():
            if chunk_id in corrected and isinstance(spo, dict) and "subject" in spo:
                corrected[chunk_id] = {
                    "subject": spo.get("subject", "").strip(),
                    "subject_type": spo.get("subject_type", "unknown"),
                    "predicate": spo.get("predicate", "").strip(),
                    "object": spo.get("object", "").strip(),
                    "object_type": spo.get("object_type", "unknown"),
                }
                changes += 1
        if verbose:
            print(f"    [Validation] Applied corrections to {changes} chunk(s).")
        return corrected

    if verbose:
        print("    [Validation] WARN: Could not parse correction, keeping original.")
    return spo_by_chunk


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — Drop-in replacement for lexical_graph.extract_subjects_simple
# ═══════════════════════════════════════════════════════════════════════════════

def extract_spo_triplets_advanced(
    chunks: list[dict],
    llm_client: AzureOpenAI,
    lance_table: lancedb.table.Table,
    embedding_client,
    verbose: bool = True,
    validate: bool = True,
) -> dict:
    """
    Advanced abstract/ontology-level SPO triplet extraction using a ReAct agent per document.

    Same output contract as lexical_graph.extract_spo_triplets_simple — returns
    {chunk_id: {"subject", "subject_type", "predicate", "object", "object_type"}}.

    Subjects and objects are ABSTRACT CATEGORIES / ROLES / CONCEPT CLASSES,
    not specific proper nouns or instance-level identifiers. This produces
    a conceptual ontology graph rather than a factual knowledge graph.

    The agent can explore the LanceDB table to find cross-references
    and build more accurate abstract SPO triplets.

    Args:
        chunks:           List of chunk dicts (chunk_id, doc_name, text, index).
        llm_client:       AzureOpenAI client for chat completions.
        lance_table:      LanceDB table with embedded chunks.
        embedding_client: AzureOpenAI client for embeddings.
        verbose:          Print agent reasoning trace.
        validate:         Run cross-document validation after extraction.

    Returns:
        {chunk_id: {"subject": str, "subject_type": str, "predicate": str,
                    "object": str, "object_type": str}}
    """
    tool = VectorDBQueryTool(lance_table, embedding_client)
    spo_by_chunk = {}

    # Group chunks by document
    docs = {}
    for chunk in chunks:
        docs.setdefault(chunk["doc_name"], []).append(chunk)

    for doc_name, doc_chunks in docs.items():
        if verbose:
            print(f"\n    -- Agent extracting SPO triplets: {doc_name} "
                  f"({len(doc_chunks)} chunks) --")

        agent = LexicalEnrichmentAgent(
            llm_client=llm_client,
            tool=tool,
            target_chunks=doc_chunks,
            doc_name=doc_name,
            verbose=verbose,
        )

        result = agent.run()
        spo_by_chunk.update(result)

        if verbose:
            filled = sum(1 for v in result.values() if v.get("subject"))
            print(f"    Result: {filled}/{len(result)} chunks have SPO triplets")

    # Cross-document validation
    if validate:
        spo_by_chunk = validate_extraction(
            spo_by_chunk, chunks, llm_client, verbose
        )

    return spo_by_chunk
