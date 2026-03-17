"""
enrich_advanced.py — Advanced ReAct Agent-based Entity Resolution for Layer 3

Instead of pure embedding cosine similarity (subject_graph.resolve_correspondences_simple),
this module uses a custom ReAct agent that iteratively explores the Neo4j graph
(both Layer 1 and Layer 2) and optionally the LanceDB vector store to produce
richer, more accurate entity resolution with cross-entity awareness.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                  ResolutionAgent                             │
  │                                                              │
  │  THOUGHT ──► ACTION (graph_query_tool) ──► OBSERVE           │
  │     ▲                                          │             │
  │     └──────────────────────────────────────────┘             │
  │                                                              │
  │  Repeats until FINAL_ANSWER or max iterations reached        │
  └──────────────────────────────────────────────────────────────┘

The agent has one tool — GraphQueryTool — which gives it access to the
Neo4j knowledge graph (DomainEntity, Document, Subject nodes) and
optionally LanceDB for semantic search across document chunks.

Usage:
    from agents.subject_agent import resolve_correspondences_advanced

    correspondences = resolve_correspondences_advanced(
        subjects, domain_entities, llm_client, driver,
        lance_table=lance_table, embedding_client=embedding_client,
    )
    # Returns same list format as subject_graph.resolve_correspondences_simple
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
from utils.cosmos_helpers import run_gremlin, esc, make_vertex_id, gval


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph Query Tool (Neo4j + optional LanceDB)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphQueryTool:
    """
    Read-only tool to explore the Cosmos DB knowledge graph (Gremlin API)
    for the subject agent — covers Layer 1 DomainEntity + Layer 2 Subject/Document
    and optionally the LanceDB vector store.

    Supported actions:
      - list_subjects()               → all Subject vertices
      - list_domain_entities()        → all DomainEntity vertices
      - get_subject_context(name)     → Subject + MENTIONS docs + SPO triplets
      - get_domain_entity_detail(name)→ DomainEntity + relationships + columns
      - search_similar(query, n)      → embedding-based LanceDB search
      - query_graph(gremlin)          → arbitrary read-only Gremlin traversal
    """

    TOOL_DESCRIPTION = textwrap.dedent("""\
    graph_query_tool — Read-only access to the Cosmos DB knowledge graph (Gremlin) and vector store.

    Available actions (pass as JSON):

    1. {"action": "list_subjects"}
       Returns: all Subject vertices with name, type, description, mention_count.

    2. {"action": "list_domain_entities"}
       Returns: all DomainEntity vertices with name, description, domain, key_columns.

    3. {"action": "get_subject_context", "name": "<subject_name>"}
       Returns: Subject vertex + Documents that MENTION it + SPO triplets + context.

    4. {"action": "get_domain_entity_detail", "name": "<entity_name>"}
       Returns: DomainEntity vertex + relationships + column info.

    5. {"action": "search_similar", "query": "<natural language query>", "n": 5}
       Returns: top-N text chunks from LanceDB most similar to the query.
       NOTE: Only available if vector store is connected.

    6. {"action": "query_graph", "gremlin": "<Gremlin traversal>"}
       Runs a read-only Gremlin traversal. Example: g.V().hasLabel('Subject').values('name')
       Only g.V() / g.E() reads are allowed.
    """)

    def __init__(self, client, lance_table=None, embedding_client=None):
        self.client = client
        self.lance_table = lance_table
        self.embedding_client = embedding_client

    def execute(self, action_input: dict) -> str:
        action = action_input.get("action", "").strip().lower()
        try:
            if action == "list_subjects":
                return self._list_subjects()
            elif action == "list_domain_entities":
                return self._list_domain_entities()
            elif action == "get_subject_context":
                return self._get_subject_context(action_input["name"])
            elif action == "get_domain_entity_detail":
                return self._get_domain_entity_detail(action_input["name"])
            elif action == "search_similar":
                return self._search_similar(
                    action_input.get("query", ""),
                    action_input.get("n", 5),
                )
            elif action == "query_graph":
                return self._query_graph(action_input["gremlin"])
            else:
                return (
                    f"ERROR: Unknown action '{action}'. "
                    "Available: list_subjects, list_domain_entities, get_subject_context, "
                    "get_domain_entity_detail, search_similar, query_graph"
                )
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _list_subjects(self) -> str:
        records = run_gremlin(self.client,
            "g.V().hasLabel('Subject').valueMap('name','type','description','mention_count')"
        )
        subjects = []
        for r in records:
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

    def _get_subject_context(self, name: str) -> str:
        sid = make_vertex_id("Subject", name)
        subj_recs = run_gremlin(self.client,
            f"g.V('{esc(sid)}').valueMap('name','type','description','mention_count')"
        )
        if not subj_recs:
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

        return json.dumps({
            "subject": subject,
            "mentioned_in_documents": docs,
            "total_documents": len(docs),
            "spo_triplets": spo,
        }, indent=2)

    def _get_domain_entity_detail(self, name: str) -> str:
        vid = make_vertex_id("DomainEntity", name)
        de_recs = run_gremlin(self.client,
            f"g.V('{esc(vid)}').valueMap('name','description','domain','key_columns','column_info','row_count')"
        )
        if not de_recs:
            return f"ERROR: DomainEntity '{name}' not found."

        r = de_recs[0]
        entity = {
            "name": gval(r, "name", name),
            "description": gval(r, "description", ""),
            "domain": gval(r, "domain", ""),
            "key_columns": gval(r, "key_columns", "[]"),
            "column_info": gval(r, "column_info", "[]"),
            "row_count": gval(r, "row_count", 0),
        }

        out_rels = run_gremlin(self.client,
            f"g.V('{esc(vid)}').outE()"
            ".where(inV().hasLabel('DomainEntity'))"
            ".project('rel_type','target','reason')"
            ".by(label())"
            ".by(inV().values('name'))"
            ".by(coalesce(values('reason'), constant('')))"
        )

        in_rels = run_gremlin(self.client,
            f"g.V('{esc(vid)}').inE()"
            ".where(outV().hasLabel('DomainEntity'))"
            ".project('rel_type','source','reason')"
            ".by(label())"
            ".by(outV().values('name'))"
            ".by(coalesce(values('reason'), constant('')))"
        )

        return json.dumps({
            "entity": entity,
            "outgoing_relationships": out_rels,
            "incoming_relationships": in_rels,
        }, indent=2)

    def _search_similar(self, query: str, n: int) -> str:
        if self.lance_table is None or self.embedding_client is None:
            return "ERROR: Vector store not available. Use other actions to explore the graph."
        if not query:
            return "ERROR: 'query' parameter is required for search_similar."
        q_embedding = embed_texts(self.embedding_client, [query])[0]
        total = len(self.lance_table)
        n = min(n, total)
        if n == 0:
            return json.dumps({"results": [], "message": "Table is empty"})
        results = (
            self.lance_table.search(q_embedding)
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
                "text": row["text"][:400],
                "metadata": {"doc_name": row["doc_name"], "chunk_index": row["chunk_index"]},
            })
        return json.dumps({"results": output}, indent=2)

    def _query_graph(self, gremlin_query: str) -> str:
        stripped = gremlin_query.strip().lower()
        write_keywords = ["addv(", "adde(", ".drop(", ".property("]
        if any(kw in stripped for kw in write_keywords):
            return "ERROR: Only read traversals are allowed."
        try:
            records = run_gremlin(self.client, gremlin_query)
            if len(records) > 30:
                records = records[:30]
                return json.dumps(records, indent=2) + "\n... (truncated to 30 results)"
            return json.dumps(records, indent=2)
        except Exception as e:
            return f"ERROR: Gremlin query failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  ReAct Resolution Agent
# ═══════════════════════════════════════════════════════════════════════════════

class ResolutionAgent:
    """
    Custom ReAct (Reason → Act → Observe) agent that iteratively explores
    the Neo4j knowledge graph and vector store to determine which DomainEntity
    nodes a given Subject corresponds to.

    Workflow per subject:
      1. Receives target subject name + list of available domain entities
      2. THINKS about what to explore
      3. ACTS by calling graph_query_tool with an action
      4. OBSERVES the result
      5. Repeats steps 2-4 until it has enough context
      6. Produces FINAL_ANSWER — the correspondence matches JSON

    This is a custom agent — no frameworks, no LangChain, no CrewAI.
    """

    MAX_ITERATIONS = 10

    def __init__(
        self,
        llm_client: AzureOpenAI,
        tool: GraphQueryTool,
        target_subject: dict = None,
        all_domain_entities: list[dict] = None,
        target_domain_entity: dict = None,
        all_subjects: list[dict] = None,
        direction: str = "subject",
        verbose: bool = True,
    ):
        self.client = llm_client
        self.tool = tool
        self.direction = direction
        self.target_subject = target_subject
        self.all_domain_entities = all_domain_entities or []
        self.target_domain_entity = target_domain_entity
        self.all_subjects = all_subjects or []
        self.verbose = verbose
        self.messages: list[dict] = []
        self.trace: list[dict] = []

    # ── Public API ───────────────────────────────────────────────────────

    def run(self) -> list[dict]:
        """
        Execute the ReAct loop and return resolved correspondences.

        Returns:
            [
                {
                    "domain_entity": str,
                    "confidence": float (0-1),
                    "reason": str
                },
                ...
            ]
            Empty list if no correspondence found.
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
                    print(f"      → FINAL_ANSWER reached after {iteration} iteration(s)")
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
                        "ACTION: graph_query_tool\n"
                        "ACTION_INPUT: <valid JSON>\n\n"
                        "Or for your final answer:\n"
                        "THOUGHT: <your final reasoning>\n"
                        "FINAL_ANSWER:\n<valid JSON — see required schema>"
                    ),
                })

        # Max iterations exhausted
        if self.verbose:
            print(f"      → Max iterations ({self.MAX_ITERATIONS}) reached, forcing final answer...")
        return self._force_final_answer()

    # ── Prompt Construction ──────────────────────────────────────────────

    def _build_system_prompt(self):
        if self.direction == "domain_entity":
            return self._build_system_prompt_domain_entity()
        de_list = "\n".join(
            f"  - {e['name']}: {e['description'][:80]} (domain: {e['domain']})"
            for e in self.all_domain_entities
        )
        de_names = [e["name"] for e in self.all_domain_entities]

        system = textwrap.dedent(f"""\
        You are a senior entity resolution agent for an airlines knowledge graph.

        Your task: Determine which structured database table(s) the subject
        "{self.target_subject['name']}" (type: {self.target_subject['type']})
        corresponds to.

        The "CORRESPONDS_TO" relationship means: data about this concept can be
        found in or is semantically related to the database table. For example,
        a "Boeing 737" subject CORRESPONDS_TO the "aircraft" table because it
        is a type of aircraft tracked in that table.

        You work in a ReAct loop — you THINK, then ACT (use a tool to explore),
        then OBSERVE the result, and repeat until you have enough context.

        ── YOUR TOOL ──────────────────────────────────────────────────────────
        {self.tool.TOOL_DESCRIPTION}
        ── AVAILABLE DATABASE TABLES ──────────────────────────────────────────
        {de_list}

        ── RESPONSE FORMAT ────────────────────────────────────────────────────

        On each turn, respond in EXACTLY one of these two formats:

        FORMAT A — Take an action:
        THOUGHT: <your reasoning about what to explore next>
        ACTION: graph_query_tool
        ACTION_INPUT: <valid JSON object for the tool>

        FORMAT B — Provide final answer:
        THOUGHT: <your final reasoning about the correspondence>
        FINAL_ANSWER:
        <valid JSON — see schema below>

        ── REQUIRED OUTPUT SCHEMA (for FINAL_ANSWER) ──────────────────────────

        [
          {{
            "domain_entity": "<table name from {de_names}>",
            "confidence": 0.0 to 1.0,
            "reason": "1-2 sentence explanation of WHY this subject corresponds to this table"
          }}
        ]

        Return an EMPTY array [] if the subject does NOT correspond to any table.
        Return MULTIPLE items if the subject corresponds to more than one table.

        ── STRATEGY ───────────────────────────────────────────────────────────

        1. Start by understanding the subject — get its context from documents.
        2. Examine each candidate domain entity to understand what it contains.
        3. Look for semantic overlap: does the subject's document context mention
           things that live in a particular table?
        4. Consider indirect links: a crew member's name might correspond to the
           "crew" table even if the text doesn't mention "crew table".
        5. Use search_similar to find related document mentions if helpful.
        6. When confident, produce your FINAL_ANSWER.
        7. Aim for 2-5 exploration steps. Don't over-explore.

        ── CONFIDENCE GUIDELINES ──────────────────────────────────────────────

        - 0.9-1.0: The subject is an exact instance of what the table tracks
          (e.g., "Boeing 737" → aircraft table)
        - 0.7-0.9: The subject is strongly related to the table's domain
          (e.g., "engine maintenance" → maintenance table)
        - 0.5-0.7: The subject has a meaningful connection to the table
          (e.g., "turbulence event" → incidents or flights table)
        - Below 0.5: Weak or tangential — prefer not to link

        ── IMPORTANT RULES ────────────────────────────────────────────────────

        - Respond with EXACTLY one format per turn (A or B). Never both.
        - ACTION_INPUT must be valid parseable JSON on a SINGLE line.
        - FINAL_ANSWER JSON must be a valid array.
        - Only reference domain entities that actually exist: {de_names}
        - Ground your confidence scores in observed evidence, not guesses.
        """)

        self.messages.append({"role": "system", "content": system})

    def _build_system_prompt_domain_entity(self):
        """System prompt for per-domain-entity direction."""
        subj_list = "\n".join(
            f"  - {s['name']}: type={s['type']}, mentions={s['mention_count']}"
            for s in self.all_subjects
        )
        subj_names = [s["name"] for s in self.all_subjects]
        de = self.target_domain_entity

        system = textwrap.dedent(f"""\
        You are a senior entity resolution agent for an airlines knowledge graph.

        Your task: Determine which document subject(s) correspond to the
        structured database table "{de['name']}" (domain: {de['domain']}).
        Table description: {de['description']}

        The "CORRESPONDS_TO" relationship means: the subject, as discussed in
        the document corpus, is semantically related to or has data represented
        in this database table. For example, the subjects "flight delay" and
        "route cancellation" might CORRESPOND_TO the "flights" table because they
        are events tracked in that table.

        You work in a ReAct loop — you THINK, then ACT (use a tool to explore),
        then OBSERVE the result, and repeat until you have enough context.

        ── YOUR TOOL ──────────────────────────────────────────────────────────
        {self.tool.TOOL_DESCRIPTION}
        ── AVAILABLE SUBJECTS FROM DOCUMENTS ──────────────────────────────────
        {subj_list}

        ── RESPONSE FORMAT ────────────────────────────────────────────────────

        On each turn, respond in EXACTLY one of these two formats:

        FORMAT A — Take an action:
        THOUGHT: <your reasoning about what to explore next>
        ACTION: graph_query_tool
        ACTION_INPUT: <valid JSON object for the tool>

        FORMAT B — Provide final answer:
        THOUGHT: <your final reasoning about the correspondences>
        FINAL_ANSWER:
        <valid JSON — see schema below>

        ── REQUIRED OUTPUT SCHEMA (for FINAL_ANSWER) ──────────────────────────

        [
          {{
            "subject_name": "<subject name from {subj_names}>",
            "confidence": 0.0 to 1.0,
            "reason": "1-2 sentence explanation"
          }}
        ]

        Return an EMPTY array [] if no subjects correspond to this table.
        Return MULTIPLE items if several subjects correspond to this table.

        ── STRATEGY ───────────────────────────────────────────────────────────

        1. Start by understanding the table — examine its columns and sample data
           via get_domain_entity_detail.
        2. Then explore candidate subjects — check their document context to see
           which ones discuss concepts relevant to this table.
        3. Look for semantic overlap: does the subject's context mention data that
           lives in this table?
        4. Consider indirect links: a subject named "fuel efficiency" might
           correspond to the "flights" table if fuel consumption data is there.
        5. Use search_similar to find related document mentions if helpful.
        6. When confident, produce your FINAL_ANSWER.
        7. Aim for 2-5 exploration steps.

        ── CONFIDENCE GUIDELINES ──────────────────────────────────────────────

        - 0.9-1.0: The subject is an exact instance of what the table tracks
        - 0.7-0.9: The subject is strongly related to the table's domain
        - 0.5-0.7: The subject has a meaningful connection to the table
        - Below 0.5: Weak or tangential — prefer not to link

        ── IMPORTANT RULES ────────────────────────────────────────────────────

        - Respond with EXACTLY one format per turn (A or B). Never both.
        - ACTION_INPUT must be valid parseable JSON on a SINGLE line.
        - FINAL_ANSWER JSON must be a valid array.
        - Only reference subjects that actually exist: {subj_names}
        - Ground your confidence scores in observed evidence, not guesses.
        """)

        self.messages.append({"role": "system", "content": system})

    def _build_initial_user_prompt(self):
        if self.direction == "domain_entity":
            return self._build_initial_user_prompt_domain_entity()
        # Include basic subject info to kickstart
        subj = self.target_subject
        context_preview = ""
        if subj.get("doc_contexts"):
            snippets = []
            for cc in subj["doc_contexts"][:3]:
                if cc.get("mention_context"):
                    snippets.append(cc["mention_context"])
                elif cc.get("topic_summary"):
                    snippets.append(cc["topic_summary"])
            context_preview = "; ".join(snippets)

        prompt = textwrap.dedent(f"""\
        Begin your analysis of the subject "{subj['name']}".

        Subject info:
          - Name: {subj['name']}
          - Type: {subj['type']}
          - Description: {subj.get('description', 'N/A')}
          - Mention count: {subj['mention_count']}
          - Context preview: {context_preview[:300] if context_preview else 'N/A'}

        Start by exploring the subject's context in the graph, then examine
        the candidate domain entities to determine correspondences.
        """)
        self.messages.append({"role": "user", "content": prompt})

    def _build_initial_user_prompt_domain_entity(self):
        """Initial user prompt for per-domain-entity direction."""
        de = self.target_domain_entity
        cols = ""
        if de.get("columns"):
            cols = ", ".join(de["columns"][:10])

        prompt = textwrap.dedent(f"""\
        Begin your analysis of the database table "{de['name']}".

        Table info:
          - Name: {de['name']}
          - Domain: {de['domain']}
          - Description: {de['description']}
          - Columns: {cols if cols else 'N/A'}

        Explore the subjects from the document corpus and determine which ones
        correspond to this table. Use get_subject_context to understand what
        each subject is about, and get_domain_entity_detail to examine the
        table's structure.
        """)
        self.messages.append({"role": "user", "content": prompt})

    # ── LLM Interaction ──────────────────────────────────────────────────

    def _call_llm(self) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=self.messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content

    # ── Response Parsing ─────────────────────────────────────────────────

    def _parse_response(self, text: str) -> dict:
        """
        Parse LLM response into one of:
          - {"type": "action", "thought": str, "action_input": dict}
          - {"type": "final_answer", "result": list}
          - {"type": "error", "error": str}
        """
        text = text.strip()

        if "FINAL_ANSWER" in text:
            return self._parse_final_answer(text)

        if "ACTION_INPUT" in text:
            return self._parse_action(text)

        # Fallback: check if there's embedded JSON that looks like a correspondence list
        json_obj = self._try_extract_json(text)
        if json_obj is not None:
            if isinstance(json_obj, list):
                return {"type": "final_answer", "result": self._normalize_result(json_obj)}
            if isinstance(json_obj, dict) and ("domain_entity" in json_obj or "subject_name" in json_obj):
                return {"type": "final_answer", "result": self._normalize_result([json_obj])}

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
                if isinstance(result, list):
                    return {"type": "final_answer", "result": self._normalize_result(result)}
                if isinstance(result, dict) and ("domain_entity" in result or "subject_name" in result):
                    return {"type": "final_answer", "result": self._normalize_result([result])}
            except json.JSONDecodeError:
                continue

        return {"type": "error", "error": f"Could not parse FINAL_ANSWER JSON: {raw[:300]}"}

    def _json_candidates(self, text: str) -> list[str]:
        """Generate candidate JSON strings from text."""
        candidates = []

        # Strategy 1: The entire text
        candidates.append(text.strip())

        # Strategy 2: First line only
        first_line = text.split("\n")[0].strip()
        candidates.append(first_line)

        # Strategy 3: Find first [ ... ] block (array)
        bracket_start = text.find("[")
        if bracket_start >= 0:
            depth = 0
            for i in range(bracket_start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[bracket_start : i + 1])
                        break

        # Strategy 4: Find first { ... } block (single object)
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[brace_start : i + 1])
                        break

        # Strategy 5: Greedy regex match
        brace_match = re.search(r"\[.*\]", text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group())

        return candidates

    def _try_extract_json(self, text: str):
        """Try to extract JSON from text."""
        for candidate in self._json_candidates(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _normalize_result(self, result: list) -> list:
        """Ensure result has correct structure based on direction."""
        if self.direction == "domain_entity":
            return self._normalize_result_domain_entity(result)

        # ── Per-subject direction: [{domain_entity, confidence, reason}] ──
        de_names = {e["name"] for e in self.all_domain_entities}
        normalized = []

        for item in result:
            if not isinstance(item, dict):
                continue
            de_name = item.get("domain_entity", "")
            if de_name not in de_names:
                # Try case-insensitive match
                for name in de_names:
                    if name.lower() == de_name.lower():
                        de_name = name
                        break
                else:
                    continue  # skip unknown entities

            confidence = item.get("confidence", 0.5)
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5

            normalized.append({
                "domain_entity": de_name,
                "confidence": round(confidence, 4),
                "reason": item.get("reason", ""),
            })

        return normalized

    def _normalize_result_domain_entity(self, result: list) -> list:
        """Normalize results for per-domain-entity direction: [{subject_name, confidence, reason}]."""
        subj_names = {s["name"] for s in self.all_subjects}
        normalized = []

        for item in result:
            if not isinstance(item, dict):
                continue
            subj_name = item.get("subject_name", "") or item.get("subject", "")
            if subj_name not in subj_names:
                # Try case-insensitive match
                for name in subj_names:
                    if name.lower() == subj_name.lower():
                        subj_name = name
                        break
                else:
                    continue  # skip unknown subjects

            confidence = item.get("confidence", 0.5)
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5

            normalized.append({
                "subject_name": subj_name,
                "confidence": round(confidence, 4),
                "reason": item.get("reason", ""),
            })

        return normalized

    # ── Fallback ─────────────────────────────────────────────────────────

    def _force_final_answer(self) -> list:
        """Force a final answer if max iterations reached."""
        observations_summary = "\n\n".join(
            f"Step {t['iteration']}: {t['thought'][:150]}" for t in self.trace
        )

        if self.direction == "domain_entity":
            subj_names = [s["name"] for s in self.all_subjects]
            target_name = self.target_domain_entity["name"]
            candidate_list = subj_names
            target_label = "domain entity (table)"
        else:
            target_name = self.target_subject["name"]
            candidate_list = [e["name"] for e in self.all_domain_entities]
            target_label = "subject"

        self.messages.append({
            "role": "user",
            "content": textwrap.dedent(f"""\
            You have run out of exploration steps. Based on everything you observed,
            produce your FINAL_ANSWER now for {target_label} "{target_name}".

            Summary of your exploration:
            {observations_summary}

            Available candidates: {candidate_list}

            Respond with ONLY:
            FINAL_ANSWER:
            <valid JSON array of correspondences, or empty array []>
            """),
        })

        response = self._call_llm()
        parsed = self._parse_response(response)

        if parsed["type"] == "final_answer":
            return parsed["result"]

        # Absolute fallback — no correspondences
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-Subject Validation (Satisfaction Checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_correspondences(
    all_correspondences: list[dict],
    subjects: list[dict],
    domain_entities: list[dict],
    client: AzureOpenAI,
    verbose: bool = True,
    direction: str = "subject",
) -> list[dict]:
    """
    Post-resolution validation: review all correspondences together for
    consistency, completeness, and correctness. The LLM acts as a reviewer.

    This is the "satisfaction checkpoint" — ensures the full mapping is
    coherent, not just per-subject or per-table.
    """
    if verbose:
        print("\n    [Validation] Cross-entity correspondence check...")

    subject_names = [s["name"] for s in subjects]
    de_names = [e["name"] for e in domain_entities]

    # Build a summary of current correspondences — grouped by subject
    corr_summary = {}
    linked_de_set = set()
    for corr in all_correspondences:
        subj = corr["subject_name"]
        if subj not in corr_summary:
            corr_summary[subj] = []
        corr_summary[subj].append({
            "domain_entity": corr["domain_entity_name"],
            "confidence": corr["confidence"],
            "reason": corr["reason"],
        })
        linked_de_set.add(corr["domain_entity_name"])

    # Unlinked entities
    unlinked_subjects = [s for s in subject_names if s not in corr_summary]
    unlinked_tables = [d for d in de_names if d not in linked_de_set]

    # Build direction-aware prompt framing
    if direction == "domain_entity":
        direction_note = (
            "Resolution was run per-table (each table was matched to relevant subjects). "
            "Pay special attention to tables WITHOUT any correspondence — every table that "
            "has related subjects in the documents should have at least one link."
        )
    else:
        direction_note = (
            "Resolution was run per-subject (each subject was matched to relevant tables). "
            "Pay special attention to subjects WITHOUT any correspondence — every subject "
            "that represents something tracked in the database should have a link."
        )

    prompt = textwrap.dedent(f"""\
    You are a senior knowledge graph architect reviewing entity resolution results
    for an airlines company.

    Below is the mapping between Subject nodes (extracted from documents) and
    DomainEntity nodes (structured database tables). {direction_note}

    Review for:
    1. Missing links: Are there entities that SHOULD be linked but aren't?
    2. Incorrect links: Are there correspondences that don't make sense?
    3. Confidence scores: Are they appropriate given the semantic relationship?
    4. Completeness: Every meaningful relationship should be captured.

    Subjects: {subject_names}
    Domain Entities (tables): {de_names}

    Current correspondences (grouped by subject):
    {json.dumps(corr_summary, indent=2)}

    Subjects WITHOUT any correspondence: {unlinked_subjects}
    Tables WITHOUT any correspondence: {unlinked_tables}

    If the mapping is good, respond with EXACTLY:
    VALIDATED

    If you want to fix issues, respond with a corrected JSON object where keys are
    subject names and values are arrays of {{domain_entity, confidence, reason}}.
    Include ALL subjects (even unchanged ones). No extra text — just the JSON.
    """)

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    response = completion.choices[0].message.content.strip()

    if "VALIDATED" in response.upper() and len(response) < 50:
        if verbose:
            print("    [Validation] ✓ Correspondences validated — no changes needed.")
        return all_correspondences

    # Try to parse corrected JSON
    parsed = parse_llm_json(response)
    if parsed and isinstance(parsed, dict):
        corrected_list = []
        changes = 0
        for subj_name, mappings in parsed.items():
            if not isinstance(mappings, list):
                continue
            for m in mappings:
                if isinstance(m, dict) and "domain_entity" in m:
                    de_name = m["domain_entity"]
                    # Validate domain entity exists
                    if de_name not in de_names:
                        continue
                    corrected_list.append({
                        "subject_name": subj_name,
                        "domain_entity_name": de_name,
                        "confidence": round(float(m.get("confidence", 0.5)), 4),
                        "method": "llm_validated",
                        "reason": m.get("reason", ""),
                    })

        if corrected_list:
            # Count changes
            old_pairs = {(c["subject_name"], c["domain_entity_name"]) for c in all_correspondences}
            new_pairs = {(c["subject_name"], c["domain_entity_name"]) for c in corrected_list}
            added = new_pairs - old_pairs
            removed = old_pairs - new_pairs

            if verbose:
                print(f"    [Validation] ✓ Correspondences corrected by reviewer.")
                if added:
                    print(f"      + Added {len(added)} new correspondence(s)")
                    for a in added:
                        print(f"        + {a[0]} → {a[1]}")
                if removed:
                    print(f"      - Removed {len(removed)} correspondence(s)")
                    for r in removed:
                        print(f"        - {r[0]} → {r[1]}")
            return corrected_list

        if verbose:
            print("    [Validation] WARN: Corrected JSON was empty, keeping original.")
        return all_correspondences

    if verbose:
        print("    [Validation] WARN: Could not parse correction, keeping original.")
    return all_correspondences


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — Drop-in replacement for subject_graph.resolve_correspondences_simple
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_correspondences_advanced(
    subjects: list[dict],
    domain_entities: list[dict],
    llm_client: AzureOpenAI,
    gremlin_client,
    lance_table=None,
    embedding_client=None,
    verbose: bool = True,
    validate: bool = True,
    direction: str = "subject",
) -> list[dict]:
    """
    Advanced entity resolution using a ReAct agent with Cosmos DB Gremlin.

    direction controls the outer loop:
      - "subject" (default): One agent per subject, finds matching tables.
      - "domain_entity": One agent per table, finds matching subjects.

    Args:
        subjects:          List of subject dicts (from fetch_subjects).
        domain_entities:   List of domain entity dicts (from fetch_domain_entities).
        llm_client:        AzureOpenAI client for chat completions.
        gremlin_client:    Cosmos DB Gremlin client.
        lance_table:       Optional LanceDB table for vector search.
        embedding_client:  Optional AzureOpenAI client for embeddings.
        verbose:           Print agent reasoning trace.
        validate:          Run cross-subject validation after resolution.
        direction:         "subject" (per-subject loop) or "domain_entity" (per-table loop).

    Returns:
        [{"subject_name", "domain_entity_name", "confidence", "method", "reason"}, ...]
    """
    tool = GraphQueryTool(gremlin_client, lance_table, embedding_client)
    all_correspondences = []

    # ── Per-domain-entity direction ──────────────────────────────────────
    if direction == "domain_entity":
        for de in domain_entities:
            if verbose:
                print(f"\n    ── Agent resolving: table '{de['name']}' ({de['domain']}) ──────────────────")

            agent = ResolutionAgent(
                llm_client=llm_client,
                tool=tool,
                target_domain_entity=de,
                all_subjects=subjects,
                direction="domain_entity",
                verbose=verbose,
            )

            result = agent.run()

            if verbose:
                if result:
                    for match in result:
                        print(f"    Result: {de['name']} ← {match['subject_name']}  "
                              f"(confidence: {match['confidence']})")
                else:
                    print(f"    Result: No correspondence found for table {de['name']}")

            # Convert agent output to correspondence format
            for match in result:
                all_correspondences.append({
                    "subject_name": match["subject_name"],
                    "domain_entity_name": de["name"],
                    "confidence": match["confidence"],
                    "method": "react_agent",
                    "reason": match.get("reason", ""),
                })

    # ── Per-subject direction (default) ──────────────────────────────────
    else:
        for subj in subjects:
            if verbose:
                print(f"\n    ── Agent resolving: {subj['name']} ({subj['type']}) ──────────────────")

            agent = ResolutionAgent(
                llm_client=llm_client,
                tool=tool,
                target_subject=subj,
                all_domain_entities=domain_entities,
                direction="subject",
                verbose=verbose,
            )

            result = agent.run()

            if verbose:
                if result:
                    for match in result:
                        print(f"    Result: {subj['name']} → {match['domain_entity']}  "
                              f"(confidence: {match['confidence']})")
                else:
                    print(f"    Result: No correspondence found for {subj['name']}")

            # Convert agent output to correspondence format
            for match in result:
                all_correspondences.append({
                    "subject_name": subj["name"],
                    "domain_entity_name": match["domain_entity"],
                    "confidence": match["confidence"],
                    "method": "react_agent",
                    "reason": match.get("reason", ""),
                })

    # ── Cross-subject validation (satisfaction checkpoint) ──────────────
    if validate:
        all_correspondences = validate_correspondences(
            all_correspondences, subjects, domain_entities, llm_client, verbose,
            direction=direction,
        )

    return all_correspondences


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone Test
# ═══════════════════════════════════════════════════════════════════════════════

