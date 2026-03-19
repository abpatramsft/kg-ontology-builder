"""
Vanilla Planner Agent — Custom FoundryCBAgent (no framework) for Foundry deployment.

Uses the existing PlannerAgent ReAct loop (THOUGHT/ACTION/FINAL_ANSWER) as-is,
wrapped in a thin FoundryCBAgent HTTP adapter so Foundry can call it via the
Responses API on port 8088.

The ReAct agent internally calls chat.completions.create() to reason over the
knowledge graph ontology via GraphOntologyTool, then returns the final plan
as an OpenAIResponse.

Usage:
    # Local test (starts HTTP server on :8088):
    python main.py

    # Then send requests:
    curl -X POST http://localhost:8088/responses \
         -H "Content-Type: application/json" \
         -d '{"input": "Which data sources should I check for flight delay analysis?"}'
"""

import datetime
import json
import logging
import os
import re
import textwrap

from dotenv import load_dotenv

load_dotenv(override=True)

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from typing import Any, AsyncGenerator, Union

from azure.ai.agentserver.core import AgentRunContext, FoundryCBAgent
from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseOutputItemAddedEvent,
    ResponsesAssistantMessageItemResource,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from azure.ai.agentserver.core.logger import get_logger
from openai import AzureOpenAI

from cosmos_helpers import get_gremlin_client, run_gremlin, esc, make_vertex_id, gval

logger = get_logger()

# ── Configuration ───────────────────────────────────────────────────────────
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT", "")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1")


# ═══════════════════════════════════════════════════════════════════════════════
#  GraphOntologyTool — read-only Cosmos DB knowledge graph explorer
#  (inlined from the original inference_agent.py)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphOntologyTool:
    """
    Read-only tool to explore the unified 3-layer knowledge graph in Cosmos DB
    (Gremlin API, database: indigokg, container: knowledgegraph).
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
    2. {"action": "list_relationship_types"}
    3. {"action": "list_domain_entities"}
    4. {"action": "list_subjects"}
    5. {"action": "list_documents"}
    6. {"action": "get_domain_entity_detail", "name": "<entity_name>"}
    7. {"action": "get_subject_context", "name": "<subject_name>"}
    8. {"action": "get_correspondences", "name": "<subject_name>"}
    9. {"action": "find_path", "from_name": "<vertex_name>", "to_name": "<vertex_name>"}
    10. {"action": "query_graph", "gremlin": "<Gremlin traversal>"}
    """)

    def __init__(self, client):
        self.client = client

    def execute(self, action_input: dict) -> str:
        action = action_input.get("action", "").strip().lower()
        try:
            dispatch = {
                "list_node_labels":         self._list_node_labels,
                "list_relationship_types":  self._list_relationship_types,
                "list_domain_entities":     self._list_domain_entities,
                "list_subjects":            self._list_subjects,
                "list_documents":           self._list_documents,
                "get_domain_entity_detail": lambda: self._get_domain_entity_detail(action_input["name"]),
                "get_subject_context":      lambda: self._get_subject_context(action_input["name"]),
                "get_correspondences":      lambda: self._get_correspondences(action_input["name"]),
                "find_path":               lambda: self._find_path(action_input["from_name"], action_input["to_name"]),
                "query_graph":             lambda: self._query_graph(action_input["gremlin"]),
            }
            fn = dispatch.get(action)
            if fn is None:
                return f"ERROR: Unknown action '{action}'. Available: " + ", ".join(dispatch.keys())
            return fn() if callable(fn) else fn
        except KeyError as e:
            return f"ERROR: Missing required parameter: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

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
        vid = make_vertex_id("DomainEntity", name)
        de_recs = run_gremlin(self.client,
            f"g.V('{esc(vid)}').valueMap('name','description','domain','key_columns','column_info','row_count')"
        )
        if not de_recs:
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
            ".by(coalesce(values('confidence'), constant(0)))"
            ".by(coalesce(values('reason'), constant('')))"
        )

        return json.dumps({
            "entity": entity,
            "outgoing_relationships": out_rels,
            "incoming_relationships": in_rels,
            "subject_correspondences": correspondences,
        }, indent=2)

    def _get_subject_context(self, name: str) -> str:
        sid = make_vertex_id("Subject", name)
        s_recs = run_gremlin(self.client,
            f"g.V('{esc(sid)}').valueMap('name','type','description','mention_count')"
        )
        if not s_recs:
            all_s = run_gremlin(self.client,
                "g.V().hasLabel('Subject').valueMap('name','type','description','mention_count')"
            )
            s_recs = [r for r in all_s if gval(r, "name", "").lower() == name.lower()]

        if not s_recs:
            return f"ERROR: Subject '{name}' not found."

        r = s_recs[0]
        resolved_name = gval(r, "name", name)
        resolved_id = make_vertex_id("Subject", resolved_name)

        subject = {
            "name": resolved_name,
            "type": gval(r, "type", ""),
            "description": gval(r, "description", ""),
            "mention_count": gval(r, "mention_count", 0),
        }

        docs = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').in('MENTIONS')"
            ".valueMap('name','topic_summary')"
        )
        doc_list = [{"name": gval(d, "name", ""), "topic_summary": gval(d, "topic_summary", "")} for d in docs]

        spo = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').outE('RELATES_TO')"
            ".project('predicate','object','object_type')"
            ".by(values('predicate'))"
            ".by(inV().values('name'))"
            ".by(inV().coalesce(values('type'), constant('')))"
        )

        correspondences = run_gremlin(self.client,
            f"g.V('{esc(resolved_id)}').out('CORRESPONDS_TO')"
            ".valueMap('name','domain','description')"
        )
        corr_list = [
            {"name": gval(c, "name", ""), "domain": gval(c, "domain", ""), "description": gval(c, "description", "")}
            for c in correspondences
        ]

        return json.dumps({
            "subject": subject,
            "documents": doc_list,
            "spo_triplets": spo,
            "corresponding_entities": corr_list,
        }, indent=2)

    def _get_correspondences(self, name: str) -> str:
        sid = make_vertex_id("Subject", name)
        corr = run_gremlin(self.client,
            f"g.V('{esc(sid)}').outE('CORRESPONDS_TO')"
            ".project('entity','entity_domain','confidence','reason')"
            ".by(inV().values('name'))"
            ".by(inV().coalesce(values('domain'), constant('')))"
            ".by(coalesce(values('confidence'), constant(0)))"
            ".by(coalesce(values('reason'), constant('')))"
        )
        return json.dumps({"subject": name, "correspondences": corr, "count": len(corr)}, indent=2)

    def _find_path(self, from_name: str, to_name: str) -> str:
        path_result = run_gremlin(self.client,
            f"g.V().has('name', '{esc(from_name)}')"
            f".repeat(both().simplePath()).until(has('name', '{esc(to_name)}')).limit(1)"
            ".path().by(valueMap('name','label'))"
        )
        if not path_result:
            return f"No path found between '{from_name}' and '{to_name}'."
        return json.dumps({"from": from_name, "to": to_name, "path": path_result}, indent=2)

    def _query_graph(self, gremlin: str) -> str:
        lowered = gremlin.strip().lower()
        if any(kw in lowered for kw in ["drop", "addv", "adde", "property("]):
            return "ERROR: Only read-only Gremlin queries are allowed."
        result = run_gremlin(self.client, gremlin)
        return json.dumps({"query": gremlin, "results": result, "count": len(result)}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  PlannerAgent — ReAct loop (unchanged from planner_agent.py)
# ═══════════════════════════════════════════════════════════════════════════════

class PlannerAgent:
    """
    ReAct agent that explores the knowledge graph ontology and returns a
    structured plan of action — which tools to use, which tables/documents
    to query, and what joins or searches to perform — without actually
    executing any data retrieval.
    """

    MAX_ITERATIONS = 4

    def __init__(self, llm_client, gremlin_client, verbose: bool = True):
        self.client = llm_client
        self.ontology_tool = GraphOntologyTool(gremlin_client)
        self.tools = {GraphOntologyTool.NAME: self.ontology_tool}
        self.verbose = verbose
        self.messages: list[dict] = []
        self.trace: list[dict] = []

    def run(self, question: str) -> str:
        self.messages = []
        self.trace = []
        self._build_system_prompt()
        self._build_user_prompt(question)

        if self.verbose:
            logger.info(f"QUESTION: {question}")

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            if self.verbose:
                logger.info(f"[Iteration {iteration}/{self.MAX_ITERATIONS}]")

            response_text = self._call_llm()
            parsed = self._parse_response(response_text)

            if parsed["type"] == "final_answer":
                if self.verbose:
                    logger.info(f"PLAN produced after {iteration} iteration(s)")
                return parsed["answer"]

            elif parsed["type"] == "action":
                thought = parsed.get("thought", "")
                tool_name = parsed["tool_name"]
                action_input = parsed["action_input"]

                if self.verbose:
                    logger.info(f"THOUGHT: {thought[:150]}")
                    logger.info(f"ACTION: {tool_name} -> {json.dumps(action_input)}")

                tool = self.tools.get(tool_name)
                if tool is None:
                    observation = (
                        f"ERROR: Unknown tool '{tool_name}'. "
                        f"You only have: {GraphOntologyTool.NAME}"
                    )
                else:
                    observation = tool.execute(action_input)

                if self.verbose:
                    logger.info(f"OBSERVE: {observation[:250]}")

                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "tool": tool_name,
                    "action": action_input,
                    "observation": observation,
                })

                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"OBSERVATION:\n{observation}\n\n"
                        "Continue exploring the ontology if needed, or produce "
                        "your FINAL_ANSWER with the complete plan of action."
                    ),
                })

            elif parsed["type"] == "error":
                if self.verbose:
                    logger.warning(f"PARSE ERROR: {parsed.get('error', 'unknown')}")
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        "I could not parse your response. Please follow the exact format:\n\n"
                        "For an action:\n"
                        "THOUGHT: <your reasoning>\n"
                        "ACTION: graph_ontology_tool\n"
                        "ACTION_INPUT: <valid JSON>\n\n"
                        "Or for your final plan:\n"
                        "THOUGHT: <your final reasoning>\n"
                        "FINAL_ANSWER: <your complete plan of action>"
                    ),
                })

        if self.verbose:
            logger.info(f"Max iterations ({self.MAX_ITERATIONS}) reached, forcing plan...")
        return self._force_final_answer(question)

    # ── Prompt Construction ──────────────────────────────────────────

    def _build_system_prompt(self):
        system = textwrap.dedent(f"""\
        You are a planning agent for IndiGo Airlines data analysis. Your job is
        to explore the knowledge graph ontology and produce a detailed PLAN OF
        ACTION — you do NOT execute data queries yourself.

        You work in a ReAct (Reason -> Act -> Observe) loop:
          1. THINK about what you need to understand about the data landscape
          2. ACT by calling graph_ontology_tool to explore the ontology
          3. OBSERVE the result
          4. Repeat until you have enough context to produce a plan

        -- YOUR TOOL --
        {GraphOntologyTool.TOOL_DESCRIPTION}

        -- YOUR GOAL --

        After exploring the ontology, produce a PLAN OF ACTION that describes:

        1. **Relevant Data Sources**: Which domain entities (tables), documents,
           subjects, and concepts are relevant to the question.

        2. **Navigation Path**: How the data sources connect — which graph
           relationships, correspondences, and foreign keys link them.

        3. **Search Strategy**: What vector searches (queries, filters) should
           be run against the document store.

        4. **SQL Strategy**: What SQL queries (tables, joins, filters, aggregations)
           should be run against the structured database.

        5. **Synthesis Approach**: How to combine the results from different
           sources into a final answer.

        The plan should be specific and actionable — include table names, column
        names, document names, subject names, and relationship types discovered
        from the ontology.

        -- RESPONSE FORMAT --

        On each turn, respond in EXACTLY one of these two formats:

        FORMAT A — Explore the ontology:
        THOUGHT: <your reasoning about what to explore next>
        ACTION: graph_ontology_tool
        ACTION_INPUT: <valid JSON object — on a SINGLE line>

        FORMAT B — Deliver the plan:
        THOUGHT: <your final reasoning summarizing what you discovered>
        FINAL_ANSWER: <your complete, structured plan of action>

        -- IMPORTANT RULES --
        - You can ONLY use graph_ontology_tool.
        - Respond with EXACTLY one format per turn (A or B). Never both.
        - ACTION_INPUT must be valid JSON on a SINGLE line.
        - Be thorough: explore entities, subjects, documents, and their
          connections before producing the plan.
        - The plan should be detailed enough for another agent to execute.
        """)

        self.messages.append({"role": "system", "content": system})

    def _build_user_prompt(self, question: str):
        self.messages.append({
            "role": "user",
            "content": (
                f"Please create a plan of action for answering the following question:\n\n"
                f"{question}\n\n"
                f"Start by exploring the knowledge graph ontology to understand "
                f"what data sources are available and how they connect."
            ),
        })

    # ── LLM Interaction ──────────────────────────────────────────────

    def _call_llm(self) -> str:
        completion = self.client.chat.completions.create(
            model=MODEL_DEPLOYMENT_NAME,
            messages=self.messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content

    # ── Response Parsing ─────────────────────────────────────────────

    def _parse_response(self, text: str) -> dict:
        text = text.strip()

        if "FINAL_ANSWER:" in text:
            return self._parse_final_answer(text)

        if "ACTION_INPUT:" in text:
            return self._parse_action(text)

        if len(text) > 200 and "THOUGHT:" not in text:
            return {"type": "final_answer", "answer": text}

        return {"type": "error", "error": "Could not parse THOUGHT/ACTION or FINAL_ANSWER"}

    def _parse_action(self, text: str) -> dict:
        thought = ""
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\nACTION:)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        tool_match = re.search(r"ACTION:\s*(\S+)", text)
        if not tool_match:
            return {"type": "error", "error": "Found ACTION_INPUT but no ACTION tool name"}
        tool_name = tool_match.group(1).strip()

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
        if answer.startswith("```"):
            answer = re.sub(r"^```\w*\n?", "", answer)
            answer = re.sub(r"\n?```$", "", answer)
            answer = answer.strip()

        return {"type": "final_answer", "answer": answer}

    def _json_candidates(self, text: str) -> list[str]:
        candidates = []
        candidates.append(text.strip())
        first_line = text.split("\n")[0].strip()
        candidates.append(first_line)

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

    def _force_final_answer(self, question: str) -> str:
        observations_summary = "\n\n".join(
            f"Step {t['iteration']} ({t['tool']}): {t['thought'][:200]}\n"
            f"  Result preview: {t['observation'][:200]}"
            for t in self.trace
        )

        self.messages.append({
            "role": "user",
            "content": textwrap.dedent(f"""\
            You have run out of exploration steps. Based on everything you have
            observed so far, produce your FINAL_ANSWER with a plan of action now.

            Original question: {question}

            Summary of your exploration:
            {observations_summary}

            Respond with ONLY:
            FINAL_ANSWER: <your complete plan of action>
            """),
        })

        response = self._call_llm()
        parsed = self._parse_response(response)

        if parsed["type"] == "final_answer":
            return parsed["answer"]

        return (
            "I was unable to fully produce a plan after exploring the "
            "available ontology. Here is what I found:\n\n"
            + observations_summary
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FoundryCBAgent wrapper — thin HTTP adapter for the Responses API
# ═══════════════════════════════════════════════════════════════════════════════

class PlannerCBAgent(FoundryCBAgent):
    """
    Thin FoundryCBAgent that receives Responses API calls on :8088,
    delegates to the PlannerAgent ReAct loop, and returns an OpenAIResponse.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set up OpenAI client
        if PROJECT_ENDPOINT:
            self.project_client = AIProjectClient(
                endpoint=PROJECT_ENDPOINT,
                credential=DefaultAzureCredential(),
            )
            self.openai_client = self.project_client.get_openai_client()
            logger.info("Using AIProjectClient OpenAI client.")
        else:
            self.openai_client = AzureOpenAI(
                api_version=os.getenv("OPENAI_API_VERSION", "2025-11-15-preview"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            )
            logger.info("Using AzureOpenAI client with key-based auth.")

        # Set up Gremlin + PlannerAgent
        logger.info("Connecting to Cosmos DB Gremlin...")
        self.gremlin_client = get_gremlin_client()
        run_gremlin(self.gremlin_client, "g.V().limit(1).count()")
        logger.info("Cosmos DB Gremlin connected (indigokg/knowledgegraph)")

        self.planner = PlannerAgent(
            llm_client=self.openai_client,
            gremlin_client=self.gremlin_client,
            verbose=True,
        )

    def _build_response(self, final_text: str, context: AgentRunContext) -> OpenAIResponse:
        """Build a non-streaming OpenAIResponse."""
        return OpenAIResponse({
            "object": "response",
            "agent": context.get_agent_id_object(),
            "conversation": context.get_conversation_object(),
            "metadata": {},
            "id": context.response_id,
            "created_at": int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
            "output": [
                ResponsesAssistantMessageItemResource(
                    id=context.id_generator.generate_message_id(),
                    status="completed",
                    content=[ItemContentOutputText(text=final_text, annotations=[])],
                )
            ],
            "status": "completed",
        })

    def _stream_response(self, final_text: str, context: AgentRunContext):
        """Return an async generator of SSE streaming events."""
        async def _generate():
            seq = 0

            yield ResponseCreatedEvent(
                sequence_number=seq,
                response=OpenAIResponse(
                    output=[],
                    conversation=context.get_conversation_object(),
                    agent=context.get_agent_id_object(),
                    id=context.response_id,
                ),
            )
            seq += 1

            item_id = context.id_generator.generate_message_id()
            yield ResponseOutputItemAddedEvent(
                sequence_number=seq,
                output_index=0,
                item=ResponsesAssistantMessageItemResource(
                    id=item_id,
                    status="in_progress",
                    content=[ItemContentOutputText(text="", annotations=[])],
                ),
            )
            seq += 1

            # Stream the text word-by-word
            assembled = ""
            words = final_text.split(" ")
            for idx, token in enumerate(words):
                piece = token if idx == len(words) - 1 else token + " "
                assembled += piece
                yield ResponseTextDeltaEvent(
                    sequence_number=seq,
                    output_index=0,
                    content_index=0,
                    delta=piece,
                )
                seq += 1

            yield ResponseTextDoneEvent(
                sequence_number=seq,
                output_index=0,
                content_index=0,
                text=assembled,
            )
            seq += 1

            yield ResponseCompletedEvent(
                sequence_number=seq,
                response=OpenAIResponse(
                    agent=context.get_agent_id_object(),
                    conversation=context.get_conversation_object(),
                    metadata={},
                    id=context.response_id,
                    created_at=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
                    output=[
                        ResponsesAssistantMessageItemResource(
                            id=item_id,
                            status="completed",
                            content=[ItemContentOutputText(text=assembled, annotations=[])],
                        )
                    ],
                ),
            )

        return _generate()

    async def agent_run(
        self, context: AgentRunContext
    ) -> Union[OpenAIResponse, AsyncGenerator[ResponseStreamEvent, Any]]:
        """Handle a Responses API request — delegate to the ReAct planner."""
        is_stream = context.request.get("stream", False)
        request_input = context.request.get("input")
        logger.info(f"Received input: {request_input}")

        # Extract user question from input
        if isinstance(request_input, str):
            question = request_input
        elif isinstance(request_input, list):
            # Find the last user message
            question = ""
            for item in reversed(request_input):
                if isinstance(item, dict):
                    if item.get("role") == "user":
                        question = item.get("content", "")
                        break
                    elif item.get("type") == "message" and item.get("role") == "user":
                        question = item.get("content", "")
                        break
            if not question:
                question = str(request_input)
        else:
            question = str(request_input)

        # Run the ReAct planner (synchronous)
        plan = self.planner.run(question)

        if is_stream:
            return self._stream_response(plan, context)
        else:
            return self._build_response(plan, context)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    agent = PlannerCBAgent()
    agent.run()  # starts HTTP server on :8088
