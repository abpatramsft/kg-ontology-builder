"""
planner_agent.py — Planning Agent over the Unified Knowledge Graph

A ReAct agent that uses ONLY the GraphOntologyTool to explore the knowledge
graph ontology, then produces a structured plan of action describing how to
navigate and search the available data sources to answer a user question.

It does NOT execute data queries — it just returns the plan.

Architecture:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                       PlannerAgent                                       │
  │                                                                          │
  │  THOUGHT ──► ACTION (graph_ontology_tool only) ──► OBSERVE               │
  │     ▲                                              │                     │
  │     └──────────────────────────────────────────────┘                     │
  │                                                                          │
  │  Tool:                                                                   │
  │    1. graph_ontology_tool — Navigate the Neo4j KG ontology               │
  │                                                                          │
  │  The agent's workflow:                                                   │
  │    a) Explore the ontology to understand what data exists & how it       │
  │       connects  (graph_ontology_tool)                                    │
  │    b) Produce a step-by-step plan for answering the question             │
  └──────────────────────────────────────────────────────────────────────────┘

Usage:
    python agents/planner_agent.py
    # Opens an interactive REPL. Ask questions to get a navigation plan.

    # Or programmatic usage:
    from agents.planner_agent import PlannerAgent
    agent = PlannerAgent(llm_client, gremlin_client)
    plan = agent.run("Which suppliers provide parts for the A320neo landing gear?")
"""

import json
import os
import re
import sys
import textwrap

from openai import AzureOpenAI

# Ensure src/ is on sys.path for utils imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # agents/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   # project root
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.llm import get_llm_client
from utils.cosmos_helpers import get_gremlin_client, run_gremlin

# Reuse the GraphOntologyTool from inference_agent
try:
    from agents.inference_agent import GraphOntologyTool
except ModuleNotFoundError:
    from inference_agent import GraphOntologyTool


# ═══════════════════════════════════════════════════════════════════════════════
#  Planner Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlannerAgent:
    """
    ReAct agent that explores the knowledge graph ontology and returns a
    structured plan of action — which tools to use, which tables/documents
    to query, and what joins or searches to perform — without actually
    executing any data retrieval.
    """

    MAX_ITERATIONS = 4

    def __init__(
        self,
        llm_client: AzureOpenAI,
        gremlin_client,
        verbose: bool = True,
    ):
        self.client = llm_client
        self.ontology_tool = GraphOntologyTool(gremlin_client)
        self.tools = {GraphOntologyTool.NAME: self.ontology_tool}
        self.verbose = verbose
        self.messages: list[dict] = []
        self.trace: list[dict] = []

    # ── Public API ───────────────────────────────────────────────────

    def run(self, question: str) -> str:
        """
        Explore the ontology and return a plan of action for answering
        the given question.
        """
        self.messages = []
        self.trace = []
        self._build_system_prompt()
        self._build_user_prompt(question)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  QUESTION: {question}")
            print(f"{'='*70}")

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            if self.verbose:
                print(f"\n  [Iteration {iteration}/{self.MAX_ITERATIONS}]")

            response_text = self._call_llm()
            parsed = self._parse_response(response_text)

            if parsed["type"] == "final_answer":
                if self.verbose:
                    print(f"  → PLAN produced after {iteration} iteration(s)\n")
                return parsed["answer"]

            elif parsed["type"] == "action":
                thought = parsed.get("thought", "")
                tool_name = parsed["tool_name"]
                action_input = parsed["action_input"]

                if self.verbose:
                    print(f"  THOUGHT: {thought[:150]}{'...' if len(thought) > 150 else ''}")
                    print(f"  ACTION:  {tool_name} → {json.dumps(action_input)}")

                tool = self.tools.get(tool_name)
                if tool is None:
                    observation = (
                        f"ERROR: Unknown tool '{tool_name}'. "
                        f"You only have: {GraphOntologyTool.NAME}"
                    )
                else:
                    observation = tool.execute(action_input)

                if self.verbose:
                    obs_preview = observation[:250] + ("..." if len(observation) > 250 else "")
                    print(f"  OBSERVE: {obs_preview}")

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
                    print(f"  PARSE ERROR: {parsed.get('error', 'unknown')}")
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

        # Max iterations — force a plan
        if self.verbose:
            print(f"\n  → Max iterations ({self.MAX_ITERATIONS}) reached, forcing plan...")
        return self._force_final_answer(question)

    # ── Prompt Construction ──────────────────────────────────────────

    def _build_system_prompt(self):
        system = textwrap.dedent(f"""\
        You are a planning agent for IndiGo Airlines data analysis. Your job is
        to explore the knowledge graph ontology and produce a detailed PLAN OF
        ACTION — you do NOT execute data queries yourself.

        You work in a ReAct (Reason → Act → Observe) loop:
          1. THINK about what you need to understand about the data landscape
          2. ACT by calling graph_ontology_tool to explore the ontology
          3. OBSERVE the result
          4. Repeat until you have enough context to produce a plan

        ── YOUR TOOL ──────────────────────────────────────────────────────────
        {GraphOntologyTool.TOOL_DESCRIPTION}

        ── YOUR GOAL ──────────────────────────────────────────────────────────

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

        ── RESPONSE FORMAT ────────────────────────────────────────────────────

        On each turn, respond in EXACTLY one of these two formats:

        FORMAT A — Explore the ontology:
        THOUGHT: <your reasoning about what to explore next>
        ACTION: graph_ontology_tool
        ACTION_INPUT: <valid JSON object — on a SINGLE line>

        FORMAT B — Deliver the plan:
        THOUGHT: <your final reasoning summarizing what you discovered>
        FINAL_ANSWER: <your complete, structured plan of action>

        ── IMPORTANT RULES ────────────────────────────────────────────────────

        - You can ONLY use graph_ontology_tool. Do NOT attempt to use
          vector_search_tool or sql_query_tool.
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
            model="gpt-4.1",
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

    # ── Fallback ─────────────────────────────────────────────────────

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
#  Interactive REPL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive planning loop."""
    print("=" * 70)
    print("  IndiGo Airlines — Knowledge Graph Planner Agent")
    print("  Explores ontology and produces a plan of action")
    print("=" * 70)

    # ── Connect to data sources ────────────────────────────────────────
    print("\n[1/2] Connecting to Cosmos DB (Gremlin)...")
    gremlin = get_gremlin_client()
    try:
        run_gremlin(gremlin, "g.V().limit(1).count()")
        print("  ✓ Cosmos DB Gremlin connected (indigokg/knowledgegraph)")
    except Exception as e:
        print(f"  ✗ Cosmos DB connection failed: {e}")
        print("    Check COSMOS_DB_KEY in your .env file.")
        return

    print("[2/2] Initializing Azure OpenAI client...")
    llm_client = get_llm_client()
    print("  ✓ LLM client ready")

    # ── Build agent ────────────────────────────────────────────────────
    agent = PlannerAgent(llm_client=llm_client, gremlin_client=gremlin, verbose=True)
    print(f"\n  Tool loaded: {GraphOntologyTool.NAME}")
    print("-" * 70)
    print("  Type your question and press Enter. Type 'quit' or 'exit' to stop.")
    print("  Type 'trace' to see the last agent trace.")
    print("-" * 70)

    last_trace = []

    while True:
        try:
            question = input("\n❓ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "trace":
            if last_trace:
                print(json.dumps(last_trace, indent=2, default=str))
            else:
                print("No trace available yet.")
            continue

        plan = agent.run(question)
        last_trace = agent.trace

        print(f"\n{'─'*70}")
        print(f"📋 Plan of Action:\n")
        print(plan)
        print(f"{'─'*70}")
        print(f"  ({len(agent.trace)} ontology exploration steps)")


if __name__ == "__main__":
    main()
