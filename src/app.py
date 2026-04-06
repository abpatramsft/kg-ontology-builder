"""
app.py — Flask web UI for the Knowledge Graph Inference Agent

Streams intermediate agent steps (THOUGHT → ACTION → OBSERVE) to the
browser via Server-Sent Events so each step appears in real time.

Usage:
    python src/app.py          # starts on http://localhost:5050
"""

import json
import os
import sys
import time
import queue
import threading

from flask import Flask, render_template, request, Response, jsonify, stream_with_context

# Ensure src/ is on sys.path for package imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # repo root
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.inference_agent import (
    InferenceAgent,
    GraphOntologyTool,
    VectorSearchTool,
    SQLQueryTool,
    build_tools,
)
from utils.llm import get_llm_client, get_embedding_client, embed_texts
from utils.cosmos_helpers import get_gremlin_client, run_gremlin
from utils.cosmos_vector_helpers import get_vector_container, get_collection_stats


# ═══════════════════════════════════════════════════════════════════════════════
#  Streaming wrapper — captures each agent step into an event queue
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingInferenceAgent(InferenceAgent):
    """
    Subclass that pushes each intermediate step as an SSE event
    into a queue that the Flask route consumes.
    """

    def __init__(self, llm_client, tools, event_queue: queue.Queue):
        super().__init__(llm_client, tools, verbose=True)
        self.eq = event_queue

    def _emit(self, event_type: str, data: dict):
        self.eq.put({"event": event_type, "data": data})

    def run(self, question: str) -> str:
        self.messages = []
        self.trace = []
        self._build_system_prompt()
        self._build_user_prompt(question)

        self._emit("status", {"message": "Agent started", "question": question})

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            self._emit("status", {"message": f"Iteration {iteration}/{self.MAX_ITERATIONS}"})

            response_text = self._call_llm()
            parsed = self._parse_response(response_text)

            if parsed["type"] == "final_answer":
                self._emit("final_answer", {
                    "answer": parsed["answer"],
                    "iterations": iteration,
                    "total_steps": len(self.trace),
                })
                return parsed["answer"]

            elif parsed["type"] == "action":
                thought = parsed.get("thought", "")
                tool_name = parsed["tool_name"]
                action_input = parsed["action_input"]

                # Emit thought
                self._emit("thought", {
                    "iteration": iteration,
                    "thought": thought,
                })

                # Emit action
                self._emit("action", {
                    "iteration": iteration,
                    "tool": tool_name,
                    "action_input": action_input,
                })

                # Execute the tool
                tool = self.tools.get(tool_name)
                if tool is None:
                    observation = (
                        f"ERROR: Unknown tool '{tool_name}'. Available: "
                        + ", ".join(self.tools.keys())
                    )
                else:
                    observation = tool.execute(action_input)

                # Emit observation
                self._emit("observation", {
                    "iteration": iteration,
                    "tool": tool_name,
                    "observation": observation,
                })

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
                        "Continue your analysis. You may take another ACTION "
                        "(using any of the 3 tools) or provide your FINAL_ANSWER."
                    ),
                })

            elif parsed["type"] == "error":
                self._emit("parse_error", {
                    "iteration": iteration,
                    "error": parsed.get("error", "unknown"),
                })
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        "I could not parse your response. Please follow the exact format:\n\n"
                        "For an action:\n"
                        "THOUGHT: <your reasoning>\n"
                        "ACTION: <tool_name>\n"
                        "ACTION_INPUT: <valid JSON>\n\n"
                        "Or for your final answer:\n"
                        "THOUGHT: <your final reasoning>\n"
                        "FINAL_ANSWER: <your complete answer>"
                    ),
                })

        # Max iterations — force answer
        answer = self._force_final_answer(question)
        self._emit("final_answer", {
            "answer": answer,
            "iterations": self.MAX_ITERATIONS,
            "total_steps": len(self.trace),
        })
        return answer


# ═══════════════════════════════════════════════════════════════════════════════
#  Flask Application
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Globals — initialized once at startup
_gremlin = None
_vector_container = None
_llm_client = None
_embedding_client = None
_db_path = None
_tools = None


def init_resources():
    """Connect to all data sources once at startup."""
    global _gremlin, _vector_container, _llm_client, _embedding_client, _db_path, _tools

    print("[1/4] Connecting to Cosmos DB (Gremlin)...")
    _gremlin = get_gremlin_client()
    run_gremlin(_gremlin, "g.V().limit(1).count()")
    print("  OK (indigokg/knowledgegraph)")

    print("[2/4] Connecting to Cosmos DB vector store...")
    _vector_container = get_vector_container()
    stats = get_collection_stats(_vector_container)
    print(f"  OK ({stats['total_chunks']} chunks)")

    print("[3/4] Azure OpenAI clients...")
    _llm_client = get_llm_client()
    _embedding_client = get_embedding_client()
    print("  OK")

    print("[4/4] SQLite database...")
    _db_path = os.path.join(PROJECT_ROOT, "source_data", "airlines.db")
    assert os.path.exists(_db_path), f"DB not found: {_db_path}"
    print("  OK")

    _tools = build_tools(_gremlin, _vector_container, _embedding_client, _db_path)
    print(f"\nReady — tools: {', '.join(_tools.keys())}")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["GET"])
def ask():
    """SSE endpoint — streams agent steps as events."""
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    eq = queue.Queue()

    def run_agent():
        try:
            agent = StreamingInferenceAgent(_llm_client, _tools, eq)
            agent.run(question)
        except Exception as e:
            eq.put({"event": "error", "data": {"message": str(e)}})
        finally:
            eq.put(None)  # sentinel

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    def generate():
        while True:
            item = eq.get()
            if item is None:
                # Send a done event so the JS knows to close
                yield f"event: done\ndata: {{}}\n\n"
                break
            event_type = item["event"]
            payload = json.dumps(item["data"], default=str)
            yield f"event: {event_type}\ndata: {payload}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/sample-trace")
def sample_trace():
    """Return the saved test_inference_results.json if it exists."""
    path = os.path.join(PROJECT_ROOT, "test_inference_results.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify([])


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_resources()
    print("\n  Starting web UI at  http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
