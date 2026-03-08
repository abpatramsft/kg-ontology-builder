"""
test_inference.py — Test script for the Knowledge Graph Inference Agent

Runs a set of sample questions through the InferenceAgent, printing every
intermediate THOUGHT → ACTION → OBSERVATION step in a clear, readable format.

Usage:
    python test_inference.py                 # run all test questions
    python test_inference.py --question 2    # run only question #2
    python test_inference.py --custom "..."  # run a custom question
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime

# Ensure project root on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agent_inference import InferenceAgent, build_tools
from utils.llm import get_llm_client, get_embedding_client
from utils.neo4j_helpers import get_neo4j_driver, run_cypher

import lancedb


# ═══════════════════════════════════════════════════════════════════════════════
#  Test questions — covering all 3 tools and cross-source queries
# ═══════════════════════════════════════════════════════════════════════════════

TEST_QUESTIONS = [
    # Q1: Ontology + SQL — follows FK chain through structured DB
    "Which suppliers provide parts for the A320neo landing gear assembly, and what is each part's material and cost?",

    # Q2: Ontology + Vector search — finds document insights about a concept
    "What do the maintenance and quality review documents say about engine-related issues?",

    # Q3: Cross-source — bridges a document concept to structured data via CORRESPONDS_TO
    "The documents mention brake assemblies — which structured database tables hold data about brakes, and what specific parts and suppliers are involved?",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Rich verbose printing (wraps the agent's default verbose with extra detail)
# ═══════════════════════════════════════════════════════════════════════════════

SEPARATOR     = "═" * 78
THIN_SEP      = "─" * 78
SECTION_SEP   = "┄" * 78


def print_header():
    print()
    print(SEPARATOR)
    print("  IndiGo Airlines — Knowledge Graph Inference Agent  ·  TEST RUN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEPARATOR)


def print_connection_status(label, ok, detail=""):
    status = "✓" if ok else "✗"
    msg = f"  {status} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def print_question_banner(idx, total, question):
    print()
    print(SEPARATOR)
    print(f"  QUESTION {idx}/{total}")
    print(SEPARATOR)
    print(f"  {question}")
    print(THIN_SEP)


def print_trace_summary(trace):
    """Print a compact summary table of all agent steps."""
    print()
    print(SECTION_SEP)
    print("  AGENT TRACE SUMMARY")
    print(SECTION_SEP)
    print(f"  {'Step':>4}  {'Tool':<24}  {'Action':<30}  Observation (preview)")
    print(f"  {'────':>4}  {'────────────────────────':<24}  {'──────────────────────────────':<30}  {'─'*30}")

    for step in trace:
        action_str = json.dumps(step["action"].get("action", step["action"]))
        if len(action_str) > 28:
            action_str = action_str[:28] + ".."
        obs_preview = step["observation"].replace("\n", " ")[:30]
        print(f"  {step['iteration']:>4}  {step['tool']:<24}  {action_str:<30}  {obs_preview}...")

    print(SECTION_SEP)


def print_full_trace(trace):
    """Print the complete trace with full details."""
    print()
    print(SECTION_SEP)
    print("  FULL AGENT TRACE (all intermediate steps)")
    print(SECTION_SEP)

    for step in trace:
        print(f"\n  ┌── Step {step['iteration']} ──────────────────────────────────────")
        print(f"  │ THOUGHT: {step['thought']}")
        print(f"  │ TOOL:    {step['tool']}")
        print(f"  │ ACTION:  {json.dumps(step['action'], indent=2).replace(chr(10), chr(10) + '  │          ')}")
        print(f"  │")

        # Print observation with nice wrapping (truncate at 1500 chars for readability)
        obs = step["observation"]
        if len(obs) > 1500:
            obs = obs[:1500] + f"\n  ... (truncated, {len(step['observation'])} chars total)"
        for line in obs.split("\n"):
            print(f"  │ OBSERVE: {line}")

        print(f"  └{'─'*60}")


def print_answer(answer, elapsed):
    print()
    print(SEPARATOR)
    print("  FINAL ANSWER")
    print(SEPARATOR)
    print()
    print(answer)
    print()
    print(THIN_SEP)
    print(f"  Time: {elapsed:.1f}s")
    print(THIN_SEP)


# ═══════════════════════════════════════════════════════════════════════════════
#  Setup — connect to all data sources
# ═══════════════════════════════════════════════════════════════════════════════

def setup():
    """Connect to Neo4j, LanceDB, Azure OpenAI, and SQLite. Returns all clients."""
    print_header()
    print("\n  Connecting to data sources...\n")

    # Neo4j
    driver = get_neo4j_driver()
    try:
        run_cypher(driver, "RETURN 1 AS ok")
        print_connection_status("Neo4j", True, "bolt://localhost:7687")
    except Exception as e:
        print_connection_status("Neo4j", False, str(e))
        print("\n  ERROR: Neo4j must be running. Start it with:  docker compose up -d")
        sys.exit(1)

    # LanceDB
    lance_db_path = os.path.join(BASE_DIR, "data", "lancedb_store")
    db = lancedb.connect(lance_db_path)
    try:
        lance_table = db.open_table("lexical_chunks")
        print_connection_status("LanceDB", True, f"lexical_chunks — {len(lance_table)} rows")
    except Exception as e:
        print_connection_status("LanceDB", False, str(e))
        print("\n  ERROR: Run the lexical_graph pipeline first.")
        sys.exit(1)

    # Azure OpenAI
    llm_client = get_llm_client()
    embedding_client = get_embedding_client()
    print_connection_status("Azure OpenAI (LLM)", True, "gpt-4.1")
    print_connection_status("Azure OpenAI (Embeddings)", True, "text-embedding-3-small")

    # SQLite
    db_path = os.path.join(BASE_DIR, "data", "manufacturing.db")
    if not os.path.exists(db_path):
        print_connection_status("SQLite", False, f"Not found: {db_path}")
        print("\n  ERROR: Run:  python data/setup_db.py")
        sys.exit(1)
    print_connection_status("SQLite", True, db_path)

    return driver, lance_table, llm_client, embedding_client, db_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Run a single question through the agent
# ═══════════════════════════════════════════════════════════════════════════════

def run_question(agent: InferenceAgent, question: str, idx: int, total: int):
    """Run a single question, printing all intermediate steps and the final answer."""
    print_question_banner(idx, total, question)

    start = time.time()
    answer = agent.run(question)
    elapsed = time.time() - start

    # Print compact summary table
    print_trace_summary(agent.trace)

    # Print full trace with all details
    print_full_trace(agent.trace)

    # Print the final answer
    print_answer(answer, elapsed)

    return {
        "question": question,
        "answer": answer,
        "steps": len(agent.trace),
        "elapsed": round(elapsed, 1),
        "trace": agent.trace,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test the KG Inference Agent")
    parser.add_argument(
        "--question", "-q", type=int, default=None,
        help="Run only test question N (1-indexed)",
    )
    parser.add_argument(
        "--custom", "-c", type=str, default=None,
        help="Run a custom question instead of the test suite",
    )
    parser.add_argument(
        "--save-trace", "-s", action="store_true",
        help="Save full trace to test_inference_results.json",
    )
    args = parser.parse_args()

    # Setup connections
    driver, lance_table, llm_client, embedding_client, db_path = setup()

    # Build tools + agent
    tools = build_tools(driver, lance_table, embedding_client, db_path)
    agent = InferenceAgent(llm_client=llm_client, tools=tools, verbose=True)

    print(f"\n  Tools: {', '.join(tools.keys())}")

    # Determine which questions to run
    if args.custom:
        questions = [args.custom]
    elif args.question is not None:
        if 1 <= args.question <= len(TEST_QUESTIONS):
            questions = [TEST_QUESTIONS[args.question - 1]]
        else:
            print(f"\n  ERROR: --question must be 1-{len(TEST_QUESTIONS)}")
            sys.exit(1)
    else:
        questions = TEST_QUESTIONS

    print(f"  Questions to run: {len(questions)}")
    print(THIN_SEP)

    # Run each question
    results = []
    for i, q in enumerate(questions, 1):
        result = run_question(agent, q, i, len(questions))
        results.append(result)

    # ── Final summary ──────────────────────────────────────────────────
    print()
    print(SEPARATOR)
    print("  TEST RUN COMPLETE — SUMMARY")
    print(SEPARATOR)
    print(f"\n  {'#':>3}  {'Steps':>5}  {'Time':>7}  Question")
    print(f"  {'───':>3}  {'─────':>5}  {'───────':>7}  {'─'*50}")
    for i, r in enumerate(results, 1):
        print(f"  {i:>3}  {r['steps']:>5}  {r['elapsed']:>5.1f}s  {r['question'][:50]}{'...' if len(r['question']) > 50 else ''}")

    total_time = sum(r["elapsed"] for r in results)
    total_steps = sum(r["steps"] for r in results)
    print(f"\n  Total: {total_steps} steps, {total_time:.1f}s across {len(results)} question(s)")
    print(SEPARATOR)

    # ── Save trace if requested ────────────────────────────────────────
    if args.save_trace:
        out_path = os.path.join(BASE_DIR, "test_inference_results.json")
        # Serialize results (strip observation to save space)
        serializable = []
        for r in results:
            serializable.append({
                "question": r["question"],
                "answer": r["answer"],
                "steps": r["steps"],
                "elapsed": r["elapsed"],
                "trace": [
                    {
                        "iteration": t["iteration"],
                        "thought": t["thought"],
                        "tool": t["tool"],
                        "action": t["action"],
                        "observation_preview": t["observation"][:500],
                    }
                    for t in r["trace"]
                ],
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n  Trace saved to: {out_path}")


if __name__ == "__main__":
    main()
