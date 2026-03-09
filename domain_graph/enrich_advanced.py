"""
enrich_advanced.py — Advanced ReAct Agent-based Schema Enrichment

Instead of a single LLM call per table (domain_graph.enrich_with_llm), this
module uses a custom ReAct (Reason → Act → Observe) agent that iteratively
explores the SQL database to build richer, more accurate semantic metadata.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                  EnrichmentAgent                        │
  │                                                         │
  │  THOUGHT ──► ACTION (sql_db_query_tool) ──► OBSERVE     │
  │     ▲                                          │        │
  │     └──────────────────────────────────────────┘        │
  │                                                         │
  │  Repeats until FINAL_ANSWER or max iterations reached   │
  └─────────────────────────────────────────────────────────┘

The agent has one tool — SQLDBQueryTool — which gives it full read-only
access to the SQLite database: list tables, describe schemas, sample rows,
run arbitrary SELECT queries, and navigate FK chains.

Usage:
    from enrich_advanced import enrich_with_llm_advanced

    enriched = enrich_with_llm_advanced(schema, client)
    # Returns same dict format as domain_graph.enrich_with_llm
"""

import json
import os
import re
import sqlite3
import textwrap

from openai import AzureOpenAI


# ═══════════════════════════════════════════════════════════════════════════════
#  SQL Database Query Tool
# ═══════════════════════════════════════════════════════════════════════════════

class SQLDBQueryTool:
    """
    Read-only tool that gives an agent full access to explore a SQLite database.

    Supported actions:
      - list_tables            → list all table names
      - describe_table(table)  → columns, types, PKs, FKs, row count
      - sample_rows(table, n)  → first N rows (default 5)
      - query(sql)             → run an arbitrary read-only SQL query
      - get_foreign_keys(table)→ FK relationships for a table
      - distinct_values(table, column, n) → distinct values in a column (default 20)
    """

    TOOL_DESCRIPTION = textwrap.dedent("""\
    sql_db_query_tool — Full read-only access to the SQLite database.

    Available actions (pass as JSON):

    1. {"action": "list_tables"}
       Returns: list of all table names in the database.

    2. {"action": "describe_table", "table": "<table_name>"}
       Returns: columns (name, type, nullable, PK), foreign keys, row count.

    3. {"action": "sample_rows", "table": "<table_name>", "limit": 5}
       Returns: first N rows from the table (default 5). Useful to understand
       what kind of data the table holds.

    4. {"action": "query", "sql": "<SELECT ...>"}
       Runs an arbitrary read-only SQL query. Use this for JOINs, aggregations,
       filtering, or any exploration that the other actions don't cover.
       ONLY SELECT statements are allowed.

    5. {"action": "get_foreign_keys", "table": "<table_name>"}
       Returns: foreign key relationships (from_col → to_table.to_col).

    6. {"action": "distinct_values", "table": "<table_name>", "column": "<col>", "limit": 20}
       Returns: distinct values in a column (up to limit). Good for understanding
       categorical data and enumerations.
    """)

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, action_input: dict) -> str:
        """
        Execute a tool action and return a string result.

        Args:
            action_input: dict with "action" key and action-specific parameters.

        Returns:
            String representation of the result (fed back to LLM as OBSERVATION).
        """
        action = action_input.get("action", "").strip().lower()

        try:
            if action == "list_tables":
                return self._list_tables()
            elif action == "describe_table":
                return self._describe_table(action_input["table"])
            elif action == "sample_rows":
                limit = action_input.get("limit", 5)
                return self._sample_rows(action_input["table"], limit)
            elif action == "query":
                return self._query(action_input["sql"])
            elif action == "get_foreign_keys":
                return self._get_foreign_keys(action_input["table"])
            elif action == "distinct_values":
                limit = action_input.get("limit", 20)
                return self._distinct_values(
                    action_input["table"], action_input["column"], limit
                )
            else:
                return f"ERROR: Unknown action '{action}'. Available: list_tables, describe_table, sample_rows, query, get_foreign_keys, distinct_values"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    # ── Action implementations ───────────────────────────────────────────

    def _list_tables(self) -> str:
        conn = self._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        conn.close()
        names = [t["name"] for t in tables]
        return json.dumps({"tables": names}, indent=2)

    def _describe_table(self, table: str) -> str:
        conn = self._get_conn()
        cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        fks = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
        row_count = conn.execute(f"SELECT COUNT(*) as cnt FROM [{table}]").fetchone()["cnt"]
        conn.close()

        columns = []
        for c in cols:
            columns.append({
                "name": c["name"],
                "type": c["type"],
                "notnull": bool(c["notnull"]),
                "primary_key": bool(c["pk"]),
            })

        foreign_keys = []
        for fk in fks:
            foreign_keys.append({
                "from_column": fk["from"],
                "to_table": fk["table"],
                "to_column": fk["to"],
            })

        result = {
            "table": table,
            "row_count": row_count,
            "columns": columns,
            "foreign_keys": foreign_keys,
        }
        return json.dumps(result, indent=2)

    def _sample_rows(self, table: str, limit: int) -> str:
        conn = self._get_conn()
        rows = conn.execute(f"SELECT * FROM [{table}] LIMIT {int(limit)}").fetchall()
        conn.close()
        result = [dict(r) for r in rows]
        return json.dumps(result, indent=2)

    def _query(self, sql: str) -> str:
        # Safety: only allow SELECT
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return "ERROR: Only SELECT queries are allowed (read-only access)."

        conn = self._get_conn()
        try:
            rows = conn.execute(sql).fetchall()
            result = [dict(r) for r in rows]
            # Truncate if too many rows
            if len(result) > 50:
                result = result[:50]
                return json.dumps(result, indent=2) + f"\n... (truncated, {len(result)} of many rows shown)"
            return json.dumps(result, indent=2)
        finally:
            conn.close()

    def _get_foreign_keys(self, table: str) -> str:
        conn = self._get_conn()
        fks = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
        conn.close()

        result = []
        for fk in fks:
            result.append({
                "from_column": fk["from"],
                "to_table": fk["table"],
                "to_column": fk["to"],
            })
        return json.dumps({"table": table, "foreign_keys": result}, indent=2)

    def _distinct_values(self, table: str, column: str, limit: int) -> str:
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT DISTINCT [{column}] FROM [{table}] LIMIT {int(limit)}"
        ).fetchall()
        conn.close()
        values = [dict(r)[column] for r in rows]
        return json.dumps({"table": table, "column": column, "distinct_values": values}, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  ReAct Enrichment Agent
# ═══════════════════════════════════════════════════════════════════════════════

class EnrichmentAgent:
    """
    Custom ReAct (Reason → Act → Observe) agent that iteratively explores
    a SQL database to produce rich semantic metadata for a given table.

    Workflow per table:
      1. Receives target table name + high-level schema overview
      2. THINKS about what to explore (e.g., "I should look at sample rows")
      3. ACTS by calling sql_db_query_tool with an action
      4. OBSERVES the result
      5. Repeats steps 2-4 until it has enough context
      6. Produces FINAL_ANSWER — the enriched metadata JSON

    This is a custom agent — no frameworks, no LangChain, no CrewAI.
    Just structured prompting + a parse-execute loop.
    """

    MAX_ITERATIONS = 10  # safety cap per table

    def __init__(
        self,
        llm_client: AzureOpenAI,
        tool: SQLDBQueryTool,
        target_table: str,
        all_tables: list[str],
        verbose: bool = True,
    ):
        self.client = llm_client
        self.tool = tool
        self.target_table = target_table
        self.all_tables = all_tables
        self.verbose = verbose
        self.messages: list[dict] = []
        self.trace: list[dict] = []  # (thought, action, observation) tuples

    # ── Public API ───────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the ReAct loop and return enriched metadata.

        Returns:
            {
                "description": str,
                "domain": str,
                "semantic_relationships": [
                    {"target_table": str, "relationship_type": str, "reason": str}
                ]
            }
        """
        self._build_system_prompt()
        self._build_initial_user_prompt()

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            if self.verbose:
                print(f"      [Iteration {iteration}/{self.MAX_ITERATIONS}]")

            # Get LLM response
            response_text = self._call_llm()

            # Parse: is it a FINAL_ANSWER or an ACTION?
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

                # Execute the tool
                observation = self.tool.execute(action_input)

                if self.verbose:
                    obs_preview = observation[:200] + ("..." if len(observation) > 200 else "")
                    print(f"      OBSERVE: {obs_preview}")

                # Record trace
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action_input,
                    "observation": observation,
                })

                # Feed observation back to LLM
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": f"OBSERVATION:\n{observation}\n\nContinue your analysis. Either take another ACTION or provide your FINAL_ANSWER.",
                })

            elif parsed["type"] == "error":
                # LLM produced something we can't parse — nudge it
                if self.verbose:
                    print(f"      PARSE ERROR: {parsed.get('error', 'unknown')}")
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({
                    "role": "user",
                    "content": (
                        "I could not parse your response. Please follow the exact format:\n\n"
                        "For an action:\n"
                        "THOUGHT: <your reasoning>\n"
                        "ACTION: sql_db_query_tool\n"
                        "ACTION_INPUT: <valid JSON>\n\n"
                        "Or for your final answer:\n"
                        "THOUGHT: <your final reasoning>\n"
                        "FINAL_ANSWER:\n<valid JSON with description, domain, semantic_relationships>"
                    ),
                })

        # Max iterations exhausted — force a final answer
        if self.verbose:
            print(f"      → Max iterations ({self.MAX_ITERATIONS}) reached, forcing final answer...")
        return self._force_final_answer()

    # ── Prompt Construction ──────────────────────────────────────────────

    def _build_system_prompt(self):
        system = textwrap.dedent(f"""\
        You are a senior data architect agent analyzing a database for an airlines company.

        Your task: Produce rich semantic metadata for the table "{self.target_table}".

        You work in a ReAct loop — you THINK, then ACT (use a tool), then OBSERVE
        the result, and repeat until you have enough context for a high-quality answer.

        ── YOUR TOOL ─────────────────────────────────────────────────────────
        {self.tool.TOOL_DESCRIPTION}
        ── RESPONSE FORMAT ───────────────────────────────────────────────────

        On each turn, respond in EXACTLY one of these two formats:

        FORMAT A — Take an action:
        THOUGHT: <your reasoning about what to explore next and why>
        ACTION: sql_db_query_tool
        ACTION_INPUT: <valid JSON object for the tool>

        FORMAT B — Provide final answer (when you've explored enough):
        THOUGHT: <your final reasoning summarizing what you learned>
        FINAL_ANSWER:
        <valid JSON object — see required schema below>

        ── REQUIRED OUTPUT SCHEMA (for FINAL_ANSWER) ─────────────────────────

        {{
          "description": "2-3 sentences: what this table represents, what kind of data it holds, its business purpose in the context of airlines operations",
          "domain": "single domain label (e.g., fleet_management, flight_operations, crew_management, bookings, revenue, maintenance, route_network, passenger_services)",
          "semantic_relationships": [
            {{
              "target_table": "<name of related table from {self.all_tables}>",
              "relationship_type": "UPPER_SNAKE_CASE label (e.g., OPERATES_ON, ASSIGNED_TO, BOOKED_FOR, BELONGS_TO, COMPOSED_OF)",
              "reason": "1 sentence explaining WHY this semantic relationship exists, grounded in the data you observed"
            }}
          ],
          "concepts": [
            {{
              "name": "<Abstract business concept derived from this table, e.g. 'Flight Schedule', 'Crew Assignment', 'Revenue Stream'>",
              "description": "1 sentence: what this concept represents in the airlines domain",
              "derived_from": ["<column_name_1>", "<column_name_2>"]
            }}
          ]
        }}

        CONCEPT GUIDELINES:
        - Extract 2-5 abstract business concepts per table.
        - Concepts are HIGH-LEVEL ideas (e.g., 'Route Coverage', 'Booking Pattern', 'Crew Utilization'), NOT raw column names or data values.
        - derived_from lists the column(s) that evidence this concept.
        - Think about what business themes this table's data represents.

        ── STRATEGY ──────────────────────────────────────────────────────────

        1. Start by understanding the target table's schema and sample rows.
        2. Follow foreign key chains to related tables — understand how they connect.
        3. Look at actual data values to understand real-world semantics (not just column names).
        4. Use JOINs or aggregations if useful to understand cardinality and patterns.
        5. When you have a clear picture, produce your FINAL_ANSWER.
        6. Aim for 3-6 exploration steps. Don't over-explore, but don't rush either.

        ── IMPORTANT RULES ───────────────────────────────────────────────────

        - Respond with EXACTLY one format per turn (A or B). Never both.
        - ACTION_INPUT must be valid parseable JSON on a SINGLE line.
        - FINAL_ANSWER JSON must be valid and parseable.
        - Only reference tables that actually exist: {self.all_tables}
        - Relationship types should be semantically meaningful UPPER_SNAKE_CASE labels.
        - Ground your descriptions in observed data, not assumptions.
        """)

        self.messages.append({"role": "system", "content": system})

    def _build_initial_user_prompt(self):
        prompt = textwrap.dedent(f"""\
        Begin your analysis of the table "{self.target_table}".

        Database tables available: {self.all_tables}

        Start by exploring the target table's structure and data. Use the
        sql_db_query_tool to gather context. Take your time — explore related
        tables as needed before producing your FINAL_ANSWER.
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
          - {"type": "final_answer", "result": dict}
          - {"type": "error", "error": str}
        """
        text = text.strip()

        # ── Try to detect FINAL_ANSWER ───────────────────────────────
        if "FINAL_ANSWER" in text:
            return self._parse_final_answer(text)

        # ── Try to detect ACTION ─────────────────────────────────────
        if "ACTION_INPUT" in text:
            return self._parse_action(text)

        # ── Fallback: check if there's embedded JSON that looks like final answer
        json_match = re.search(r'\{[^{}]*"description"[^{}]*"domain"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if "description" in result and "domain" in result:
                    return {"type": "final_answer", "result": self._normalize_result(result)}
            except json.JSONDecodeError:
                pass

        return {"type": "error", "error": "Could not parse THOUGHT/ACTION or FINAL_ANSWER from response"}

    def _parse_action(self, text: str) -> dict:
        """Extract THOUGHT and ACTION_INPUT from a response."""
        thought = ""
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\nACTION)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Find ACTION_INPUT — extract the JSON
        input_match = re.search(r"ACTION_INPUT:\s*(.+)", text, re.DOTALL)
        if not input_match:
            return {"type": "error", "error": "Found ACTION but no ACTION_INPUT"}

        raw_input = input_match.group(1).strip()

        # Try to parse JSON — the input might span multiple lines or have trailing text
        # First try: take everything until end or next section
        for candidate in self._json_candidates(raw_input):
            try:
                action_input = json.loads(candidate)
                if isinstance(action_input, dict):
                    return {"type": "action", "thought": thought, "action_input": action_input}
            except json.JSONDecodeError:
                continue

        return {"type": "error", "error": f"Could not parse ACTION_INPUT as JSON: {raw_input[:200]}"}

    def _parse_final_answer(self, text: str) -> dict:
        """Extract FINAL_ANSWER JSON from a response."""
        # Find everything after "FINAL_ANSWER:"
        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.DOTALL)
        if not fa_match:
            return {"type": "error", "error": "Found FINAL_ANSWER marker but no content after it"}

        raw = fa_match.group(1).strip()

        # Strip markdown fences if present
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
        """
        Generate candidate JSON strings from text — tries multiple extraction
        strategies to handle common LLM formatting quirks.
        """
        candidates = []

        # Strategy 1: The entire text is valid JSON
        candidates.append(text.strip())

        # Strategy 2: First line only (for single-line JSON)
        first_line = text.split("\n")[0].strip()
        candidates.append(first_line)

        # Strategy 3: Find the first { ... } block (greedy)
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group())

        # Strategy 4: Find JSON by brace matching
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

        return candidates

    def _normalize_result(self, result: dict) -> dict:
        """Ensure the result has all required fields with correct types."""
        normalized = {
            "description": result.get("description", f"Database table '{self.target_table}'."),
            "domain": result.get("domain", "unknown"),
            "semantic_relationships": [],
            "concepts": [],
        }

        for rel in result.get("semantic_relationships", []):
            if isinstance(rel, dict) and "target_table" in rel:
                # Only keep relationships to tables that actually exist
                if rel["target_table"] in self.all_tables:
                    normalized["semantic_relationships"].append({
                        "target_table": rel["target_table"],
                        "relationship_type": rel.get("relationship_type", "RELATED_TO"),
                        "reason": rel.get("reason", ""),
                    })

        # Validate and normalize concepts
        for concept in result.get("concepts", []):
            if isinstance(concept, dict) and concept.get("name"):
                normalized["concepts"].append({
                    "name": concept.get("name", "Unknown"),
                    "description": concept.get("description", ""),
                    "derived_from": concept.get("derived_from", []),
                })

        return normalized

    # ── Fallback: Force Final Answer ─────────────────────────────────────

    def _force_final_answer(self) -> dict:
        """
        If the agent runs out of iterations, force it to summarize
        what it has learned so far into a final answer.
        """
        # Compile observations so far
        observations_summary = "\n\n".join(
            f"Step {t['iteration']}: {t['thought'][:150]}"
            for t in self.trace
        )

        self.messages.append({
            "role": "user",
            "content": textwrap.dedent(f"""\
            You have run out of exploration steps. Based on everything you have
            observed so far, produce your FINAL_ANSWER now.

            Summary of your exploration:
            {observations_summary}

            Respond with ONLY:
            FINAL_ANSWER:
            <valid JSON with description, domain, semantic_relationships>
            """),
        })

        response = self._call_llm()
        parsed = self._parse_response(response)

        if parsed["type"] == "final_answer":
            return parsed["result"]

        # Absolute fallback — return a minimal result
        return {
            "description": f"Database table '{self.target_table}'.",
            "domain": "unknown",
            "semantic_relationships": [],
            "concepts": [],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-Table Validation (Satisfaction Checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_enrichments(
    enriched: dict,
    all_tables: list[str],
    client: AzureOpenAI,
    verbose: bool = True,
) -> dict:
    """
    Post-enrichment validation step: review all table enrichments together
    for consistency, completeness, and correctness. The LLM acts as a
    reviewer that can fix issues across the full set.

    This is the "satisfaction checkpoint" — it ensures the enrichment
    output is coherent as a whole, not just per-table.
    """
    if verbose:
        print("\n    [Validation] Cross-table consistency check...")

    enrichment_dump = json.dumps(enriched, indent=2)

    prompt = textwrap.dedent(f"""\
    You are a senior data architect reviewing enrichment metadata for an airlines database.

    Below is the enriched metadata for all tables. Review it for:
    1. Consistency: Are domains aligned? Are relationship types consistent across tables?
       (e.g., if table A says A→B is PART_OF, table B should acknowledge the inverse)
    2. Completeness: Are any obvious semantic relationships missing?
    3. Quality: Are descriptions accurate, specific, and grounded in the airlines/aviation domain?
    4. Concepts: Does each table have 2-5 meaningful abstract concepts? Are concept names
       appropriately abstract (not raw column names)? Are derived_from columns correct?

    Tables: {all_tables}

    Current enrichment:
    {enrichment_dump}

    If the enrichment is good, respond with EXACTLY:
    VALIDATED

    If you want to fix issues, respond with the COMPLETE corrected JSON (same schema,
    all tables included, including the "concepts" array for each table) — no extra text, just the JSON object.
    """)

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    response = completion.choices[0].message.content.strip()

    if "VALIDATED" in response.upper() and len(response) < 50:
        if verbose:
            print("    [Validation] ✓ Enrichments validated — no changes needed.")
        return enriched

    # Try to parse corrected JSON
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        corrected = json.loads(cleaned)
        # Verify it has the right structure
        if isinstance(corrected, dict) and all(t in corrected for t in all_tables):
            if verbose:
                print("    [Validation] ✓ Enrichments corrected by reviewer.")
                # Show what changed
                for table in all_tables:
                    old_desc = enriched.get(table, {}).get("description", "")
                    new_desc = corrected.get(table, {}).get("description", "")
                    if old_desc != new_desc:
                        print(f"      ~ {table}: description updated")
                    old_rels = len(enriched.get(table, {}).get("semantic_relationships", []))
                    new_rels = len(corrected.get(table, {}).get("semantic_relationships", []))
                    if old_rels != new_rels:
                        print(f"      ~ {table}: relationships {old_rels} → {new_rels}")
            return corrected
        else:
            if verbose:
                print("    [Validation] WARN: Corrected JSON missing tables, keeping original.")
            return enriched
    except json.JSONDecodeError:
        if verbose:
            print("    [Validation] WARN: Could not parse corrected JSON, keeping original.")
        return enriched


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — Drop-in replacement for domain_graph.enrich_with_llm
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_with_llm_advanced(
    schema: dict,
    client: AzureOpenAI,
    db_path: str = None,
    verbose: bool = True,
    validate: bool = True,
) -> dict:
    """
    Advanced enrichment using a ReAct agent per table.

    Same input/output contract as domain_graph.enrich_with_llm — can be used
    as a drop-in replacement.

    Args:
        schema:   Introspected schema dict (from introspect_sqlite).
        client:   AzureOpenAI client instance.
        db_path:  Path to SQLite database. If None, uses default from domain_graph.
        verbose:  Print agent reasoning trace.
        validate: Run cross-table validation step after enrichment.

    Returns:
        {
            "table_name": {
                "description": str,
                "domain": str,
                "semantic_relationships": [
                    {"target_table": str, "relationship_type": str, "reason": str}
                ]
            }
        }
    """
    if db_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)  # one level up from domain_graph/
        db_path = os.path.join(project_root, "source_data", "airlines.db")

    all_tables = list(schema.keys())
    tool = SQLDBQueryTool(db_path)
    enriched = {}

    for table_name in all_tables:
        if verbose:
            print(f"\n    ── Agent enriching: {table_name} ──────────────────")

        agent = EnrichmentAgent(
            llm_client=client,
            tool=tool,
            target_table=table_name,
            all_tables=all_tables,
            verbose=verbose,
        )

        result = agent.run()
        enriched[table_name] = result

        if verbose:
            print(f"    Result: {result.get('domain', '?')} — {result.get('description', '')[:80]}")
            rel_count = len(result.get("semantic_relationships", []))
            print(f"    Relationships: {rel_count} semantic edge(s)")
            for rel in result.get("semantic_relationships", []):
                print(f"      → {rel['relationship_type']} → {rel['target_table']}")

    # ── Cross-table validation (satisfaction checkpoint) ────────────────
    if validate:
        enriched = validate_enrichments(enriched, all_tables, client, verbose)

    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Run the advanced enrichment standalone for testing."""
    from domain_graph import get_llm_client, introspect_sqlite, print_enrichment

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)  # one level up from domain_graph/
    db_path = os.path.join(project_root, "source_data", "airlines.db")

    print("=" * 70)
    print("  ADVANCED ENRICHMENT — ReAct Agent")
    print("  Airlines Database Schema Enrichment")
    print("=" * 70)

    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        print("Run 'python source_data/setup_new_db.py' first.")
        exit(1)

    print("\n[1] Introspecting schema...")
    schema = introspect_sqlite(db_path)
    print(f"    Found {len(schema)} table(s): {list(schema.keys())}")

    print("\n[2] Creating LLM client...")
    client = get_llm_client()

    print("\n[3] Running ReAct agent enrichment...")
    enriched = enrich_with_llm_advanced(schema, client, db_path=db_path)

    print("\n[4] Final enrichment results:")
    print_enrichment(enriched)

    # Save trace to file for inspection
    trace_path = os.path.join(project_root, "source_data", "enrichment_trace.json")
    with open(trace_path, "w") as f:
        json.dump(enriched, f, indent=2)
    print(f"\n  Saved enrichment to {trace_path}")
