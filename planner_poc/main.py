"""
Planner Agent — Explores the Cosmos DB knowledge graph ontology and produces
structured plans of action for IndiGo Airlines data analysis.

Uses Microsoft Agent Framework with Azure AI Foundry.
Ready for deployment to Foundry Hosted Agent service.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import textwrap
from typing import Annotated

from dotenv import load_dotenv

load_dotenv(override=True)

from agent_framework import Agent
from agent_framework.azure import AzureAIAgentClient
from azure.ai.agentserver.agentframework import from_agent_framework
from azure.identity.aio import DefaultAzureCredential

from cosmos_helpers import get_gremlin_client, run_gremlin, esc, make_vertex_id, gval

# ── Configuration ───────────────────────────────────────────────────────────
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Gremlin singleton & thread pool ────────────────────────────────────────
_gremlin_client = None
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _ensure_gremlin():
    """Lazy-init the Gremlin client (sync — call from thread pool)."""
    global _gremlin_client
    if _gremlin_client is None:
        logger.info("Connecting to Cosmos DB Gremlin …")
        _gremlin_client = get_gremlin_client()
        run_gremlin(_gremlin_client, "g.V().limit(1).count()")
        logger.info("Cosmos DB Gremlin connected (indigokg/knowledgegraph)")
    return _gremlin_client


# ═══════════════════════════════════════════════════════════════════════════════
#  GraphOntologyTool — read-only Cosmos DB knowledge graph explorer
# ═══════════════════════════════════════════════════════════════════════════════

class GraphOntologyTool:
    """
    Read-only tool to explore the unified 3-layer knowledge graph in Cosmos DB
    (Gremlin API, database: indigokg, container: knowledgegraph).

    Supported actions:
      list_node_labels, list_relationship_types, list_domain_entities,
      list_subjects, list_documents, get_domain_entity_detail,
      get_subject_context, get_correspondences, find_path, query_graph
    """

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
#  Function tool: explore_ontology
# ═══════════════════════════════════════════════════════════════════════════════

_ontology_tool = None


def _run_explore_sync(action, name, from_name, to_name, gremlin) -> str:
    """Synchronous helper — runs in thread pool to avoid event-loop conflicts."""
    global _ontology_tool
    client = _ensure_gremlin()
    if _ontology_tool is None:
        _ontology_tool = GraphOntologyTool(client)
    action_input = {"action": action}
    if name:
        action_input["name"] = name
    if from_name:
        action_input["from_name"] = from_name
    if to_name:
        action_input["to_name"] = to_name
    if gremlin:
        action_input["gremlin"] = gremlin
    return _ontology_tool.execute(action_input)


async def explore_ontology(
    action: Annotated[str, "The ontology action to perform. One of: "
        "list_node_labels, list_relationship_types, list_domain_entities, "
        "list_subjects, list_documents, get_domain_entity_detail, "
        "get_subject_context, get_correspondences, find_path, query_graph"],
    name: Annotated[str, "Entity or subject name (for get_domain_entity_detail, "
        "get_subject_context, get_correspondences)"] = "",
    from_name: Annotated[str, "Source vertex name (for find_path)"] = "",
    to_name: Annotated[str, "Target vertex name (for find_path)"] = "",
    gremlin: Annotated[str, "Read-only Gremlin traversal (for query_graph)"] = "",
) -> str:
    """Explore the Cosmos DB knowledge graph ontology.

    This tool gives you read-only access to the unified 3-layer knowledge graph
    (Domain Graph + Lexical Graph + Subject Graph) stored in Cosmos DB Gremlin API.

    Graph model (database 'indigokg', container 'knowledgegraph'):
    - DomainEntity vertices (structured DB tables) connected by HAS_FK/SEMANTIC edges
    - Concept vertices connected to DomainEntity via HAS_CONCEPT; cross-linked via RELATED_CONCEPT
    - Document vertices -> Subject vertices (via MENTIONS edges)
    - Subject vertices -> Object vertices (via RELATES_TO {predicate} edges)
    - Subject -> DomainEntity (via CORRESPONDS_TO) bridges unstructured and structured data

    Use this tool to understand what data sources exist and how they connect
    before producing a plan of action.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, _run_explore_sync, action, name, from_name, to_name, gremlin,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  System instructions
# ═══════════════════════════════════════════════════════════════════════════════

PLANNER_INSTRUCTIONS = textwrap.dedent("""\
You are a planning agent for IndiGo Airlines data analysis. Your job is
to explore the knowledge graph ontology and produce a detailed PLAN OF
ACTION — you do NOT execute data queries yourself.

You have ONE tool: explore_ontology — use it to navigate the Cosmos DB
knowledge graph and understand the data landscape.

Your workflow:
  1. Explore the ontology (list entities, subjects, documents, relationships)
  2. Drill into relevant entities and subjects for detail
  3. Produce a structured plan of action

Your plan must include:

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
names, document names, subject names, and relationship types you discovered.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent creation & HTTP server entry point
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Main function to run the agent as a web server."""
    # Pre-initialize the Gremlin connection in a thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _ensure_gremlin)

    async with (
        DefaultAzureCredential() as credential,
        AzureAIAgentClient(
            project_endpoint=PROJECT_ENDPOINT,
            model_deployment_name=MODEL_DEPLOYMENT_NAME,
            credential=credential,
        ) as client,
    ):
        agent = Agent(
            client,
            name="PlannerAgent",
            instructions=PLANNER_INSTRUCTIONS,
            tools=[explore_ontology],
        )

        print("Planner Agent Server running on http://localhost:8088")
        server = from_agent_framework(agent)
        await server.run_async()


if __name__ == "__main__":
    asyncio.run(main())
