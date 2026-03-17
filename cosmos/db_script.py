"""
Knowledge Graph with Azure Cosmos DB Gremlin API
=================================================
Demonstrates how to:
  1. Connect to Cosmos DB Gremlin endpoint
  2. Add vertices (entities) and edges (relationships)
  3. Query the graph using Gremlin traversals
"""

import os
import sys

from dotenv import load_dotenv
from gremlin_python.driver import client as gremlin_client, serializer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (loaded from .env)
# ---------------------------------------------------------------------------
ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
KEY = os.getenv("COSMOS_DB_KEY")
DATABASE = os.getenv("COSMOS_DB_DATABASE")
GRAPH = os.getenv("COSMOS_DB_GRAPH")

if not all([ENDPOINT, KEY, DATABASE, GRAPH]):
    sys.exit("ERROR: Missing one or more env vars. Check your .env file.")


def _create_client() -> gremlin_client.Client:
    """Return an authenticated Gremlin client for Cosmos DB."""
    return gremlin_client.Client(
        url=f"wss://{ENDPOINT}:443/",
        traversal_source="g",
        username=f"/dbs/{DATABASE}/colls/{GRAPH}",
        password=KEY,
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )


def run_query(client: gremlin_client.Client, query: str):
    """Submit a Gremlin query and return the result set."""
    print(f"  > {query}")
    callback = client.submitAsync(query)
    result = callback.result()
    return result.all().result()


# ---------------------------------------------------------------------------
# Sample: build a small knowledge graph
# ---------------------------------------------------------------------------
# Vertices represent concepts; edges represent relationships.
# Partition key property = "category"
CLEANUP_QUERY = "g.V().drop()"

VERTICES = [
    "g.addV('language').property('id', 'python').property('category', 'programming').property('name', 'Python')",
    "g.addV('language').property('id', 'javascript').property('category', 'programming').property('name', 'JavaScript')",
    "g.addV('framework').property('id', 'django').property('category', 'programming').property('name', 'Django')",
    "g.addV('framework').property('id', 'react').property('category', 'programming').property('name', 'React')",
    "g.addV('database').property('id', 'cosmosdb').property('category', 'data').property('name', 'Azure Cosmos DB')",
    "g.addV('database').property('id', 'neo4j').property('category', 'data').property('name', 'Neo4j')",
    "g.addV('concept').property('id', 'graph-theory').property('category', 'cs').property('name', 'Graph Theory')",
]

EDGES = [
    "g.V('python').addE('has_framework').to(g.V('django'))",
    "g.V('javascript').addE('has_framework').to(g.V('react'))",
    "g.V('django').addE('can_use').to(g.V('cosmosdb'))",
    "g.V('react').addE('can_use').to(g.V('cosmosdb'))",
    "g.V('cosmosdb').addE('implements').to(g.V('graph-theory'))",
    "g.V('neo4j').addE('implements').to(g.V('graph-theory'))",
]

# Sample read queries
READ_QUERIES = [
    # All vertices
    ("All vertices", "g.V().valueMap(true)"),
    # All edges
    ("All edges", "g.E()"),
    # Frameworks reachable from Python
    ("Frameworks for Python", "g.V('python').out('has_framework').values('name')"),
    # Databases that implement Graph Theory
    ("Databases implementing Graph Theory", "g.V('graph-theory').in('implements').values('name')"),
    # Full path: Python -> framework -> database
    ("Python -> framework -> database path", "g.V('python').out('has_framework').out('can_use').values('name')"),
]


def main():
    print("Connecting to Cosmos DB Gremlin endpoint…")
    gremlin = _create_client()

    try:
        # 1. Clean up any previous data
        print("\n--- Cleaning up existing data ---")
        run_query(gremlin, CLEANUP_QUERY)

        # 2. Insert vertices
        print("\n--- Adding vertices ---")
        for v in VERTICES:
            run_query(gremlin, v)

        # 3. Insert edges
        print("\n--- Adding edges ---")
        for e in EDGES:
            run_query(gremlin, e)

        # 4. Run read queries
        print("\n--- Querying the knowledge graph ---")
        for label, q in READ_QUERIES:
            results = run_query(gremlin, q)
            print(f"  Result ({label}): {results}\n")

        print("Done! Your knowledge graph is stored in Cosmos DB.")

    finally:
        gremlin.close()


if __name__ == "__main__":
    main()
