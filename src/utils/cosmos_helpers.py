"""
cosmos_helpers.py — Shared Azure Cosmos DB Gremlin connection & query helpers.

Replaces neo4j_helpers.py — all graph operations now target the Cosmos DB
Gremlin API (database: indigokg, container: knowledgegraph).
"""

import os
import re

from dotenv import load_dotenv
from gremlin_python.driver import client as gremlin_client, serializer

load_dotenv()

COSMOS_ENDPOINT = os.getenv(
    "COSMOS_DB_ENDPOINT",
    "cosmosdb-gremlin-abpatra.gremlin.cosmos.azure.com",
)
COSMOS_KEY = os.getenv("COSMOS_DB_KEY")
COSMOS_DATABASE = "indigokg"
COSMOS_GRAPH = "knowledgegraph"


def get_gremlin_client() -> gremlin_client.Client:
    """Create and return an authenticated Gremlin client for Cosmos DB."""
    if not COSMOS_KEY:
        raise RuntimeError(
            "COSMOS_DB_KEY environment variable not set. "
            "Add it to your project .env file: COSMOS_DB_KEY=<your-key>"
        )
    return gremlin_client.Client(
        url=f"wss://{COSMOS_ENDPOINT}:443/",
        traversal_source="g",
        username=f"/dbs/{COSMOS_DATABASE}/colls/{COSMOS_GRAPH}",
        password=COSMOS_KEY,
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )


def run_gremlin(client: gremlin_client.Client, query: str) -> list:
    """Submit a Gremlin query and return all results as a Python list."""
    callback = client.submitAsync(query)
    result = callback.result()
    return result.all().result()


def run_gremlin_write(client: gremlin_client.Client, query: str) -> list:
    """Submit a Gremlin write query (alias for run_gremlin; kept for semantic clarity)."""
    return run_gremlin(client, query)


def esc(value) -> str:
    """
    Escape a value for safe embedding inside a Gremlin single-quoted string.
    Cosmos DB Gremlin does not support query parameters, so all values must be
    inlined — this function prevents injection via backslash and quote escaping.
    """
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def make_vertex_id(label: str, name: str) -> str:
    """
    Create a deterministic, stable vertex ID from label + name.
    Stable IDs enable idempotent upsert writes and efficient lookups by g.V('id').
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(name))[:80]
    return f"{label.lower()}__{safe_name}"


def gval(result_map: dict, key: str, default=None):
    """
    Extract a single value from a Gremlin valueMap() result.

    Cosmos DB Gremlin's valueMap() wraps every property value in a list
    (e.g., {'name': ['passengers'], 'domain': ['operations']}).
    The special keys 'id' and 'label' come through unwrapped.
    This helper unwraps list values and returns the first element.
    """
    val = result_map.get(key)
    if val is None:
        return default
    if isinstance(val, list):
        return val[0] if val else default
    return val
