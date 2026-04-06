"""
cosmos_vector_helpers.py — Centralized Cosmos DB NoSQL vector search operations.

Provides all vector database operations (search, retrieve, upsert) used by
the agents. Replaces the previous LanceDB-based vector store with Cosmos DB
NoSQL + VectorDistance.

Configuration (env vars with defaults):
  COSMOS_VECTOR_ENDPOINT   — Cosmos DB NoSQL endpoint
  COSMOS_VECTOR_DATABASE   — database name
  COSMOS_VECTOR_CONTAINER  — container name
"""

import os

from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

# ── Configuration ────────────────────────────────────────────────────────────
COSMOS_VECTOR_ENDPOINT = os.environ.get(
    "COSMOS_VECTOR_ENDPOINT",
    "https://cosmosdb-vectors.documents.azure.com:443/",
)
COSMOS_VECTOR_DATABASE = os.environ.get("COSMOS_VECTOR_DATABASE", "lexical_vector_db")
COSMOS_VECTOR_CONTAINER = os.environ.get("COSMOS_VECTOR_CONTAINER", "lexical_chunks")

# Module-level singleton
_container_client = None


def get_vector_container():
    """Return a cached Cosmos DB container client for the vector store."""
    global _container_client
    if _container_client is None:
        credential = DefaultAzureCredential()
        client = CosmosClient(url=COSMOS_VECTOR_ENDPOINT, credential=credential)
        database = client.get_database_client(COSMOS_VECTOR_DATABASE)
        _container_client = database.get_container_client(COSMOS_VECTOR_CONTAINER)
    return _container_client


# ═════════════════════════════════════════════════════════════════════════════
#  Vector Search
# ═════════════════════════════════════════════════════════════════════════════

def vector_search(container, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Top-K vector similarity search using Cosmos DB VectorDistance.

    Returns list of dicts with keys:
        chunk_id, doc_name, chunk_index, text, text_preview, char_count,
        similarity_score
    """
    query = """
        SELECT TOP @topK
            c.chunk_id, c.doc_name, c.chunk_index, c.text,
            c.text_preview, c.char_count,
            VectorDistance(c.embedding, @queryVector) AS similarity_score
        FROM c
        ORDER BY VectorDistance(c.embedding, @queryVector)
    """
    parameters = [
        {"name": "@topK", "value": top_k},
        {"name": "@queryVector", "value": query_embedding},
    ]
    results = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))
    return results


def vector_search_filtered(
    container,
    query_embedding: list[float],
    doc_name: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Top-K vector similarity search filtered to a specific document.
    """
    query = """
        SELECT TOP @topK
            c.chunk_id, c.doc_name, c.chunk_index, c.text,
            c.text_preview, c.char_count,
            VectorDistance(c.embedding, @queryVector) AS similarity_score
        FROM c
        WHERE c.doc_name = @docName
        ORDER BY VectorDistance(c.embedding, @queryVector)
    """
    parameters = [
        {"name": "@topK", "value": top_k},
        {"name": "@queryVector", "value": query_embedding},
        {"name": "@docName", "value": doc_name},
    ]
    results = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  Single-item retrieval
# ═════════════════════════════════════════════════════════════════════════════

def get_chunk(container, chunk_id: str) -> dict | None:
    """Retrieve a single chunk by chunk_id."""
    query = """
        SELECT c.chunk_id, c.doc_name, c.chunk_index, c.text,
               c.text_preview, c.char_count
        FROM c
        WHERE c.chunk_id = @chunkId
    """
    parameters = [{"name": "@chunkId", "value": chunk_id}]
    results = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))
    return results[0] if results else None


def get_chunks_by_doc(container, doc_name: str) -> list[dict]:
    """Retrieve all chunks for a given document, ordered by chunk_index."""
    query = """
        SELECT c.chunk_id, c.doc_name, c.chunk_index, c.text,
               c.text_preview, c.char_count
        FROM c
        WHERE c.doc_name = @docName
        ORDER BY c.chunk_index
    """
    parameters = [{"name": "@docName", "value": doc_name}]
    return list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))


def list_documents(container) -> list[str]:
    """Return distinct doc_name values."""
    query = "SELECT DISTINCT VALUE c.doc_name FROM c"
    return sorted(list(container.query_items(
        query=query,
        parameters=[],
        enable_cross_partition_query=True,
    )))


def get_collection_stats(container) -> dict:
    """Return total chunk count and document count."""
    count_query = "SELECT VALUE COUNT(1) FROM c"
    total = list(container.query_items(
        query=count_query,
        parameters=[],
        enable_cross_partition_query=True,
    ))
    total_chunks = total[0] if total else 0

    doc_names = list_documents(container)
    return {
        "total_chunks": total_chunks,
        "documents": len(doc_names),
        "document_names": doc_names,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Write operations (used by setup script)
# ═════════════════════════════════════════════════════════════════════════════

def upsert_chunks(container, chunks: list[dict]):
    """
    Upsert a batch of chunk documents into the Cosmos DB container.

    Each chunk dict must include at minimum:
        id, chunk_id, doc_name, chunk_index, text, text_preview, char_count, embedding
    """
    for i, chunk in enumerate(chunks):
        container.upsert_item(chunk)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  {i + 1}/{len(chunks)} chunks upserted")
