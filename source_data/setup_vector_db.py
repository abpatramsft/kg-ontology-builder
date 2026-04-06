"""
setup_vector_db.py — Provision Cosmos DB vector container and populate it with
embedded document chunks.

This script is run ONCE during setup (after setup_new_db.py). It:
  1. Creates the Cosmos DB database + container with vector index policy
     (idempotent — skips if already exists)
  2. Loads .txt documents from source_data/
  3. Chunks them using the same section-based chunker as lexical_graph.py
  4. Generates embeddings via Azure OpenAI (text-embedding-3-small)
  5. Upserts all chunks with embeddings into the Cosmos DB container

Usage:
    python source_data/setup_vector_db.py
"""

import os
import sys

import requests
from azure.identity import DefaultAzureCredential

# ── Ensure src/ is importable ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.llm import get_embedding_client, embed_texts
from utils.cosmos_vector_helpers import (
    COSMOS_VECTOR_ENDPOINT,
    COSMOS_VECTOR_DATABASE,
    COSMOS_VECTOR_CONTAINER,
    get_vector_container,
    upsert_chunks,
)
from src.lexical_graph.lexical_graph import load_documents, chunk_document

# ── ARM provisioning config ─────────────────────────────────────────────────
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "ea5dfd63-50a1-4e2f-8144-5f6c7088c0c7")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "rg-abpatra-7946")
ACCOUNT_NAME = os.environ.get("COSMOS_VECTOR_ACCOUNT", "cosmosdb-vectors")
ARM_API_VERSION = "2024-05-15"
ARM_BASE = (
    f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    f"/providers/Microsoft.DocumentDB/databaseAccounts/{ACCOUNT_NAME}"
)


# ═════════════════════════════════════════════════════════════════════════════
#  Step 1 — Provision Cosmos DB database + container (idempotent)
# ═════════════════════════════════════════════════════════════════════════════

def _get_arm_token() -> str:
    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default")
    return token.token


def provision_database(token: str):
    """Create the SQL database via ARM REST API (no-op if exists)."""
    url = f"{ARM_BASE}/sqlDatabases/{COSMOS_VECTOR_DATABASE}?api-version={ARM_API_VERSION}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "location": "East US",
        "properties": {"resource": {"id": COSMOS_VECTOR_DATABASE}},
    }
    print(f"  Creating database '{COSMOS_VECTOR_DATABASE}'...")
    resp = requests.put(url, json=body, headers=headers)
    if resp.status_code in (200, 201, 202):
        print(f"  OK — database '{COSMOS_VECTOR_DATABASE}' ready")
    else:
        print(f"  FAILED ({resp.status_code}): {resp.text}")
        resp.raise_for_status()


def provision_container(token: str):
    """Create the container with vector embedding policy and index (no-op if exists)."""
    url = (
        f"{ARM_BASE}/sqlDatabases/{COSMOS_VECTOR_DATABASE}"
        f"/containers/{COSMOS_VECTOR_CONTAINER}?api-version={ARM_API_VERSION}"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "location": "East US",
        "properties": {
            "resource": {
                "id": COSMOS_VECTOR_CONTAINER,
                "partitionKey": {
                    "paths": ["/doc_name"],
                    "kind": "Hash",
                    "version": 2,
                },
                "indexingPolicy": {
                    "indexingMode": "consistent",
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}],
                    "excludedPaths": [
                        {"path": "/embedding/*"},
                        {"path": "/\"_etag\"/?"},
                    ],
                    "vectorIndexes": [
                        {"path": "/embedding", "type": "quantizedFlat"},
                    ],
                },
                "vectorEmbeddingPolicy": {
                    "vectorEmbeddings": [
                        {
                            "path": "/embedding",
                            "dataType": "float32",
                            "distanceFunction": "cosine",
                            "dimensions": 1536,
                        }
                    ]
                },
            }
        },
    }
    print(f"  Creating container '{COSMOS_VECTOR_CONTAINER}' with vector index...")
    resp = requests.put(url, json=body, headers=headers)
    if resp.status_code in (200, 201, 202):
        print(f"  OK — container '{COSMOS_VECTOR_CONTAINER}' ready")
        print(f"       Partition key : /doc_name")
        print(f"       Vector index  : quantizedFlat on /embedding (1536 dims, cosine)")
    else:
        print(f"  FAILED ({resp.status_code}): {resp.text}")
        resp.raise_for_status()


def provision_cosmos_db():
    """Provision database and container via ARM (idempotent)."""
    print("\n[Step 1] Provisioning Cosmos DB vector store...")
    token = _get_arm_token()
    provision_database(token)
    provision_container(token)


# ═════════════════════════════════════════════════════════════════════════════
#  Step 2-4 — Load, chunk, embed, upsert
# ═════════════════════════════════════════════════════════════════════════════

def ingest_documents():
    """Load documents, chunk, embed, and upsert into Cosmos DB."""

    # Step 2: Load documents
    data_dir = os.path.join(PROJECT_ROOT, "source_data")
    print(f"\n[Step 2] Loading .txt documents from {data_dir}...")
    documents = load_documents(data_dir)
    if not documents:
        print("  ERROR: No .txt files found.")
        return
    for doc in documents:
        print(f"  {doc['name']} ({len(doc['content'])} chars)")

    # Step 3: Chunk documents (same section-based chunker as lexical_graph.py)
    print("\n[Step 3] Chunking documents (section-based)...")
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  {doc['name']} → {len(doc_chunks)} chunks")
    print(f"  Total: {len(all_chunks)} chunks")

    # Step 4: Generate embeddings
    print("\n[Step 4] Generating embeddings (text-embedding-3-small, 1536 dims)...")
    embedding_client = get_embedding_client()
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(embedding_client, texts)
    print(f"  {len(embeddings)} embeddings generated")

    # Build Cosmos DB documents
    records = []
    for chunk, emb in zip(all_chunks, embeddings):
        records.append({
            "id": chunk["chunk_id"],           # Cosmos DB required `id` field
            "chunk_id": chunk["chunk_id"],
            "doc_name": chunk["doc_name"],
            "chunk_index": chunk["index"],
            "text": chunk["text"],
            "text_preview": chunk["text"][:200],
            "char_count": len(chunk["text"]),
            "embedding": emb,
        })

    # Step 5: Upsert into Cosmos DB
    print(f"\n[Step 5] Upserting {len(records)} chunks into Cosmos DB "
          f"({COSMOS_VECTOR_DATABASE}/{COSMOS_VECTOR_CONTAINER})...")
    container = get_vector_container()
    upsert_chunks(container, records)
    print(f"\n  Done! {len(records)} chunks stored in Cosmos DB vector store.")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  VECTOR DB SETUP — Cosmos DB NoSQL with Vector Search")
    print(f"  Endpoint : {COSMOS_VECTOR_ENDPOINT}")
    print(f"  Database : {COSMOS_VECTOR_DATABASE}")
    print(f"  Container: {COSMOS_VECTOR_CONTAINER}")
    print("=" * 70)

    provision_cosmos_db()
    ingest_documents()

    print("\n" + "=" * 70)
    print("  Vector DB setup complete!")
    print("  Chunks are indexed and ready for vector search.")
    print("=" * 70)


if __name__ == "__main__":
    main()
