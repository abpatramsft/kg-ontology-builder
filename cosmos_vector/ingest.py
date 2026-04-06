"""
ingest.py — Generate embeddings for document chunks and upload to Cosmos DB.

PIPELINE OVERVIEW:
  1. chunker.py   →  load .txt files → list of chunk dicts
  2. llm.py       →  embed each chunk's text → 1536-dim vector
  3. cosmos DB    →  upsert each chunk (with its embedding) into the container

WHAT GETS STORED IN COSMOS DB (one document per chunk):
  {
      "id":        "a1b2c3d4e5f6g7h8",
      "source":    "maintenance_incidents.txt",
      "section":   "Fleet Maintenance Overview",
      "chunk_idx": 0,
      "text":      "A total of 20 maintenance events ...",
      "embedding": [0.012, -0.034, ...]   ← 1536 floats
  }

USAGE:
  python ingest.py
"""

import os
import time
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

from chunker import load_and_chunk
from llm import get_embedding_client, embed_texts

# ── Cosmos DB config (same as cosmos_setup.py) ──────────────────────────────
COSMOS_ENDPOINT = os.environ.get(
    "COSMOS_ENDPOINT",
    "https://cosmosdb-vectors.documents.azure.com:443/"
)
DATABASE_NAME = "vector_demo"
CONTAINER_NAME = "documents"


def get_container():
    """Connect to the existing Cosmos DB container."""
    credential = DefaultAzureCredential()
    client = CosmosClient(url=COSMOS_ENDPOINT, credential=credential)
    database = client.get_database_client(DATABASE_NAME)
    return database.get_container_client(CONTAINER_NAME)


def ingest():
    """Full ingestion pipeline: chunk → embed → upsert."""

    # ── Step 1: Chunk the documents ─────────────────────────────────────────
    print("📄 Loading and chunking documents...")
    chunks = load_and_chunk()
    print(f"   {len(chunks)} chunks generated from {len(set(c['source'] for c in chunks))} files\n")

    # ── Step 2: Generate embeddings ─────────────────────────────────────────
    print("🔢 Generating embeddings (text-embedding-3-small, 1536 dims)...")
    emb_client = get_embedding_client()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(emb_client, texts)
    print(f"   {len(embeddings)} embeddings generated\n")

    # Attach embedding vectors to chunk dicts
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    # ── Step 3: Upsert into Cosmos DB ───────────────────────────────────────
    print("☁️  Upserting into Cosmos DB...")
    container = get_container()

    for i, chunk in enumerate(chunks):
        container.upsert_item(chunk)
        # Progress indicator
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"   {i + 1}/{len(chunks)} documents upserted")

    print(f"\n✅ Ingestion complete! {len(chunks)} chunks stored in "
          f"Cosmos DB ({DATABASE_NAME}/{CONTAINER_NAME})")


if __name__ == "__main__":
    ingest()
