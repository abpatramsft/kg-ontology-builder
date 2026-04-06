"""
search.py — Vector similarity search + RAG Q&A over Cosmos DB.

HOW VECTOR SEARCH WORKS IN COSMOS DB:
  Cosmos DB for NoSQL provides a built-in SQL function:

    VectorDistance(document.embedding, @queryVector)

  This computes the cosine similarity (or other metric you configured) between
  the stored embedding and your query vector. Combined with ORDER BY and TOP,
  this gives you the K most relevant chunks for any natural-language question.

RAG (Retrieval-Augmented Generation):
  1. Embed the user's question  →  query vector
  2. Vector search in Cosmos DB  →  top-K relevant chunks
  3. Feed chunks + question to GPT-4.1  →  grounded answer

USAGE:
  python search.py "What maintenance was done on the A321neo?"
  python search.py                          # ← interactive mode
"""

import os
import sys
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

from llm import (get_embedding_client, embed_texts,
                 get_llm_client, call_llm)

# ── Cosmos DB config ────────────────────────────────────────────────────────
COSMOS_ENDPOINT = os.environ.get(
    "COSMOS_ENDPOINT",
    "https://cosmosdb-vectors.documents.azure.com:443/"
)
DATABASE_NAME = "vector_demo"
CONTAINER_NAME = "documents"
TOP_K = 5  # number of chunks to retrieve


def get_container():
    """Connect to the Cosmos DB container."""
    credential = DefaultAzureCredential()
    client = CosmosClient(url=COSMOS_ENDPOINT, credential=credential)
    database = client.get_database_client(DATABASE_NAME)
    return database.get_container_client(CONTAINER_NAME)


def vector_search(container, query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """
    Find the top-K most similar chunks using Cosmos DB vector search.

    The query uses:
      - VectorDistance()  : computes cosine similarity between stored & query vectors
      - ORDER BY          : rank by similarity (ascending for cosine = closest first)
      - TOP               : limit results
    """
    # Cosmos DB NoSQL vector search query
    query = """
        SELECT TOP @topK
            c.id, c.source, c.section, c.chunk_idx, c.text,
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


def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Construct a prompt that feeds retrieved chunks as context to the LLM.
    """
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Section: {c['section']}]\n{c['text']}"
        for c in context_chunks
    )

    return f"""You are a helpful assistant answering questions about IndiGo Airlines operations.
Use ONLY the context below to answer. If the context doesn't contain the answer, say so.
Cite the source file and section when possible.

CONTEXT:
{context_block}

QUESTION: {question}

ANSWER:"""


def ask(question: str) -> str:
    """
    End-to-end RAG pipeline:
      question → embed → vector search → LLM answer
    """
    # 1. Embed the question
    emb_client = get_embedding_client()
    query_embedding = embed_texts(emb_client, [question])[0]

    # 2. Vector search in Cosmos DB
    container = get_container()
    results = vector_search(container, query_embedding)

    if not results:
        return "No relevant documents found in the database."

    # 3. Show retrieved chunks (for transparency)
    print(f"\n🔍 Retrieved {len(results)} chunks:")
    for r in results:
        score = r.get("similarity_score", "N/A")
        print(f"   [{r['source']} §{r['section']}] score={score}")
    print()

    # 4. RAG — send context + question to GPT-4.1
    llm_client = get_llm_client()
    prompt = build_rag_prompt(question, results)
    answer = call_llm(llm_client, prompt, temperature=0.2)
    return answer


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question from command line
        q = " ".join(sys.argv[1:])
        print(f"❓ {q}")
        print(f"\n💬 {ask(q)}")
    else:
        # Interactive mode
        print("=== IndiGo Airlines RAG Q&A (Cosmos DB Vector Search) ===")
        print("Type your question and press Enter. Type 'quit' to exit.\n")
        while True:
            q = input("❓ ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            print(f"\n💬 {ask(q)}\n")
