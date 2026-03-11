"""
llm.py — Shared LLM & Embedding clients (Azure OpenAI)

All layers import from here to avoid duplicating client setup.
"""

import json
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


# ─── Constants (update these for your environment) ─────────────────────────
LLM_ENDPOINT = "https://abpatra-7946-resource.openai.azure.com/"
EMBEDDING_ENDPOINT = "https://abpatra-7946-resource.cognitiveservices.azure.com/"
API_VERSION = "2024-02-15-preview"
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 16


# ─── LLM (Chat Completions) ────────────────────────────────────────────────

def get_llm_client() -> AzureOpenAI:
    """Create an Azure OpenAI client for chat completions."""
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, TOKEN_SCOPE)
    return AzureOpenAI(
        azure_endpoint=LLM_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION,
    )


def call_llm(client: AzureOpenAI, prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to GPT-4.1 and return the response text."""
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return completion.choices[0].message.content


def parse_llm_json(response: str) -> dict | list | None:
    """
    Parse JSON from LLM response, handling markdown fences and quirks.
    Returns parsed object or None on failure.
    """
    cleaned = response.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]  # remove first line
        cleaned = cleaned.rsplit("```", 1)[0]  # remove last fence
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON array or object within the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None


# ─── Embeddings ─────────────────────────────────────────────────────────────

def get_embedding_client() -> AzureOpenAI:
    """Create an Azure OpenAI client for embeddings."""
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, TOKEN_SCOPE)
    return AzureOpenAI(
        azure_endpoint=EMBEDDING_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION,
    )


def embed_texts(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using text-embedding-3-small.
    Batches in groups of 16 to stay within token limits.
    """
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings
