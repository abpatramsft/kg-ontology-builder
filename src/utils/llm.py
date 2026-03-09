"""
llm.py — Shared LLM & Embedding clients (Azure OpenAI)

All layers import from here to avoid duplicating client setup.
"""

import json
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


# ─── LLM (Chat Completions) ────────────────────────────────────────────────

def get_llm_client() -> AzureOpenAI:
    """Create an Azure OpenAI client for chat completions (GPT-4.1)."""
    endpoint = "https://abpatra-7946-resource.openai.azure.com/"
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-02-15-preview",
    )


def call_llm(client: AzureOpenAI, prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to GPT-4.1 and return the response text."""
    completion = client.chat.completions.create(
        model="gpt-4.1",
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
    """Create an Azure OpenAI client for embeddings (text-embedding-3-small)."""
    endpoint = "https://abpatra-7946-resource.cognitiveservices.azure.com/"
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-02-15-preview",
    )


EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using text-embedding-3-small.
    Batches in groups of 16 to stay within token limits.
    """
    all_embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings
