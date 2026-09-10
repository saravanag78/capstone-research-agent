"""Thin wrapper around the Voyage AI embeddings API.

Anthropic does not serve embeddings directly and recommends Voyage AI as
its embeddings partner for retrieval use cases. Voyage's `input_type`
parameter asymmetrically optimizes embeddings for documents vs. queries,
which improves retrieval quality over embedding both sides the same way.
"""

import os

import voyageai

DEFAULT_MODEL = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-3.5")


def embed_documents(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float]]:
    client = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
    result = client.embed(texts, model=model, input_type="document")
    return result.embeddings


def embed_query(text: str, model: str = DEFAULT_MODEL) -> list[float]:
    client = voyageai.Client()
    result = client.embed([text], model=model, input_type="query")
    return result.embeddings[0]
