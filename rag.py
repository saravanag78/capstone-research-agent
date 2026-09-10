"""Week 3: RAG pipeline — chunk documents, embed them, and index them for
semantic top-k retrieval. Ties together chunker.py, embeddings.py, and
vector_store.py behind two calls: build the index once per agent run, then
retrieve against it as many times as the agent needs.
"""

from chunker import chunk_documents
from document_loader import Document
from embeddings import embed_documents, embed_query
from vector_store import ScoredChunk, VectorStore


def build_index(documents: list[Document]) -> VectorStore:
    chunks = chunk_documents(documents)
    store = VectorStore()
    if not chunks:
        return store
    vectors = embed_documents([chunk.text for chunk in chunks])
    store.add(chunks, vectors)
    return store


def retrieve(store: VectorStore, query: str, top_k: int = 5) -> list[ScoredChunk]:
    query_vector = embed_query(query)
    return store.search(query_vector, top_k=top_k)
