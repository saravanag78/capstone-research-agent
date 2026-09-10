"""A minimal in-memory vector database: cosine-similarity search over chunk embeddings."""

from dataclasses import dataclass

import numpy as np

from chunker import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """Holds chunk embeddings in memory and answers top-k similarity queries."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        matrix = np.array(vectors, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        self._chunks.extend(chunks)
        self._vectors = matrix if self._vectors is None else np.vstack([self._vectors, matrix])

    def search(self, query_vector: list[float], top_k: int = 5) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            return []
        query = np.array(query_vector, dtype=np.float32)
        query /= np.linalg.norm(query)
        scores = self._vectors @ query  # both sides normalized, so dot product is cosine similarity
        top_indices = np.argsort(-scores)[:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices]
