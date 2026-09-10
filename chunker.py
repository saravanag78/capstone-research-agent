"""Splits loaded documents into overlapping word-window chunks for embedding.

Word-based sliding windows are simpler and more predictable than character
windows (they never cut a word in half) while still working uniformly
across the short, unstructured .txt sources in data/.
"""

from dataclasses import dataclass

from document_loader import Document

CHUNK_SIZE_WORDS = 60
CHUNK_OVERLAP_WORDS = 15


@dataclass
class Chunk:
    source_type: str
    filename: str
    chunk_index: int
    text: str


def chunk_document(document: Document) -> list[Chunk]:
    words = document.content.split()
    if not words:
        return []

    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    chunks = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + CHUNK_SIZE_WORDS]
        chunks.append(
            Chunk(source_type=document.source_type, filename=document.filename, chunk_index=index, text=" ".join(window))
        )
        if start + CHUNK_SIZE_WORDS >= len(words):
            break
        start += step
        index += 1
    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document)]
