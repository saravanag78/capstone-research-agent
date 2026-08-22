"""Loads sample source documents from the data/ directory.

Week 1 scope: documents are plain .txt files organized into subfolders by
source type (agendas, meeting_notes, pitch_books, prior_notes). Later weeks
replace this flat loader with retrieval (RAG) over a vector database.
"""

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

SOURCE_TYPES = ["agendas", "meeting_notes", "pitch_books", "prior_notes"]


@dataclass
class Document:
    source_type: str
    filename: str
    content: str


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents = []
    for source_type in SOURCE_TYPES:
        folder = data_dir / source_type
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8").strip()
            documents.append(
                Document(source_type=source_type, filename=file_path.name, content=content)
            )
    return documents
