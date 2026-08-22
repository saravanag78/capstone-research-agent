"""Builds the LLM prompt from the template and loaded documents."""

from pathlib import Path

from document_loader import Document

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "draft_note_prompt.txt"

SOURCE_TYPE_LABELS = {
    "agendas": "Meeting Agenda",
    "meeting_notes": "Meeting Notes",
    "pitch_books": "Pitch Book",
    "prior_notes": "Prior Research Note",
}


def format_source_content(documents: list[Document]) -> str:
    blocks = []
    for doc in documents:
        label = SOURCE_TYPE_LABELS.get(doc.source_type, doc.source_type)
        blocks.append(f"--- {label}: {doc.filename} ---\n{doc.content}")
    return "\n\n".join(blocks)


def build_prompt(company: str, documents: list[Document]) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    source_content = format_source_content(documents)
    return template.format(company=company, source_content=source_content)
