"""Tool abstraction layer for the ReAct agent.

Each Tool pairs the JSON schema the LLM needs to call it correctly with a
plain Python callable that performs the action. build_tools() wires up the
concrete tools against the documents loaded for a single agent run.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from document_loader import load_documents

OUTPUT_DIR = Path(__file__).parent / "output"

SOURCE_TYPE_LABELS = {
    "agendas": "Meeting Agenda",
    "meeting_notes": "Meeting Notes",
    "pitch_books": "Pitch Book",
    "prior_notes": "Prior Research Note",
}


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., str]

    def to_anthropic_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, **kwargs) -> str:
        return self.handler(**kwargs)


def build_tools(company: str) -> list[Tool]:
    """Construct the toolset for one agent run, closing over its documents."""
    documents = load_documents()
    by_filename = {doc.filename: doc for doc in documents}

    def list_documents(**_) -> str:
        if not documents:
            return "No source documents are available."
        return "\n".join(
            f"- {doc.filename} [{SOURCE_TYPE_LABELS.get(doc.source_type, doc.source_type)}]"
            for doc in documents
        )

    def read_document(filename: str) -> str:
        doc = by_filename.get(filename)
        if doc is None:
            return f"No document named '{filename}'. Use list_documents to see available files."
        return doc.content

    def search_documents(query: str) -> str:
        query_lower = query.lower()
        matches = [
            f"{doc.filename}: {line.strip()}"
            for doc in documents
            for line in doc.content.splitlines()
            if query_lower in line.lower()
        ]
        return "\n".join(matches[:20]) if matches else f"No matches for '{query}'."

    def write_draft_note(content: str) -> str:
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path = OUTPUT_DIR / f"draft_note_{company.replace(' ', '_')}_{date.today().isoformat()}.md"
        output_path.write_text(content, encoding="utf-8")
        return f"Draft note written to {output_path}"

    return [
        Tool(
            name="list_documents",
            description="List all available source documents with their type and filename.",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=list_documents,
        ),
        Tool(
            name="read_document",
            description="Read the full text content of one source document by filename.",
            input_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Exact filename, as returned by list_documents."}
                },
                "required": ["filename"],
            },
            handler=read_document,
        ),
        Tool(
            name="search_documents",
            description=(
                "Search across all source documents for a keyword or phrase; "
                "returns matching lines with their source filename."
            ),
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keyword or phrase to search for."}},
                "required": ["query"],
            },
            handler=search_documents,
        ),
        Tool(
            name="write_draft_note",
            description=(
                "Write the final draft research note to disk. Call this exactly once, "
                "as the last step, with the complete Markdown note as `content`."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Full Markdown text of the finished draft research note."}
                },
                "required": ["content"],
            },
            handler=write_draft_note,
        ),
    ]
