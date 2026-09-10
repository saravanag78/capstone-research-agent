"""Tool abstraction layer for the ReAct agent.

Each Tool pairs the JSON schema the LLM needs to call it correctly with a
plain Python callable that performs the action. build_tools() wires up the
concrete tools against the documents loaded for a single agent run.
Tool.run() catches any exception a handler raises (a missing document, a
failed embeddings call, a bad argument) and turns it into an observation
the agent can read and recover from, instead of crashing the whole run —
see README's "Week 6" notes.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from document_loader import load_documents
from memory import LongTermMemory
from rag import build_index, retrieve

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
        try:
            return self.handler(**kwargs)
        except Exception as exc:
            return f"Tool '{self.name}' failed: {exc}. Try a different argument or a different tool."


def build_tools(company: str, long_term_memory: LongTermMemory, retrieved_filenames: set[str]) -> list[Tool]:
    """Construct the toolset for one agent run, closing over its documents and memory store.

    retrieved_filenames is a mutable set the caller keeps a reference to; it's
    populated as read_document/semantic_search are used, so the caller can
    later report retrieval coverage without re-deriving it (see evaluation.py).
    """
    documents = load_documents()
    by_filename = {doc.filename: doc for doc in documents}
    vector_store = build_index(documents)

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
        retrieved_filenames.add(doc.filename)
        return doc.content

    def semantic_search(query: str, top_k: int = 5) -> str:
        results = retrieve(vector_store, query, top_k=top_k)
        if not results:
            return f"No relevant chunks found for '{query}'."
        for scored in results:
            retrieved_filenames.add(scored.chunk.filename)
        return "\n\n".join(
            f"[{scored.chunk.filename} chunk {scored.chunk.chunk_index}, similarity={scored.score:.3f}]\n{scored.chunk.text}"
            for scored in results
        )

    def write_draft_note(content: str) -> str:
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_path = OUTPUT_DIR / f"draft_note_{company.replace(' ', '_')}_{date.today().isoformat()}.md"
        output_path.write_text(content, encoding="utf-8")
        return f"Draft note written to {output_path}"

    def save_memory_note(note: str) -> str:
        long_term_memory.add_note(company, note)
        return f"Saved to long-term memory for {company}: {note}"

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
            name="semantic_search",
            description=(
                "Retrieve the top-k document chunks most semantically relevant to a natural-language "
                "query, using embedding similarity (RAG) rather than exact keyword matching. Prefer "
                "this over read_document for gathering evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language question or topic to search for."},
                    "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default 5)."},
                },
                "required": ["query"],
            },
            handler=semantic_search,
        ),
        Tool(
            name="write_draft_note",
            description=(
                "Signal that you have gathered enough evidence and are ready to draft. Call this "
                "exactly once, as the last step, with a concise synthesis (not the full note — a "
                "Tree-of-Thought process drafts, scores, and refines the actual note from your "
                "evidence and this synthesis)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "evidence_summary": {
                        "type": "string",
                        "description": "3-6 sentences synthesizing the most important facts, changes, and risks found.",
                    }
                },
                "required": ["evidence_summary"],
            },
            handler=write_draft_note,
        ),
        Tool(
            name="save_memory_note",
            description=(
                "Persist a fact worth remembering across future runs for this company (e.g., a "
                "baseline figure, a standing risk, a confirmed detail) to long-term memory. Use this "
                "for anything future sessions should already know, beyond what write_draft_note saves."
            ),
            input_schema={
                "type": "object",
                "properties": {"note": {"type": "string", "description": "The fact to remember, in one or two sentences."}},
                "required": ["note"],
            },
            handler=save_memory_note,
        ),
    ]
