"""Entry point: Week 1 linear baseline and Week 2-6 ReAct agent.

Week 1 (--mode linear): load -> prompt -> generate, a single non-agentic pass.
Week 2-6 (--mode agent, default): plan -> Thought-Action-Observation loop over
a tool abstraction layer (see agent.py, tools.py), backed by RAG retrieval
(rag.py), short/long-term memory (memory.py), Tree-of-Thought drafting
(tree_of_thought.py), and post-hoc evaluation (evaluation.py).
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
from datetime import date
from pathlib import Path

from agent import ReActAgent
from document_loader import load_documents
from llm_client import generate_draft_note
from memory import LongTermMemory
from prompt_builder import build_prompt

OUTPUT_DIR = Path(__file__).parent / "output"


def run_linear(company: str) -> None:
    documents = load_documents()
    if not documents:
        print("No source documents found under data/. Add .txt files and retry.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(documents)} source document(s):")
    for doc in documents:
        print(f"  - [{doc.source_type}] {doc.filename}")

    prompt = build_prompt(company, documents)

    print("\nCalling LLM to generate draft research note...")
    draft = generate_draft_note(prompt)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"draft_note_{company.replace(' ', '_')}_{date.today().isoformat()}.md"
    output_path.write_text(draft, encoding="utf-8")

    print(f"\nDraft research note written to {output_path}\n")
    print(draft)


def run_agent(company: str) -> None:
    goal = f"Research {company} using the available tools and produce a structured draft research note."
    agent = ReActAgent(company=company)
    result = agent.run(goal)
    print("=== Final Result ===")
    print(result)


def show_memory() -> None:
    memory = LongTermMemory.load()
    print("=== Preferences ===")
    print("\n".join(f"- {p}" for p in memory.preferences) if memory.preferences else "(none saved)")
    print("\n=== Notes ===")
    if memory.notes:
        for note in memory.notes:
            print(f"- [{note['date']}] {note['company']}: {note['note']}")
    else:
        print("(none saved)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a draft research note.")
    parser.add_argument("--company", default="XYZ Corp", help="Company name for the note")
    parser.add_argument(
        "--mode",
        choices=["agent", "linear"],
        default="agent",
        help="'agent' (default, Week 2+) runs the planning + ReAct loop; 'linear' (Week 1) runs the baseline pass.",
    )
    parser.add_argument(
        "--remember",
        metavar="TEXT",
        help="Save an analyst preference to long-term memory (applies to this and all future runs), e.g. "
        "--remember 'always lead the Executive Summary with the biggest guidance change'.",
    )
    parser.add_argument("--show-memory", action="store_true", help="Print stored long-term memory and exit.")
    args = parser.parse_args()

    if args.show_memory:
        show_memory()
        return

    if args.remember:
        LongTermMemory.load().add_preference(args.remember)
        print(f"Saved preference: {args.remember}\n")

    try:
        if args.mode == "linear":
            run_linear(args.company)
        else:
            run_agent(args.company)
    except Exception as exc:
        print(f"\nAgent run failed: {exc}", file=sys.stderr)
        print(
            "Check your API keys (ANTHROPIC_API_KEY, VOYAGE_API_KEY) and network connection.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
