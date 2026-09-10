"""Week 6: post-hoc evaluation of a finished draft.

Reuses the Tree-of-Thought judge's groundedness score (see tree_of_thought.py)
rather than paying for a second LLM call, and adds two cheap structural
checks: whether every filename cited in Source Attribution actually exists
in data/, and how much of the available corpus the agent actually looked at
before drafting. These surface failures explicitly instead of silently
shipping a draft that cites documents that don't exist or ignored most of
the available evidence.
"""

import re
from dataclasses import dataclass


def extract_section(markdown: str, heading: str) -> str:
    """Pull the body text of one '## Heading' section out of a draft note."""
    lines = markdown.splitlines()
    collected = []
    capturing = False
    for line in lines:
        if line.strip().startswith("## "):
            if capturing:
                break
            capturing = heading.lower() in line.lower()
            continue
        if capturing:
            collected.append(line)
    return "\n".join(collected).strip()


def extract_cited_filenames(draft_content: str) -> list[str]:
    section = extract_section(draft_content, "Source Attribution")
    return sorted(set(re.findall(r"[\w.\-]+\.txt", section)))


@dataclass
class EvaluationReport:
    groundedness_score: float
    groundedness_rationale: str
    cited_filenames: list[str]
    invalid_citations: list[str]
    retrieved_filenames: set[str]
    available_filenames: list[str]

    def retrieval_coverage(self) -> float:
        if not self.available_filenames:
            return 1.0
        return len(self.retrieved_filenames) / len(self.available_filenames)

    def summary(self) -> str:
        lines = [
            f"Groundedness (Tree-of-Thought judge): {self.groundedness_score:.1f}/10 — {self.groundedness_rationale}",
            f"Retrieval coverage: {len(self.retrieved_filenames)}/{len(self.available_filenames)} "
            f"available documents were retrieved this session",
            f"Source citations: {len(self.cited_filenames)} document(s) attributed",
        ]
        if self.invalid_citations:
            lines.append(f"WARNING — cited document(s) not found in data/: {', '.join(self.invalid_citations)}")
        unretrieved = set(self.available_filenames) - self.retrieved_filenames
        if unretrieved:
            lines.append(f"Note: never retrieved this session: {', '.join(sorted(unretrieved))}")
        return "\n".join(lines)


def evaluate(
    draft_content: str,
    groundedness_score: float,
    groundedness_rationale: str,
    available_filenames: list[str],
    retrieved_filenames: set[str],
) -> EvaluationReport:
    cited = extract_cited_filenames(draft_content)
    invalid = [f for f in cited if f not in available_filenames]
    return EvaluationReport(
        groundedness_score=groundedness_score,
        groundedness_rationale=groundedness_rationale,
        cited_filenames=cited,
        invalid_citations=invalid,
        retrieved_filenames=retrieved_filenames,
        available_filenames=available_filenames,
    )
