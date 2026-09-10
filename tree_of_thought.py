"""Week 5: Tree of Thoughts drafting.

Instead of writing one linear draft, generate several candidate narrative
framings from the evidence gathered during the ReAct loop, score each with
an LLM judge, and beam-search down to a single winner across two rounds: an
initial branch-and-prune (one candidate per fixed narrative angle, so the
branches are genuinely distinct rather than resampled near-duplicates),
then a refine-and-prune round on the survivors.
"""

import re
from dataclasses import dataclass
from pathlib import Path

NARRATIVE_ANGLES = [
    "Growth and expansion: lead with what's accelerating and why it matters.",
    "Risk and margin pressure: lead with what could go wrong and what's under strain.",
    "Execution and guidance credibility: lead with whether management is delivering on what they said.",
]
BEAM_WIDTH = 2
DRAFT_MAX_TOKENS = 2048
SCORE_MAX_TOKENS = 200

GENERATE_PROMPT_PATH = Path(__file__).parent / "prompts" / "tot_generate_prompt.txt"
REFINE_PROMPT_PATH = Path(__file__).parent / "prompts" / "tot_refine_prompt.txt"
SCORE_PROMPT_PATH = Path(__file__).parent / "prompts" / "tot_score_prompt.txt"


@dataclass
class Candidate:
    narrative: str
    content: str
    score: float = 0.0
    rationale: str = ""


def _split_narrative(text: str) -> tuple[str, str]:
    """Candidates start with 'NARRATIVE: <label>' on the first line, then the draft."""
    lines = text.splitlines()
    if lines and lines[0].strip().upper().startswith("NARRATIVE:"):
        return lines[0].split(":", 1)[1].strip(), "\n".join(lines[1:]).strip()
    return "(unlabeled)", text


def _parse_score(text: str) -> tuple[float, str]:
    score_match = re.search(r"SCORE:\s*([\d.]+)", text)
    rationale_match = re.search(r"RATIONALE:\s*(.+)", text)
    try:
        score = float(score_match.group(1)) if score_match else 5.0
    except ValueError:
        score = 5.0
    rationale = rationale_match.group(1).strip() if rationale_match else "(no rationale returned)"
    return min(max(score, 0.0), 10.0), rationale


def _complete(client, model: str, prompt: str, max_tokens: int) -> str:
    response = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _generate_initial_candidates(client, model, company, evidence, evidence_summary, memory_context) -> list[Candidate]:
    template = GENERATE_PROMPT_PATH.read_text(encoding="utf-8")
    candidates = []
    for angle in NARRATIVE_ANGLES:
        prompt = template.format(
            company=company, angle=angle, evidence=evidence, evidence_summary=evidence_summary, memory=memory_context
        )
        try:
            text = _complete(client, model, prompt, DRAFT_MAX_TOKENS)
        except Exception as exc:
            print(f"  (skipping angle '{angle}': generation failed — {exc})")
            continue
        narrative, content = _split_narrative(text)
        candidates.append(Candidate(narrative=narrative, content=content))
    return candidates


def _refine_candidate(client, model, company, evidence, memory_context, candidate: Candidate) -> Candidate:
    template = REFINE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        company=company,
        narrative=candidate.narrative,
        rationale=candidate.rationale,
        memory=memory_context,
        evidence=evidence,
        content=candidate.content,
    )
    try:
        text = _complete(client, model, prompt, DRAFT_MAX_TOKENS)
    except Exception as exc:
        print(f"  (refinement of '{candidate.narrative}' failed — {exc}; keeping prior draft)")
        return candidate
    narrative, content = _split_narrative(text)
    return Candidate(narrative=narrative or candidate.narrative, content=content)


def _score_candidate(client, model, company, evidence, candidate: Candidate) -> Candidate:
    template = SCORE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(company=company, evidence=evidence, narrative=candidate.narrative, content=candidate.content)
    try:
        text = _complete(client, model, prompt, SCORE_MAX_TOKENS)
        candidate.score, candidate.rationale = _parse_score(text)
    except Exception as exc:
        candidate.score, candidate.rationale = 5.0, f"(scoring failed — {exc}; defaulted to 5.0)"
    return candidate


def run(client, model: str, company: str, evidence: str, evidence_summary: str, memory_context: str) -> Candidate:
    """Generate NARRATIVE_ANGLES candidates, score, keep the top BEAM_WIDTH;
    refine each survivor once, score again, and return the single best."""
    print(f"=== Tree of Thought: generating {len(NARRATIVE_ANGLES)} candidate narratives ===")
    candidates = _generate_initial_candidates(client, model, company, evidence, evidence_summary, memory_context)
    if not candidates:
        raise RuntimeError("Tree-of-Thought drafting failed: no candidates could be generated.")
    candidates = [_score_candidate(client, model, company, evidence, c) for c in candidates]
    for c in candidates:
        print(f"  - [{c.score:.1f}] {c.narrative}: {c.rationale}")

    beam = sorted(candidates, key=lambda c: c.score, reverse=True)[:BEAM_WIDTH]
    print(f"Kept top {len(beam)}: {', '.join(c.narrative for c in beam)}\n")

    print(f"=== Tree of Thought: refining {len(beam)} surviving branch(es) ===")
    refined = [_refine_candidate(client, model, company, evidence, memory_context, c) for c in beam]
    refined = [_score_candidate(client, model, company, evidence, c) for c in refined]
    for c in refined:
        print(f"  - [{c.score:.1f}] {c.narrative}: {c.rationale}")

    winner = max(refined, key=lambda c: c.score)
    print(f"Winner: {winner.narrative} (score {winner.score:.1f})\n")
    return winner
