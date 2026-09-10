"""Week 2-6: Agent Architecture, ReAct, RAG, Memory, Tree of Thoughts, Evaluation.

Adds planning/orchestration on top of the Week 1 baseline: a short
planning pass states a strategy, then a Thought-Action-Observation loop
lets the model pick which tools to call (see tools.py) to gather evidence
before drafting, rather than Week 1's single load-everything-then-prompt
pass. Week 3 backs evidence-gathering with RAG (see rag.py). Week 4 adds
memory (see memory.py). Week 5 replaces single-shot drafting with a
Tree-of-Thought beam search over candidate narratives (see
tree_of_thought.py), triggered when the agent calls write_draft_note. Week
6 evaluates the winning draft (see evaluation.py) and makes both retrieval
and reasoning failures recoverable instead of fatal.
"""

from pathlib import Path

import anthropic

import tree_of_thought
from evaluation import evaluate, extract_section
from llm_client import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from document_loader import load_documents
from memory import LongTermMemory, ShortTermMemory
from tools import build_tools

MAX_STEPS = 8

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system_prompt.txt"
PLANNING_PROMPT_PATH = Path(__file__).parent / "prompts" / "planning_prompt.txt"


def _evidence_transcript(messages: list[dict]) -> str:
    """Flatten the ReAct message history into the observation text (tool
    results) that drafting should be grounded in."""
    lines = []
    for message in messages:
        if message["role"] != "user" or isinstance(message["content"], str):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                lines.append(str(block.get("content", "")))
    return "\n\n".join(lines) if lines else "(no evidence was gathered)"


class ReActAgent:
    """Orchestrates a planning pass and the Thought-Action-Observation loop."""

    def __init__(self, company: str, model: str = DEFAULT_MODEL):
        self.company = company
        self.model = model
        self.client = anthropic.Anthropic()
        self.long_term_memory = LongTermMemory.load()
        self.short_term_memory = ShortTermMemory()
        self.retrieved_filenames: set[str] = set()
        self.tools = build_tools(company, self.long_term_memory, self.retrieved_filenames)
        self._tool_by_name = {tool.name: tool for tool in self.tools}

    def _plan(self, goal: str, memory_context: str) -> str:
        template = PLANNING_PROMPT_PATH.read_text(encoding="utf-8")
        tool_list = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)
        prompt = template.format(goal=goal, tools=tool_list, memory=memory_context)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def _draft_and_finish(self, evidence_summary: str, messages: list[dict], step: int) -> str:
        """Run Tree-of-Thought drafting from gathered evidence, write the
        winning candidate, evaluate it, and return the observation."""
        memory_context = self.long_term_memory.context_summary(self.company)
        evidence = _evidence_transcript(messages)

        winner = tree_of_thought.run(
            client=self.client,
            model=self.model,
            company=self.company,
            evidence=evidence,
            evidence_summary=evidence_summary,
            memory_context=memory_context,
        )

        write_tool = self._tool_by_name["write_draft_note"]
        observation = write_tool.run(content=winner.content)
        print(f"--- Step {step}: Action ---\nwrite_draft_note(evidence_summary={evidence_summary!r})\n")
        print(f"--- Step {step}: Observation ---\n{observation}\n")
        self.short_term_memory.add(f"Step {step} action: write_draft_note -> {observation}")

        exec_summary = extract_section(winner.content, "Executive Summary")
        self.long_term_memory.add_note(self.company, exec_summary or observation)
        print(f"=== Short-Term Memory (this session) ===\n{self.short_term_memory.transcript()}\n")

        available_filenames = [doc.filename for doc in load_documents()]
        report = evaluate(
            draft_content=winner.content,
            groundedness_score=winner.score,
            groundedness_rationale=winner.rationale,
            available_filenames=available_filenames,
            retrieved_filenames=self.retrieved_filenames,
        )
        print(f"=== Evaluation ===\n{report.summary()}\n")

        return observation

    def run(self, goal: str) -> str:
        memory_context = self.long_term_memory.context_summary(self.company)
        print(f"=== Long-Term Memory ===\n{memory_context}\n")

        plan = self._plan(goal, memory_context)
        print(f"=== Plan ===\n{plan}\n")

        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").format(
            company=self.company, memory=memory_context
        )
        messages = [{"role": "user", "content": f"{goal}\n\nHere is your plan:\n{plan}"}]
        tool_schemas = [tool.to_anthropic_schema() for tool in self.tools]

        for step in range(1, MAX_STEPS + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=system_prompt,
                tools=tool_schemas,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            thought = "".join(block.text for block in response.content if block.type == "text").strip()
            tool_calls = [block for block in response.content if block.type == "tool_use"]

            if thought:
                print(f"--- Step {step}: Thought ---\n{thought}\n")
                self.short_term_memory.add(f"Step {step} thought: {thought}")

            if not tool_calls:
                return thought or "(agent stopped without a final answer)"

            draft_call = next((c for c in tool_calls if c.name == "write_draft_note"), None)
            if draft_call is not None:
                return self._draft_and_finish(draft_call.input.get("evidence_summary", ""), messages, step)

            tool_results = []
            for call in tool_calls:
                tool = self._tool_by_name.get(call.name)
                observation = tool.run(**call.input) if tool else f"Unknown tool '{call.name}'."

                print(f"--- Step {step}: Action ---\n{call.name}({call.input})\n")
                print(f"--- Step {step}: Observation ---\n{observation}\n")
                self.short_term_memory.add(f"Step {step} action: {call.name}({call.input}) -> {observation[:200]}")

                tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": observation})

            messages.append({"role": "user", "content": tool_results})

        print(
            f"WARNING: agent did not choose to draft within {MAX_STEPS} steps; "
            "forcing a best-effort draft from whatever evidence was gathered.\n"
        )
        return self._draft_and_finish(
            "Step limit reached before the agent decided to draft; this note is a best-effort "
            "synthesis of whatever evidence was gathered so far and may be incomplete.",
            messages,
            MAX_STEPS,
        )
