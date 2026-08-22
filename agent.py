"""Week 2: Agent Architecture & ReAct.

Adds planning/orchestration on top of the Week 1 baseline: a short
planning pass states a strategy, then a Thought-Action-Observation loop
lets the model pick which tools to call (see tools.py) to gather evidence
before drafting, rather than Week 1's single load-everything-then-prompt
pass.
"""

from pathlib import Path

import anthropic

from llm_client import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from tools import build_tools

MAX_STEPS = 8

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system_prompt.txt"
PLANNING_PROMPT_PATH = Path(__file__).parent / "prompts" / "planning_prompt.txt"


class ReActAgent:
    """Orchestrates a planning pass and the Thought-Action-Observation loop."""

    def __init__(self, company: str, model: str = DEFAULT_MODEL):
        self.company = company
        self.model = model
        self.client = anthropic.Anthropic()
        self.tools = build_tools(company)
        self._tool_by_name = {tool.name: tool for tool in self.tools}

    def _plan(self, goal: str) -> str:
        template = PLANNING_PROMPT_PATH.read_text(encoding="utf-8")
        tool_list = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)
        prompt = template.format(goal=goal, tools=tool_list)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def run(self, goal: str) -> str:
        plan = self._plan(goal)
        print(f"=== Plan ===\n{plan}\n")

        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").format(company=self.company)
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

            if not tool_calls:
                return thought or "(agent stopped without a final answer)"

            tool_results = []
            for call in tool_calls:
                tool = self._tool_by_name.get(call.name)
                observation = tool.run(**call.input) if tool else f"Unknown tool '{call.name}'."

                print(f"--- Step {step}: Action ---\n{call.name}({call.input})\n")
                print(f"--- Step {step}: Observation ---\n{observation}\n")

                tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": observation})

                if call.name == "write_draft_note":
                    return observation

            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"Agent did not reach a final answer within {MAX_STEPS} steps.")
