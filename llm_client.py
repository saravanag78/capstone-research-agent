"""Thin wrapper around the Anthropic Messages API for the Week 1 prototype."""

import os

import anthropic

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 4096


def generate_draft_note(prompt: str, model: str = DEFAULT_MODEL) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
