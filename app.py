"""Simple web UI for a research analyst to kick off a draft research note.

Run with: streamlit run app.py

No API server needed. Streamlit runs this script directly in one Python
process and re-executes it top-to-bottom on every interaction, so the UI
just calls straight into agent.py/main.py — there's no separate
frontend/backend to keep in sync, and no need for FastAPI or similar
unless another program (not a human) needs to call this agent
programmatically.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import ReActAgent
from main import run_linear
from memory import LongTermMemory

st.set_page_config(page_title="Research Note Drafting Agent", page_icon="📝", layout="wide")


class _LiveLog:
    """Redirects stdout into a Streamlit placeholder, updated as it streams."""

    def __init__(self, placeholder):
        self._placeholder = placeholder
        self.buffer = ""

    def write(self, text: str) -> None:
        self.buffer += text
        self._placeholder.code(self.buffer[-6000:] or " ")

    def flush(self) -> None:
        pass


def _extract_output_path(observation: str) -> Path | None:
    match = re.search(r"Draft note written to (.+\.md)", observation)
    return Path(match.group(1)) if match else None


st.title("📝 Research Note Drafting Agent")
st.caption(
    "Gathers evidence from source documents and drafts a structured research note. "
    "A full run takes 1-2 minutes and makes live Anthropic + Voyage AI API calls."
)

memory = LongTermMemory.load()

with st.sidebar:
    st.header("Long-Term Memory")

    st.subheader("Analyst Preferences")
    if memory.preferences:
        for pref in memory.preferences:
            st.markdown(f"- {pref}")
    else:
        st.caption("No preferences saved yet.")

    new_pref = st.text_input("Add a standing preference", placeholder="e.g. always lead with the biggest guidance change")
    if st.button("Save Preference") and new_pref.strip():
        memory.add_preference(new_pref.strip())
        st.rerun()

    st.subheader("Notes by Company")
    if memory.notes:
        by_company: dict[str, list[dict]] = {}
        for note in memory.notes:
            by_company.setdefault(note["company"], []).append(note)
        for company_name, notes in by_company.items():
            with st.expander(f"{company_name} ({len(notes)})"):
                for note in notes:
                    st.markdown(f"**{note['date']}** — {note['note']}")
    else:
        st.caption("No notes saved yet.")

company = st.text_input("Company name", value="ABC Company")

with st.expander("Advanced options"):
    mode = st.radio(
        "Mode",
        ["Agent (planning + ReAct + RAG + Tree of Thoughts + evaluation)", "Linear baseline (single-pass, no agent)"],
    )

generate = st.button("Generate Draft Research Note", type="primary")

if generate:
    if not company.strip():
        st.error("Enter a company name first.")
    elif mode.startswith("Linear"):
        with st.spinner("Generating draft (linear baseline)..."):
            draft = run_linear(company.strip())
        st.subheader("Draft Research Note")
        st.markdown(draft)
        st.download_button("Download as Markdown", draft, file_name=f"draft_note_{company.strip()}.md", mime="text/markdown")
    else:
        log_placeholder = st.empty()
        stream = _LiveLog(log_placeholder)
        with st.spinner(f"Researching {company}..."):
            with contextlib.redirect_stdout(stream):
                agent = ReActAgent(company=company.strip())
                goal = f"Research {company.strip()} using the available tools and produce a structured draft research note."
                observation = agent.run(goal)
        log_placeholder.empty()

        st.success(observation)

        output_path = _extract_output_path(observation)
        if output_path and output_path.exists():
            draft_text = output_path.read_text(encoding="utf-8")
            st.subheader("Draft Research Note")
            st.markdown(draft_text)
            st.download_button("Download as Markdown", draft_text, file_name=output_path.name, mime="text/markdown")
        else:
            st.warning("Could not locate the generated draft file; see the run log below for details.")

        with st.expander("Full run log (Thoughts, Actions, Observations, Tree of Thought, Evaluation)"):
            st.code(stream.buffer)
