# Research Note Drafting Agent

CMU Agentic AI Capstone.

- **Week 1 — Simple Drafting Agent:** a non-agentic baseline that loads sample
  source documents, fills a prompt template, and generates a draft research
  note via an LLM in a single linear pass.
- **Week 2 — Agent Architecture & ReAct:** adds a tool abstraction layer, a
  planning step, and a Thought-Action-Observation loop. The agent now decides
  which documents to inspect (via `list_documents`, `read_document`,
  `search_documents`) instead of always loading everything, and finishes by
  calling a `write_draft_note` tool.

Later weeks add RAG retrieval, memory, and Tree-of-Thought reasoning on top.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in ANTHROPIC_API_KEY
export $(cat .env | xargs)  # or use direnv/python-dotenv
```

## Project structure

```
capstone-research-agent/
├── main.py                     # entry point; --mode agent (default) or linear
├── agent.py                    # Week 2: planning + ReAct (Thought-Action-Observation) loop
├── tools.py                    # Week 2: tool abstraction layer (list/read/search docs, write note)
├── document_loader.py          # loads sample source docs from data/
├── prompt_builder.py           # Week 1: fills prompts/draft_note_prompt.txt
├── llm_client.py                # Week 1: single-shot Anthropic Messages API call
├── prompts/
│   ├── draft_note_prompt.txt    # Week 1 structured note template
│   ├── agent_system_prompt.txt  # Week 2 ReAct system prompt
│   └── planning_prompt.txt      # Week 2 planning-step prompt
├── data/
│   ├── agendas/
│   ├── meeting_notes/
│   ├── pitch_books/
│   └── prior_notes/            # sample documents for "Company XYZ"
└── output/                     # generated draft notes land here
```

## Run

```bash
python main.py --company "XYZ Corp"              # Week 2: planning + ReAct agent (default)
python main.py --company "XYZ Corp" --mode linear # Week 1: single linear pass
```

The agent prints its plan and each Thought/Action/Observation step to stdout;
the final draft is saved under `output/`.

## Notes

- Uses `claude-sonnet-5` by default (override with the `LLM_MODEL` env var,
  e.g. `claude-opus-5` for higher-quality drafts at higher cost).
- Document loading here is a flat read of `data/*/*.txt` — Week 3 replaces
  this with chunking + embeddings + vector retrieval (RAG), which the agent's
  tools will call into instead of reading whole files.
- The ReAct loop caps at `MAX_STEPS = 8` (in `agent.py`) to guard against
  runaway tool-call loops.
