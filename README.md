# Research Note Drafting Agent

CMU Agentic AI Capstone.

- **Week 1 — Simple Drafting Agent:** a non-agentic baseline that loads sample
  source documents, fills a prompt template, and generates a draft research
  note via an LLM in a single linear pass.
- **Week 2 — Agent Architecture & ReAct:** adds a tool abstraction layer, a
  planning step, and a Thought-Action-Observation loop. The agent decides
  which documents to inspect instead of always loading everything, and
  finishes by calling a `write_draft_note` tool.
- **Week 3 — RAG Integration:** replaces whole-file reads with real retrieval.
  Documents are chunked, embedded with Voyage AI, and indexed in an in-memory
  vector store; the agent's `semantic_search` tool retrieves the top-k most
  relevant chunks for each query instead of loading full files.
- **Week 4 — Memory:** adds short-term memory (a per-run scratchpad logging
  each Thought/Action, discarded when the run ends) and long-term memory
  (analyst preferences and per-company notes persisted to
  `memory/agent_memory.json`, injected into every future run's planning and
  system prompt). The agent can also write to long-term memory mid-run via
  `save_memory_note`.
- **Week 5 — Tree of Thoughts:** replaces single-shot drafting. When the
  agent calls `write_draft_note`, it now hands off a short evidence synthesis
  (not a full note) to a beam search: 3 candidate drafts are generated under
  fixed, distinct narrative angles (growth, risk, execution/credibility),
  each scored 1-10 by an LLM judge on groundedness/coverage/clarity, the top
  2 survive, each is refined once against its own judge feedback, and the
  single best-scoring refinement is written to disk.
- **Week 6 — Refinement & Evaluation:** every tool call is now failure-safe
  (`Tool.run()` catches handler exceptions and feeds the agent an observation
  it can recover from instead of crashing); hitting the step limit without
  drafting no longer crashes the run — it forces a best-effort draft from
  whatever evidence exists, clearly labeled as such. After drafting,
  `evaluation.py` reports groundedness (reusing the Week 5 judge score),
  what fraction of the available documents were actually retrieved, and
  cross-checks every filename in Source Attribution against the real
  documents in `data/`, flagging any that don't exist.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY
export $(cat .env | xargs)  # or use direnv/python-dotenv
```

Get a Voyage AI key (used for embeddings — Anthropic does not serve
embeddings directly) at https://dash.voyageai.com; it has a free tier.

## Project structure

```
capstone-research-agent/
├── main.py                     # entry point; --mode agent (default) or linear
├── agent.py                    # planning + ReAct loop; Week 4-6: memory, ToT handoff, evaluation
├── tools.py                    # tool abstraction layer (Week 6: failure-safe Tool.run())
├── rag.py                      # Week 3: chunk -> embed -> index -> retrieve pipeline
├── chunker.py                  # Week 3: splits documents into overlapping word-window chunks
├── embeddings.py                # Week 3: Voyage AI embeddings client
├── vector_store.py              # Week 3: in-memory cosine-similarity vector database
├── memory.py                    # Week 4: short-term scratchpad + long-term JSON-backed memory
├── tree_of_thought.py           # Week 5: candidate generation, LLM-judge scoring, beam search
├── evaluation.py                 # Week 6: groundedness/coverage/citation-validity report
├── document_loader.py          # loads sample source docs from data/
├── prompt_builder.py           # Week 1: fills prompts/draft_note_prompt.txt
├── llm_client.py                # Week 1: single-shot Anthropic Messages API call
├── prompts/
│   ├── draft_note_prompt.txt    # Week 1 structured note template
│   ├── agent_system_prompt.txt  # Week 2-5 ReAct system prompt
│   ├── planning_prompt.txt      # Week 2/4 planning-step prompt
│   ├── tot_generate_prompt.txt  # Week 5: initial candidate drafting, one per narrative angle
│   ├── tot_refine_prompt.txt    # Week 5: refines a surviving candidate against judge feedback
│   └── tot_score_prompt.txt     # Week 5: LLM-judge scoring rubric
├── data/
│   ├── agendas/
│   ├── meeting_notes/
│   ├── pitch_books/
│   └── prior_notes/            # sample documents for "Company XYZ"
├── memory/
│   └── agent_memory.json       # generated; long-term memory store (gitignored)
└── output/                     # generated draft notes land here
```

## Run

```bash
python main.py --company "XYZ Corp"              # Week 2-4: planning + ReAct agent (default)
python main.py --company "XYZ Corp" --mode linear # Week 1: single linear pass

python main.py --remember "always lead with the biggest guidance change"  # save a standing preference
python main.py --show-memory                                              # inspect stored long-term memory
```

The agent prints long-term memory, its plan, each Thought/Action/Observation
step, the Tree-of-Thought candidate/scoring trace, and the final evaluation
report to stdout; the winning draft is saved under `output/`. `--remember`
can be combined with `--company` to save a preference and run in the same
command.

## Notes

- Uses `claude-sonnet-5` by default (override with the `LLM_MODEL` env var,
  e.g. `claude-opus-5` for higher-quality drafts at higher cost).
- Embeddings use `voyage-3.5` by default (override with `VOYAGE_EMBED_MODEL`).
  Chunking is a 60-word sliding window with 15-word overlap
  (`CHUNK_SIZE_WORDS`/`CHUNK_OVERLAP_WORDS` in `chunker.py`).
- The vector index is rebuilt in memory each run (`build_tools()` in
  `tools.py` calls `rag.build_index()`) — fine at this data scale, but it
  means every run re-embeds all chunks. Nothing is persisted to disk.
- The ReAct loop caps at `MAX_STEPS = 8` (in `agent.py`) to guard against
  runaway tool-call loops.
- Long-term memory is a single flat JSON file, not a database — fine at this
  scale, but note that `add_preference`/`add_note` rewrite the whole file on
  every call, so it isn't safe for concurrent runs.
- Short-term memory only ever lives inside one `ReActAgent.run()` call; it is
  never written to disk, and nothing carries over between runs except what
  gets promoted into long-term memory via `write_draft_note` (automatic
  Executive Summary excerpt) or `save_memory_note` (explicit).
- Tree-of-Thought drafting costs roughly 10 extra Claude calls per run (3
  initial drafts + 3 scores + 2 refinements + 2 re-scores) on top of the
  ReAct loop itself — real but bounded, since `NARRATIVE_ANGLES` and
  `BEAM_WIDTH` in `tree_of_thought.py` are fixed, not proportional to corpus
  size.
- The evaluation report is diagnostic output only (printed to stdout); it is
  not written into the saved `.md` file and does not block a draft from
  being saved, even if citations are invalid — it's meant to be read, not
  enforced.
