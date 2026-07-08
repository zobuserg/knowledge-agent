# Knowledge Agent

**Semantic memory for your documents.** A local RAG system with a real ReAct agent on top:
point it at a folder (an Obsidian vault, a library of PDFs, your notes), and ask questions in
natural language — the agent decides how many searches to run, answers grounded in your files,
and cites its sources. Conversations persist across restarts.

Works standalone — or as the **Tier-3 knowledge upgrade** for
[Expert Agent](https://github.com/dantedanielgm/expert-agent): when your knowledge folder outgrows
plain-text search, this is the semantic layer you plug in.

---

## What it does

- **Folder sync, incremental.** Connect a folder; only new or changed files get re-indexed
  (MD5-hash tracking). Re-sync 500 files where 3 changed → 3 get embedded.
- **Agentic retrieval.** A ReAct agent with a `search_knowledge_base` tool decides — per
  question — how many retrieval calls to make and with which queries, instead of one fixed
  top-k lookup. Multi-part questions trigger multiple searches.
- **Answers with citations.** Every answer lists the source files (with relevance scores and
  excerpts) that every retrieval step touched.
- **Persistent sessions.** Chat history is saved to disk per session and survives restarts.
- **Multilingual.** Ask in Spanish about English documents (or vice versa) — embeddings are
  multilingual and the agent replies in your language.
- **Evals included.** A runnable eval suite measures retrieval hit-rate and end-to-end answer
  quality against *your* corpus — so config changes are judged by a score, not a feeling.

## Stack (what actually runs)

| Layer | Tech |
|---|---|
| UI | **Streamlit** (`ui.py`) — folder picker, multi-file upload, chat |
| API | **FastAPI** (`main.py`) — `/ingest`, `/query`, `/documents`, `/health` |
| Agent | **LlamaIndex ReActAgent** + `QueryEngineTool` |
| LLM | **OpenAI** `gpt-4o-mini` (with key) · **Ollama** `qwen2.5:3b` (local fallback) |
| Embeddings | OpenAI `text-embedding-3-small` (with key) · `nomic-embed-text` via Ollama |
| Vector store | **ChromaDB**, persistent on disk |

No key? Everything falls back to local Ollama — free, private, slower.

## Quickstart

```bash
git clone https://github.com/dantedanielgm/knowledge-agent.git
cd knowledge-agent
python -m venv .venv && .venv/Scripts/activate   # Windows (use bin/activate on Mac/Linux)
pip install -r requirements.txt

cp .env.example .env        # add your OPENAI_API_KEY (or leave empty for Ollama)

# Option A — the app (recommended)
streamlit run ui.py         # connect a folder from the UI and start asking

# Option B — CLI + API
python scripts/ingest.py --path path/to/your/docs
uvicorn main:app --reload   # then POST /query
```

## Evals — measure it, don't feel it

```bash
python evals/run_evals.py          # retrieval hit-rate (cheap: embeddings only)
python evals/run_evals.py --e2e    # + full agent answers graded (LLM cost)
```

Copy `evals/dataset.example.json` → `evals/dataset.json` and write cases that match **your**
documents: a question, the file that should be retrieved, facts the answer must state. Every time
you change the embedding model, chunk size or top-k, the score tells you whether it helped.
(The personal dataset is gitignored — commit the example, keep your cases yours.)

## Architecture

```
        ┌────────────┐        ┌──────────────┐
docs →  │ ingestion  │  → →   │  ChromaDB    │
        │ (hash sync)│        │ (persistent) │
        └────────────┘        └──────┬───────┘
                                     │ search_knowledge_base (tool)
                              ┌──────┴───────┐
 question → Streamlit/API  →  │  ReAct agent │ → answer + citations
                              │ (N searches, │
                              │  it decides) │ ←→ sessions/ (persistent memory)
                              └──────────────┘
```

## Part of an ecosystem

This repo is the knowledge layer of a two-piece system:

- **[expert-agent](https://github.com/dantedanielgm/expert-agent)** — the brain: a personal expert
  agent for Claude Code (engineer + verifier + AI operator + tutor, specialty of your choice). Its
  knowledge protocol starts with plain-text search over your folder — the right default.
- **knowledge-agent** (this repo) — the semantic memory: when the corpus grows past what keyword
  search handles (whole books, cross-language questions, paraphrase misses), this is the upgrade.

## Honest status

- ✅ Ingestion with incremental sync — in daily use over a 500+ file vault (6,900+ chunks)
- ✅ ReAct agent with decision-driven retrieval, persistent sessions, citations
- ✅ Eval suite (retrieval + e2e) runnable against your own corpus
- 🔜 Retrieval re-ranking; RAGAS-style graded metrics on top of the eval suite

---

Built by **Dante Daniel Gutiérrez Matos** ([@dantedanielgm](https://github.com/dantedanielgm)) ·
MIT License
