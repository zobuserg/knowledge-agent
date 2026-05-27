# Knowledge Agent

An agentic RAG (Retrieval-Augmented Generation) system that ingests your documents and lets you query them with an AI agent that reasons over the content and returns answers with citations.

Built as a portfolio project demonstrating production-grade AI engineering skills: document ingestion pipelines, vector search, agentic reasoning, REST API, and automated evaluation.

---

## What it does

- **Ingest** PDF, Markdown, and plain text documents into a persistent vector store
- **Query** your knowledge base in natural language — the agent retrieves the most relevant chunks and synthesizes a grounded answer
- **Cite sources** — every answer includes the source document and page/section reference
- **Evaluate** — built-in eval suite using RAGAS to measure faithfulness, answer relevance, and context precision
- **REST API** — FastAPI backend, ready to integrate with any frontend or workflow

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI Server                      │
│   POST /ingest   ·   POST /query   ·   GET /documents   │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
   ┌──────▼──────┐              ┌───────▼──────┐
   │  Ingestion  │              │  Query Agent │
   │  Pipeline   │              │              │
   │             │              │  LlamaIndex  │
   │  • Chunking │              │  ReAct Agent │
   │  • Embed    │              │  + Memory    │
   │  • Store    │              └───────┬──────┘
   └──────┬──────┘                      │
          │                             │
   ┌──────▼─────────────────────────────▼──────┐
   │              ChromaDB (vector store)       │
   │         persistent · local · fast          │
   └────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent framework | LlamaIndex |
| Vector store | ChromaDB |
| LLM | Anthropic Claude (claude-sonnet-4-5) |
| Embeddings | sentence-transformers / nomic-embed |
| API | FastAPI + uvicorn |
| Evaluation | RAGAS |
| Language | Python 3.12 |

---

## Project structure

```
knowledge-agent/
├── app/
│   ├── ingestion/       # Document loading, chunking, embedding
│   ├── query/           # Agent logic, retrieval, answer generation
│   └── memory/          # Conversation history persistence
├── data/
│   ├── raw/             # Drop your documents here
│   └── processed/       # ChromaDB persistent storage
├── evals/               # RAGAS evaluation suite + test dataset
├── scripts/             # CLI utilities (ingest, query, eval)
├── main.py              # FastAPI app entry point
├── requirements.txt
└── .env.example
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/zobuserg/knowledge-agent.git
cd knowledge-agent
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Ingest documents
python scripts/ingest.py --path data/raw/

# 4. Run the API
uvicorn main:app --reload

# 5. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics covered?"}'
```

---

## Evaluation

```bash
python evals/run_evals.py
```

Runs the RAGAS evaluation suite and reports:
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevance** — does the answer actually address the question?
- **Context Precision** — are the retrieved chunks relevant?

---

## Why this project

Most AI demos connect a PDF to ChatGPT and call it RAG. This project goes further:

1. **Agentic retrieval** — the agent decides *how many* retrieval steps to take before answering, not a fixed k-chunk lookup
2. **Persistent memory** — conversation history survives across sessions
3. **Eval-driven** — every change is validated against a test dataset before shipping
4. **Production-ready structure** — FastAPI, environment config, proper error handling

---

## Status

- [x] Project structure and README
- [ ] Ingestion pipeline (PDF + Markdown + TXT)
- [ ] ChromaDB integration
- [ ] ReAct query agent with citations
- [ ] FastAPI endpoints
- [ ] Conversation memory
- [ ] RAGAS eval suite
- [ ] Demo with sample dataset

---

*Built by [@zobuserg](https://github.com/zobuserg)*
