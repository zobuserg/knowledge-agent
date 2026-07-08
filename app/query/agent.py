"""
Query Agent
-----------
A real ReAct agent over the knowledge base: given a question, the agent decides
how many retrieval calls to make (and with which queries) before synthesizing a
grounded answer with citations.

Also persists chat history per session to disk, so conversations survive restarts.
"""

import asyncio
import json

from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent, ToolCallResult
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama

from app.config import settings
from app.ingestion.pipeline import get_vector_store, get_embed_model

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's personal knowledge base "
    "through the `search_knowledge_base` tool.\n"
    "RULES:\n"
    "- ALWAYS search the knowledge base before answering. Never answer from general "
    "knowledge alone.\n"
    "- If the first search doesn't fully answer the question, search again with a "
    "different or more specific query. You decide how many searches are enough.\n"
    "- Always cite which document(s) your answer comes from (file names).\n"
    "- If the knowledge base doesn't contain enough information, say so clearly and "
    "suggest the user add documents on that topic.\n"
    "- Always respond in the same language the user writes in."
)


def get_llm(api_key: str = ""):
    """Returns the configured LLM — OpenAI if key provided, Ollama otherwise."""
    key = api_key or settings.OPENAI_API_KEY
    if key:
        from llama_index.llms.openai import OpenAI
        return OpenAI(model=settings.OPENAI_MODEL, api_key=key)
    return Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=300.0,
    )


def build_agent(api_key: str = "") -> ReActAgent:
    """
    Builds the ReAct agent wired to the knowledge base.

    The agent gets ONE tool — semantic search over ChromaDB — and decides on its
    own how many times to call it (and with which queries) before answering.
    That decision loop is what makes this agentic rather than a fixed
    retrieve-then-answer pipeline.
    """
    _, vector_store = get_vector_store()
    embed_model = get_embed_model()

    # Load existing index from the vector store (don't re-embed)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    llm = get_llm(api_key)

    search_tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(similarity_top_k=6, llm=llm),
        name="search_knowledge_base",
        description=(
            "Semantic search over the user's documents. Input: a search query in "
            "natural language. Returns the most relevant passages with their source "
            "files. Call it as many times as needed with different queries to cover "
            "all parts of the user's question."
        ),
    )

    return ReActAgent(
        tools=[search_tool],
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )


# ── Persistent session memory ─────────────────────────────────────────────────
# Chat history lives on disk (one JSON per session), so a conversation survives
# app restarts. Kept small on purpose: the last N turns are enough context.

SESSIONS_DIR = settings.CHROMA_PERSIST_DIR / "sessions"
MAX_HISTORY_MESSAGES = 20


def _session_file(session_id: str):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return SESSIONS_DIR / f"{safe}.json"


def load_history(session_id: str) -> list[ChatMessage]:
    """Load persisted chat history for a session (empty list if none)."""
    f = _session_file(session_id)
    if not f.exists():
        return []
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        return [ChatMessage(role=m["role"], content=m["content"]) for m in raw]
    except Exception:
        return []


def save_history(session_id: str, history: list[ChatMessage]):
    """Persist chat history, trimmed to the most recent messages."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    data = [{"role": m.role.value, "content": m.content or ""} for m in trimmed]
    _session_file(session_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clear_history(session_id: str):
    """Delete a session's persisted history."""
    f = _session_file(session_id)
    if f.exists():
        f.unlink()


# ── Agent cache ───────────────────────────────────────────────────────────────
# One agent per (session, key-present). The UI calls _engine_cache.clear() after
# re-syncs or key changes to force a rebuild against the fresh index.

_engine_cache: dict[str, ReActAgent] = {}


def get_or_create_agent(session_id: str = "default", api_key: str = "") -> ReActAgent:
    """Returns cached agent for session, or builds a new one."""
    cache_key = f"{session_id}:{bool(api_key)}"
    if cache_key not in _engine_cache:
        _engine_cache[cache_key] = build_agent(api_key)
    return _engine_cache[cache_key]


# ── Query ─────────────────────────────────────────────────────────────────────

async def _aquery(question: str, session_id: str, api_key: str) -> dict:
    agent = get_or_create_agent(session_id, api_key)
    history = load_history(session_id)

    handler = agent.run(question, chat_history=history)

    # Collect sources from every retrieval the agent decided to make
    sources = []
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            raw = getattr(event.tool_output, "raw_output", None)
            for node in getattr(raw, "source_nodes", []) or []:
                sources.append({
                    "file": node.metadata.get("file_name", "unknown"),
                    "score": round(node.score, 3) if node.score else None,
                    "excerpt": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                })

    response = await handler
    answer = str(response)

    # Persist the turn
    history.append(ChatMessage(role="user", content=question))
    history.append(ChatMessage(role="assistant", content=answer))
    save_history(session_id, history)

    return {"answer": answer, "sources": sources, "session_id": session_id}


def query(question: str, session_id: str = "default", api_key: str = "") -> dict:
    """
    Answer a question using the knowledge base.

    Args:
        question: the user's question in natural language
        session_id: conversation ID (history persists across restarts)
        api_key: optional OpenAI key (falls back to env/Ollama)

    Returns:
        dict with answer, sources, and session_id
    """
    return asyncio.run(_aquery(question, session_id, api_key))
