"""
Query Agent
-----------
Takes a user question, retrieves the most relevant chunks from ChromaDB,
and uses the LLM to synthesize a grounded answer with citations.

Uses LlamaIndex's query engine with chat memory for follow-up questions.
"""

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.ollama import Ollama

from app.config import settings
from app.ingestion.pipeline import get_vector_store, get_embed_model


def build_chat_engine(session_id: str = "default") -> CondensePlusContextChatEngine:
    """
    Builds a chat engine wired to the knowledge base.

    CondensePlusContextChatEngine:
    - Condenses the conversation history into a single query
    - Retrieves relevant chunks from ChromaDB
    - Generates a grounded answer with the LLM

    Args:
        session_id: used to separate conversation histories

    Returns:
        A ready-to-use chat engine
    """
    _, vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    # Load existing index from the vector store (don't re-embed)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    llm = Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=300.0,   # 5 min — needed for multi-chunk synthesis
    )

    # Retriever: fetch top-3 most relevant chunks (fewer = faster synthesis)
    retriever = index.as_retriever(similarity_top_k=3)

    # Memory: keeps the last 4096 tokens of conversation history
    # This enables follow-up questions ("what about the second point?")
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        system_prompt=(
            "You are a helpful assistant with access to a knowledge base. "
            "Answer questions based only on the provided context. "
            "Always mention which document your answer comes from. "
            "If the context doesn't contain the answer, say so clearly."
        ),
        verbose=True,
    )

    return chat_engine


# Session cache: one engine per session_id
_engine_cache: dict[str, CondensePlusContextChatEngine] = {}


def get_or_create_engine(session_id: str = "default") -> CondensePlusContextChatEngine:
    """Returns cached engine for session, or builds a new one."""
    if session_id not in _engine_cache:
        _engine_cache[session_id] = build_chat_engine(session_id)
    return _engine_cache[session_id]


def query(question: str, session_id: str = "default") -> dict:
    """
    Answer a question using the knowledge base.

    Args:
        question: the user's question in natural language
        session_id: conversation ID (enables follow-up questions)

    Returns:
        dict with answer, sources, and session_id
    """
    engine = get_or_create_engine(session_id)
    response = engine.chat(question)

    # Extract source documents
    sources = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for node in response.source_nodes:
            sources.append({
                "file": node.metadata.get("file_name", "unknown"),
                "score": round(node.score, 3) if node.score else None,
                "excerpt": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            })

    return {
        "answer": str(response),
        "sources": sources,
        "session_id": session_id,
    }
