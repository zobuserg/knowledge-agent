"""
Query Agent
-----------
Takes a user question, retrieves the most relevant chunks from ChromaDB,
and uses Claude to synthesize a grounded answer with citations.

This is a ReAct agent: it Reasons and Acts in a loop until it has
enough information to answer confidently.
"""

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic

from app.config import settings
from app.ingestion.pipeline import get_vector_store, get_embed_model


def build_agent(session_id: str = "default") -> ReActAgent:
    """
    Builds a ReAct agent wired to the knowledge base.

    Args:
        session_id: used to separate conversation histories
                    (different users = different sessions)

    Returns:
        A ready-to-use agent that can answer questions
    """
    # Reconnect to existing ChromaDB collection (already ingested)
    _, vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    # Load the index from the existing vector store (don't re-embed!)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        storage_context=storage_context,
    )

    # Query engine: handles retrieval + response synthesis
    # similarity_top_k=5 → fetch the 5 most relevant chunks
    # response_mode="compact" → concise answers with citations
    query_engine = index.as_query_engine(
        llm=Anthropic(
            model=settings.LLM_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
        ),
        similarity_top_k=5,
        response_mode="compact",
    )

    # Wrap query engine as a tool the agent can call
    # The description tells the agent WHEN to use this tool
    knowledge_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="knowledge_base",
            description=(
                "Use this tool to search the knowledge base and answer questions "
                "about the ingested documents. Always use this tool before answering "
                "to ensure your response is grounded in the actual documents."
            ),
        ),
    )

    # Memory: keeps conversation history so follow-up questions work
    # token_limit=4096: keeps the last ~4k tokens of context
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    # Build the ReAct agent
    # ReAct = Reason + Act: the agent thinks step-by-step before answering
    agent = ReActAgent.from_tools(
        tools=[knowledge_tool],
        llm=Anthropic(
            model=settings.LLM_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
        ),
        memory=memory,
        verbose=True,   # prints reasoning steps to console (useful for debugging)
        system_prompt=(
            "You are a helpful assistant with access to a knowledge base. "
            "Always search the knowledge base before answering. "
            "Include source references in your answers (document name and section). "
            "If the knowledge base doesn't contain relevant information, say so clearly "
            "instead of making things up."
        ),
    )

    return agent


# Simple cache: one agent per session_id
# In production this would be Redis or a proper session store
_agent_cache: dict[str, ReActAgent] = {}


def get_or_create_agent(session_id: str = "default") -> ReActAgent:
    """Returns cached agent for session, or builds a new one."""
    if session_id not in _agent_cache:
        _agent_cache[session_id] = build_agent(session_id)
    return _agent_cache[session_id]


def query(question: str, session_id: str = "default") -> dict:
    """
    Main function: answer a question using the knowledge base.

    Args:
        question: the user's question in natural language
        session_id: conversation ID (enables follow-up questions)

    Returns:
        dict with answer, sources, and session_id
    """
    agent = get_or_create_agent(session_id)
    response = agent.chat(question)

    # Extract source documents from the response metadata
    sources = []
    if hasattr(response, "source_nodes"):
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
