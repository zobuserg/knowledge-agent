"""
Ingestion Pipeline
------------------
Takes raw documents (PDF, Markdown, TXT) and stores them
in a ChromaDB vector store, ready for retrieval.

Flow:
    raw file  →  load  →  chunk  →  embed  →  ChromaDB
"""

from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings


def get_vector_store(collection_name: Optional[str] = None) -> tuple:
    """
    Creates (or reopens) a ChromaDB collection and wraps it
    in a LlamaIndex VectorStore.

    Returns:
        (chroma_collection, llama_vector_store)

    Why ChromaDB?
        - Runs locally, no API key needed
        - Persistent: survives restarts (data stays on disk)
        - Fast for the scale we need (thousands of chunks)
    """
    name = collection_name or settings.COLLECTION_NAME

    # PersistentClient = data saved to disk, not lost on restart
    client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))

    # get_or_create: safe to call multiple times — won't duplicate
    collection = client.get_or_create_collection(name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return collection, vector_store


def get_embed_model():
    """
    Returns an embedding model.
    Uses Ollama locally (free, no API key, runs offline).

    Embeddings = turning text into a vector of numbers
    so we can find "similar" chunks by measuring distance.
    """
    return OllamaEmbedding(
        model_name=settings.EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def ingest_directory(path: str | Path) -> dict:
    """
    Main function: load all documents in a folder and index them.

    Args:
        path: folder containing your documents (PDF, MD, TXT)

    Returns:
        dict with stats: how many docs and chunks were processed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    print(f"[Ingestion] Loading documents from: {path}")

    # Step 1: Load — SimpleDirectoryReader handles PDF, MD, TXT automatically
    documents = SimpleDirectoryReader(
        input_dir=str(path),
        recursive=True,           # look inside subfolders too
        required_exts=[".pdf", ".md", ".txt"],
    ).load_data()

    if not documents:
        return {"docs_loaded": 0, "chunks_created": 0}

    print(f"[Ingestion] Loaded {len(documents)} document(s)")

    # Step 2: Chunk — split each doc into overlapping pieces
    # Why overlap? So a sentence at the edge of a chunk isn't cut in half.
    # chunk_size=512 tokens is a good default for most content.
    splitter = SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # Step 3: Get the vector store (ChromaDB)
    _, vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Step 4: Embed + Store — VectorStoreIndex does both in one call
    # It: splits docs → creates embeddings → stores in ChromaDB
    embed_model = get_embed_model()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    # Count how many chunks were created
    collection, _ = get_vector_store()
    chunk_count = collection.count()

    print(f"[Ingestion] Done. {chunk_count} chunks in vector store.")

    return {
        "docs_loaded": len(documents),
        "chunks_created": chunk_count,
        "collection": settings.COLLECTION_NAME,
    }


def ingest_single_file(file_path: str | Path) -> dict:
    """
    Ingest a single file instead of a whole directory.
    Useful for the API endpoint POST /ingest.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Wrap single file in a temp folder structure LlamaIndex expects
    documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()

    _, vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()
    splitter = SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    collection, _ = get_vector_store()
    return {
        "file": file_path.name,
        "docs_loaded": len(documents),
        "total_chunks_in_store": collection.count(),
    }
