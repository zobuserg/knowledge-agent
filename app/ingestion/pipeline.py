"""
Ingestion Pipeline
------------------
Loads documents into ChromaDB for retrieval.
Supports: single file, multiple files, full folder sync.
"""

from pathlib import Path
from typing import Optional
import hashlib
import json

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings

# Path to the user config (same file saved by the UI)
_USER_CONFIG = settings.CHROMA_PERSIST_DIR / ".user_config.json"

def _get_openai_key() -> str:
    """Read OpenAI key from user config file."""
    if _USER_CONFIG.exists():
        try:
            return json.loads(_USER_CONFIG.read_text(encoding="utf-8")).get("openai_api_key", "")
        except Exception:
            pass
    return settings.OPENAI_API_KEY


# ── Vector store ──────────────────────────────────────────────────────────────

def get_vector_store(collection_name: Optional[str] = None):
    name = collection_name or settings.COLLECTION_NAME
    client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
    collection = client.get_or_create_collection(name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return collection, vector_store


def get_embed_model():
    """
    Returns the best available embed model.
    - If OpenAI key is configured → text-embedding-3-small (much better for Spanish)
    - Otherwise → nomic-embed-text via Ollama (local fallback)
    """
    key = _get_openai_key()
    if key:
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(
            model=settings.OPENAI_EMBED_MODEL,
            api_key=key,
        )
    return OllamaEmbedding(
        model_name=settings.EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


# ── File tracking (avoid re-indexing unchanged files) ─────────────────────────

SYNC_LOG = settings.CHROMA_PERSIST_DIR / ".sync_log.json"

def load_sync_log() -> dict:
    """Returns {filepath: md5_hash} of already-indexed files."""
    if SYNC_LOG.exists():
        return json.loads(SYNC_LOG.read_text(encoding="utf-8"))
    return {}

def save_sync_log(log: dict):
    SYNC_LOG.parent.mkdir(parents=True, exist_ok=True)
    SYNC_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")

def file_hash(path: Path) -> str:
    """MD5 of file content — detects if a file changed since last sync."""
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── Core indexing ─────────────────────────────────────────────────────────────

def _index_documents(documents, show_progress=True) -> int:
    """Takes loaded LlamaIndex documents, embeds and stores them. Returns chunk count."""
    if not documents:
        return 0

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
        show_progress=show_progress,
    )

    collection, _ = get_vector_store()
    return collection.count()


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_files(file_paths: list[Path], show_progress=True) -> dict:
    """
    Ingest a list of files. Skips files that haven't changed since last sync.
    Returns stats dict.
    """
    SUPPORTED = {".pdf", ".md", ".txt"}
    sync_log = load_sync_log()

    to_index = []
    skipped = 0
    new_files = []

    for path in file_paths:
        path = Path(path)
        if not path.exists() or path.suffix.lower() not in SUPPORTED:
            continue
        h = file_hash(path)
        key = str(path)
        if sync_log.get(key) == h:
            skipped += 1
            continue
        to_index.append(path)
        new_files.append((key, h))

    if not to_index:
        collection, _ = get_vector_store()
        return {"new": 0, "skipped": skipped, "total_chunks": collection.count()}

    documents = SimpleDirectoryReader(input_files=[str(p) for p in to_index]).load_data()
    total = _index_documents(documents, show_progress=show_progress)

    for key, h in new_files:
        sync_log[key] = h
    save_sync_log(sync_log)

    return {"new": len(to_index), "skipped": skipped, "total_chunks": total}


def sync_folder(folder_path: str | Path, recursive: bool = True) -> dict:
    """
    Sync an entire folder. Only indexes new or changed files.
    This is the main function for 'connect your knowledge base'.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    SUPPORTED = {".pdf", ".md", ".txt"}
    if recursive:
        files = [f for f in folder.rglob("*") if f.suffix.lower() in SUPPORTED]
    else:
        files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED]

    # Save connected folder to config
    settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    folder_config = settings.CHROMA_PERSIST_DIR / ".connected_folder.txt"
    folder_config.write_text(str(folder), encoding="utf-8")

    return ingest_files(files, show_progress=True)


def get_connected_folder() -> Optional[str]:
    """Returns the last connected folder path, if any."""
    folder_config = settings.CHROMA_PERSIST_DIR / ".connected_folder.txt"
    if folder_config.exists():
        return folder_config.read_text(encoding="utf-8").strip()
    return None


def get_index_stats() -> dict:
    """Summary of what's currently indexed."""
    try:
        collection, _ = get_vector_store()
        count = collection.count()
        if count == 0:
            return {"total_chunks": 0, "documents": [], "connected_folder": get_connected_folder()}

        results = collection.get(include=["metadatas"])
        files = sorted({
            m.get("file_name") or Path(m.get("file_path", "unknown")).name
            for m in results["metadatas"]
        })

        sync_log = load_sync_log()

        return {
            "total_chunks": count,
            "total_documents": len(files),
            "documents": files,
            "connected_folder": get_connected_folder(),
            "indexed_files": len(sync_log),
        }
    except Exception:
        return {"total_chunks": 0, "documents": [], "connected_folder": None}


def ingest_single_file(file_path: str | Path) -> dict:
    """Ingest a single file. Convenience wrapper."""
    return ingest_files([Path(file_path)])
