"""
Config — reads from .env file (or environment variables).
One place for all settings. Never hardcode keys in code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama (local, free, no API key needed)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "nomic-embed-text"   # embeddings model
    LLM_MODEL: str = "qwen2.5:3b"          # fast local model (swap for 7b if you want quality)

    # ChromaDB
    CHROMA_PERSIST_DIR: Path = Path("data/processed")
    COLLECTION_NAME: str = "knowledge_base"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere instead of creating new instances
settings = Settings()
