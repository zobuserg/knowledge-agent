"""
Config — reads from .env file (or environment variables).
One place for all settings. Never hardcode keys in code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str = ""

    # Ollama (local, free)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "nomic-embed-text"   # fast, good quality, free via Ollama

    # LLM
    LLM_MODEL: str = "claude-sonnet-4-5"

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
