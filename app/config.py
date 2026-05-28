"""
Config — reads from .env file (or environment variables).
One place for all settings. Never hardcode keys in code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM provider: "openai" or "ollama"
    LLM_PROVIDER: str = "ollama"

    # OpenAI (cloud, fast, best quality)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"       # fast + cheap + smart

    # Ollama (local, free, no API key needed)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "nomic-embed-text"   # fallback when no OpenAI key
    LLM_MODEL: str = "qwen2.5:3b"          # fallback if no API key

    # OpenAI embeddings (much better for Spanish/multilingual)
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

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
