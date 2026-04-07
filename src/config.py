"""Application configuration for MedAssist AI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"
EMBEDDING_DIMENSION = 384
TOP_K_RESULTS = 3
CONFIDENCE_THRESHOLD = 0.3
MEMORY_WINDOW = 5
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.1


@dataclass(frozen=True)
class Settings:
    """Environment-backed application settings."""

    pinecone_api_key: str
    pinecone_index_name: str
    ollama_model: str
    ollama_base_url: str


def get_settings() -> Settings:
    """Load required settings from environment variables."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", LLM_MODEL).strip()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).strip()

    missing = [
        name
        for name, value in (
            ("PINECONE_API_KEY", pinecone_api_key),
            ("PINECONE_INDEX_NAME", pinecone_index_name),
            ("OLLAMA_MODEL", ollama_model),
            ("OLLAMA_BASE_URL", ollama_base_url),
        )
        if not value
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing required environment variables: {missing_text}. "
            "Please configure them in your environment or .env file."
        )

    return Settings(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
    )
