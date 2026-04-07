"""Embedding model management for MedAssist AI."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME
from src.logger import setup_logger

logger = setup_logger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize and return the HuggingFace embedding model."""
    logger.info("Initializing embedding model '%s'.", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

