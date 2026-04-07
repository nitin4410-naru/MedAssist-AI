"""Chunking helpers for MedAssist AI."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.logger import setup_logger

logger = setup_logger(__name__)


def get_text_chunks(documents: list[Document]) -> list[Document]:
    """Split documents into metadata-preserving chunks."""
    if not documents:
        logger.warning("No documents received for splitting.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %s documents into %s chunks.", len(documents), len(chunks))
    return chunks

