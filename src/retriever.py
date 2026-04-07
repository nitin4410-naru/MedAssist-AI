"""Retrieval helpers for MedAssist AI."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from src.config import CONFIDENCE_THRESHOLD, TOP_K_RESULTS
from src.logger import setup_logger

logger = setup_logger(__name__)


def get_retriever(
    vector_store: PineconeVectorStore,
    top_k: int = TOP_K_RESULTS,
):
    """Create a retriever from the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def retrieve_with_scores(
    vector_store: PineconeVectorStore,
    query: str,
    top_k: int = TOP_K_RESULTS,
) -> list[tuple[Document, float]]:
    """Retrieve relevant documents and their similarity scores."""
    logger.info("Running similarity search for query: %s", query)
    return vector_store.similarity_search_with_relevance_scores(query, k=top_k)


def filter_by_confidence(
    results: list[tuple[Document, float]],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[tuple[Document, float]]:
    """Filter retrieval results by minimum relevance threshold."""
    filtered = [item for item in results if item[1] >= threshold]
    logger.info(
        "Filtered retrieval results from %s to %s using threshold %.2f.",
        len(results),
        len(filtered),
        threshold,
    )
    return filtered

