"""Pinecone vector store integration for MedAssist AI."""

from __future__ import annotations

import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import EMBEDDING_DIMENSION, get_settings
from src.logger import setup_logger

logger = setup_logger(__name__)


def initialize_pinecone() -> Pinecone:
    """Initialize and return a Pinecone client."""
    settings = get_settings()
    logger.info("Initializing Pinecone client.")
    return Pinecone(api_key=settings.pinecone_api_key)


def create_index_if_not_exists(
    pc_client: Pinecone,
    index_name: str,
    dimension: int = EMBEDDING_DIMENSION,
) -> None:
    """Create the Pinecone index if it does not already exist."""
    index_list = pc_client.list_indexes()
    if hasattr(index_list, "names"):
        existing_indexes = set(index_list.names())
    else:
        existing_indexes = {
            index_info.get("name")
            for index_info in index_list
            if isinstance(index_info, dict)
        }
    if index_name in existing_indexes:
        logger.info("Pinecone index '%s' already exists.", index_name)
        return

    logger.info("Creating Pinecone index '%s'.", index_name)
    pc_client.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    while not pc_client.describe_index(index_name).status["ready"]:
        logger.info("Waiting for Pinecone index '%s' to become ready...", index_name)
        time.sleep(1)


def get_vector_store(
    pc_client: Pinecone,
    index_name: str,
    embedding_model: HuggingFaceEmbeddings,
) -> PineconeVectorStore:
    """Return a LangChain-compatible Pinecone vector store."""
    index = pc_client.Index(index_name)
    logger.info("Connecting LangChain vector store to Pinecone index '%s'.", index_name)
    return PineconeVectorStore(index=index, embedding=embedding_model)


def upsert_documents(
    vector_store: PineconeVectorStore,
    documents: list[Document],
) -> list[str]:
    """Upsert document chunks into Pinecone."""
    if not documents:
        logger.warning("No documents supplied for vector upsert.")
        return []

    logger.info("Upserting %s document chunks into Pinecone.", len(documents))
    return vector_store.add_documents(documents)
