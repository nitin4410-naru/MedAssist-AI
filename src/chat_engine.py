"""Chat orchestration for MedAssist AI."""

from __future__ import annotations

import time
from typing import Any

from pinecone import Pinecone

from src.chain_builder import build_rag_chain, get_llm
from src.config import CONFIDENCE_THRESHOLD, TOP_K_RESULTS, get_settings
from src.embedding_manager import get_embedding_model
from src.logger import QueryAnalytics, setup_logger
from src.retriever import filter_by_confidence, get_retriever, retrieve_with_scores
from src.vector_store import (
    create_index_if_not_exists,
    get_vector_store,
    initialize_pinecone,
)

logger = setup_logger(__name__)


class ChatEngine:
    """Coordinates retrieval, generation, and analytics for chat requests."""

    def __init__(self) -> None:
        """Initialize the full MedAssist AI stack."""
        self.settings = get_settings()
        self.analytics = QueryAnalytics()
        self.embedding_model = get_embedding_model()
        self.pc_client: Pinecone = initialize_pinecone()
        create_index_if_not_exists(
            self.pc_client, self.settings.pinecone_index_name
        )
        self.vector_store = get_vector_store(
            self.pc_client,
            self.settings.pinecone_index_name,
            self.embedding_model,
        )
        self.retriever = get_retriever(self.vector_store, top_k=TOP_K_RESULTS)
        self.llm = get_llm()
        self._verify_ollama_connection()
        self.chain = build_rag_chain(self.retriever, self.llm)

    def _verify_ollama_connection(self) -> None:
        """Validate that the configured Ollama server and model are reachable."""
        try:
            self.llm.invoke("ping")
        except Exception as exc:
            logger.exception("Ollama connectivity check failed: %s", exc)
            raise RuntimeError(
                "Failed to reach the local Ollama server. Make sure Ollama is "
                f"running and the model '{self.settings.ollama_model}' is pulled "
                f"at '{self.settings.ollama_base_url}'."
            ) from exc

    def has_indexed_documents(self) -> bool:
        """Return whether the Pinecone index currently contains vectors."""
        stats = self.pc_client.Index(
            self.settings.pinecone_index_name
        ).describe_index_stats()
        total_vectors = int(stats.get("total_vector_count", 0))
        return total_vectors > 0

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a question using retrieved medical document context."""
        start_time = time.perf_counter()

        try:
            scored_results = retrieve_with_scores(
                self.vector_store, question, top_k=TOP_K_RESULTS
            )
            filtered_results = filter_by_confidence(
                scored_results, threshold=CONFIDENCE_THRESHOLD
            )
        except Exception as exc:
            logger.exception("Retrieval failed for question '%s': %s", question, exc)
            raise RuntimeError(f"Failed to retrieve relevant context: {exc}") from exc

        response_time_ms = (time.perf_counter() - start_time) * 1000

        if not filtered_results:
            sources: list[dict[str, Any]] = []
            self.analytics.log_query(
                question=question,
                response_time=response_time_ms,
                top_score=0.0,
                sources=sources,
            )
            return {
                "answer": (
                    "I don't have sufficient information in the indexed medical "
                    "documents to answer that question."
                ),
                "sources": sources,
                "confidence_scores": [],
                "response_time_ms": round(response_time_ms, 2),
            }

        try:
            chain_response = self.chain.invoke({"question": question})
        except Exception as exc:
            logger.exception("Generation failed for question '%s': %s", question, exc)
            raise RuntimeError(f"Failed to generate response: {exc}") from exc

        response_time_ms = (time.perf_counter() - start_time) * 1000
        sources = [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "page": int(doc.metadata.get("page", 0)),
            }
            for doc, _score in filtered_results
        ]
        confidence_scores = [round(score, 3) for _doc, score in filtered_results]
        top_score = max(confidence_scores, default=0.0)

        self.analytics.log_query(
            question=question,
            response_time=response_time_ms,
            top_score=top_score,
            sources=sources,
        )

        answer = chain_response.get("answer", "").strip()
        if not answer:
            answer = (
                "I don't have sufficient information in the indexed medical "
                "documents to answer that question."
            )

        return {
            "answer": answer,
            "sources": sources,
            "confidence_scores": confidence_scores,
            "response_time_ms": round(response_time_ms, 2),
        }
