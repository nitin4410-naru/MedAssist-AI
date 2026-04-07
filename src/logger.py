"""Logging utilities and query analytics for MedAssist AI."""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import Any

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"


def setup_logger(name: str = "medassist_ai") -> logging.Logger:
    """Configure and return an application logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class QueryAnalytics:
    """Tracks query-level analytics for the active session."""

    def __init__(self) -> None:
        """Initialize in-memory analytics storage."""
        self._queries: list[dict[str, Any]] = []

    def log_query(
        self,
        question: str,
        response_time: float,
        top_score: float,
        sources: list[dict[str, Any]],
    ) -> None:
        """Persist query analytics for the current session."""
        self._queries.append(
            {
                "question": question,
                "response_time_ms": response_time,
                "top_score": top_score,
                "sources": sources,
            }
        )

    def get_session_stats(self) -> dict[str, float | int]:
        """Return aggregate analytics for the current session."""
        if not self._queries:
            return {
                "total_queries": 0,
                "avg_response_time_ms": 0.0,
                "avg_confidence": 0.0,
            }

        return {
            "total_queries": len(self._queries),
            "avg_response_time_ms": round(
                mean(item["response_time_ms"] for item in self._queries), 2
            ),
            "avg_confidence": round(
                mean(item["top_score"] for item in self._queries), 3
            ),
        }

