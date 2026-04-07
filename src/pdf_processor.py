"""PDF ingestion helpers for MedAssist AI."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PyPDF2 import PdfReader
from langchain_core.documents import Document

from src.logger import setup_logger

logger = setup_logger(__name__)


def _extract_documents_from_reader(
    file_stream: BinaryIO, filename: str
) -> list[Document]:
    """Read a PDF stream and return page-level LangChain documents."""
    documents: list[Document] = []
    try:
        reader = PdfReader(file_stream)
    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", filename, exc)
        return documents

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Skipping unreadable page %s in '%s': %s",
                page_index,
                filename,
                exc,
            )
            continue

        cleaned_text = page_text.strip()
        if not cleaned_text:
            logger.info("Skipping empty page %s in '%s'.", page_index, filename)
            continue

        documents.append(
            Document(
                page_content=cleaned_text,
                metadata={"source": filename, "page": page_index},
            )
        )

    if not documents:
        logger.warning("No extractable text found in '%s'.", filename)
    return documents


def load_pdfs_from_directory(dir_path: str) -> list[Document]:
    """Load every PDF from a directory into LangChain documents."""
    documents: list[Document] = []
    pdf_dir = Path(dir_path)

    if not pdf_dir.exists():
        logger.warning("PDF directory does not exist: %s", pdf_dir)
        return documents

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            with pdf_path.open("rb") as pdf_file:
                extracted = _extract_documents_from_reader(pdf_file, pdf_path.name)
                documents.extend(extracted)
                logger.info(
                    "Loaded %s documents from '%s'.", len(extracted), pdf_path.name
                )
        except Exception as exc:
            logger.error("Failed to process '%s': %s", pdf_path.name, exc)

    return documents


def load_uploaded_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """Load a user-uploaded PDF into LangChain documents."""
    if not file_bytes:
        logger.warning("Uploaded file '%s' is empty.", filename)
        return []

    return _extract_documents_from_reader(BytesIO(file_bytes), filename)

