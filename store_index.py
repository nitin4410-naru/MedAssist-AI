"""Batch indexing script for MedAssist AI."""

from __future__ import annotations

from pathlib import Path

from src.chat_engine import ChatEngine
from src.pdf_processor import load_pdfs_from_directory
from src.text_splitter import get_text_chunks
from src.vector_store import upsert_documents


def main() -> None:
    """Load PDFs from disk and upsert them into Pinecone."""
    pdf_directory = Path("data/medical_pdfs")
    print(f"Looking for PDFs in: {pdf_directory.resolve()}")

    try:
        documents = load_pdfs_from_directory(str(pdf_directory))
        if not documents:
            print("No PDF documents found or no extractable text available.")
            return

        print(f"Loaded {len(documents)} page-level documents.")
        chunks = get_text_chunks(documents)
        if not chunks:
            print("Chunking produced no output. Nothing to index.")
            return

        print(f"Generated {len(chunks)} chunks. Initializing chat engine...")
        engine = ChatEngine()
        upsert_documents(engine.vector_store, chunks)
        print("Indexing complete.")
    except Exception as exc:
        print(f"Batch indexing failed: {exc}")


if __name__ == "__main__":
    main()
