"""Streamlit frontend for MedAssist AI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.chat_engine import ChatEngine
from src.logger import QueryAnalytics, setup_logger
from src.pdf_processor import load_uploaded_pdf
from src.text_splitter import get_text_chunks
from src.vector_store import upsert_documents

logger = setup_logger(__name__)

st.set_page_config(page_title="MedAssist AI", page_icon="🏥", layout="wide")


def initialize_session_state() -> None:
    """Initialize Streamlit session state objects once."""
    if "chat_engine" not in st.session_state:
        try:
            st.session_state.chat_engine = ChatEngine()
            st.session_state.engine_init_error = None
        except Exception as exc:
            logger.exception("Failed to initialize chat engine: %s", exc)
            st.session_state.chat_engine = None
            st.session_state.engine_init_error = str(exc)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = []
    if "analytics" not in st.session_state:
        st.session_state.analytics = (
            st.session_state.chat_engine.analytics
            if st.session_state.chat_engine is not None
            else QueryAnalytics()
        )


def render_sidebar(analytics: QueryAnalytics) -> None:
    """Render the application sidebar."""
    with st.sidebar:
        st.title("MedAssist AI")
        st.caption(
            "Upload medical PDFs, index them into Pinecone, and ask evidence-backed questions."
        )

        uploaded_files = st.file_uploader(
            "Upload medical PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Process & Index PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF before indexing.")
            else:
                progress_bar = st.progress(0, text="Preparing documents...")
                all_documents = []

                for idx, uploaded_file in enumerate(uploaded_files, start=1):
                    file_bytes = uploaded_file.getvalue()
                    documents = load_uploaded_pdf(file_bytes, uploaded_file.name)
                    all_documents.extend(documents)
                    progress_bar.progress(
                        int((idx / max(len(uploaded_files), 1)) * 40),
                        text=f"Extracted {uploaded_file.name}",
                    )

                chunks = get_text_chunks(all_documents)
                progress_bar.progress(65, text="Chunking complete. Uploading vectors...")

                if not chunks:
                    st.error("No extractable content found in the uploaded PDFs.")
                elif st.session_state.chat_engine is None:
                    st.error(
                        "The chat engine is not initialized. Please configure your "
                        "environment variables and refresh the app."
                    )
                else:
                    try:
                        upsert_documents(st.session_state.chat_engine.vector_store, chunks)
                        st.session_state.processed_pdfs = sorted(
                            {uploaded_file.name for uploaded_file in uploaded_files}
                        )
                        progress_bar.progress(100, text="Indexing complete.")
                        st.success("PDFs processed and indexed successfully.")
                    except Exception as exc:
                        logger.exception("Failed to upsert uploaded PDFs: %s", exc)
                        st.error(f"Failed to index PDFs: {exc}")

        st.subheader("Processed PDFs")
        if st.session_state.processed_pdfs:
            for pdf_name in st.session_state.processed_pdfs:
                st.write(f"- {pdf_name}")
        else:
            st.caption("No PDFs processed in this session yet.")

        stats = analytics.get_session_stats()
        st.subheader("Session Statistics")
        st.metric("Total Queries", stats["total_queries"])
        st.metric("Avg Response Time (ms)", stats["avg_response_time_ms"])
        st.metric("Avg Confidence", stats["avg_confidence"])

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chat_engine is not None:
                st.session_state.chat_engine.chain.memory.clear()
            st.success("Chat history cleared.")


def render_chat() -> None:
    """Render the main chat experience."""
    st.title("MedAssist AI")
    st.write(
        "Ask medical questions grounded in your uploaded PDF knowledge base."
    )

    if st.session_state.chat_engine is None:
        st.error(
            "Chat engine initialization failed. Check your `.env` configuration for "
            "`PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `OLLAMA_MODEL`, and "
            "`OLLAMA_BASE_URL`."
        )
        if st.session_state.engine_init_error:
            st.caption(st.session_state.engine_init_error)
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources & Confidence"):
                    for source, score in zip(
                        message["sources"], message["confidence_scores"]
                    ):
                        st.write(
                            f"File: {source['file']} | Page: {source['page']} | "
                            f"Confidence: {score}"
                        )

    if not st.session_state.chat_engine.has_indexed_documents():
        st.info(
            "No PDFs are indexed yet. Upload and process medical PDFs from the sidebar to begin."
        )
        return

    prompt = st.chat_input("Ask a medical question...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_engine.ask(prompt)
            except Exception as exc:
                logger.exception("Chat request failed: %s", exc)
                st.error(f"Something went wrong while answering your question: {exc}")
                return

            st.markdown(response["answer"])
            if response["sources"]:
                with st.expander("📚 Sources & Confidence"):
                    for source, score in zip(
                        response["sources"], response["confidence_scores"]
                    ):
                        st.write(
                            f"File: {source['file']} | Page: {source['page']} | "
                            f"Confidence: {score}"
                        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "confidence_scores": response["confidence_scores"],
        }
    )


def main() -> None:
    """Application entry point."""
    Path("data/medical_pdfs").mkdir(parents=True, exist_ok=True)
    initialize_session_state()
    render_sidebar(st.session_state.analytics)
    render_chat()


if __name__ == "__main__":
    main()
