"""RAG chain construction for MedAssist AI."""

from __future__ import annotations

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from src.config import LLM_TEMPERATURE, MEMORY_WINDOW, get_settings
from src.logger import setup_logger

logger = setup_logger(__name__)

PROMPT_TEMPLATE = """You are MedAssist AI, a helpful medical knowledge assistant. Answer the question based ONLY on the provided context. If the context doesn't contain enough information to answer, clearly state that you don't have sufficient information. Always be accurate and cite which part of the context you're using.

Context: {context}
Chat History: {chat_history}
Question: {question}

Helpful Answer:"""


def get_llm() -> ChatOllama:
    """Initialize the Ollama chat model."""
    settings = get_settings()
    logger.info(
        "Initializing ChatOllama model '%s' via '%s'.",
        settings.ollama_model,
        settings.ollama_base_url,
    )
    return ChatOllama(
        model=settings.ollama_model,
        temperature=LLM_TEMPERATURE,
        base_url=settings.ollama_base_url,
    )


def build_rag_chain(retriever, llm: ChatOllama) -> ConversationalRetrievalChain:
    """Build and return the conversational RAG chain."""
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        input_key="question",
        return_messages=False,
        k=MEMORY_WINDOW,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"],
    )

    logger.info("Building conversational retrieval chain.")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        rephrase_question=False,
    )
