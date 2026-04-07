"""Setup configuration for MedAssist AI."""

from setuptools import find_packages, setup


setup(
    name="medassist-ai",
    version="1.0.0",
    description="Medical RAG chatbot powered by Ollama, LangChain, and Pinecone.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
