"""
Interfaces module for external dependencies.

Provides abstract interfaces and concrete implementations for:
- Vector databases (FAISS, Chroma)
- Large Language Models (Ollama, OpenAI)
- Embedding models
"""

from agents.interfaces.base import (
    VectorDBInterface,
    LLMInterface,
    EmbeddingInterface,
    RetrievedContext,
    LLMResponse,
)

from agents.interfaces.vector_db import (
    FAISSVectorDB,
    HuggingFaceEmbedding,
)

from agents.interfaces.llm import (
    OllamaLLM,
    MockLLM,
    OpenAILLM,
    GroqLLM,
)

__all__ = [
    # Abstract interfaces
    "VectorDBInterface",
    "LLMInterface", 
    "EmbeddingInterface",
    # Data classes
    "RetrievedContext",
    "LLMResponse",
    # Concrete implementations
    "FAISSVectorDB",
    "HuggingFaceEmbedding",
    "OllamaLLM",
    "MockLLM",
    "OpenAILLM",
    "GroqLLM",
]
