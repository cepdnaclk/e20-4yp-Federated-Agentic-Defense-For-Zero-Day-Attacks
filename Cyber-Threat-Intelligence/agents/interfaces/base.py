"""
Abstract interfaces for external dependencies.

This module provides abstract base classes for LLM and Vector Database
integrations, following dependency injection principles for testability
and flexibility.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievedContext:
    """
    Represents a retrieved document from the vector database.
    
    Attributes:
        content: The document text content.
        metadata: Associated metadata (source, category, etc.).
        similarity_score: Similarity score to the query.
        document_id: Unique identifier for the document.
    """
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    document_id: Optional[str] = None


@dataclass
class LLMResponse:
    """
    Response from an LLM.
    
    Attributes:
        content: Generated text content.
        model: Model name used for generation.
        tokens_used: Number of tokens consumed.
        metadata: Additional response metadata.
    """
    content: str
    model: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorDBInterface(ABC):
    """
    Abstract interface for vector database operations.
    
    Implementations should provide similarity search capabilities
    for retrieving contextually similar attack information.
    
    Example implementations:
        - FAISSVectorDB
        - ChromaVectorDB
        - PineconeVectorDB
    """
    
    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Adds documents to the vector database.
        
        Args:
            texts: List of document texts to add.
            metadatas: Optional metadata for each document.
            ids: Optional unique IDs for documents.
        
        Returns:
            List of document IDs.
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Searches for similar documents.
        
        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter: Optional metadata filter.
        
        Returns:
            List of RetrievedContext objects.
        """
        pass
    
    @abstractmethod
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Searches using a pre-computed embedding vector.
        
        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter.
        
        Returns:
            List of RetrievedContext objects.
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Deletes documents by IDs."""
        pass
    
    @abstractmethod
    def persist(self) -> None:
        """Persists the database to disk."""
        pass
    
    @property
    @abstractmethod
    def count(self) -> int:
        """Returns the number of documents in the database."""
        pass


class LLMInterface(ABC):
    """
    Abstract interface for Large Language Model operations.
    
    Implementations should provide text generation capabilities
    for reasoning about threats.
    
    Example implementations:
        - OllamaLLM (local Llama, Mistral, etc.)
        - OpenAILLM
        - HuggingFaceLLM
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates text response from the LLM.
        
        Args:
            prompt: User prompt/query.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model-specific parameters.
        
        Returns:
            LLMResponse with generated content.
        """
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates response using retrieved context (RAG).
        
        Args:
            query: User query.
            context: List of context documents.
            system_prompt: Optional system instructions.
            **kwargs: Additional parameters.
        
        Returns:
            LLMResponse with generated content.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model name."""
        pass


class EmbeddingInterface(ABC):
    """
    Abstract interface for text embedding generation.
    
    Used for converting text to vectors for similarity search.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generates embedding for a single text.
        
        Args:
            text: Input text.
        
        Returns:
            Embedding vector.
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for multiple texts.
        
        Args:
            texts: List of input texts.
        
        Returns:
            List of embedding vectors.
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding dimension."""
        pass
