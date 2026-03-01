"""
FAISS Vector Database Implementation.

This module provides a FAISS-backed vector database for storing
and retrieving attack knowledge using similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from agents.interfaces.base import (
    VectorDBInterface, 
    RetrievedContext,
    EmbeddingInterface,
)

logger = logging.getLogger(__name__)


class FAISSVectorDB(VectorDBInterface):
    """
    FAISS-based vector database implementation using LangChain.
    
    This implementation uses LangChain's FAISS wrapper for efficient
    similarity search over attack knowledge documents.
    
    Features:
        - Efficient similarity search with FAISS
        - Metadata filtering
        - Persistence to disk
        - Multiple embedding model support
    
    Example:
        >>> from langchain_community.embeddings import HuggingFaceEmbeddings
        >>> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        >>> db = FAISSVectorDB(embeddings)
        >>> db.add_documents(["DoS attacks flood networks..."])
        >>> results = db.similarity_search("denial of service", k=3)
    """
    
    def __init__(
        self,
        embedding_function: Any,
        persist_directory: Optional[str] = None,
    ):
        """
        Initializes the FAISS vector database.
        
        Args:
            embedding_function: LangChain-compatible embedding function.
            persist_directory: Optional directory for persistence.
        """
        self._embedding_function = embedding_function
        self._persist_directory = Path(persist_directory) if persist_directory else None
        self._vectorstore = None
        self._documents: List[Dict[str, Any]] = []
        self._doc_count = 0
        
        # Try to load existing database
        if self._persist_directory and self._persist_directory.exists():
            self._load()
        
        logger.info("FAISSVectorDB initialized")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Adds documents to the FAISS index.
        
        Args:
            texts: List of document texts.
            metadatas: Optional metadata for each document.
            ids: Optional unique IDs (generated if not provided).
        
        Returns:
            List of document IDs.
        """
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{self._doc_count + i}" for i in range(len(texts))]
        
        # Create metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add IDs to metadata
        for i, meta in enumerate(metadatas):
            meta["doc_id"] = ids[i]
        
        # Create LangChain documents
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        
        # Add to vectorstore
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(
                documents, self._embedding_function
            )
        else:
            self._vectorstore.add_documents(documents)
        
        self._doc_count += len(texts)
        
        # Store documents for reference
        self._documents.extend([
            {"id": id_, "text": text, "metadata": meta}
            for id_, text, meta in zip(ids, texts, metadatas)
        ])
        
        logger.info("Added %d documents to FAISS index", len(texts))
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Searches for similar documents.
        
        Args:
            query: Query text.
            k: Number of results.
            filter: Optional metadata filter.
        
        Returns:
            List of RetrievedContext objects.
        """
        if self._vectorstore is None:
            logger.warning("Vector store is empty")
            return []
        
        # Perform search with scores
        results = self._vectorstore.similarity_search_with_score(
            query, k=k, filter=filter
        )
        
        contexts = []
        for doc, score in results:
            contexts.append(RetrievedContext(
                content=doc.page_content,
                metadata=doc.metadata,
                similarity_score=float(1 - score),  # Convert distance to similarity
                document_id=doc.metadata.get("doc_id"),
            ))
        
        return contexts
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedContext]:
        """
        Searches using a pre-computed embedding.
        
        Args:
            embedding: Query embedding vector.
            k: Number of results.
            filter: Optional metadata filter.
        
        Returns:
            List of RetrievedContext objects.
        """
        if self._vectorstore is None:
            logger.warning("Vector store is empty")
            return []
        
        results = self._vectorstore.similarity_search_by_vector(
            embedding, k=k, filter=filter
        )
        
        contexts = []
        for doc in results:
            # Note: FAISS by vector doesn't return scores directly
            contexts.append(RetrievedContext(
                content=doc.page_content,
                metadata=doc.metadata,
                similarity_score=1.0,  # Unknown score
                document_id=doc.metadata.get("doc_id"),
            ))
        
        return contexts
    
    def delete(self, ids: List[str]) -> None:
        """
        Deletes documents by IDs.
        
        Note: FAISS doesn't support direct deletion. This recreates the index.
        """
        # Filter out deleted documents
        self._documents = [d for d in self._documents if d["id"] not in ids]
        
        # Recreate index if documents remain
        if self._documents:
            texts = [d["text"] for d in self._documents]
            metadatas = [d["metadata"] for d in self._documents]
            doc_ids = [d["id"] for d in self._documents]
            
            self._vectorstore = None
            self._doc_count = 0
            self.add_documents(texts, metadatas, doc_ids)
        else:
            self._vectorstore = None
            self._doc_count = 0
        
        logger.info("Deleted %d documents", len(ids))
    
    def persist(self) -> None:
        """Saves the database to disk."""
        if self._persist_directory is None:
            raise ValueError("No persist directory configured")
        
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._vectorstore:
            self._vectorstore.save_local(str(self._persist_directory / "faiss_index"))
        
        # Save metadata
        metadata = {
            "doc_count": self._doc_count,
            "documents": self._documents,
        }
        with open(self._persist_directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("FAISS database persisted to: %s", self._persist_directory)
    
    def _load(self) -> None:
        """Loads the database from disk."""
        from langchain_community.vectorstores import FAISS
        
        index_path = self._persist_directory / "faiss_index"
        metadata_path = self._persist_directory / "metadata.json"
        
        if index_path.exists():
            self._vectorstore = FAISS.load_local(
                str(index_path),
                self._embedding_function,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index from: %s", index_path)
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self._doc_count = metadata.get("doc_count", 0)
            self._documents = metadata.get("documents", [])
    
    @property
    def count(self) -> int:
        """Returns the number of documents."""
        return self._doc_count
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns all stored documents with metadata."""
        return self._documents.copy()


class HuggingFaceEmbedding(EmbeddingInterface):
    """
    HuggingFace embedding implementation.
    
    Uses sentence-transformers for efficient text embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.
        
        Args:
            model_name: HuggingFace model name.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded embedding model: %s (dim=%d)", model_name, self._dimension)
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )
    
    def embed_text(self, text: str) -> List[float]:
        """Generates embedding for a single text."""
        embedding = self._model.encode(text)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for multiple texts."""
        embeddings = self._model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        """Returns embedding dimension."""
        return self._dimension
