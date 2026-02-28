# agents/A3_federation_agent/rag_updater.py
"""
RAG Update Utility for Federated Signature Injection

This module provides functions to inject federated zero-day signatures into 
the LLM agents' RAG vector stores, enabling real-time knowledge sharing.

When a new signature is received from the federation, this module:
1. Converts the signature to a Document format
2. Adds it to the triage agent's FAISS vector store
3. Optionally adds to suspicious agent's vector store
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Shared embedding model (must match agent embedding models)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model = None


def get_embedding_model():
    """Get or create shared embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )
    return _embedding_model


def signature_to_document(signature: Dict) -> Document:
    """
    Convert a federated signature to a LangChain Document for RAG injection.
    
    Args:
        signature: Dict with id, attack_description, confidence, metadata, etc.
    
    Returns:
        Document with formatted content and metadata
    """
    metadata = signature.get("metadata", {})
    
    # Format the signature as informative text for the LLM
    content_parts = [
        f"FEDERATED ATTACK SIGNATURE [{signature.get('id', 'unknown')}]",
        f"Category: {metadata.get('potential_category', 'Unknown')}",
        f"Description: {signature.get('attack_description', 'No description available')}",
        f"Confidence: {signature.get('confidence', 0.0):.1%}",
        f"Status: {signature.get('state', 'Candidate')}",
    ]
    
    # Add distinguishing features if available
    features = metadata.get("distinguishing_features", [])
    if features:
        content_parts.append(f"Key Indicators: {', '.join(features)}")
    
    # Add network context if available
    if metadata.get("protocol"):
        content_parts.append(f"Protocol: {metadata.get('protocol')}")
    if metadata.get("service"):
        content_parts.append(f"Service: {metadata.get('service')}")
    
    # Add source info
    content_parts.append(f"Source: Federated Agent {signature.get('source_agent', 'unknown')}")
    
    content = "\n".join(content_parts)
    
    doc_metadata = {
        "type": "federated_signature",
        "signature_id": signature.get("id"),
        "source_agent": signature.get("source_agent"),
        "confidence": signature.get("confidence", 0.0),
        "state": signature.get("state"),
        "potential_category": metadata.get("potential_category"),
        "timestamp": signature.get("timestamp", datetime.utcnow().isoformat())
    }
    
    return Document(page_content=content, metadata=doc_metadata)


def inject_signatures_to_vector_store(
    vector_store,
    signatures: List[Dict],
    deduplicate: bool = True
) -> int:
    """
    Inject federated signatures into an existing FAISS vector store.
    
    Args:
        vector_store: FAISS vector store instance
        signatures: List of signature dicts from federation
        deduplicate: If True, skip signatures already in the store
    
    Returns:
        Number of signatures added
    """
    if not signatures:
        return 0
    
    documents = []
    existing_ids = set()
    
    # Try to get existing signature IDs (for deduplication)
    if deduplicate:
        try:
            # Check docstore for existing federated signatures
            for doc_id, doc in vector_store.docstore._dict.items():
                if hasattr(doc, 'metadata') and doc.metadata.get('type') == 'federated_signature':
                    sig_id = doc.metadata.get('signature_id')
                    if sig_id:
                        existing_ids.add(sig_id)
        except Exception:
            pass  # If we can't check, just add all
    
    for sig in signatures:
        sig_id = sig.get('id')
        if deduplicate and sig_id in existing_ids:
            continue
        documents.append(signature_to_document(sig))
    
    if documents:
        vector_store.add_documents(documents)
        print(f"[RAGUpdater] Injected {len(documents)} federated signatures")
    
    return len(documents)


def update_triage_agent_rag(signatures: List[Dict]) -> int:
    """
    Update the triage agent's RAG with new federated signatures.
    
    Args:
        signatures: List of signatures from federation
    
    Returns:
        Number of signatures added
    """
    try:
        import agents.A1_triage_agent.triage_agent as triage_agent
        
        # Ensure KB is initialized
        triage_agent.initialize_triage_kb()
        
        if triage_agent.triage_vector_store is None:
            print("[RAGUpdater] Warning: Triage vector store not initialized")
            return 0
        
        count = inject_signatures_to_vector_store(
            triage_agent.triage_vector_store,
            signatures,
            deduplicate=True
        )
        
        print(f"[RAGUpdater] Updated triage agent RAG with {count} signatures")
        return count
        
    except Exception as e:
        print(f"[RAGUpdater] Error updating triage RAG: {e}")
        return 0


def update_suspicious_agent_rag(signatures: List[Dict]) -> int:
    """
    Update the suspicious agent's RAG with new federated signatures.
    
    Args:
        signatures: List of signatures from federation
    
    Returns:
        Number of signatures added
    """
    try:
        import agents.A2_suspicious_agent.suspicious_agent as suspicious_agent
        
        # Ensure KB is initialized
        suspicious_agent.initialize_kb()
        
        if suspicious_agent.suspicious_vector_store is None:
            print("[RAGUpdater] Warning: Suspicious vector store not initialized")
            return 0
        
        count = inject_signatures_to_vector_store(
            suspicious_agent.suspicious_vector_store,
            signatures,
            deduplicate=True
        )
        
        print(f"[RAGUpdater] Updated suspicious agent RAG with {count} signatures")
        return count
        
    except Exception as e:
        print(f"[RAGUpdater] Error updating suspicious RAG: {e}")
        return 0


def update_all_agent_rags(signatures: List[Dict]) -> Dict[str, int]:
    """
    Update all agent RAGs with new federated signatures.
    
    This is the main callback to be registered with kb_sync_daemon.
    
    Args:
        signatures: List of signatures from federation
    
    Returns:
        Dict with counts for each agent
    """
    results = {
        "triage_agent": 0,
        "suspicious_agent": 0,
        "total": 0
    }
    
    if not signatures:
        return results
    
    print(f"[RAGUpdater] Updating all agent RAGs with {len(signatures)} new signatures")
    
    results["triage_agent"] = update_triage_agent_rag(signatures)
    results["suspicious_agent"] = update_suspicious_agent_rag(signatures)
    results["total"] = results["triage_agent"] + results["suspicious_agent"]
    
    return results


# Export the callback function for kb_sync_daemon
def rag_update_callback(signatures: List[Dict]):
    """
    Callback for kb_sync_daemon to update all RAGs when new signatures arrive.
    """
    return update_all_agent_rags(signatures)
