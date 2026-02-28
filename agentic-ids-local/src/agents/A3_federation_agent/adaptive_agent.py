# agents/A3_federation_agent/adaptive_agent.py
"""
AdaptiveRAG Agent - Handles zero-day attack classification and federated signature generation.

Key responsibilities:
1. Generate signatures from confirmed zero-day detections
2. Submit signatures to FL server immediately (N_min=1, no cross-validation)
3. Fetch and integrate federated signatures into local RAG context
4. Update LLM agent knowledge base with new attack patterns
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.A3_federation_agent.fl_client import FLClient
from agents.knowledge_base import LocalKnowledgeBase

load_dotenv()


# Configuration
FL_SERVER_URL = os.getenv("FL_SERVER_URL", "http://localhost:5000")
AGENT_ID = os.getenv("AGENT_ID", f"agent_{hashlib.md5(os.urandom(8)).hexdigest()[:8]}")
LOCAL_KB_PATH = Path(__file__).resolve().parent.parent / "local_knowledge_base.json"
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to generate signature


@dataclass
class ZeroDaySignature:
    """Represents a zero-day attack signature for federation."""
    signature_id: str
    embedding: List[float]  # Latent embedding from autoencoder bottleneck
    recon_error: float       # Reconstruction error score
    feature_vector: List[float]  # Original scaled features
    attack_description: str  # LLM-generated description
    confidence: float        # Detection confidence
    timestamp: str
    source_agent: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.signature_id,
            "embedding": self.embedding,
            "recon_error": self.recon_error,
            "feature_vector": self.feature_vector,
            "attack_description": self.attack_description,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "state": "Candidate",  # Single agent = direct candidate (no cross-validation)
            "metadata": self.metadata
        }


class AdaptiveAgentState:
    """State container for the AdaptiveRAG Agent."""
    def __init__(self):
        self.pending_signatures: List[ZeroDaySignature] = []
        self.submitted_signatures: List[str] = []
        self.federated_signatures: List[Dict] = []
        self.last_sync_version: int = 0
        self.sync_errors: List[str] = []


class ZeroDayAssessment(BaseModel):
    """LLM-structured output for zero-day assessment."""
    is_zero_day: bool = Field(description="True if this is a novel/unknown attack pattern")
    confidence: float = Field(description="Confidence score 0.0-1.0 for zero-day classification")
    attack_description: str = Field(description="Technical description of the attack pattern")
    distinguishing_features: List[str] = Field(description="Key features that make this pattern unique")
    potential_category: str = Field(description="Best guess category: Exploit, DoS, Backdoor, Worms, Shellcode, Generic")
    reasoning: str = Field(description="Explanation for classification decision")


class AdaptiveRAGAgent:
    """
    AdaptiveRAG Agent for zero-day handling and federated learning.
    
    This agent:
    1. Assesses anomalies marked as ZERO-DAY by triage agent
    2. Generates unique signatures using latent embeddings
    3. Immediately submits to FL server (single-agent sharing, N_min=1)
    4. Fetches federated signatures and updates local RAG context
    """

    def __init__(self, agent_id: str = None, fl_server_url: str = None):
        self.agent_id = agent_id or AGENT_ID
        self.fl_server_url = fl_server_url or FL_SERVER_URL
        self.state = AdaptiveAgentState()
        self.local_kb = LocalKnowledgeBase.load(str(LOCAL_KB_PATH))
        
        # Initialize FL client
        self.fl_client = FLClient(server_url=self.fl_server_url, timeout=10)
        
        # Initialize LLM for zero-day assessment
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        
        # Register with FL server
        self._register_agent()
        
        print(f"[AdaptiveRAG] Initialized agent: {self.agent_id}")

    def _register_agent(self):
        """Register this agent with the FL server."""
        result = self.fl_client.register(
            agent_id=self.agent_id,
            capabilities={
                "autoencoder": True,
                "llm_triage": True,
                "signature_generation": True
            }
        )
        if result:
            print(f"[AdaptiveRAG] Registered with FL server: {result}")
        else:
            print(f"[AdaptiveRAG] Warning: Could not register with FL server")

    def generate_signature_id(self, embedding: List[float], recon_error: float) -> str:
        """Generate unique signature ID from embedding hash."""
        data = json.dumps({
            "embedding": embedding[:10],  # First 10 dims for hash
            "recon_error": round(recon_error, 4),
            "agent": self.agent_id
        }, sort_keys=True)
        return f"sig_{hashlib.sha256(data.encode()).hexdigest()[:16]}"

    async def assess_zero_day(self, anomaly_data: Dict) -> ZeroDayAssessment:
        """
        Use LLM to assess if anomaly is truly a novel zero-day attack.
        
        Args:
            anomaly_data: Output from orchestrator containing:
                - latent_embedding: From autoencoder bottleneck
                - anomaly_score: Reconstruction error
                - feature_vector: Scaled input features
                - features: Raw network flow features
                - triage_result: Classification from triage agent
        """
        features = anomaly_data.get("features", {})
        anomaly_score = anomaly_data.get("anomaly_score", 0)
        triage_result = anomaly_data.get("triage_result", {})
        
        # Build context for LLM
        feature_summary = self._format_features_for_llm(features)
        
        system_prompt = """You are a network security expert specializing in zero-day attack detection.
Analyze the following anomaly data and determine if it represents a novel, previously unknown attack pattern.

Consider:
1. Does the pattern match known attack categories (Exploits, DoS, Reconnaissance, Backdoors, etc.)?
2. Are there unusual feature combinations that suggest a new attack technique?
3. Is the reconstruction error significantly high, indicating the autoencoder hasn't seen similar patterns?

Be conservative: only classify as zero-day if there's strong evidence of novelty."""

        user_prompt = f"""Analyze this anomaly detection result:

## Triage Classification
- Classification: {triage_result.get('classification', 'ZERO-DAY')}
- Triage Confidence: {triage_result.get('confidence', 'N/A')}
- Triage Reasoning: {triage_result.get('reasoning', 'N/A')}

## Anomaly Metrics
- Reconstruction Error (Anomaly Score): {anomaly_score:.6f}
- Above normal threshold: Yes (detected as anomaly)

## Network Flow Features
{feature_summary}

Based on this data, provide your zero-day assessment."""

        try:
            structured_llm = self.llm.with_structured_output(ZeroDayAssessment)
            result = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return result
        except Exception as e:
            print(f"[AdaptiveRAG] LLM assessment error: {e}")
            # Fallback: if LLM fails, assume it's a zero-day with moderate confidence
            return ZeroDayAssessment(
                is_zero_day=True,
                confidence=0.6,
                attack_description="Unknown attack pattern - LLM assessment failed",
                distinguishing_features=["high_reconstruction_error"],
                potential_category="Generic",
                reasoning="Fallback classification due to LLM error"
            )

    def _format_features_for_llm(self, features: Dict) -> str:
        """Format network flow features for LLM consumption."""
        key_features = [
            ("Source IP", features.get("srcip", "N/A")),
            ("Destination IP", features.get("dstip", "N/A")),
            ("Source Port", features.get("sport", "N/A")),
            ("Destination Port", features.get("dsport", "N/A")),
            ("Protocol", features.get("proto", "N/A")),
            ("Service", features.get("service", "-")),
            ("State", features.get("state", "N/A")),
            ("Duration", features.get("dur", "N/A")),
            ("Src Packets", features.get("spkts", "N/A")),
            ("Dst Packets", features.get("dpkts", "N/A")),
            ("Src Bytes", features.get("sbytes", "N/A")),
            ("Dst Bytes", features.get("dbytes", "N/A")),
            ("Src TTL", features.get("sttl", "N/A")),
            ("Dst TTL", features.get("dttl", "N/A")),
            ("Src Load", features.get("sload", "N/A")),
            ("Dst Load", features.get("dload", "N/A")),
            ("Src Mean Pkt Size", features.get("smeansz", "N/A")),
            ("Dst Mean Pkt Size", features.get("dmeansz", "N/A")),
            ("TCP RTT Syn-Ack", features.get("synack", "N/A")),
            ("TCP RTT Ack-Data", features.get("ackdat", "N/A")),
        ]
        return "\n".join([f"- {name}: {value}" for name, value in key_features])

    def create_signature(
        self,
        anomaly_data: Dict,
        assessment: ZeroDayAssessment
    ) -> ZeroDaySignature:
        """
        Create a shareable signature from confirmed zero-day detection.
        
        Args:
            anomaly_data: Contains latent_embedding, anomaly_score, features
            assessment: LLM zero-day assessment result
        """
        embedding = anomaly_data.get("latent_embedding", [])
        recon_error = anomaly_data.get("anomaly_score", 0.0)
        feature_vector = anomaly_data.get("feature_vector", [])
        features = anomaly_data.get("features", {})
        
        signature = ZeroDaySignature(
            signature_id=self.generate_signature_id(embedding, recon_error),
            embedding=embedding,
            recon_error=recon_error,
            feature_vector=feature_vector,
            attack_description=assessment.attack_description,
            confidence=assessment.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_agent=self.agent_id,
            metadata={
                "potential_category": assessment.potential_category,
                "distinguishing_features": assessment.distinguishing_features,
                "reasoning": assessment.reasoning,
                "source_ip": features.get("srcip"),
                "dest_ip": features.get("dstip"),
                "protocol": features.get("proto"),
                "service": features.get("service")
            }
        )
        
        self.state.pending_signatures.append(signature)
        print(f"[AdaptiveRAG] Created signature: {signature.signature_id}")
        return signature

    def submit_signature_to_federation(self, signature: ZeroDaySignature) -> bool:
        """
        Submit signature to FL server immediately (N_min=1, no cross-validation).
        
        This implements single-agent sharing: any agent that detects a new attack
        immediately shares it with all other agents through the federation server.
        """
        print(f"[AdaptiveRAG] Submitting signature to federation: {signature.signature_id}")
        
        result = self.fl_client.submit_update(
            agent_id=self.agent_id,
            weights=None,  # No model weights for signature-only update
            sample_count=1,
            anomaly_stats={
                "total_anomalies": 1,
                "zero_day_count": 1,
                "avg_recon_error": signature.recon_error
            },
            signatures=[{
                "id": signature.signature_id,
                "embedding": signature.embedding,
                "recon_error": signature.recon_error,
                "attack_description": signature.attack_description,
                "confidence": signature.confidence,
                "metadata": signature.metadata
            }],
            round_end=True  # Submit immediately, don't wait for round
        )
        
        if result:
            self.state.submitted_signatures.append(signature.signature_id)
            # Save to local KB
            self.local_kb.anomaly_signatures[signature.signature_id] = signature.to_dict()
            self.local_kb.save(str(LOCAL_KB_PATH))
            print(f"[AdaptiveRAG] Successfully submitted signature: {signature.signature_id}")
            return True
        else:
            self.state.sync_errors.append(f"Failed to submit {signature.signature_id}")
            print(f"[AdaptiveRAG] Failed to submit signature: {signature.signature_id}")
            return False

    def fetch_federated_signatures(self) -> List[Dict]:
        """
        Fetch new signatures from FL server and update local RAG context.
        
        Returns list of new signatures received from federation.
        """
        print(f"[AdaptiveRAG] Fetching federated signatures (since v{self.state.last_sync_version})")
        
        result = self.fl_client.fetch_signatures(since_version=self.state.last_sync_version)
        
        if not result:
            print("[AdaptiveRAG] No signature updates from federation")
            return []
        
        new_signatures = result.get("signatures", [])
        new_version = result.get("version", self.state.last_sync_version)
        
        # Filter out our own signatures
        external_signatures = [
            sig for sig in new_signatures
            if sig.get("source_agent") != self.agent_id
        ]
        
        if external_signatures:
            # Merge into local KB
            changes = self.local_kb.merge_updates({
                "signature_version": new_version,
                "signatures": external_signatures
            })
            self.local_kb.save(str(LOCAL_KB_PATH))
            
            print(f"[AdaptiveRAG] Merged {len(changes['added'])} new, {len(changes['updated'])} updated signatures")
            self.state.federated_signatures.extend(external_signatures)
        
        self.state.last_sync_version = new_version
        return external_signatures

    def get_rag_context_updates(self) -> List[Dict]:
        """
        Get signature updates formatted for RAG vector store injection.
        
        Returns list of documents to add to local LLM RAG context.
        """
        rag_documents = []
        
        for sig_id, sig_data in self.local_kb.anomaly_signatures.items():
            if sig_data.get("deprecated"):
                continue
            
            # Format as document for RAG embedding
            doc = {
                "id": sig_id,
                "content": self._format_signature_for_rag(sig_data),
                "metadata": {
                    "type": "federated_signature",
                    "source_agent": sig_data.get("source_agent", "unknown"),
                    "confidence": sig_data.get("confidence", 0.0),
                    "state": sig_data.get("state", "Candidate"),
                    "potential_category": sig_data.get("metadata", {}).get("potential_category", "Unknown")
                }
            }
            rag_documents.append(doc)
        
        return rag_documents

    def _format_signature_for_rag(self, sig_data: Dict) -> str:
        """Format signature as text for RAG vector store."""
        metadata = sig_data.get("metadata", {})
        
        text = f"""FEDERATED ZERO-DAY SIGNATURE
ID: {sig_data.get('id', 'unknown')}
Category: {metadata.get('potential_category', 'Unknown')}
Description: {sig_data.get('attack_description', 'No description')}
Confidence: {sig_data.get('confidence', 0.0):.2f}
Source Agent: {sig_data.get('source_agent', 'unknown')}
Distinguishing Features: {', '.join(metadata.get('distinguishing_features', []))}
Protocol: {metadata.get('protocol', 'N/A')}
Service: {metadata.get('service', 'N/A')}
Reconstruction Error: {sig_data.get('recon_error', 0.0):.6f}
State: {sig_data.get('state', 'Candidate')}
"""
        return text.strip()

    async def process_zero_day_anomaly(self, anomaly_data: Dict) -> Dict:
        """
        Main entry point: Process a zero-day classified anomaly.
        
        Flow:
        1. LLM assesses if truly zero-day
        2. If confirmed, create signature
        3. Submit to federation immediately (N_min=1)
        4. Return result for orchestrator
        
        Args:
            anomaly_data: From orchestrator, must contain:
                - latent_embedding
                - anomaly_score
                - feature_vector
                - features
                - triage_result
        """
        print(f"[AdaptiveRAG] Processing potential zero-day anomaly")
        
        # Step 1: LLM assessment
        assessment = await self.assess_zero_day(anomaly_data)
        
        result = {
            "is_zero_day": assessment.is_zero_day,
            "confidence": assessment.confidence,
            "assessment": assessment.model_dump() if hasattr(assessment, 'model_dump') else assessment.__dict__,
            "signature_created": False,
            "federation_submitted": False
        }
        
        if not assessment.is_zero_day:
            print(f"[AdaptiveRAG] Not classified as zero-day (confidence: {assessment.confidence})")
            return result
        
        if assessment.confidence < CONFIDENCE_THRESHOLD:
            print(f"[AdaptiveRAG] Confidence too low: {assessment.confidence} < {CONFIDENCE_THRESHOLD}")
            return result
        
        # Step 2: Create signature
        signature = self.create_signature(anomaly_data, assessment)
        result["signature_created"] = True
        result["signature_id"] = signature.signature_id
        
        # Step 3: Submit to federation immediately (N_min=1 - single agent can share)
        submitted = self.submit_signature_to_federation(signature)
        result["federation_submitted"] = submitted
        
        print(f"[AdaptiveRAG] Zero-day processing complete: {result}")
        return result

    def sync_process_zero_day(self, anomaly_data: Dict) -> Dict:
        """
        Synchronous wrapper for process_zero_day_anomaly.
        Use this when calling from non-async code.
        """
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.process_zero_day_anomaly(anomaly_data)
                )
                return future.result()
        else:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.process_zero_day_anomaly(anomaly_data))


# Singleton instance for import
_adaptive_agent_instance: Optional[AdaptiveRAGAgent] = None


def get_adaptive_agent() -> AdaptiveRAGAgent:
    """Get or create singleton AdaptiveRAG agent instance."""
    global _adaptive_agent_instance
    if _adaptive_agent_instance is None:
        _adaptive_agent_instance = AdaptiveRAGAgent()
    return _adaptive_agent_instance
