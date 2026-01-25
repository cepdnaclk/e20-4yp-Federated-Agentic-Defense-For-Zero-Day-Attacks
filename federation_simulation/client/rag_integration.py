"""
RAG (Retrieval-Augmented Generation) Integration Module
Integrates LLM-based policy generation with the Federated Learning simulation.

This module provides:
1. RAGPolicyGenerator - Uses RAG to generate dynamic mitigation policies
2. LLMClient - Abstract interface for different LLM providers
3. VectorStore - Simple vector similarity search for context retrieval
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    llm_provider: str = "openai"  # "openai", "anthropic", "ollama", "huggingface"
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    max_context_docs: int = 3
    temperature: float = 0.3
    max_tokens: int = 500
    
    # For local models (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    
    # For HuggingFace
    hf_model_id: str = "microsoft/DialoGPT-medium"


# ============================================================================
# LLM Client Interface
# ============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass in config.")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI Chat API"""
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.config.model_name),
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert specializing in network intrusion detection and mitigation strategies."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
        )
        return response.choices[0].message.content
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI Embeddings API"""
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return response.data[0].embedding


class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY env var.")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Claude API"""
        message = self.client.messages.create(
            model=kwargs.get("model", "claude-3-sonnet-20240229"),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are a cybersecurity expert specializing in network intrusion detection and mitigation strategies."
        )
        return message.content[0].text
    
    def get_embedding(self, text: str) -> List[float]:
        """Anthropic doesn't have embeddings API - use simple hash-based embedding"""
        return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generate a deterministic pseudo-embedding from text"""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        return np.random.randn(dim).tolist()


class OllamaClient(LLMClient):
    """Ollama local LLM client"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.base_url = config.ollama_base_url
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using local Ollama model"""
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": kwargs.get("model", self.config.model_name),
                "prompt": f"You are a cybersecurity expert. {prompt}",
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature)
                }
            },
            timeout=60
        )
        return response.json().get("response", "")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Ollama embeddings endpoint"""
        response = self.requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": "nomic-embed-text",  # or another embedding model
                "prompt": text
            },
            timeout=30
        )
        return response.json().get("embedding", [0.0] * 384)


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response based on detected attack type"""
        # Extract attack type from prompt
        attack_keywords = {
            "fuzzers": "Implement input validation, enable WAF rules, rate limit API endpoints",
            "dos": "Enable DDoS protection, scale horizontally, implement circuit breakers",
            "reconnaissance": "Deploy honeypots, mask network topology, enable IDS alerts",
            "exploits": "Apply security patches, enable virtual patching, segment networks",
            "backdoor": "Rotate credentials, audit access logs, isolate compromised systems",
            "shellcode": "Enable DEP/ASLR, sandbox execution, block suspicious processes",
            "worms": "Quarantine hosts, block lateral movement, update signatures",
            "analysis": "Increase logging, deploy deception tech, monitor data exfiltration",
            "generic": "Apply defense-in-depth, enable anomaly detection, alert SOC"
        }
        
        prompt_lower = prompt.lower()
        for attack, policy in attack_keywords.items():
            if attack in prompt_lower:
                return f"""## Mitigation Policy for {attack.title()} Attack

**Immediate Actions:**
1. {policy.split(', ')[0]}
2. {policy.split(', ')[1] if len(policy.split(', ')) > 1 else 'Monitor for additional indicators'}
3. {policy.split(', ')[2] if len(policy.split(', ')) > 2 else 'Document incident timeline'}

**Long-term Recommendations:**
- Review and update firewall rules
- Conduct security awareness training
- Implement continuous monitoring

**Risk Level:** HIGH
**Confidence:** 85%

Generated by RAG-enhanced policy engine."""
        
        return "Apply standard security controls and monitor for anomalies."
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embedding"""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).tolist()


# ============================================================================
# Vector Store for Context Retrieval
# ============================================================================

class SimpleVectorStore:
    """
    Simple in-memory vector store for RAG context retrieval.
    In production, replace with FAISS, Pinecone, Chroma, etc.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the vector store"""
        embedding = self.llm_client.get_embedding(text)
        self.documents.append({
            "text": text,
            "metadata": metadata or {},
            "id": len(self.documents)
        })
        self.embeddings.append(embedding)
    
    def add_security_knowledge_base(self):
        """Pre-populate with cybersecurity knowledge"""
        knowledge_docs = [
            {
                "text": """Fuzzing Attack Mitigation: Fuzzers send malformed or random data to find vulnerabilities.
                Mitigations: 1) Implement strict input validation, 2) Use WAF with fuzzing detection rules,
                3) Rate limit requests from single sources, 4) Enable application-level anomaly detection,
                5) Deploy input sanitization at all entry points.""",
                "metadata": {"attack_type": "Fuzzers", "category": "input_attacks"}
            },
            {
                "text": """DoS/DDoS Attack Response: Denial of Service attacks overwhelm resources.
                Mitigations: 1) Enable cloud-based DDoS protection (Cloudflare, AWS Shield),
                2) Implement rate limiting and traffic shaping, 3) Use CDN for traffic distribution,
                4) Configure auto-scaling policies, 5) Establish incident response playbook.""",
                "metadata": {"attack_type": "DoS", "category": "availability_attacks"}
            },
            {
                "text": """Reconnaissance Defense: Attackers scan networks to identify targets.
                Mitigations: 1) Minimize exposed services, 2) Use port knocking or SPA,
                3) Deploy honeypots to detect scanning, 4) Enable IDS/IPS with scan detection,
                5) Implement network segmentation to limit visibility.""",
                "metadata": {"attack_type": "Reconnaissance", "category": "information_gathering"}
            },
            {
                "text": """Exploit Prevention: Attackers leverage known vulnerabilities.
                Mitigations: 1) Maintain patch management program, 2) Enable virtual patching via WAF/IPS,
                3) Use application allowlisting, 4) Implement least privilege access,
                5) Deploy endpoint detection and response (EDR) solutions.""",
                "metadata": {"attack_type": "Exploits", "category": "vulnerability_exploitation"}
            },
            {
                "text": """Backdoor Detection and Response: Persistent unauthorized access mechanisms.
                Mitigations: 1) Conduct regular integrity checks, 2) Monitor for unusual network connections,
                3) Implement file integrity monitoring (FIM), 4) Review authentication logs,
                5) Use behavioral analysis to detect anomalous activities.""",
                "metadata": {"attack_type": "Backdoor", "category": "persistence"}
            },
            {
                "text": """Shellcode Defense: Malicious code injection into running processes.
                Mitigations: 1) Enable DEP (Data Execution Prevention), 2) Use ASLR,
                3) Implement Control Flow Integrity (CFI), 4) Use sandboxing for untrusted code,
                5) Deploy memory protection technologies.""",
                "metadata": {"attack_type": "Shellcode", "category": "code_execution"}
            },
            {
                "text": """Worm Containment: Self-propagating malware spreading across networks.
                Mitigations: 1) Segment networks to limit spread, 2) Block unnecessary lateral movement,
                3) Update antivirus signatures urgently, 4) Isolate infected hosts immediately,
                5) Disable vulnerable services network-wide.""",
                "metadata": {"attack_type": "Worms", "category": "malware"}
            },
            {
                "text": """Network Analysis Attack Defense: Deep inspection and traffic analysis by attackers.
                Mitigations: 1) Encrypt all traffic (TLS 1.3), 2) Use VPNs for sensitive communications,
                3) Implement traffic obfuscation, 4) Deploy decoy traffic,
                5) Monitor for data exfiltration patterns.""",
                "metadata": {"attack_type": "Analysis", "category": "traffic_analysis"}
            }
        ]
        
        for doc in knowledge_docs:
            self.add_document(doc["text"], doc["metadata"])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for most relevant documents using cosine similarity"""
        if not self.embeddings:
            return []
        
        query_embedding = self.llm_client.get_embedding(query)
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            results.append({
                "document": self.documents[idx],
                "score": score
            })
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================================
# RAG Policy Generator
# ============================================================================

class RAGPolicyGenerator:
    """
    RAG-based mitigation policy generator.
    Retrieves relevant context and uses LLM to generate dynamic policies.
    """
    
    def __init__(self, config: RAGConfig = None, use_mock: bool = True):
        """
        Initialize RAG Policy Generator.
        
        Args:
            config: RAG configuration
            use_mock: If True, use mock LLM (no API calls). Set False for real LLM.
        """
        self.config = config or RAGConfig()
        
        # Initialize LLM client
        if use_mock:
            self.llm_client = MockLLMClient(self.config)
            print("🤖 RAG: Using Mock LLM (no API calls)")
        else:
            self.llm_client = self._create_llm_client()
            print(f"🤖 RAG: Using {self.config.llm_provider} ({self.config.model_name})")
        
        # Initialize vector store with security knowledge
        self.vector_store = SimpleVectorStore(self.llm_client)
        self.vector_store.add_security_knowledge_base()
        
        print(f"📚 RAG: Loaded {len(self.vector_store.documents)} security knowledge documents")
    
    def _create_llm_client(self) -> LLMClient:
        """Create appropriate LLM client based on config"""
        provider = self.config.llm_provider.lower()
        
        if provider == "openai":
            return OpenAIClient(self.config)
        elif provider == "anthropic":
            return AnthropicClient(self.config)
        elif provider == "ollama":
            return OllamaClient(self.config)
        else:
            print(f"⚠️  Unknown provider '{provider}', using mock client")
            return MockLLMClient(self.config)
    
    def generate_policy(
        self,
        attack_category: str,
        features: List[float],
        agent_id: str = "Unknown"
    ) -> str:
        """
        Generate mitigation policy using RAG.
        
        Args:
            attack_category: Type of attack detected
            features: Attack signature (feature vector)
            agent_id: ID of the agent that detected the attack
            
        Returns:
            Generated mitigation policy string
        """
        # Step 1: Create query for retrieval
        query = f"Mitigation strategies for {attack_category} network attack"
        
        # Step 2: Retrieve relevant context
        retrieved_docs = self.vector_store.search(query, top_k=self.config.max_context_docs)
        
        context = "\n\n".join([
            f"[Source {i+1}] {doc['document']['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Step 3: Build prompt with context
        feature_summary = self._summarize_features(features)
        
        prompt = f"""You are a cybersecurity incident responder. Based on the following attack detection and security knowledge, generate a specific mitigation policy.

## Attack Detection Details
- **Attack Type:** {attack_category}
- **Detected by:** {agent_id}
- **Signature Analysis:** {feature_summary}

## Relevant Security Knowledge
{context}

## Task
Generate a concise, actionable mitigation policy for this specific attack. Include:
1. Immediate actions (within 5 minutes)
2. Short-term mitigations (within 1 hour)
3. Recommended long-term improvements

Format the response as a clear, prioritized action list."""

        # Step 4: Generate policy using LLM
        try:
            policy = self.llm_client.generate(prompt)
            return policy
        except Exception as e:
            print(f"❌ RAG: Error generating policy: {e}")
            return f"[{attack_category}] Apply standard security controls. Error in RAG generation."
    
    def _summarize_features(self, features: List[float]) -> str:
        """Create human-readable summary of attack signature"""
        if not features or len(features) < 10:
            return "Insufficient feature data"
        
        # Feature names (matching UNSW_NB15 dataset)
        feature_names = [
            "duration", "src_packets", "dst_packets", "src_bytes", "dst_bytes",
            "rate", "src_ttl", "dst_ttl", "src_load", "dst_load"
        ]
        
        summary_parts = []
        for i, (name, value) in enumerate(zip(feature_names, features[:10])):
            if value > 0:
                summary_parts.append(f"{name}={value:.2f}")
        
        return ", ".join(summary_parts[:5]) + "..." if summary_parts else "No significant features"
    
    def add_custom_knowledge(self, text: str, metadata: Dict[str, Any] = None):
        """Add custom security knowledge to the vector store"""
        self.vector_store.add_document(text, metadata)
        print(f"📚 RAG: Added custom knowledge document")


# ============================================================================
# Factory Function
# ============================================================================

def create_rag_generator(
    provider: str = "mock",
    api_key: str = None,
    model: str = None
) -> RAGPolicyGenerator:
    """
    Factory function to create RAG Policy Generator.
    
    Args:
        provider: "mock", "openai", "anthropic", or "ollama"
        api_key: API key for cloud providers
        model: Model name to use
        
    Returns:
        Configured RAGPolicyGenerator instance
    """
    config = RAGConfig(
        llm_provider=provider,
        api_key=api_key,
        model_name=model or ("gpt-3.5-turbo" if provider == "openai" else "claude-3-sonnet-20240229")
    )
    
    use_mock = provider.lower() == "mock"
    return RAGPolicyGenerator(config=config, use_mock=use_mock)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing RAG Policy Generator")
    print("="*60 + "\n")
    
    # Test with mock LLM (no API calls needed)
    rag = create_rag_generator(provider="mock")
    
    # Test policy generation for different attack types
    test_cases = [
        ("Fuzzers", [0.001, 6, 6, 258, 218, 863.7, 62, 253, 0, 0]),
        ("DoS", [0.5, 1000, 10, 50000, 100, 2000, 64, 64, 95.5, 0.5]),
        ("Reconnaissance", [0.0, 1, 0, 60, 0, 0, 64, 0, 0, 0]),
    ]
    
    for attack_type, features in test_cases:
        print(f"\n{'─'*40}")
        print(f"Attack Type: {attack_type}")
        print(f"{'─'*40}")
        
        policy = rag.generate_policy(
            attack_category=attack_type,
            features=features,
            agent_id="Test_Agent"
        )
        
        print(policy)
        print()
