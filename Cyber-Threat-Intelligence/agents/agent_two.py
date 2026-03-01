"""
Agent Two: Classification and Reasoning Agent for Flagged Anomalies.

This module provides AgentTwo, which combines:
    - XGBoost classification for known threat categorization
    - LangChain + Vector DB for contextual retrieval
    - LLM reasoning for unknown/zero-day threat analysis

Design Pattern:
    - Dependency Injection: LLM and VectorDB are injected for flexibility
    - Strategy Pattern: Different LLMs/DBs can be swapped without code changes
    - Single Responsibility: Classification, retrieval, and reasoning are separate

Example:
    >>> from agents import AgentTwo
    >>> from agents.interfaces import FAISSVectorDB, OllamaLLM
    >>> 
    >>> # Create with injected dependencies
    >>> agent = AgentTwo(
    ...     classifier=ThreatClassifier.load("models/agent_two"),
    ...     vector_db=FAISSVectorDB(...),
    ...     llm=OllamaLLM(model="llama3"),
    ... )
    >>> 
    >>> # Analyze a flagged anomaly
    >>> result = agent.analyze_threat(anomaly_features)
    >>> print(result.summary)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
import json

import numpy as np

from agents.models.xgboost_classifier import ThreatClassifier, ClassificationResult
from agents.interfaces.base import VectorDBInterface, LLMInterface, RetrievedContext

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ThreatAnalysisResult:
    """
    Complete threat analysis result from Agent Two.
    
    Attributes:
        classification: XGBoost classification result.
        is_zero_day: Whether the threat was classified as unknown.
        retrieved_contexts: Similar attacks from vector DB (if zero-day).
        llm_reasoning: LLM-generated analysis (if zero-day).
        feature_summary: Human-readable summary of key features.
        recommended_actions: List of recommended response actions.
        severity: Assessed severity level (critical/high/medium/low).
        confidence_score: Overall confidence in the analysis.
    """
    classification: ClassificationResult
    is_zero_day: bool
    retrieved_contexts: List[RetrievedContext] = field(default_factory=list)
    llm_reasoning: Optional[str] = None
    feature_summary: Optional[str] = None
    recommended_actions: List[str] = field(default_factory=list)
    severity: str = "unknown"
    confidence_score: float = 0.0
    
    @property
    def summary(self) -> str:
        """Returns a formatted summary of the analysis."""
        lines = [
            "=" * 60,
            "THREAT ANALYSIS REPORT",
            "=" * 60,
            f"Classification: {self.classification.predicted_category}",
            f"Confidence: {self.classification.confidence:.1%}",
            f"Zero-day: {'Yes' if self.is_zero_day else 'No'}",
            f"Severity: {self.severity.upper()}",
            "-" * 60,
        ]
        
        if self.feature_summary:
            lines.append("Key Features:")
            lines.append(self.feature_summary)
            lines.append("-" * 60)
        
        if self.is_zero_day and self.llm_reasoning:
            lines.append("AI Analysis:")
            lines.append(self.llm_reasoning)
            lines.append("-" * 60)
        
        if self.recommended_actions:
            lines.append("Recommended Actions:")
            for i, action in enumerate(self.recommended_actions, 1):
                lines.append(f"  {i}. {action}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts result to dictionary for serialization."""
        return {
            "classification": {
                "predicted_category": self.classification.predicted_category,
                "confidence": self.classification.confidence,
                "is_zero_day": self.classification.is_zero_day,
                "all_probabilities": self.classification.all_probabilities,
            },
            "is_zero_day": self.is_zero_day,
            "retrieved_contexts": [
                {
                    "content": ctx.content[:500],
                    "similarity": ctx.similarity_score,
                    "metadata": ctx.metadata,
                }
                for ctx in self.retrieved_contexts
            ],
            "llm_reasoning": self.llm_reasoning,
            "feature_summary": self.feature_summary,
            "recommended_actions": self.recommended_actions,
            "severity": self.severity,
            "confidence_score": self.confidence_score,
        }


class AgentTwo:
    """
    Classification and Reasoning Agent for Network Threat Analysis.
    
    AgentTwo works downstream of AgentOne (anomaly detection). When AgentOne
    flags a network flow as anomalous, AgentTwo:
    
    1. **Classifies** the threat using an optimized XGBoost model
    2. **Retrieves** similar historical attacks from a vector database
    3. **Reasons** about unknown threats using an LLM (for zero-day detection)
    
    Architecture:
        - XGBoost: Fast, accurate classification of known attack types
        - Vector DB: Semantic search over historical attack knowledge
        - LLM: Natural language reasoning for unknown patterns
    
    Design Principles:
        - Dependency Injection: External services injected, not hardcoded
        - Interface Segregation: Uses abstract interfaces for flexibility
        - Open/Closed: New LLMs/DBs can be added without modifying agent
    
    Example:
        >>> # Initialize with dependencies
        >>> agent = AgentTwo(
        ...     classifier=ThreatClassifier.load("models/agent_two"),
        ...     vector_db=faiss_db,
        ...     llm=ollama_llm,
        ... )
        >>> 
        >>> # Analyze flagged anomaly
        >>> result = agent.analyze_threat(features, feature_names)
        >>> if result.is_zero_day:
        ...     print("Unknown threat detected!")
        ...     print(result.llm_reasoning)
    """
    
    # Feature name -> Human readable description mapping
    FEATURE_DESCRIPTIONS = {
        "dur": "Connection duration",
        "spkts": "Source packets",
        "dpkts": "Destination packets", 
        "sbytes": "Source bytes",
        "dbytes": "Destination bytes",
        "rate": "Connection rate",
        "sttl": "Source TTL",
        "dttl": "Destination TTL",
        "sload": "Source load (bps)",
        "dload": "Destination load (bps)",
        "sloss": "Source packet loss",
        "dloss": "Destination packet loss",
        "proto": "Protocol",
        "service": "Service",
        "state": "Connection state",
    }
    
    # Severity mappings based on attack category
    SEVERITY_MAP = {
        "Normal": "low",
        "Reconnaissance": "medium",
        "Fuzzers": "medium",
        "Analysis": "medium",
        "DoS": "high",
        "Exploits": "high",
        "Backdoor": "critical",
        "Shellcode": "critical",
        "Worms": "critical",
        "Generic": "medium",
        "Unknown/Zero-day": "high",
    }
    
    def __init__(
        self,
        classifier: ThreatClassifier,
        vector_db: Optional[VectorDBInterface] = None,
        llm: Optional[LLMInterface] = None,
        feature_names: Optional[List[str]] = None,
        zero_day_context_k: int = 5,
    ):
        """
        Initializes Agent Two with injected dependencies.
        
        Args:
            classifier: Trained ThreatClassifier (XGBoost) for categorization.
            vector_db: Vector database for retrieving similar attacks.
                      Required for zero-day reasoning.
            llm: Large Language Model for generating threat analysis.
                Required for zero-day reasoning.
            feature_names: Names of input features for interpretability.
            zero_day_context_k: Number of similar attacks to retrieve
                               for zero-day analysis.
        
        Raises:
            ValueError: If classifier is not fitted.
            
        Note:
            vector_db and llm are optional - if not provided, zero-day
            threats will be flagged but without detailed reasoning.
        """
        if not classifier.is_fitted:
            raise ValueError("Classifier must be fitted before use")
        
        self._classifier = classifier
        self._vector_db = vector_db
        self._llm = llm
        self._feature_names = feature_names
        self._zero_day_context_k = zero_day_context_k
        
        # Warn if zero-day reasoning won't be available
        if vector_db is None or llm is None:
            logger.warning(
                "AgentTwo initialized without vector_db or llm - "
                "zero-day reasoning will be limited"
            )
        
        logger.info(
            "AgentTwo initialized: classifier=%s, vector_db=%s, llm=%s",
            type(classifier).__name__,
            type(vector_db).__name__ if vector_db else "None",
            type(llm).__name__ if llm else "None",
        )
    
    @property
    def classifier(self) -> ThreatClassifier:
        """Returns the threat classifier."""
        return self._classifier
    
    @property
    def has_reasoning_capability(self) -> bool:
        """Returns whether zero-day reasoning is available."""
        return self._vector_db is not None and self._llm is not None
    
    def analyze_threat(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        include_raw_features: bool = False,
    ) -> ThreatAnalysisResult:
        """
        Analyzes a flagged network anomaly.
        
        This method performs the complete analysis pipeline:
        1. Classify the threat using XGBoost
        2. If zero-day, retrieve similar attacks and generate LLM reasoning
        3. Determine severity and recommended actions
        
        Args:
            features: Feature vector from the anomalous network flow.
                     Shape: (n_features,) or (1, n_features).
            feature_names: Names of features (uses stored names if None).
            include_raw_features: Include original features in result.
        
        Returns:
            ThreatAnalysisResult with complete analysis.
        
        Example:
            >>> result = agent.analyze_threat(anomaly_features)
            >>> print(f"Threat type: {result.classification.predicted_category}")
            >>> if result.is_zero_day:
            ...     print(f"Zero-day analysis: {result.llm_reasoning}")
        """
        # Use provided or stored feature names
        if feature_names is None:
            feature_names = self._feature_names
        
        # Step 1: Classify with XGBoost
        classification = self._classifier.predict(
            features, 
            return_features=include_raw_features
        )
        
        # Determine if this is a zero-day threat
        is_zero_day = classification.is_zero_day
        
        # Generate feature summary
        feature_summary = self._generate_feature_summary(features, feature_names)
        
        # Initialize result
        result = ThreatAnalysisResult(
            classification=classification,
            is_zero_day=is_zero_day,
            feature_summary=feature_summary,
            severity=self.SEVERITY_MAP.get(
                classification.predicted_category, "medium"
            ),
            confidence_score=classification.confidence,
        )
        
        # Step 2: If zero-day and reasoning available, use RAG
        if is_zero_day and self.has_reasoning_capability:
            result = self._perform_zero_day_analysis(result, features, feature_names)
        
        # Step 3: Generate recommended actions
        result.recommended_actions = self._generate_recommendations(
            classification.predicted_category,
            result.severity,
            is_zero_day,
        )
        
        logger.info(
            "Threat analysis complete: category=%s, confidence=%.2f, zero_day=%s",
            classification.predicted_category,
            classification.confidence,
            is_zero_day,
        )
        
        return result
    
    def analyze_threats_batch(
        self,
        features_batch: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> List[ThreatAnalysisResult]:
        """
        Analyzes multiple threats in batch.
        
        Note: Zero-day reasoning is only performed for individual threats,
        not batch-processed, to avoid overwhelming the LLM.
        
        Args:
            features_batch: Batch of feature vectors, shape (n_samples, n_features).
            feature_names: Optional feature names.
        
        Returns:
            List of ThreatAnalysisResult objects.
        """
        results = []
        
        # First, batch classify all threats
        classifications = self._classifier.predict_batch(features_batch)
        
        for i, classification in enumerate(classifications):
            features = features_batch[i]
            
            # Only do full analysis (with LLM) for zero-day threats
            if classification.is_zero_day and self.has_reasoning_capability:
                # Full analysis for zero-day
                result = self.analyze_threat(features, feature_names)
            else:
                # Quick analysis for known threats
                result = ThreatAnalysisResult(
                    classification=classification,
                    is_zero_day=classification.is_zero_day,
                    feature_summary=self._generate_feature_summary(features, feature_names),
                    severity=self.SEVERITY_MAP.get(
                        classification.predicted_category, "medium"
                    ),
                    confidence_score=classification.confidence,
                    recommended_actions=self._generate_recommendations(
                        classification.predicted_category,
                        self.SEVERITY_MAP.get(classification.predicted_category, "medium"),
                        classification.is_zero_day,
                    ),
                )
            
            results.append(result)
        
        return results
    
    def _perform_zero_day_analysis(
        self,
        result: ThreatAnalysisResult,
        features: np.ndarray,
        feature_names: Optional[List[str]],
    ) -> ThreatAnalysisResult:
        """
        Performs RAG-based analysis for zero-day threats.
        
        Args:
            result: Initial analysis result.
            features: Feature vector.
            feature_names: Feature names for context.
        
        Returns:
            Enhanced result with LLM reasoning.
        """
        # Build query from feature data
        query = self._build_threat_query(features, feature_names, result.classification)
        
        # Retrieve similar attacks from vector DB
        try:
            contexts = self._vector_db.similarity_search(
                query, k=self._zero_day_context_k
            )
            result.retrieved_contexts = contexts
            
            logger.info("Retrieved %d similar attacks for analysis", len(contexts))
        except Exception as e:
            logger.error("Vector DB search failed: %s", str(e))
            contexts = []
        
        # Generate LLM reasoning
        if contexts and self._llm:
            try:
                context_texts = [ctx.content for ctx in contexts]
                llm_response = self._llm.generate_with_context(
                    query=query,
                    context=context_texts,
                    temperature=0.3,  # Lower temperature for more focused analysis
                    max_tokens=1024,
                )
                result.llm_reasoning = llm_response.content
                
                logger.info("LLM reasoning generated successfully")
            except Exception as e:
                logger.error("LLM generation failed: %s", str(e))
                result.llm_reasoning = (
                    "Unable to generate detailed analysis. "
                    "Please review similar attacks manually."
                )
        
        return result
    
    def _build_threat_query(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]],
        classification: ClassificationResult,
    ) -> str:
        """Builds a query string for vector DB search and LLM."""
        # Get top probabilities
        top_probs = sorted(
            classification.all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        query_parts = [
            "Network threat analysis required.",
            f"Top predicted categories: {', '.join([f'{cat}({p:.1%})' for cat, p in top_probs])}",
            f"Confidence is low ({classification.confidence:.1%}), suggesting potential zero-day or novel attack.",
        ]
        
        # Add key feature values
        if feature_names:
            features_flat = np.atleast_1d(features).flatten()
            key_features = []
            for i, (name, value) in enumerate(zip(feature_names, features_flat)):
                if name in self.FEATURE_DESCRIPTIONS:
                    desc = self.FEATURE_DESCRIPTIONS[name]
                    key_features.append(f"{desc}: {value:.4f}")
            
            if key_features:
                query_parts.append("Key features: " + ", ".join(key_features[:10]))
        
        return " ".join(query_parts)
    
    def _generate_feature_summary(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]],
    ) -> str:
        """Generates human-readable feature summary."""
        if feature_names is None:
            return "Feature names not available."
        
        features_flat = np.atleast_1d(features).flatten()
        
        lines = []
        for name, value in zip(feature_names, features_flat):
            if name in self.FEATURE_DESCRIPTIONS:
                desc = self.FEATURE_DESCRIPTIONS[name]
                lines.append(f"  {desc}: {value:.4f}")
        
        return "\n".join(lines[:15]) if lines else "No key features identified."
    
    def _generate_recommendations(
        self,
        category: str,
        severity: str,
        is_zero_day: bool,
    ) -> List[str]:
        """Generates recommended response actions."""
        base_actions = {
            "Normal": [
                "No action required - traffic appears normal",
                "Continue monitoring for pattern changes",
            ],
            "DoS": [
                "Implement rate limiting on affected services",
                "Enable DDoS protection mechanisms",
                "Consider traffic scrubbing services",
                "Block source IPs if attack is targeted",
            ],
            "Exploits": [
                "Isolate affected systems immediately",
                "Check for successful compromise indicators",
                "Patch vulnerable software",
                "Review access logs for lateral movement",
            ],
            "Backdoor": [
                "CRITICAL: Isolate system from network immediately",
                "Preserve evidence for forensic analysis",
                "Change all credentials for affected systems",
                "Scan for additional compromise indicators",
            ],
            "Reconnaissance": [
                "Review firewall rules for information leakage",
                "Enable logging on probed services",
                "Consider honeypot deployment",
                "Monitor for follow-up attack attempts",
            ],
            "Shellcode": [
                "CRITICAL: Isolate affected system",
                "Check for process injection indicators",
                "Review memory dumps if available",
                "Enable enhanced endpoint monitoring",
            ],
            "Worms": [
                "CRITICAL: Network segment isolation",
                "Check for propagation to other systems",
                "Deploy emergency patches",
                "Enable network-level blocking",
            ],
            "Fuzzers": [
                "Review application input validation",
                "Check for crashes or errors in logs",
                "Update WAF/IDS signatures",
            ],
            "Analysis": [
                "Review for data exfiltration attempts",
                "Check database query logs",
                "Monitor for unusual data access patterns",
            ],
            "Generic": [
                "Continue monitoring for specific patterns",
                "Collect additional context data",
                "Escalate to security team for review",
            ],
        }
        
        actions = base_actions.get(category, [
            "Investigate the flagged traffic",
            "Collect packet captures if possible",
            "Review related system logs",
        ])
        
        # Add zero-day specific actions
        if is_zero_day:
            actions = [
                "ALERT: Potentially novel/zero-day attack detected",
                "Preserve all evidence for detailed analysis",
                "Consider engaging incident response team",
            ] + actions
        
        # Add severity-specific prefix
        if severity == "critical":
            actions.insert(0, "IMMEDIATE ACTION REQUIRED")
        elif severity == "high":
            actions.insert(0, "High priority - respond within 1 hour")
        
        return actions
    
    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        vector_db: Optional[VectorDBInterface] = None,
        llm: Optional[LLMInterface] = None,
        **kwargs,
    ) -> "AgentTwo":
        """
        Loads AgentTwo from a saved model directory.
        
        Args:
            model_dir: Directory containing saved classifier.
            vector_db: Optional vector database for reasoning.
            llm: Optional LLM for reasoning.
            **kwargs: Additional arguments for AgentTwo.
        
        Returns:
            Loaded AgentTwo instance.
        """
        model_dir = Path(model_dir)
        
        # Load classifier
        classifier = ThreatClassifier.load(model_dir / "classifier")
        
        # Load feature names if saved
        feature_names = None
        features_path = model_dir / "feature_names.json"
        if features_path.exists():
            with open(features_path, "r") as f:
                feature_names = json.load(f)
        
        return cls(
            classifier=classifier,
            vector_db=vector_db,
            llm=llm,
            feature_names=feature_names,
            **kwargs,
        )
    
    def save(self, model_dir: Union[str, Path]) -> None:
        """
        Saves AgentTwo components to disk.
        
        Args:
            model_dir: Directory to save model files.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        self._classifier.save(model_dir / "classifier")
        
        # Save feature names
        if self._feature_names:
            with open(model_dir / "feature_names.json", "w") as f:
                json.dump(self._feature_names, f)
        
        # Note: Vector DB and LLM are not saved as they are injected dependencies
        
        logger.info("AgentTwo saved to: %s", model_dir)
    
    def get_config(self) -> Dict[str, Any]:
        """Returns agent configuration."""
        return {
            "classifier_config": self._classifier.get_config(),
            "has_vector_db": self._vector_db is not None,
            "has_llm": self._llm is not None,
            "llm_model": self._llm.model_name if self._llm else None,
            "zero_day_context_k": self._zero_day_context_k,
            "feature_names_count": len(self._feature_names) if self._feature_names else 0,
        }
