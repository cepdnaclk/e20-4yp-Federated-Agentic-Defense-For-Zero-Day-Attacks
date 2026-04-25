"""Agent Two: XGBoost Classification Agent.

Per the current architecture, AgentTwo is responsible only for XGBoost-based
threat categorization and emitting a lightweight classification signal.

RAG retrieval + LLM-based action recommendations have been separated into
AgentThree (see `agents/agent_three.py`).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
import json

import numpy as np

from agents.models.xgboost_classifier import ThreatClassifier, ClassificationResult
from agents.interfaces.base import RetrievedContext

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ThreatAnalysisResult:
    """
    Complete threat analysis result from Agent Two.
    
    Attributes:
        classification: XGBoost classification result.
        is_zero_day: Whether the threat was classified as unknown.
        retrieved_contexts: Deprecated (kept for backward compatibility).
        llm_reasoning: Deprecated (moved to AgentThree).
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
    XGBoost Classification Agent for Network Threat Analysis.
    
    AgentTwo works downstream of AgentOne (anomaly detection). When AgentOne
    flags a network flow as anomalous, AgentTwo:
    
    1. **Classifies** the threat using an optimized XGBoost model
    
    RAG retrieval + LLM-based action recommendations are provided by AgentThree.
    
    Example:
        >>> agent = AgentTwo(classifier=ThreatClassifier.load("models/agent_two"))
        >>> result = agent.analyze_threat(features, feature_names)
        >>> print(result.classification.predicted_category)
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
        vector_db: Optional[Any] = None,
        llm: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        zero_day_context_k: int = 5,
    ):
        """
        Initializes Agent Two with injected dependencies.
        
        Args:
            classifier: Trained ThreatClassifier (XGBoost) for categorization.
            vector_db: Deprecated (ignored). Use AgentThree.
            llm: Deprecated (ignored). Use AgentThree.
            feature_names: Names of input features for interpretability.
            zero_day_context_k: Number of similar attacks to retrieve
                               for zero-day analysis.
        
        Raises:
            ValueError: If classifier is not fitted.
            
        Note:
            `vector_db` and `llm` are accepted for backward compatibility but
            are not used by AgentTwo.
        """
        if not classifier.is_fitted:
            raise ValueError("Classifier must be fitted before use")
        
        self._classifier = classifier
        self._feature_names = feature_names
        self._zero_day_context_k = zero_day_context_k

        if vector_db is not None or llm is not None:
            logger.warning(
                "AgentTwo no longer performs RAG/LLM reasoning; "
                "vector_db/llm args are ignored. Use AgentThree instead."
            )
        
        logger.info("AgentTwo initialized: classifier=%s", type(classifier).__name__)
    
    @property
    def classifier(self) -> ThreatClassifier:
        """Returns the threat classifier."""
        return self._classifier
    
    def classify(self, features: np.ndarray) -> Dict[str, Any]:
        """Lightweight classification API used by the coordinator."""
        classification = self._classifier.predict(features)
        return {
            "category": classification.predicted_category,
            "confidence": float(classification.confidence),
            "is_zero_day": bool(classification.is_zero_day),
            "all_probabilities": dict(classification.all_probabilities or {}),
        }
    
    def analyze_threat(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        include_raw_features: bool = False,
    ) -> ThreatAnalysisResult:
        """
        Analyzes a flagged network anomaly.
        
        This method performs the analysis pipeline:
        1. Classify the threat using XGBoost
        2. Determine severity (no LLM/RAG calls)
        
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
        
        # AgentTwo does not generate response actions; AgentThree does.
        result.recommended_actions = []
        
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
        
        Note: AgentTwo does not perform LLM calls.
        
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
            
            result = ThreatAnalysisResult(
                classification=classification,
                is_zero_day=classification.is_zero_day,
                feature_summary=self._generate_feature_summary(features, feature_names),
                severity=self.SEVERITY_MAP.get(
                    classification.predicted_category, "medium"
                ),
                confidence_score=classification.confidence,
                recommended_actions=[],
            )
            
            results.append(result)
        
        return results
    
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
        vector_db: Optional[Any] = None,
        llm: Optional[Any] = None,
        **kwargs,
    ) -> "AgentTwo":
        """
        Loads AgentTwo from a saved model directory.
        
        Args:
            model_dir: Directory containing saved classifier.
            vector_db: Deprecated (ignored). Use AgentThree.
            llm: Deprecated (ignored). Use AgentThree.
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
            "has_vector_db": False,
            "has_llm": False,
            "llm_model": None,
            "zero_day_context_k": self._zero_day_context_k,
            "feature_names_count": len(self._feature_names) if self._feature_names else 0,
        }
