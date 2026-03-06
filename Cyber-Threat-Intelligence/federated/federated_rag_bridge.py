"""
Federated Learning - RAG Bridge Module

This module provides the critical integration between federated learning weight updates
and the RAG-based threat explanation system. This is the core novelty of the framework:
translating globally-updated model parameters into locally-grounded threat intelligence.

The bridge handles:
1. Model Update Semantics: Interpreting what federated weight changes mean for detection
2. RAG Context Updates: Triggering knowledge base refreshes based on model drift
3. Explanation Re-generation: Re-analyzing held samples when models improve
4. Confidence Calibration: Adjusting explanation confidence based on federated rounds

Architecture:
    
    [FL Server] -> [FederatedRAGBridge] -> [IntegrationCoordinator]
         |                  |                       |
         v                  v                       v
   Global Weights     Context Triggers       RAG Explanations
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from threading import Lock
import hashlib
import json

logger = logging.getLogger(__name__)


class ModelUpdateType(Enum):
    """Types of federated model updates."""
    AGENT_ONE_AUTOENCODER = "autoencoder"
    AGENT_TWO_XGBOOST = "xgboost"
    COMBINED = "combined"


class UpdateSignificance(Enum):
    """Significance level of model updates for RAG triggering."""
    MINIMAL = "minimal"      # < 1% weight drift
    MODERATE = "moderate"    # 1-5% weight drift
    SIGNIFICANT = "significant"  # 5-15% weight drift
    MAJOR = "major"         # > 15% weight drift


# Numeric ordering for UpdateSignificance comparison
_SIGNIFICANCE_ORDER = {
    UpdateSignificance.MINIMAL: 0,
    UpdateSignificance.MODERATE: 1,
    UpdateSignificance.SIGNIFICANT: 2,
    UpdateSignificance.MAJOR: 3,
}


def significance_gte(a: UpdateSignificance, b: UpdateSignificance) -> bool:
    """Check if significance level a >= b."""
    return _SIGNIFICANCE_ORDER[a] >= _SIGNIFICANCE_ORDER[b]


def significance_gt(a: UpdateSignificance, b: UpdateSignificance) -> bool:
    """Check if significance level a > b."""
    return _SIGNIFICANCE_ORDER[a] > _SIGNIFICANCE_ORDER[b]


@dataclass
class FederatedRound:
    """Represents a single federated learning round."""
    round_number: int
    timestamp: datetime
    participating_clients: int
    model_updates: Dict[str, Any]
    weight_drift: float  # L2 norm of weight change
    significance: UpdateSignificance
    new_attack_patterns_detected: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_number,
            "timestamp": self.timestamp.isoformat(),
            "clients": self.participating_clients,
            "drift": self.weight_drift,
            "significance": self.significance.value,
            "new_patterns": self.new_attack_patterns_detected,
        }


@dataclass
class RAGTriggerEvent:
    """Event triggered when RAG re-analysis is needed."""
    trigger_type: str  # "weight_update", "new_pattern", "confidence_drop"
    federated_round: int
    affected_categories: List[str]
    priority: int  # 1-5, with 1 being highest
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationRequest:
    """Request for threat explanation with federated context."""
    sample_id: str
    features: np.ndarray
    detection_result: Dict[str, Any]
    federated_round: int
    prior_explanation: Optional[str] = None
    requires_comparison: bool = False


@dataclass
class ExplanationResult:
    """Result of RAG-based threat explanation."""
    sample_id: str
    explanation: str
    confidence: float
    federated_round: int
    cve_references: List[str]
    mitre_techniques: List[str]
    is_zero_day: bool
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "fl_round": self.federated_round,
            "cves": self.cve_references,
            "mitre": self.mitre_techniques,
            "zero_day": self.is_zero_day,
            "time_ms": self.processing_time_ms,
        }


class UpdateInterpreter(ABC):
    """Base class for interpreting model-specific weight updates."""
    
    @abstractmethod
    def interpret_update(
        self,
        old_weights: Dict[str, Any],
        new_weights: Dict[str, Any],
    ) -> Tuple[UpdateSignificance, List[str]]:
        """
        Interpret what a weight update means semantically.
        
        Returns:
            Tuple of (significance level, list of affected attack categories)
        """
        pass


class AutoencoderUpdateInterpreter(UpdateInterpreter):
    """Interprets autoencoder weight changes for anomaly detection threshold shifts."""
    
    def interpret_update(
        self,
        old_weights: Dict[str, Any],
        new_weights: Dict[str, Any],
    ) -> Tuple[UpdateSignificance, List[str]]:
        """
        Autoencoder weight changes affect reconstruction thresholds.
        Large changes = detection sensitivity has shifted.
        """
        try:
            # Calculate L2 norm of weight drift
            total_drift = 0.0
            count = 0
            
            for key in new_weights:
                if key in old_weights:
                    old_val = np.array(old_weights[key])
                    new_val = np.array(new_weights[key])
                    drift = np.linalg.norm(new_val - old_val) / (np.linalg.norm(old_val) + 1e-8)
                    total_drift += drift
                    count += 1
            
            if count == 0:
                return UpdateSignificance.MINIMAL, []
            
            avg_drift = total_drift / count
            
            # Map drift to significance
            if avg_drift < 0.01:
                significance = UpdateSignificance.MINIMAL
            elif avg_drift < 0.05:
                significance = UpdateSignificance.MODERATE
            elif avg_drift < 0.15:
                significance = UpdateSignificance.SIGNIFICANT
            else:
                significance = UpdateSignificance.MAJOR
            
            # For autoencoder, all categories could be affected
            affected = ["all"] if significance_gte(significance, UpdateSignificance.SIGNIFICANT) else []
            
            return significance, affected
            
        except Exception as e:
            logger.error(f"Autoencoder interpretation error: {e}")
            return UpdateSignificance.MINIMAL, []


class XGBoostUpdateInterpreter(UpdateInterpreter):
    """
    Interprets XGBoost model updates.
    
    NOTE: This is a conceptual interpreter. XGBoost trees cannot be averaged like neural network
    weights. In practice, this would need to use:
    - Feature importance drift analysis
    - Prediction distribution comparison
    - Tree structure similarity metrics
    """
    
    def __init__(self, category_mapping: Optional[Dict[int, str]] = None):
        self.category_mapping = category_mapping or {
            0: "Normal",
            1: "DoS",
            2: "Reconnaissance",
            3: "Exploits",
            4: "Fuzzers",
            5: "Generic",
            6: "Analysis",
            7: "Backdoor",
            8: "Shellcode",
            9: "Worms",
        }
    
    def interpret_update(
        self,
        old_weights: Dict[str, Any],
        new_weights: Dict[str, Any],
    ) -> Tuple[UpdateSignificance, List[str]]:
        """
        For XGBoost, we analyze feature importance shifts to determine
        which attack categories are most affected.
        """
        try:
            # Compare feature importances if available
            old_importance = old_weights.get("feature_importance", {})
            new_importance = new_weights.get("feature_importance", {})
            
            if not old_importance or not new_importance:
                # Fall back to basic drift calculation
                return self._basic_drift(old_weights, new_weights)
            
            # Calculate importance drift per feature
            importance_drift = {}
            for feature in set(old_importance.keys()) | set(new_importance.keys()):
                old_val = old_importance.get(feature, 0)
                new_val = new_importance.get(feature, 0)
                drift = abs(new_val - old_val)
                importance_drift[feature] = drift
            
            # High drift features indicate changed detection patterns
            max_drift = max(importance_drift.values()) if importance_drift else 0
            
            if max_drift < 0.01:
                significance = UpdateSignificance.MINIMAL
            elif max_drift < 0.05:
                significance = UpdateSignificance.MODERATE
            elif max_drift < 0.15:
                significance = UpdateSignificance.SIGNIFICANT
            else:
                significance = UpdateSignificance.MAJOR
            
            # Map affected features to attack categories (domain-specific logic)
            affected_categories = self._features_to_categories(importance_drift)
            
            return significance, affected_categories
            
        except Exception as e:
            logger.error(f"XGBoost interpretation error: {e}")
            return UpdateSignificance.MINIMAL, []
    
    def _basic_drift(
        self,
        old_weights: Dict[str, Any],
        new_weights: Dict[str, Any],
    ) -> Tuple[UpdateSignificance, List[str]]:
        """Basic comparison when detailed info unavailable."""
        # Hash-based change detection
        old_hash = hashlib.md5(json.dumps(old_weights, sort_keys=True, default=str).encode()).hexdigest()
        new_hash = hashlib.md5(json.dumps(new_weights, sort_keys=True, default=str).encode()).hexdigest()
        
        if old_hash == new_hash:
            return UpdateSignificance.MINIMAL, []
        else:
            return UpdateSignificance.MODERATE, []
    
    def _features_to_categories(self, importance_drift: Dict[str, float]) -> List[str]:
        """
        Map feature importance changes to affected attack categories.
        This is domain-specific based on which features indicate which attacks.
        """
        affected = []
        
        # Network traffic features -> DoS, Reconnaissance
        network_features = ["sbytes", "dbytes", "spkts", "dpkts", "dur", "rate"]
        network_drift = sum(importance_drift.get(f, 0) for f in network_features)
        if network_drift > 0.1:
            affected.extend(["DoS", "Reconnaissance"])
        
        # Port/service features -> Exploits, Analysis
        port_features = ["sport", "dport", "service", "state"]
        port_drift = sum(importance_drift.get(f, 0) for f in port_features)
        if port_drift > 0.1:
            affected.extend(["Exploits", "Analysis"])
        
        # Content features -> Backdoor, Shellcode
        content_features = ["ct_srv_src", "ct_srv_dst", "ct_state_ttl"]
        content_drift = sum(importance_drift.get(f, 0) for f in content_features)
        if content_drift > 0.1:
            affected.extend(["Backdoor", "Shellcode"])
        
        return list(set(affected))


class FederatedRAGBridge:
    """
    The bridge between federated learning updates and RAG-based explanations.
    
    This is the core innovation: translating global model parameter updates
    into locally-grounded, human-readable threat intelligence.
    
    Flow:
    1. Receive federated update notification
    2. Interpret update semantics (what changed, how much)
    3. Determine if RAG re-analysis is needed
    4. Trigger coordinator to re-analyze held samples
    5. Track explanation quality across federated rounds
    """
    
    def __init__(
        self,
        coordinator: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
        auto_trigger: bool = True,
    ):
        """
        Initialize the bridge.
        
        Args:
            coordinator: IntegrationCoordinator instance for RAG processing
            knowledge_base: ThreatKnowledgeBase for context retrieval
            auto_trigger: Whether to auto-trigger RAG on significant updates
        """
        self.coordinator = coordinator
        self.knowledge_base = knowledge_base
        self.auto_trigger = auto_trigger
        
        # Interpreters for different model types
        self._interpreters: Dict[ModelUpdateType, UpdateInterpreter] = {
            ModelUpdateType.AGENT_ONE_AUTOENCODER: AutoencoderUpdateInterpreter(),
            ModelUpdateType.AGENT_TWO_XGBOOST: XGBoostUpdateInterpreter(),
        }
        
        # State tracking
        self._current_round = 0
        self._round_history: List[FederatedRound] = []
        self._trigger_queue: List[RAGTriggerEvent] = []
        self._explanation_cache: Dict[str, ExplanationResult] = {}
        self._lock = Lock()
        
        # Previous weights for drift calculation
        self._previous_weights: Dict[ModelUpdateType, Dict[str, Any]] = {}
        
        # Callbacks
        self._on_trigger: Optional[Callable[[RAGTriggerEvent], None]] = None
        self._on_explanation: Optional[Callable[[ExplanationResult], None]] = None
        
        logger.info("FederatedRAGBridge initialized")
    
    def set_coordinator(self, coordinator: Any) -> None:
        """Set the integration coordinator."""
        self.coordinator = coordinator
        if self.knowledge_base and hasattr(coordinator, 'set_knowledge_base'):
            coordinator.set_knowledge_base(self.knowledge_base)
    
    def set_knowledge_base(self, knowledge_base: Any) -> None:
        """Set the knowledge base."""
        self.knowledge_base = knowledge_base
        if self.coordinator and hasattr(self.coordinator, 'set_knowledge_base'):
            self.coordinator.set_knowledge_base(knowledge_base)
    
    def register_interpreter(
        self,
        model_type: ModelUpdateType,
        interpreter: UpdateInterpreter,
    ) -> None:
        """Register a custom interpreter for a model type."""
        self._interpreters[model_type] = interpreter
    
    def on_federated_round_complete(
        self,
        round_number: int,
        model_updates: Dict[str, Dict[str, Any]],
        participating_clients: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FederatedRound:
        """
        Called when a federated learning round completes.
        
        This is the main entry point for FL -> RAG integration.
        
        Args:
            round_number: The federated round number
            model_updates: Dict mapping model type string to weight dicts
            participating_clients: Number of clients in this round
            metadata: Optional additional metadata
            
        Returns:
            FederatedRound object with analysis results
        """
        with self._lock:
            self._current_round = round_number
            timestamp = datetime.now()
            
            # Analyze each model update
            total_drift = 0.0
            all_affected_categories = []
            max_significance = UpdateSignificance.MINIMAL
            
            for model_type_str, weights in model_updates.items():
                try:
                    model_type = ModelUpdateType(model_type_str)
                except ValueError:
                    logger.warning(f"Unknown model type: {model_type_str}")
                    continue
                
                interpreter = self._interpreters.get(model_type)
                if interpreter is None:
                    continue
                
                # Get previous weights for comparison
                prev_weights = self._previous_weights.get(model_type, {})
                
                if prev_weights:
                    # Interpret the update
                    significance, affected = interpreter.interpret_update(prev_weights, weights)
                    
                    # Track drift
                    drift = self._calculate_drift(prev_weights, weights)
                    total_drift += drift
                    all_affected_categories.extend(affected)
                    
                    if significance_gt(significance, max_significance):
                        max_significance = significance
                    
                    logger.info(
                        f"Round {round_number} - {model_type.value}: "
                        f"significance={significance.value}, drift={drift:.4f}"
                    )
                
                # Store for next round comparison
                self._previous_weights[model_type] = weights.copy()
            
            # Create round record
            fl_round = FederatedRound(
                round_number=round_number,
                timestamp=timestamp,
                participating_clients=participating_clients,
                model_updates={k: "stored" for k in model_updates},
                weight_drift=total_drift,
                significance=max_significance,
                new_attack_patterns_detected=list(set(all_affected_categories)),
            )
            
            self._round_history.append(fl_round)
            
            # Determine if RAG trigger is needed
            if self.auto_trigger and significance_gte(max_significance, UpdateSignificance.MODERATE):
                self._create_trigger_event(fl_round, all_affected_categories)
            
            # Update coordinator's federated round tracking
            # Note: on_federated_update is called separately by FL server with actual weights
            if self.coordinator:
                if hasattr(self.coordinator, 'set_federated_round'):
                    self.coordinator.set_federated_round(round_number)
                elif hasattr(self.coordinator, '_federated_round'):
                    self.coordinator._federated_round = round_number
            
            return fl_round
    
    def _calculate_drift(
        self,
        old_weights: Dict[str, Any],
        new_weights: Dict[str, Any],
    ) -> float:
        """Calculate overall weight drift as normalized L2 distance."""
        try:
            drift_sum = 0.0
            count = 0
            
            for key in new_weights:
                if key in old_weights:
                    try:
                        old_arr = np.array(old_weights[key], dtype=float)
                        new_arr = np.array(new_weights[key], dtype=float)
                        diff = np.linalg.norm(new_arr - old_arr)
                        norm = np.linalg.norm(old_arr) + 1e-8
                        drift_sum += diff / norm
                        count += 1
                    except (ValueError, TypeError):
                        continue
            
            return drift_sum / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Drift calculation error: {e}")
            return 0.0
    
    def _create_trigger_event(
        self,
        fl_round: FederatedRound,
        affected_categories: List[str],
    ) -> None:
        """Create and queue a RAG trigger event."""
        priority = {
            UpdateSignificance.MODERATE: 3,
            UpdateSignificance.SIGNIFICANT: 2,
            UpdateSignificance.MAJOR: 1,
        }.get(fl_round.significance, 4)
        
        event = RAGTriggerEvent(
            trigger_type="weight_update",
            federated_round=fl_round.round_number,
            affected_categories=affected_categories,
            priority=priority,
            metadata={
                "drift": fl_round.weight_drift,
                "clients": fl_round.participating_clients,
            }
        )
        
        self._trigger_queue.append(event)
        
        if self._on_trigger:
            self._on_trigger(event)
        
        # Process trigger if coordinator available
        if self.coordinator:
            self._process_trigger(event)
    
    def _process_trigger(self, event: RAGTriggerEvent) -> None:
        """Process a RAG trigger event by re-analyzing pending samples."""
        if not self.coordinator:
            logger.warning("No coordinator set, cannot process trigger")
            return
        
        logger.info(
            f"Processing RAG trigger - round {event.federated_round}, "
            f"categories: {event.affected_categories}"
        )
        
        # Trigger re-analysis through coordinator
        if hasattr(self.coordinator, '_trigger_rag_reanalysis'):
            self.coordinator._trigger_rag_reanalysis(
                affected_categories=event.affected_categories,
                reason=f"federated_round_{event.federated_round}",
            )
    
    def request_explanation(
        self,
        request: ExplanationRequest,
    ) -> ExplanationResult:
        """
        Request a threat explanation for a detection result.
        
        This method routes through the coordinator to produce
        RAG-grounded explanations with federated context.
        
        Args:
            request: ExplanationRequest with sample details
            
        Returns:
            ExplanationResult with explanation and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{request.sample_id}_{self._current_round}"
            if cache_key in self._explanation_cache and not request.requires_comparison:
                return self._explanation_cache[cache_key]
            
            # Route through coordinator
            if self.coordinator and hasattr(self.coordinator, 'analyze_with_federated_context'):
                result_dict = self.coordinator.analyze_with_federated_context(
                    features=request.features,
                    detection_result=request.detection_result,
                    federated_round=request.federated_round or self._current_round,
                )
            else:
                # Fallback minimal result
                result_dict = {
                    "explanation": "Coordinator unavailable",
                    "confidence": 0.0,
                    "cve_references": [],
                    "mitre_techniques": [],
                    "is_zero_day": False,
                }
            
            processing_time = (time.time() - start_time) * 1000
            
            # Build result
            result = ExplanationResult(
                sample_id=request.sample_id,
                explanation=result_dict.get("explanation", ""),
                confidence=result_dict.get("confidence", 0.0),
                federated_round=self._current_round,
                cve_references=result_dict.get("cve_references", []),
                mitre_techniques=result_dict.get("mitre_techniques", []),
                is_zero_day=result_dict.get("is_zero_day", False),
                processing_time_ms=processing_time,
            )
            
            # Cache result
            self._explanation_cache[cache_key] = result
            
            # Callback
            if self._on_explanation:
                self._on_explanation(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Explanation request error: {e}")
            return ExplanationResult(
                sample_id=request.sample_id,
                explanation=f"Error generating explanation: {e}",
                confidence=0.0,
                federated_round=self._current_round,
                cve_references=[],
                mitre_techniques=[],
                is_zero_day=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def compare_explanations(
        self,
        sample_id: str,
        round_a: int,
        round_b: int,
    ) -> Dict[str, Any]:
        """
        Compare explanations for the same sample across federated rounds.
        
        Useful for evaluating how model updates affect explanation quality.
        """
        cache_key_a = f"{sample_id}_{round_a}"
        cache_key_b = f"{sample_id}_{round_b}"
        
        result_a = self._explanation_cache.get(cache_key_a)
        result_b = self._explanation_cache.get(cache_key_b)
        
        if not result_a or not result_b:
            return {"error": "One or both explanations not cached"}
        
        return {
            "sample_id": sample_id,
            "round_a": {
                "round": round_a,
                "confidence": result_a.confidence,
                "cve_count": len(result_a.cve_references),
                "is_zero_day": result_a.is_zero_day,
            },
            "round_b": {
                "round": round_b,
                "confidence": result_b.confidence,
                "cve_count": len(result_b.cve_references),
                "is_zero_day": result_b.is_zero_day,
            },
            "confidence_change": result_b.confidence - result_a.confidence,
            "zero_day_change": result_b.is_zero_day != result_a.is_zero_day,
        }
    
    def get_round_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get history of federated rounds."""
        history = self._round_history[-last_n:] if last_n else self._round_history
        return [r.to_dict() for r in history]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "current_round": self._current_round,
            "total_rounds": len(self._round_history),
            "pending_triggers": len(self._trigger_queue),
            "cached_explanations": len(self._explanation_cache),
            "significance_distribution": self._get_significance_distribution(),
        }
    
    def _get_significance_distribution(self) -> Dict[str, int]:
        """Calculate distribution of update significance levels."""
        dist = {s.value: 0 for s in UpdateSignificance}
        for r in self._round_history:
            dist[r.significance.value] += 1
        return dist
    
    def set_trigger_callback(
        self,
        callback: Callable[[RAGTriggerEvent], None],
    ) -> None:
        """Set callback for trigger events."""
        self._on_trigger = callback
    
    def set_explanation_callback(
        self,
        callback: Callable[[ExplanationResult], None],
    ) -> None:
        """Set callback for explanation completions."""
        self._on_explanation = callback
    
    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        with self._lock:
            self._explanation_cache.clear()


def create_federated_rag_bridge(
    coordinator: Optional[Any] = None,
    knowledge_base: Optional[Any] = None,
    auto_trigger: bool = True,
) -> FederatedRAGBridge:
    """
    Factory function to create a configured FederatedRAGBridge.
    
    Args:
        coordinator: IntegrationCoordinator for RAG processing
        knowledge_base: ThreatKnowledgeBase for context
        auto_trigger: Whether to auto-trigger RAG updates
        
    Returns:
        Configured FederatedRAGBridge instance
    """
    bridge = FederatedRAGBridge(
        coordinator=coordinator,
        knowledge_base=knowledge_base,
        auto_trigger=auto_trigger,
    )
    
    logger.info("FederatedRAGBridge created via factory")
    return bridge


# Flower integration hook
def create_flower_strategy_callback(bridge: FederatedRAGBridge) -> Callable:
    """
    Create a callback function for Flower strategy integration.
    
    This callback should be called in the FedAvg strategy's
    aggregate_fit method to notify the bridge of updates.
    
    Usage in server.py:
        bridge = create_federated_rag_bridge(...)
        callback = create_flower_strategy_callback(bridge)
        
        class NetworkDefenseStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(self, ...):
                result = super().aggregate_fit(...)
                callback(round_number, weights, num_clients)
                return result
    """
    def on_aggregate_complete(
        round_number: int,
        aggregated_weights: Dict[str, Dict[str, Any]],
        num_clients: int,
    ) -> FederatedRound:
        return bridge.on_federated_round_complete(
            round_number=round_number,
            model_updates=aggregated_weights,
            participating_clients=num_clients,
        )
    
    return on_aggregate_complete
