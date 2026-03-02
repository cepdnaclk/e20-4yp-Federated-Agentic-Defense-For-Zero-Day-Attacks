"""
Privacy Metrics Collector for Federated Learning

Tracks and calculates privacy-related metrics for the FL system including:
- Differential Privacy (epsilon, delta, noise scale)
- Data Leakage Risk (weight update analysis, gradient similarity)
- Communication Privacy (data volume, embedding abstraction)
- Federation Health (participation, convergence)
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class PrivacyRoundMetrics:
    """Metrics for a single federation round"""
    round_id: int
    timestamp: str
    
    # Differential Privacy Metrics
    epsilon: float = 0.0  # Privacy budget consumed this round
    delta: float = 1e-5   # Privacy failure probability
    noise_scale: float = 0.0  # Noise multiplier applied
    cumulative_epsilon: float = 0.0  # Total privacy budget spent
    
    # Data Leakage Risk Metrics
    weight_update_magnitude: float = 0.0  # L2 norm of weight updates
    weight_update_sparsity: float = 0.0  # Fraction of non-zero updates
    gradient_similarity: float = 0.0  # Similarity between agent updates
    information_exposure_risk: float = 0.0  # Risk score (0-1)
    
    # Communication Privacy Metrics
    bytes_transmitted: int = 0  # Total bytes sent this round
    embedding_dimension: int = 0  # Dimensionality of shared embeddings
    abstraction_level: float = 0.0  # How abstracted the data is (0-1)
    signature_count: int = 0  # Number of signatures shared
    
    # Federation Health Metrics
    participating_agents: int = 0
    total_samples: int = 0
    model_convergence_delta: float = 0.0  # Change from previous round
    zero_day_candidates_found: int = 0
    
    # Agent-specific privacy
    agent_privacy_scores: Dict[str, float] = field(default_factory=dict)


class PrivacyMetricsCollector:
    """
    Collects and tracks privacy metrics across federation rounds.
    
    Usage:
        collector = PrivacyMetricsCollector()
        collector.start_round(round_id=1)
        collector.record_agent_update(agent_id, weights, sample_count)
        collector.record_signatures(agent_id, embeddings)
        metrics = collector.end_round()
    """
    
    def __init__(
        self,
        log_path: str = "./privacy_logs",
        target_epsilon: float = 10.0,  # Target total privacy budget
        target_delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        clip_norm: float = 1.0
    ):
        self.log_path = log_path
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        
        self._rounds: List[PrivacyRoundMetrics] = []
        self._current_round: Optional[PrivacyRoundMetrics] = None
        self._current_updates: List[Dict] = []
        self._previous_global_weights: Optional[List[np.ndarray]] = None
        self._cumulative_epsilon: float = 0.0
        
        os.makedirs(log_path, exist_ok=True)
    
    def start_round(self, round_id: int) -> None:
        """Start tracking a new federation round"""
        self._current_round = PrivacyRoundMetrics(
            round_id=round_id,
            timestamp=datetime.now().isoformat()
        )
        self._current_updates = []
    
    def record_agent_update(
        self,
        agent_id: str,
        weights: Optional[List[np.ndarray]],
        sample_count: int,
        raw_bytes: int = 0
    ) -> None:
        """Record a model weight update from an agent"""
        if self._current_round is None:
            return
        
        self._current_round.participating_agents += 1
        self._current_round.total_samples += sample_count
        self._current_round.bytes_transmitted += raw_bytes
        
        if weights is not None:
            # Calculate weight update magnitude
            magnitude = sum(np.linalg.norm(w) for w in weights)
            
            # Calculate sparsity
            total_params = sum(w.size for w in weights)
            non_zero = sum(np.count_nonzero(w) for w in weights)
            sparsity = non_zero / max(total_params, 1)
            
            self._current_updates.append({
                "agent_id": agent_id,
                "weights": weights,
                "magnitude": magnitude,
                "sparsity": sparsity,
                "sample_count": sample_count
            })
    
    def record_signatures(
        self,
        agent_id: str,
        embeddings: np.ndarray,
        recon_errors: np.ndarray
    ) -> None:
        """Record signature submissions from an agent"""
        if self._current_round is None:
            return
        
        if embeddings.size > 0:
            self._current_round.signature_count += len(embeddings)
            self._current_round.embedding_dimension = embeddings.shape[1] if embeddings.ndim > 1 else 0
            
            # Calculate abstraction level based on reconstruction error distribution
            # Higher recon errors suggest more abstract/compressed representations
            if len(recon_errors) > 0:
                mean_recon = float(np.mean(recon_errors))
                # Normalize to 0-1 scale (assuming typical recon errors are 0-2)
                abstraction = min(1.0, mean_recon / 2.0)
                # Update running average
                if self._current_round.abstraction_level == 0:
                    self._current_round.abstraction_level = abstraction
                else:
                    self._current_round.abstraction_level = (
                        self._current_round.abstraction_level + abstraction
                    ) / 2
    
    def calculate_dp_metrics(self, aggregated_weights: List[np.ndarray]) -> Tuple[float, float]:
        """
        Calculate differential privacy metrics using the Gaussian mechanism.
        
        Returns:
            (epsilon, noise_scale) for this round
        """
        if not self._current_updates:
            return 0.0, 0.0
        
        # Calculate sensitivity (max L2 norm of clipped gradients)
        max_magnitude = max(u["magnitude"] for u in self._current_updates)
        sensitivity = min(max_magnitude, self.clip_norm * 2)
        
        # Calculate noise scale needed to achieve (epsilon, delta)-DP
        # Using the standard Gaussian mechanism formula
        # sigma >= sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        if self.target_epsilon > 0:
            min_noise = (
                np.sqrt(2 * np.log(1.25 / self.target_delta)) 
                * sensitivity 
                / self.target_epsilon
            )
        else:
            min_noise = 1.0
        
        actual_noise = self.noise_multiplier * sensitivity
        
        # Calculate epsilon achieved with actual noise
        if actual_noise > 0:
            epsilon_achieved = (
                np.sqrt(2 * np.log(1.25 / self.target_delta)) 
                * sensitivity 
                / actual_noise
            )
        else:
            epsilon_achieved = float('inf')
        
        return epsilon_achieved, actual_noise
    
    def calculate_leakage_risk(self) -> Tuple[float, float]:
        """
        Calculate information leakage risk metrics.
        
        Returns:
            (gradient_similarity, exposure_risk)
        """
        if len(self._current_updates) < 2:
            return 0.0, 0.0
        
        # Calculate pairwise gradient similarity
        similarities = []
        for i, u1 in enumerate(self._current_updates[:-1]):
            for u2 in self._current_updates[i+1:]:
                # Flatten and normalize weights
                w1 = np.concatenate([w.flatten() for w in u1["weights"]])
                w2 = np.concatenate([w.flatten() for w in u2["weights"]])
                
                # Cosine similarity
                norm1, norm2 = np.linalg.norm(w1), np.linalg.norm(w2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(w1, w2) / (norm1 * norm2)
                    similarities.append(abs(sim))
        
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        
        # Calculate exposure risk based on:
        # 1. High gradient similarity (suggests memorization)
        # 2. Large update magnitudes (more information leaked)
        # 3. Low noise (less privacy protection)
        
        avg_magnitude = np.mean([u["magnitude"] for u in self._current_updates])
        normalized_magnitude = min(1.0, avg_magnitude / 100)  # Normalize
        
        exposure_risk = (
            0.4 * avg_similarity +
            0.3 * normalized_magnitude +
            0.3 * (1 - min(1.0, self.noise_multiplier))
        )
        
        return avg_similarity, exposure_risk
    
    def end_round(
        self,
        aggregated_weights: Optional[List[np.ndarray]] = None,
        zero_day_count: int = 0
    ) -> PrivacyRoundMetrics:
        """End the current round and calculate final metrics"""
        if self._current_round is None:
            raise ValueError("No round in progress. Call start_round() first.")
        
        # Calculate DP metrics
        if aggregated_weights and self._current_updates:
            epsilon, noise_scale = self.calculate_dp_metrics(aggregated_weights)
            self._current_round.epsilon = epsilon
            self._current_round.noise_scale = noise_scale
            self._current_round.delta = self.target_delta
            
            # Update cumulative epsilon (using simple composition)
            self._cumulative_epsilon += epsilon
            self._current_round.cumulative_epsilon = self._cumulative_epsilon
            
            # Calculate weight update metrics
            magnitudes = [u["magnitude"] for u in self._current_updates]
            sparsities = [u["sparsity"] for u in self._current_updates]
            self._current_round.weight_update_magnitude = float(np.mean(magnitudes))
            self._current_round.weight_update_sparsity = float(np.mean(sparsities))
            
            # Calculate leakage risk
            grad_sim, exposure = self.calculate_leakage_risk()
            self._current_round.gradient_similarity = grad_sim
            self._current_round.information_exposure_risk = exposure
            
            # Calculate model convergence
            if self._previous_global_weights is not None:
                delta = sum(
                    np.linalg.norm(new - old)
                    for new, old in zip(aggregated_weights, self._previous_global_weights)
                )
                self._current_round.model_convergence_delta = float(delta)
            
            self._previous_global_weights = [w.copy() for w in aggregated_weights]
            
            # Per-agent privacy scores
            for update in self._current_updates:
                # Higher sparsity and lower magnitude = better privacy
                score = 1.0 - (
                    0.5 * min(1.0, update["magnitude"] / 100) +
                    0.5 * update["sparsity"]
                )
                self._current_round.agent_privacy_scores[update["agent_id"]] = score
        
        self._current_round.zero_day_candidates_found = zero_day_count
        
        # Save metrics
        self._rounds.append(self._current_round)
        self._save_round(self._current_round)
        
        result = self._current_round
        self._current_round = None
        return result
    
    def _save_round(self, metrics: PrivacyRoundMetrics) -> None:
        """Save round metrics to JSON file"""
        filepath = os.path.join(
            self.log_path,
            f"privacy_round_{metrics.round_id}.json"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, cls=NumpyEncoder)
    
    def get_all_metrics(self) -> List[Dict]:
        """Get all collected metrics as dictionaries"""
        return [asdict(r) for r in self._rounds]
    
    def get_privacy_summary(self) -> Dict:
        """Get summary statistics across all rounds"""
        if not self._rounds:
            return {"status": "no_data"}
        
        return {
            "total_rounds": len(self._rounds),
            "cumulative_epsilon": self._cumulative_epsilon,
            "target_epsilon": self.target_epsilon,
            "privacy_budget_consumed": f"{(self._cumulative_epsilon / self.target_epsilon) * 100:.1f}%",
            "avg_exposure_risk": float(np.mean([
                r.information_exposure_risk for r in self._rounds
            ])),
            "avg_gradient_similarity": float(np.mean([
                r.gradient_similarity for r in self._rounds
            ])),
            "total_signatures_shared": sum(r.signature_count for r in self._rounds),
            "total_bytes_transmitted": sum(r.bytes_transmitted for r in self._rounds),
            "avg_participants_per_round": float(np.mean([
                r.participating_agents for r in self._rounds
            ])),
        }
    
    def load_from_logs(self) -> None:
        """Load previously saved metrics from log files"""
        self._rounds = []
        if not os.path.exists(self.log_path):
            return
        
        for filename in sorted(os.listdir(self.log_path)):
            if filename.startswith("privacy_round_") and filename.endswith(".json"):
                filepath = os.path.join(self.log_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    metrics = PrivacyRoundMetrics(**data)
                    self._rounds.append(metrics)
        
        if self._rounds:
            self._cumulative_epsilon = self._rounds[-1].cumulative_epsilon
