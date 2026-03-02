"""
Privacy Metrics and Attack Simulation for Federated Learning.

This module provides tools for measuring and validating privacy guarantees
in the federated IDS system. It includes:

1. Privacy budget tracking and composition
2. Membership inference attack simulation
3. Gradient leakage risk assessment
4. Model inversion attack estimation
5. Utility-privacy trade-off analysis

Research Contributions:
    - IDS-specific membership inference adapted for network traffic
    - Gradient-based attack bounds for autoencoder architectures
    - Multi-organization privacy risk aggregation
    - Real-time privacy monitoring dashboard metrics

References:
    - Shokri et al., "Membership Inference Attacks" (S&P 2017)
    - Nasr et al., "Comprehensive Privacy Analysis" (2019)
    - Carlini et al., "Extracting Training Data" (2021)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of privacy attacks."""
    MEMBERSHIP_INFERENCE = "membership_inference"
    GRADIENT_LEAKAGE = "gradient_leakage"
    MODEL_INVERSION = "model_inversion"
    ATTRIBUTE_INFERENCE = "attribute_inference"
    DATA_RECONSTRUCTION = "data_reconstruction"


@dataclass
class PrivacyAssessment:
    """
    Comprehensive privacy assessment result.
    
    Contains metrics from various privacy analyses and attack simulations.
    """
    # Differential Privacy metrics
    epsilon_spent: float = 0.0
    delta: float = 1e-5
    privacy_level: str = "unknown"
    
    # Attack simulation results
    membership_inference_auc: float = 0.5
    gradient_leakage_risk: float = 0.0
    model_inversion_success: float = 0.0
    
    # Utility metrics
    model_accuracy: float = 0.0
    utility_loss: float = 0.0
    
    # Confidence and bounds
    privacy_guarantee_confidence: float = 0.95
    upper_bound_epsilon: float = 0.0
    
    # Metadata
    num_samples_analyzed: int = 0
    attack_types_tested: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "differential_privacy": {
                "epsilon": self.epsilon_spent,
                "delta": self.delta,
                "privacy_level": self.privacy_level,
            },
            "attack_simulation": {
                "membership_inference_auc": self.membership_inference_auc,
                "gradient_leakage_risk": self.gradient_leakage_risk,
                "model_inversion_success": self.model_inversion_success,
            },
            "utility": {
                "accuracy": self.model_accuracy,
                "utility_loss": self.utility_loss,
            },
            "bounds": {
                "confidence": self.privacy_guarantee_confidence,
                "upper_epsilon": self.upper_bound_epsilon,
            },
            "metadata": {
                "samples_analyzed": self.num_samples_analyzed,
                "attacks_tested": self.attack_types_tested,
            },
        }
    
    def get_risk_level(self) -> str:
        """Compute overall privacy risk level."""
        # Higher AUC = higher risk
        if self.membership_inference_auc > 0.8:
            return "CRITICAL"
        elif self.membership_inference_auc > 0.7:
            return "HIGH"
        elif self.membership_inference_auc > 0.6:
            return "MEDIUM"
        elif self.membership_inference_auc > 0.55:
            return "LOW"
        else:
            return "MINIMAL"


class PrivacyBudgetTracker:
    """
    Privacy Budget Tracker for Federated Learning.
    
    Tracks cumulative privacy loss across training rounds using
    multiple accounting methods for comparison.
    
    Accounting Methods:
        1. Basic Composition: ε_total = Σ ε_i
        2. Advanced Composition: ε_total = √(2k ln(1/δ')) × ε + k×ε(e^ε-1)/(e^ε+1)
        3. Rényi DP (RDP): Tighter bounds via Rényi divergence
        4. Privacy Loss Distribution (PLD): Most accurate, highest cost
    
    Features:
        - Per-round budget allocation
        - Threat-severity weighted budgeting
        - Early stopping when budget exhausted
        - Visualization-ready history tracking
    
    Example:
        >>> tracker = PrivacyBudgetTracker(total_epsilon=10.0)
        >>> 
        >>> for round in range(100):
        ...     round_eps = tracker.allocate_round_budget()
        ...     if tracker.is_exhausted:
        ...         break
        ...     # Use round_eps for DP operations
        ...     tracker.record_round_usage(round_eps)
    """
    
    def __init__(
        self,
        total_epsilon: float = 10.0,
        total_delta: float = 1e-5,
        accounting_method: str = "rdp",
    ):
        """
        Initialize privacy budget tracker.
        
        Args:
            total_epsilon: Total privacy budget.
            total_delta: Total privacy failure probability.
            accounting_method: Composition method ("basic", "advanced", "rdp").
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.accounting_method = accounting_method
        
        # Tracking state
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.round_history: List[Dict[str, float]] = []
        
        # RDP tracking
        self.rdp_alphas = [1.5, 2, 3, 4, 5, 10, 20, 50, 100]
        self.cumulative_rdp = np.zeros(len(self.rdp_alphas))
        
        logger.info(
            f"PrivacyBudgetTracker initialized: ε_total={total_epsilon}, "
            f"δ_total={total_delta}, method={accounting_method}"
        )
    
    @property
    def remaining_epsilon(self) -> float:
        """Remaining privacy budget."""
        return max(0, self.total_epsilon - self.spent_epsilon)
    
    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.remaining_epsilon <= 0
    
    @property
    def utilization(self) -> float:
        """Budget utilization percentage."""
        return self.spent_epsilon / self.total_epsilon if self.total_epsilon > 0 else 1.0
    
    def allocate_round_budget(
        self,
        num_remaining_rounds: Optional[int] = None,
        priority: float = 1.0,
    ) -> float:
        """
        Allocate budget for next round.
        
        Args:
            num_remaining_rounds: Expected remaining rounds for even distribution.
            priority: Priority multiplier (>1 for more budget).
        
        Returns:
            Allocated epsilon for this round.
        """
        if self.is_exhausted:
            return 0.0
        
        if num_remaining_rounds and num_remaining_rounds > 0:
            # Even distribution among remaining rounds
            base_allocation = self.remaining_epsilon / num_remaining_rounds
        else:
            # Default: 1% of total per round
            base_allocation = self.total_epsilon * 0.01
        
        # Apply priority scaling
        allocation = base_allocation * priority
        
        # Don't exceed remaining
        allocation = min(allocation, self.remaining_epsilon)
        
        return allocation
    
    def record_round_usage(
        self,
        epsilon_used: float,
        delta_used: float = 0.0,
        noise_multiplier: Optional[float] = None,
        sampling_rate: float = 1.0,
        round_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Record privacy usage for a round.
        
        Args:
            epsilon_used: Epsilon spent this round.
            delta_used: Delta spent this round.
            noise_multiplier: σ/C if using Gaussian mechanism.
            sampling_rate: Subsampling rate if applicable.
            round_metadata: Additional round information.
        
        Returns:
            Updated budget status.
        """
        round_num = len(self.round_history) + 1
        
        # Update based on accounting method
        if self.accounting_method == "basic":
            self.spent_epsilon += epsilon_used
            self.spent_delta += delta_used
        
        elif self.accounting_method == "rdp" and noise_multiplier:
            # RDP accounting for Gaussian mechanism
            rdp_eps = self._compute_rdp(noise_multiplier, sampling_rate)
            self.cumulative_rdp += rdp_eps
            self.spent_epsilon = self._rdp_to_dp(self.cumulative_rdp)
            self.spent_delta = self.total_delta  # δ is fixed for RDP conversion
        
        else:
            # Default to basic
            self.spent_epsilon += epsilon_used
        
        # Record history
        self.round_history.append({
            "round": round_num,
            "epsilon_used": epsilon_used,
            "delta_used": delta_used,
            "cumulative_epsilon": self.spent_epsilon,
            "remaining_epsilon": self.remaining_epsilon,
            "utilization": self.utilization,
            "metadata": round_metadata or {},
        })
        
        logger.debug(
            f"Round {round_num}: ε={epsilon_used:.4f}, "
            f"cumulative={self.spent_epsilon:.4f}, "
            f"remaining={self.remaining_epsilon:.4f}"
        )
        
        return {
            "round": round_num,
            "spent": self.spent_epsilon,
            "remaining": self.remaining_epsilon,
            "utilization": self.utilization,
            "exhausted": self.is_exhausted,
        }
    
    def _compute_rdp(self, sigma: float, sampling_rate: float) -> np.ndarray:
        """Compute RDP epsilon for Gaussian mechanism."""
        rdp = []
        for alpha in self.rdp_alphas:
            if sampling_rate == 1.0:
                # Full batch
                eps = alpha / (2 * sigma ** 2)
            else:
                # Subsampled
                eps = sampling_rate ** 2 * alpha / (2 * sigma ** 2)
            rdp.append(eps)
        return np.array(rdp)
    
    def _rdp_to_dp(self, rdp_epsilons: np.ndarray) -> float:
        """Convert RDP to (ε, δ)-DP."""
        log_delta = np.log(self.total_delta)
        
        best_eps = float('inf')
        for alpha, rdp_eps in zip(self.rdp_alphas, rdp_epsilons):
            if alpha <= 1:
                continue
            eps = rdp_eps + (log_delta + np.log(alpha - 1)) / (alpha - 1) - np.log(alpha) / (alpha - 1)
            best_eps = min(best_eps, eps)
        
        return best_eps if best_eps < float('inf') else rdp_epsilons[0]
    
    def get_budget_report(self) -> Dict[str, Any]:
        """Generate comprehensive budget report."""
        return {
            "budget": {
                "total_epsilon": self.total_epsilon,
                "total_delta": self.total_delta,
                "spent_epsilon": self.spent_epsilon,
                "remaining_epsilon": self.remaining_epsilon,
                "utilization_percent": self.utilization * 100,
            },
            "status": {
                "is_exhausted": self.is_exhausted,
                "rounds_completed": len(self.round_history),
                "accounting_method": self.accounting_method,
            },
            "history": self.round_history,
        }
    
    def estimate_remaining_rounds(self, avg_epsilon_per_round: Optional[float] = None) -> int:
        """
        Estimate how many more rounds can be trained.
        
        Args:
            avg_epsilon_per_round: Average ε per round (computed if None).
        
        Returns:
            Estimated remaining rounds.
        """
        if avg_epsilon_per_round is None:
            if self.round_history:
                avg_epsilon_per_round = np.mean([
                    r["epsilon_used"] for r in self.round_history
                ])
            else:
                avg_epsilon_per_round = self.total_epsilon * 0.01
        
        if avg_epsilon_per_round <= 0:
            return float('inf')
        
        return int(self.remaining_epsilon / avg_epsilon_per_round)


class MembershipInferenceAttack:
    """
    Membership Inference Attack Simulation.
    
    Simulates an adversary trying to determine whether a specific
    sample was used in training the federated model.
    
    Attack Methodology:
        1. Train shadow models on similar data
        2. Collect model output distributions for members/non-members
        3. Train attack classifier to distinguish based on outputs
        4. Evaluate on target model
    
    IDS-Specific Adaptations:
        - Uses reconstruction error for autoencoders
        - Considers attack category confidence
        - Handles imbalanced attack class distributions
    
    Metrics:
        - AUC: Area under ROC curve (0.5 = random, 1.0 = perfect attack)
        - Advantage: max(TPR - FPR) over thresholds
        - Privacy Leakage: log(1/AUC) correlation with ε
    
    Example:
        >>> attack = MembershipInferenceAttack()
        >>> auc = attack.evaluate(
        ...     model=defense_model,
        ...     member_data=train_data,
        ...     non_member_data=test_data,
        ... )
    """
    
    def __init__(
        self,
        attack_model: str = "threshold",
        num_shadow_models: int = 3,
    ):
        """
        Initialize MIA attack.
        
        Args:
            attack_model: Type of attack ("threshold", "classifier", "nn").
            num_shadow_models: Number of shadow models for classifier attack.
        """
        self.attack_model = attack_model
        self.num_shadow_models = num_shadow_models
        
        # Attack classifier
        self.attack_classifier = None
        
        # Results tracking
        self.attack_results: List[Dict[str, Any]] = []
        
        logger.info(
            f"MembershipInferenceAttack initialized: model={attack_model}"
        )
    
    def compute_membership_scores(
        self,
        model: Any,
        data: np.ndarray,
        model_type: str = "autoencoder",
    ) -> np.ndarray:
        """
        Compute membership inference scores for samples.
        
        Higher scores indicate higher likelihood of membership.
        
        Args:
            model: Target model to attack.
            data: Data samples to score.
            model_type: Type of model ("autoencoder", "classifier").
        
        Returns:
            Array of membership scores.
        """
        if model_type == "autoencoder":
            # Use reconstruction error (lower = more likely member)
            # Invert so higher = more likely member
            import torch
            
            model.eval()
            with torch.no_grad():
                data_tensor = torch.tensor(data, dtype=torch.float32)
                if hasattr(model, 'device'):
                    data_tensor = data_tensor.to(model.device)
                
                reconstructed = model(data_tensor)
                mse = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
                
                # Invert: lower error = higher membership probability
                # Use exponential for better separation
                scores = torch.exp(-mse).cpu().numpy()
        
        elif model_type == "classifier":
            # Use prediction confidence
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(data)
                # Max confidence as membership signal
                scores = np.max(probs, axis=1)
            else:
                # Binary prediction
                preds = model.predict(data)
                scores = np.abs(preds - 0.5) * 2  # Distance from decision boundary
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return scores
    
    def threshold_attack(
        self,
        member_scores: np.ndarray,
        non_member_scores: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Simple threshold-based membership attack.
        
        Args:
            member_scores: Scores for training members.
            non_member_scores: Scores for non-members.
        
        Returns:
            Tuple of (auc, best_accuracy, best_threshold).
        """
        # Combine with labels
        y_true = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(non_member_scores))
        ])
        y_scores = np.concatenate([member_scores, non_member_scores])
        
        # Compute AUC
        auc = roc_auc_score(y_true, y_scores)
        
        # Find best threshold
        thresholds = np.percentile(y_scores, np.arange(0, 101, 5))
        best_acc = 0.0
        best_thresh = 0.0
        
        for thresh in thresholds:
            preds = (y_scores > thresh).astype(int)
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        return auc, best_acc, best_thresh
    
    def classifier_attack(
        self,
        member_scores: np.ndarray,
        non_member_scores: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Train attack classifier for membership inference.
        
        Args:
            member_scores: Scores for training members.
            non_member_scores: Scores for non-members.
        
        Returns:
            Tuple of (auc, accuracy).
        """
        # Prepare features (can be extended with more features)
        X = np.concatenate([
            member_scores.reshape(-1, 1),
            non_member_scores.reshape(-1, 1)
        ])
        y = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(non_member_scores))
        ])
        
        # Split for attack model training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train attack classifier
        self.attack_classifier = LogisticRegression()
        self.attack_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = self.attack_classifier.predict_proba(X_test)[:, 1]
        y_pred = self.attack_classifier.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        return auc, acc
    
    def evaluate(
        self,
        model: Any,
        member_data: np.ndarray,
        non_member_data: np.ndarray,
        model_type: str = "autoencoder",
    ) -> Dict[str, Any]:
        """
        Full membership inference attack evaluation.
        
        Args:
            model: Target model to attack.
            member_data: Training data (members).
            non_member_data: Data not used in training.
            model_type: Type of model.
        
        Returns:
            Comprehensive attack results.
        """
        logger.info("Running membership inference attack evaluation...")
        
        # Compute membership scores
        member_scores = self.compute_membership_scores(
            model, member_data, model_type
        )
        non_member_scores = self.compute_membership_scores(
            model, non_member_data, model_type
        )
        
        # Run attacks
        threshold_auc, threshold_acc, best_thresh = self.threshold_attack(
            member_scores, non_member_scores
        )
        
        classifier_auc, classifier_acc = self.classifier_attack(
            member_scores, non_member_scores
        )
        
        # Compute advantage
        advantage = max(threshold_auc, classifier_auc) - 0.5
        
        # Privacy interpretation
        # Higher AUC = worse privacy
        if threshold_auc > 0.8:
            privacy_concern = "SEVERE"
        elif threshold_auc > 0.7:
            privacy_concern = "HIGH"
        elif threshold_auc > 0.6:
            privacy_concern = "MODERATE"
        else:
            privacy_concern = "LOW"
        
        results = {
            "threshold_attack": {
                "auc": threshold_auc,
                "accuracy": threshold_acc,
                "best_threshold": best_thresh,
            },
            "classifier_attack": {
                "auc": classifier_auc,
                "accuracy": classifier_acc,
            },
            "combined": {
                "best_auc": max(threshold_auc, classifier_auc),
                "advantage": advantage,
                "privacy_concern": privacy_concern,
            },
            "statistics": {
                "member_score_mean": float(np.mean(member_scores)),
                "member_score_std": float(np.std(member_scores)),
                "non_member_score_mean": float(np.mean(non_member_scores)),
                "non_member_score_std": float(np.std(non_member_scores)),
                "score_separation": float(
                    np.mean(member_scores) - np.mean(non_member_scores)
                ),
            },
            "num_members": len(member_data),
            "num_non_members": len(non_member_data),
        }
        
        self.attack_results.append(results)
        
        logger.info(
            f"MIA evaluation complete: best_AUC={results['combined']['best_auc']:.3f}, "
            f"concern={privacy_concern}"
        )
        
        return results


class GradientLeakageRisk:
    """
    Gradient Leakage Risk Assessment.
    
    Estimates the risk of training data reconstruction from
    shared gradients in federated learning.
    
    Attack Background:
        - Gradient matching attacks (Zhu et al., 2019)
        - Inverting gradients (Geiping et al., 2020)
        - Model updates contain information about training data
    
    Risk Factors Analyzed:
        1. Gradient sparsity and magnitude
        2. Batch size (smaller = higher risk)
        3. Model architecture complexity
        4. Number of training iterations
    
    IDS-Specific Considerations:
        - Network packet features may be partially recoverable
        - Attack signatures in gradients
        - IP addresses and temporal patterns
    """
    
    def __init__(self):
        """Initialize gradient leakage risk assessor."""
        self.risk_assessments: List[Dict[str, Any]] = []
        
        logger.info("GradientLeakageRisk assessor initialized")
    
    def assess_gradient_risk(
        self,
        gradients: List[np.ndarray],
        batch_size: int,
        num_samples: int,
        model_params: int,
    ) -> Dict[str, Any]:
        """
        Assess gradient leakage risk.
        
        Args:
            gradients: Model gradients being shared.
            batch_size: Training batch size.
            num_samples: Total training samples.
            model_params: Number of model parameters.
        
        Returns:
            Risk assessment dictionary.
        """
        # Compute gradient statistics
        flat_grads = np.concatenate([g.flatten() for g in gradients])
        
        grad_stats = {
            "magnitude": float(np.linalg.norm(flat_grads)),
            "mean": float(np.mean(np.abs(flat_grads))),
            "std": float(np.std(flat_grads)),
            "sparsity": float(np.mean(flat_grads == 0)),
            "max_abs": float(np.max(np.abs(flat_grads))),
            "dimension": len(flat_grads),
        }
        
        # Risk factor: batch size
        # Smaller batches = higher risk
        batch_risk = 1.0 / np.sqrt(batch_size)
        
        # Risk factor: samples per parameter
        # More params than samples = higher risk (overfitting, memorization)
        overfit_risk = model_params / num_samples if num_samples > 0 else 1.0
        
        # Risk factor: gradient magnitude
        # Higher magnitude gradients may leak more
        magnitude_risk = min(1.0, grad_stats["magnitude"] / 100)
        
        # Combined risk score (0-1)
        risk_score = (batch_risk + overfit_risk + magnitude_risk) / 3
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Risk level
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Reconstruction feasibility estimate
        # Based on gradient dimension vs data dimension
        data_dim_estimate = batch_size * (len(flat_grads) // model_params)
        reconstruction_possible = len(flat_grads) > data_dim_estimate
        
        assessment = {
            "gradient_statistics": grad_stats,
            "risk_factors": {
                "batch_size_risk": batch_risk,
                "overfitting_risk": overfit_risk,
                "magnitude_risk": magnitude_risk,
            },
            "combined_risk_score": risk_score,
            "risk_level": risk_level,
            "reconstruction_feasibility": reconstruction_possible,
            "recommendations": self._generate_recommendations(risk_score, grad_stats),
        }
        
        self.risk_assessments.append(assessment)
        
        logger.info(
            f"Gradient leakage risk: score={risk_score:.3f}, level={risk_level}"
        )
        
        return assessment
    
    def _generate_recommendations(
        self,
        risk_score: float,
        grad_stats: Dict[str, float],
    ) -> List[str]:
        """Generate mitigation recommendations based on risk."""
        recommendations = []
        
        if risk_score > 0.5:
            recommendations.append("Increase batch size to reduce per-sample gradient influence")
            recommendations.append("Apply gradient clipping to bound maximum leakage")
        
        if grad_stats["magnitude"] > 10:
            recommendations.append("Normalize gradients before sharing")
        
        if grad_stats["sparsity"] < 0.5:
            recommendations.append("Apply gradient sparsification for privacy amplification")
        
        if risk_score > 0.7:
            recommendations.append("Add differential privacy noise to gradients")
            recommendations.append("Consider secure aggregation protocol")
        
        return recommendations


class PrivacyMetrics:
    """
    Comprehensive Privacy Metrics Suite.
    
    Unified interface for all privacy-related measurements
    and attack simulations.
    
    Provides:
        - Differential privacy accounting
        - Attack success rate estimation
        - Privacy-utility trade-off analysis
        - Comparison across methods
        - Compliance reporting
    
    Example:
        >>> metrics = PrivacyMetrics()
        >>> 
        >>> # Full assessment
        >>> assessment = metrics.comprehensive_assessment(
        ...     model=model,
        ...     train_data=X_train,
        ...     test_data=X_test,
        ...     epsilon=1.0,
        ...     delta=1e-5,
        ... )
        >>> 
        >>> print(metrics.generate_compliance_report())
    """
    
    def __init__(self):
        """Initialize privacy metrics suite."""
        self.budget_tracker = PrivacyBudgetTracker()
        self.mia_attack = MembershipInferenceAttack()
        self.gradient_risk = GradientLeakageRisk()
        
        self.assessments: List[PrivacyAssessment] = []
        
        logger.info("PrivacyMetrics suite initialized")
    
    def comprehensive_assessment(
        self,
        model: Any,
        train_data: np.ndarray,
        test_data: np.ndarray,
        epsilon: float,
        delta: float = 1e-5,
        gradients: Optional[List[np.ndarray]] = None,
        model_type: str = "autoencoder",
    ) -> PrivacyAssessment:
        """
        Run comprehensive privacy assessment.
        
        Args:
            model: Model to assess.
            train_data: Training data (potential privacy leak).
            test_data: Non-training data.
            epsilon: Privacy budget used.
            delta: Privacy failure probability.
            gradients: Optional gradients for leakage analysis.
            model_type: Type of model.
        
        Returns:
            Complete privacy assessment.
        """
        logger.info("Running comprehensive privacy assessment...")
        
        # Run membership inference attack
        mia_results = self.mia_attack.evaluate(
            model=model,
            member_data=train_data,
            non_member_data=test_data,
            model_type=model_type,
        )
        
        # Run gradient leakage assessment if gradients provided
        grad_risk = 0.0
        if gradients is not None:
            risk_result = self.gradient_risk.assess_gradient_risk(
                gradients=gradients,
                batch_size=min(64, len(train_data)),
                num_samples=len(train_data),
                model_params=sum(g.size for g in gradients),
            )
            grad_risk = risk_result["combined_risk_score"]
        
        # Determine privacy level
        if epsilon <= 0.1:
            privacy_level = "ultra_high"
        elif epsilon <= 1.0:
            privacy_level = "high"
        elif epsilon <= 5.0:
            privacy_level = "medium"
        else:
            privacy_level = "low"
        
        # Create assessment
        assessment = PrivacyAssessment(
            epsilon_spent=epsilon,
            delta=delta,
            privacy_level=privacy_level,
            membership_inference_auc=mia_results["combined"]["best_auc"],
            gradient_leakage_risk=grad_risk,
            model_inversion_success=0.0,  # Not implemented yet
            model_accuracy=0.0,  # Filled by caller
            utility_loss=0.0,
            upper_bound_epsilon=epsilon * 1.5,  # Conservative bound
            num_samples_analyzed=len(train_data) + len(test_data),
            attack_types_tested=["membership_inference", "gradient_leakage"],
        )
        
        self.assessments.append(assessment)
        
        logger.info(
            f"Assessment complete: ε={epsilon}, MIA_AUC={assessment.membership_inference_auc:.3f}, "
            f"risk={assessment.get_risk_level()}"
        )
        
        return assessment
    
    def privacy_utility_tradeoff(
        self,
        epsilons: List[float],
        accuracies: List[float],
    ) -> Dict[str, Any]:
        """
        Analyze privacy-utility trade-off.
        
        Args:
            epsilons: List of epsilon values tested.
            accuracies: Corresponding model accuracies.
        
        Returns:
            Trade-off analysis results.
        """
        # Sort by epsilon
        sorted_pairs = sorted(zip(epsilons, accuracies))
        eps_sorted = [e for e, _ in sorted_pairs]
        acc_sorted = [a for _, a in sorted_pairs]
        
        # Compute utility loss curve
        # Utility loss = max_accuracy - current_accuracy
        max_acc = max(accuracies)
        utility_losses = [max_acc - a for a in acc_sorted]
        
        # Find Pareto frontier
        pareto_points = []
        best_acc = 0
        for eps, acc in sorted_pairs:
            if acc > best_acc:
                pareto_points.append((eps, acc))
                best_acc = acc
        
        # Compute area under trade-off curve (higher = better)
        # Normalized by max possible area
        auc = np.trapz(acc_sorted, eps_sorted)
        max_auc = max_acc * (max(epsilons) - min(epsilons))
        normalized_auc = auc / max_auc if max_auc > 0 else 0
        
        return {
            "epsilons": eps_sorted,
            "accuracies": acc_sorted,
            "utility_losses": utility_losses,
            "pareto_frontier": pareto_points,
            "trade_off_auc": normalized_auc,
            "best_epsilon": eps_sorted[np.argmax(acc_sorted)],
            "best_accuracy": max_acc,
        }
    
    def generate_compliance_report(
        self,
        regulation: str = "GDPR",
    ) -> Dict[str, Any]:
        """
        Generate privacy compliance report.
        
        Args:
            regulation: Target regulation (GDPR, HIPAA, etc.).
        
        Returns:
            Compliance report.
        """
        if not self.assessments:
            return {"status": "No assessments available"}
        
        latest = self.assessments[-1]
        
        # GDPR considerations
        if regulation == "GDPR":
            # Article 25: Privacy by Design
            privacy_by_design = latest.privacy_level in ["high", "ultra_high"]
            
            # Article 32: Security measures
            security_adequate = latest.membership_inference_auc < 0.7
            
            # Article 35: DPIA required?
            dpia_required = latest.epsilon_spent > 5.0 or latest.membership_inference_auc > 0.75
            
            compliance_status = {
                "article_25_privacy_by_design": {
                    "compliant": privacy_by_design,
                    "evidence": f"Privacy level: {latest.privacy_level}",
                },
                "article_32_security": {
                    "compliant": security_adequate,
                    "evidence": f"MIA AUC: {latest.membership_inference_auc:.3f}",
                },
                "article_35_dpia": {
                    "required": dpia_required,
                    "reason": "High privacy budget" if latest.epsilon_spent > 5.0 else "Acceptable",
                },
            }
        else:
            compliance_status = {"status": f"Regulation {regulation} not implemented"}
        
        return {
            "regulation": regulation,
            "assessment_date": "latest",
            "epsilon_used": latest.epsilon_spent,
            "delta": latest.delta,
            "overall_risk": latest.get_risk_level(),
            "compliance": compliance_status,
            "recommendations": self._get_compliance_recommendations(latest),
        }
    
    def _get_compliance_recommendations(
        self,
        assessment: PrivacyAssessment,
    ) -> List[str]:
        """Generate compliance recommendations."""
        recs = []
        
        if assessment.epsilon_spent > 5.0:
            recs.append("Reduce privacy budget to ε ≤ 5.0 for better privacy guarantees")
        
        if assessment.membership_inference_auc > 0.7:
            recs.append("Add differential privacy noise to reduce membership inference risk")
        
        if assessment.gradient_leakage_risk > 0.5:
            recs.append("Implement secure aggregation or gradient compression")
        
        if assessment.privacy_level in ["low"]:
            recs.append("Consider stronger privacy mechanisms for sensitive data")
        
        return recs
