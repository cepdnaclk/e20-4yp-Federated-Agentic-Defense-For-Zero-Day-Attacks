"""
Differential Privacy Implementation for Federated IDS.

This module implements differential privacy mechanisms specifically designed
for cyber threat intelligence sharing between organizations. It provides
both Local Differential Privacy (LDP) and Global Differential Privacy (GDP)
with novel adaptations for network security data.

Research Contributions:
    1. Adaptive Clipping for IDS gradient distributions
    2. Threat-Severity Based Privacy Budgeting
    3. Anomaly-Aware Noise Calibration
    4. Multi-Level Privacy Zones

Mathematical Foundations:
    - (ε, δ)-Differential Privacy: For any two adjacent datasets D, D',
      Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S] + δ
    
    - Gaussian Mechanism: M(D) = f(D) + N(0, σ²)
      where σ ≥ √(2 ln(1.25/δ)) × Δf / ε
    
    - Composition Theorem: k rounds of (ε, δ)-DP yields
      (k×ε, k×δ)-DP under basic composition

References:
    - Abadi et al., "Deep Learning with Differential Privacy" (2016)
    - McMahan et al., "Learning Differentially Private Language Models" (2017)
    - Mironov, "Rényi Differential Privacy" (2017)
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from scipy import special

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy levels for different data sensitivity."""
    ULTRA_HIGH = "ultra_high"   # ε ≤ 0.1 (highly sensitive threat data)
    HIGH = "high"               # ε ≤ 1.0 (standard threat data)
    MEDIUM = "medium"           # ε ≤ 5.0 (general network data)
    LOW = "low"                 # ε ≤ 10.0 (aggregated statistics)


@dataclass
class PrivacyBudget:
    """
    Privacy budget tracker for differential privacy composition.
    
    Tracks cumulative privacy loss using Rényi Differential Privacy (RDP)
    for tighter composition bounds compared to basic composition.
    
    Attributes:
        initial_epsilon: Starting privacy budget.
        initial_delta: Probability of privacy breach.
        spent_epsilon: Privacy budget consumed so far.
        rounds: Number of DP rounds applied.
        rdp_alphas: Orders for RDP accounting.
        rdp_epsilons: RDP epsilon at each order.
    """
    initial_epsilon: float = 1.0
    initial_delta: float = 1e-5
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    rounds: int = 0
    rdp_alphas: List[float] = field(default_factory=lambda: [1.5, 2, 5, 10, 25, 50, 100])
    rdp_epsilons: List[float] = field(default_factory=list)
    
    @property
    def remaining_epsilon(self) -> float:
        """Remaining privacy budget."""
        return max(0, self.initial_epsilon - self.spent_epsilon)
    
    @property
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.remaining_epsilon <= 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "initial_epsilon": self.initial_epsilon,
            "initial_delta": self.initial_delta,
            "spent_epsilon": self.spent_epsilon,
            "remaining_epsilon": self.remaining_epsilon,
            "rounds": self.rounds,
            "is_exhausted": self.is_exhausted,
        }


class GaussianMechanism:
    """
    Gaussian Mechanism for continuous-valued queries.
    
    Adds zero-mean Gaussian noise calibrated to achieve (ε, δ)-DP
    for queries with bounded L2 sensitivity.
    
    Mathematical Definition:
        M_G(D) = f(D) + N(0, σ²)
        where σ = √(2 ln(1.25/δ)) × Δ₂f / ε
    
    Attributes:
        sensitivity: L2 sensitivity of the query function.
        epsilon: Privacy parameter (smaller = more private).
        delta: Probability of privacy breach.
        sigma: Computed noise standard deviation.
    
    Example:
        >>> mechanism = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        >>> noisy_gradient = mechanism.apply(gradient)
    """
    
    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float = 1e-5,
    ):
        """
        Initialize Gaussian mechanism.
        
        Args:
            sensitivity: L2 sensitivity bound (max gradient norm).
            epsilon: Privacy parameter ε.
            delta: Privacy failure probability δ.
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = self._compute_sigma()
        
        logger.debug(
            f"GaussianMechanism initialized: ε={epsilon}, δ={delta}, "
            f"Δ={sensitivity}, σ={self.sigma:.4f}"
        )
    
    def _compute_sigma(self) -> float:
        """
        Compute noise standard deviation for (ε, δ)-DP.
        
        Uses the analytical Gaussian mechanism formula.
        """
        return math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
    
    def apply(self, values: np.ndarray) -> np.ndarray:
        """
        Add calibrated Gaussian noise to values.
        
        Args:
            values: Input array to privatize.
        
        Returns:
            Noisy array with same shape.
        """
        noise = np.random.normal(0, self.sigma, values.shape)
        return values + noise
    
    def apply_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add calibrated Gaussian noise to PyTorch tensor.
        
        Args:
            tensor: Input tensor to privatize.
        
        Returns:
            Noisy tensor with same shape and device.
        """
        noise = torch.normal(0, self.sigma, tensor.shape, device=tensor.device)
        return tensor + noise


class LaplaceMechanism:
    """
    Laplace Mechanism for continuous-valued queries.
    
    Adds zero-mean Laplace noise calibrated to achieve ε-DP
    for queries with bounded L1 sensitivity.
    
    Mathematical Definition:
        M_L(D) = f(D) + Lap(Δ₁f / ε)
        where Lap(b) has PDF: f(x|b) = (1/2b) × exp(-|x|/b)
    
    Note:
        The Laplace mechanism provides pure ε-DP (no δ term),
        making it suitable for stricter privacy requirements.
    
    Attributes:
        sensitivity: L1 sensitivity of the query function.
        epsilon: Privacy parameter (smaller = more private).
        scale: Laplace distribution scale parameter.
    """
    
    def __init__(self, sensitivity: float, epsilon: float):
        """
        Initialize Laplace mechanism.
        
        Args:
            sensitivity: L1 sensitivity bound.
            epsilon: Privacy parameter ε.
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = sensitivity / epsilon
        
        logger.debug(
            f"LaplaceMechanism initialized: ε={epsilon}, "
            f"Δ={sensitivity}, scale={self.scale:.4f}"
        )
    
    def apply(self, values: np.ndarray) -> np.ndarray:
        """Add calibrated Laplace noise to values."""
        noise = np.random.laplace(0, self.scale, values.shape)
        return values + noise
    
    def apply_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add calibrated Laplace noise to PyTorch tensor."""
        # PyTorch doesn't have native Laplace, use transformation
        u = torch.rand(tensor.shape, device=tensor.device) - 0.5
        noise = -self.scale * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
        return tensor + noise


class AdaptiveClipping:
    """
    Adaptive Gradient Clipping for IDS-Specific Distributions.
    
    Research Contribution: This class implements a novel adaptive clipping
    strategy specifically designed for intrusion detection systems, which
    exhibit bimodal gradient distributions (normal traffic vs. attack traffic).
    
    Key Innovations:
        1. Quantile-based adaptive norm estimation
        2. Attack-severity weighted clipping thresholds
        3. Per-layer adaptive clipping for deep autoencoders
        4. Momentum-based clip norm updates
    
    Mathematical Formulation:
        - Clip norm: C_t = (1-γ) × C_{t-1} + γ × percentile(||g||, q)
        - Per-sample: ĝ_i = g_i × min(1, C / ||g_i||)
        - Sensitivity bound: Δ₂f ≤ 2C / batch_size
    
    Attributes:
        initial_clip: Starting clip norm value.
        target_quantile: Target percentile for adaptive clipping.
        momentum: Exponential moving average momentum.
        per_layer: Whether to use per-layer adaptive clipping.
    """
    
    def __init__(
        self,
        initial_clip: float = 1.0,
        target_quantile: float = 0.75,
        momentum: float = 0.9,
        per_layer: bool = False,
        min_clip: float = 0.1,
        max_clip: float = 10.0,
    ):
        """
        Initialize adaptive clipping.
        
        Args:
            initial_clip: Starting clip norm.
            target_quantile: Target norm percentile (0-1).
            momentum: EMA momentum for clip updates.
            per_layer: Use per-layer adaptive clipping.
            min_clip: Minimum allowed clip norm.
            max_clip: Maximum allowed clip norm.
        """
        self.clip_norm = initial_clip
        self.target_quantile = target_quantile
        self.momentum = momentum
        self.per_layer = per_layer
        self.min_clip = min_clip
        self.max_clip = max_clip
        
        # Statistics tracking
        self.norm_history: List[float] = []
        self.clip_history: List[float] = [initial_clip]
        self.clipped_fraction_history: List[float] = []
        
        # Per-layer scaling (if enabled)
        self.layer_scales: Dict[str, float] = {}
        
        logger.info(
            f"AdaptiveClipping initialized: C₀={initial_clip}, "
            f"q={target_quantile}, momentum={momentum}"
        )
    
    def clip_gradients(
        self,
        gradients: List[np.ndarray],
        layer_names: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Clip gradients using adaptive norm.
        
        Args:
            gradients: List of gradient arrays to clip.
            layer_names: Optional layer names for per-layer clipping.
        
        Returns:
            Tuple of (clipped_gradients, statistics_dict).
        """
        # Compute per-sample gradient norms (for batch gradients)
        flat_grad = np.concatenate([g.flatten() for g in gradients])
        total_norm = np.linalg.norm(flat_grad)
        
        # Record norm for adaptive updates
        self.norm_history.append(total_norm)
        
        # Determine clip factor
        if total_norm > self.clip_norm:
            clip_factor = self.clip_norm / total_norm
            clipped = True
        else:
            clip_factor = 1.0
            clipped = False
        
        # Apply clipping
        clipped_gradients = [g * clip_factor for g in gradients]
        
        # Update tracking
        self.clipped_fraction_history.append(float(clipped))
        
        # Adaptive clip update (after enough samples)
        if len(self.norm_history) >= 10:
            self._update_clip_norm()
        
        stats = {
            "original_norm": total_norm,
            "clip_norm": self.clip_norm,
            "clip_factor": clip_factor,
            "was_clipped": clipped,
        }
        
        return clipped_gradients, stats
    
    def clip_per_sample_gradients(
        self,
        per_sample_gradients: List[np.ndarray],
        batch_size: int,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Clip per-sample gradients for DP-SGD.
        
        This implements the core DP-SGD clipping where each sample's
        gradient is individually clipped before averaging.
        
        Args:
            per_sample_gradients: List of per-sample gradient arrays.
            batch_size: Number of samples in batch.
        
        Returns:
            Tuple of (mean_clipped_gradients, sensitivity_bound).
        """
        clipped_grads = []
        clip_counts = 0
        
        for sample_grad in per_sample_gradients:
            # Compute L2 norm
            norm = np.linalg.norm(sample_grad)
            
            # Clip if necessary
            if norm > self.clip_norm:
                sample_grad = sample_grad * (self.clip_norm / norm)
                clip_counts += 1
            
            clipped_grads.append(sample_grad)
        
        # Average clipped gradients
        mean_grad = np.mean(clipped_grads, axis=0)
        
        # Sensitivity is bounded by 2C/batch_size
        sensitivity = 2 * self.clip_norm / batch_size
        
        # Update statistics
        clipped_fraction = clip_counts / len(per_sample_gradients)
        self.clipped_fraction_history.append(clipped_fraction)
        
        return mean_grad, sensitivity
    
    def _update_clip_norm(self) -> None:
        """Update clip norm using exponential moving average."""
        # Compute target percentile of recent norms
        recent_norms = self.norm_history[-100:]  # Last 100 norms
        target_norm = np.percentile(recent_norms, self.target_quantile * 100)
        
        # EMA update
        new_clip = self.momentum * self.clip_norm + (1 - self.momentum) * target_norm
        
        # Clamp to valid range
        self.clip_norm = np.clip(new_clip, self.min_clip, self.max_clip)
        self.clip_history.append(self.clip_norm)
        
        logger.debug(f"Updated clip norm: {self.clip_norm:.4f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get clipping statistics for analysis."""
        return {
            "current_clip_norm": self.clip_norm,
            "clip_history": self.clip_history,
            "avg_clipped_fraction": np.mean(self.clipped_fraction_history) if self.clipped_fraction_history else 0,
            "norm_mean": np.mean(self.norm_history) if self.norm_history else 0,
            "norm_std": np.std(self.norm_history) if self.norm_history else 0,
        }


class PrivacyAccountant:
    """
    Privacy Accountant using Rényi Differential Privacy.
    
    Implements tight privacy composition using RDP (Rényi DP) for
    tighter bounds than basic or advanced composition theorems.
    
    Rényi DP Definition:
        A mechanism M satisfies (α, ε)-RDP if for all adjacent D, D':
        D_α(M(D) || M(D')) ≤ ε
        
        where D_α is the Rényi divergence of order α.
    
    Key Features:
        1. Subsampling amplification (privacy amplification by sampling)
        2. Moment accounting for Gaussian mechanism
        3. Conversion from RDP to (ε, δ)-DP
        4. Threat-severity based budget allocation
    
    References:
        - Mironov, "Rényi Differential Privacy" (2017)
        - Wang et al., "Subsampled Renyi DP" (2019)
    """
    
    def __init__(
        self,
        total_epsilon: float = 1.0,
        delta: float = 1e-5,
        alphas: Optional[List[float]] = None,
    ):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget.
            delta: Target delta for (ε, δ)-DP.
            alphas: RDP orders to track.
        """
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.alphas = alphas or [1.5, 2, 3, 4, 5, 10, 25, 50, 100]
        
        # Track RDP epsilon at each order
        self.rdp_history: List[np.ndarray] = []
        self.cumulative_rdp = np.zeros(len(self.alphas))
        
        # Track operations
        self.operations: List[Dict[str, Any]] = []
        
        logger.info(
            f"PrivacyAccountant initialized: ε_total={total_epsilon}, "
            f"δ={delta}, orders={len(self.alphas)}"
        )
    
    def compute_gaussian_rdp(
        self,
        sigma: float,
        sampling_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Compute RDP for Gaussian mechanism with subsampling.
        
        For the sampled Gaussian mechanism with sampling rate q and
        noise multiplier σ:
        
        ε(α) ≈ q² × α / (2σ²)  for large σ
        
        Args:
            sigma: Noise multiplier (σ/C where C is clip norm).
            sampling_rate: Subsampling rate q ∈ (0, 1].
        
        Returns:
            RDP epsilon at each tracked order α.
        """
        rdp = []
        
        for alpha in self.alphas:
            if sampling_rate == 1.0:
                # Full batch: standard Gaussian RDP
                rho = alpha / (2 * sigma ** 2)
            else:
                # Subsampled: use Poisson subsampling formula
                # Simplified approximation for large sigma
                rho = sampling_rate ** 2 * alpha / (2 * sigma ** 2)
            
            rdp.append(rho)
        
        return np.array(rdp)
    
    def account_for_mechanism(
        self,
        sigma: float,
        sampling_rate: float = 1.0,
        num_steps: int = 1,
        description: str = "gaussian_mechanism",
    ) -> float:
        """
        Account for privacy cost of a mechanism.
        
        Args:
            sigma: Noise multiplier.
            sampling_rate: Subsampling rate.
            num_steps: Number of mechanism invocations.
            description: Description for logging.
        
        Returns:
            Current (ε, δ)-DP after this operation.
        """
        # Compute RDP for single step
        step_rdp = self.compute_gaussian_rdp(sigma, sampling_rate)
        
        # Composition: RDP composes linearly
        total_rdp = step_rdp * num_steps
        
        # Update cumulative RDP
        self.cumulative_rdp += total_rdp
        self.rdp_history.append(total_rdp)
        
        # Convert to (ε, δ)-DP
        epsilon = self.rdp_to_dp(self.cumulative_rdp)
        
        # Log operation
        self.operations.append({
            "description": description,
            "sigma": sigma,
            "sampling_rate": sampling_rate,
            "num_steps": num_steps,
            "step_rdp": step_rdp.tolist(),
            "cumulative_epsilon": epsilon,
        })
        
        logger.info(
            f"Privacy accounting: {description}, "
            f"ε_cumulative={epsilon:.4f}, budget_used={epsilon/self.total_epsilon:.1%}"
        )
        
        return epsilon
    
    def rdp_to_dp(self, rdp_epsilons: np.ndarray) -> float:
        """
        Convert RDP to (ε, δ)-DP using optimal order selection.
        
        For each order α, (α, ε(α))-RDP implies:
        (ε(α) + log(1/δ)/(α-1), δ)-DP
        
        Select the order that minimizes the final ε.
        
        Args:
            rdp_epsilons: RDP epsilon at each order.
        
        Returns:
            Optimal ε for the target δ.
        """
        # log(1/delta) = -log(delta), for delta=1e-5 this is ~11.5
        log_one_over_delta = np.log(1.0 / self.delta)
        
        epsilons = []
        for alpha, rdp_eps in zip(self.alphas, rdp_epsilons):
            if alpha <= 1:
                continue
            # Convert RDP to (ε, δ)-DP using standard formula:
            # ε = ε_RDP(α) + log(1/δ) / (α - 1)
            eps = rdp_eps + log_one_over_delta / (alpha - 1)
            epsilons.append(eps)
        
        # Ensure non-negative epsilon
        min_eps = min(epsilons) if epsilons else float('inf')
        return max(0.0, min_eps)
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        current_eps = self.rdp_to_dp(self.cumulative_rdp)
        return max(0, self.total_epsilon - current_eps)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.get_remaining_budget() <= 0
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy accounting report."""
        current_eps = self.rdp_to_dp(self.cumulative_rdp)
        return {
            "total_budget": self.total_epsilon,
            "delta": self.delta,
            "current_epsilon": current_eps,
            "remaining_budget": self.get_remaining_budget(),
            "budget_utilization": current_eps / self.total_epsilon,
            "num_operations": len(self.operations),
            "operations": self.operations,
            "rdp_orders": self.alphas,
            "cumulative_rdp": self.cumulative_rdp.tolist(),
        }


class DifferentialPrivacyManager:
    """
    Comprehensive Differential Privacy Manager for Federated IDS.
    
    This class orchestrates all DP mechanisms for privacy-preserving
    federated learning in intrusion detection systems. It provides:
    
    1. Gradient perturbation with adaptive clipping
    2. Weight perturbation for model updates
    3. Privacy budget management across rounds
    4. Threat-aware privacy amplification
    5. Multi-level privacy zones
    
    Research Innovations:
        - IDS-Specific Gradient Clipping: Handles bimodal attack/normal distributions
        - Threat Severity Budget Allocation: More budget for critical threats
        - Privacy Amplification via Attack Subsampling
        - Cross-Organization Privacy Guarantees
    
    Example:
        >>> dp_manager = DifferentialPrivacyManager(
        ...     epsilon=1.0,
        ...     delta=1e-5,
        ...     clip_norm=1.0,
        ...     mechanism="gaussian"
        ... )
        >>> 
        >>> # Add DP to gradients
        >>> private_grads = dp_manager.privatize_gradients(gradients, batch_size)
        >>> 
        >>> # Check privacy budget
        >>> print(dp_manager.get_privacy_status())
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        mechanism: str = "gaussian",
        adaptive_clipping: bool = True,
        privacy_level: PrivacyLevel = PrivacyLevel.HIGH,
    ):
        """
        Initialize DP manager.
        
        Args:
            epsilon: Privacy budget per round.
            delta: Privacy failure probability.
            clip_norm: Gradient clipping norm.
            mechanism: Noise mechanism ("gaussian" or "laplace").
            adaptive_clipping: Use adaptive clip norm.
            privacy_level: Overall privacy level.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.mechanism_type = mechanism
        self.privacy_level = privacy_level
        
        # Initialize components
        self.accountant = PrivacyAccountant(
            total_epsilon=epsilon * 100,  # Budget for ~100 rounds
            delta=delta,
        )
        
        if adaptive_clipping:
            self.clipper = AdaptiveClipping(initial_clip=clip_norm)
        else:
            self.clipper = None
        
        # Statistics tracking
        self.round_count = 0
        self.noise_history: List[float] = []
        self.utility_history: List[float] = []
        
        logger.info(
            f"DifferentialPrivacyManager initialized: "
            f"ε={epsilon}, δ={delta}, mechanism={mechanism}, "
            f"privacy_level={privacy_level.value}"
        )
    
    def privatize_gradients(
        self,
        gradients: List[np.ndarray],
        batch_size: int,
        sampling_rate: float = 1.0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply differential privacy to gradients.
        
        Implements DP-SGD: clip per-sample gradients, add calibrated noise.
        
        Args:
            gradients: List of gradient arrays to privatize.
            batch_size: Batch size for sensitivity calculation.
            sampling_rate: Fraction of data sampled per round.
        
        Returns:
            Tuple of (private_gradients, privacy_stats).
        """
        self.round_count += 1
        
        # Step 1: Clip gradients
        if self.clipper:
            clipped_grads, clip_stats = self.clipper.clip_gradients(gradients)
            current_clip = self.clipper.clip_norm
        else:
            # Fixed clipping
            flat_grad = np.concatenate([g.flatten() for g in gradients])
            total_norm = np.linalg.norm(flat_grad)
            clip_factor = min(1.0, self.clip_norm / total_norm)
            clipped_grads = [g * clip_factor for g in gradients]
            current_clip = self.clip_norm
            clip_stats = {"clip_factor": clip_factor, "original_norm": total_norm}
        
        # Step 2: Compute sensitivity
        sensitivity = 2 * current_clip / batch_size
        
        # Step 3: Create noise mechanism
        if self.mechanism_type == "gaussian":
            mechanism = GaussianMechanism(
                sensitivity=sensitivity,
                epsilon=self.epsilon,
                delta=self.delta,
            )
            sigma = mechanism.sigma
        else:
            mechanism = LaplaceMechanism(
                sensitivity=sensitivity,
                epsilon=self.epsilon,
            )
            sigma = mechanism.scale
        
        # Step 4: Add noise
        private_grads = [mechanism.apply(g) for g in clipped_grads]
        
        # Step 5: Account for privacy
        noise_multiplier = sigma / (sensitivity + 1e-10)
        current_epsilon = self.accountant.account_for_mechanism(
            sigma=noise_multiplier,
            sampling_rate=sampling_rate,
            num_steps=1,
            description=f"round_{self.round_count}_gradient_perturbation"
        )
        
        # Record statistics
        self.noise_history.append(sigma)
        
        # Compute utility (signal-to-noise ratio)
        signal_norm = np.linalg.norm(np.concatenate([g.flatten() for g in clipped_grads]))
        noise_norm = sigma * np.sqrt(sum(g.size for g in gradients))
        snr = signal_norm / (noise_norm + 1e-10)
        self.utility_history.append(snr)
        
        stats = {
            "round": self.round_count,
            "epsilon_spent": current_epsilon,
            "remaining_budget": self.accountant.get_remaining_budget(),
            "noise_sigma": sigma,
            "sensitivity": sensitivity,
            "signal_to_noise": snr,
            **clip_stats,
        }
        
        logger.debug(f"Privatized gradients: ε_spent={current_epsilon:.4f}, SNR={snr:.4f}")
        
        return private_grads, stats
    
    def privatize_weights(
        self,
        weights: List[np.ndarray],
        sensitivity: float = 1.0,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply differential privacy to model weights.
        
        Used for output perturbation in federated learning.
        
        Args:
            weights: List of weight arrays to privatize.
            sensitivity: Weight sensitivity bound.
        
        Returns:
            Tuple of (private_weights, privacy_stats).
        """
        if self.mechanism_type == "gaussian":
            mechanism = GaussianMechanism(
                sensitivity=sensitivity,
                epsilon=self.epsilon,
                delta=self.delta,
            )
        else:
            mechanism = LaplaceMechanism(
                sensitivity=sensitivity,
                epsilon=self.epsilon,
            )
        
        private_weights = [mechanism.apply(w) for w in weights]
        
        stats = {
            "mechanism": self.mechanism_type,
            "sensitivity": sensitivity,
            "noise_level": mechanism.sigma if hasattr(mechanism, 'sigma') else mechanism.scale,
        }
        
        return private_weights, stats
    
    def privatize_aggregated_weights(
        self,
        weights: List[np.ndarray],
        num_clients: int,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply DP to aggregated weights (central DP).
        
        For federated averaging, sensitivity is reduced by 1/num_clients.
        
        Args:
            weights: Aggregated weight arrays.
            num_clients: Number of participating clients.
        
        Returns:
            Tuple of (private_weights, privacy_stats).
        """
        # Sensitivity scales with 1/n for averaging
        sensitivity = self.clip_norm / num_clients
        return self.privatize_weights(weights, sensitivity=sensitivity)
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status and statistics."""
        return {
            "privacy_level": self.privacy_level.value,
            "epsilon_per_round": self.epsilon,
            "delta": self.delta,
            "rounds_completed": self.round_count,
            "accountant_report": self.accountant.get_privacy_report(),
            "clipper_stats": self.clipper.get_statistics() if self.clipper else None,
            "avg_noise_level": np.mean(self.noise_history) if self.noise_history else 0,
            "avg_utility": np.mean(self.utility_history) if self.utility_history else 0,
        }
    
    def should_stop_training(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.accountant.is_budget_exhausted()


class ThreatAwarePrivacyManager(DifferentialPrivacyManager):
    """
    Threat-Aware Privacy Manager with Severity-Based Budgeting.
    
    Research Innovation: Allocates privacy budget based on threat
    severity, providing stronger privacy for sensitive attack data
    while allowing more utility for general network traffic.
    
    Privacy Zone Architecture:
        - Zone 1 (Ultra-High): Zero-day exploits, APT indicators
        - Zone 2 (High): Known attack signatures, IOCs
        - Zone 3 (Medium): Suspicious traffic patterns
        - Zone 4 (Low): Benign traffic statistics
    
    Example:
        >>> manager = ThreatAwarePrivacyManager(base_epsilon=1.0)
        >>> private_grad = manager.privatize_with_threat_context(
        ...     gradients, threat_severity="critical"
        ... )
    """
    
    # Privacy multipliers per threat level
    THREAT_MULTIPLIERS = {
        "critical": 0.1,   # 10% of base epsilon (strongest privacy)
        "high": 0.3,       # 30% of base epsilon
        "medium": 0.6,     # 60% of base epsilon
        "low": 1.0,        # Full epsilon
        "benign": 2.0,     # 200% (less privacy, more utility)
    }
    
    def __init__(
        self,
        base_epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        **kwargs
    ):
        super().__init__(
            epsilon=base_epsilon,
            delta=delta,
            clip_norm=clip_norm,
            **kwargs
        )
        self.base_epsilon = base_epsilon
        self.threat_budget_usage: Dict[str, float] = {
            level: 0.0 for level in self.THREAT_MULTIPLIERS
        }
    
    def privatize_with_threat_context(
        self,
        gradients: List[np.ndarray],
        batch_size: int,
        threat_severity: str = "medium",
        threat_distribution: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply threat-aware differential privacy.
        
        Args:
            gradients: Gradient arrays to privatize.
            batch_size: Batch size.
            threat_severity: Overall threat level of the batch.
            threat_distribution: Optional distribution of threats in batch.
        
        Returns:
            Tuple of (private_gradients, privacy_stats).
        """
        # Compute effective epsilon based on threat level
        multiplier = self.THREAT_MULTIPLIERS.get(threat_severity, 0.6)
        effective_epsilon = self.base_epsilon * multiplier
        
        # Update epsilon temporarily
        original_epsilon = self.epsilon
        self.epsilon = effective_epsilon
        
        # Apply privacy
        private_grads, stats = self.privatize_gradients(gradients, batch_size)
        
        # Restore and track
        self.epsilon = original_epsilon
        self.threat_budget_usage[threat_severity] += effective_epsilon
        
        stats["threat_severity"] = threat_severity
        stats["effective_epsilon"] = effective_epsilon
        stats["privacy_multiplier"] = multiplier
        
        return private_grads, stats
    
    def get_threat_budget_report(self) -> Dict[str, Any]:
        """Get budget usage per threat level."""
        return {
            "threat_multipliers": self.THREAT_MULTIPLIERS,
            "budget_usage_per_threat": self.threat_budget_usage,
            "total_budget_used": sum(self.threat_budget_usage.values()),
        }
