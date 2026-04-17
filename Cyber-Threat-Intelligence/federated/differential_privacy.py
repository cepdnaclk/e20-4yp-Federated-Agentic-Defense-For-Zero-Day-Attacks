"""
Differential Privacy Engine for Federated Learning.

This module implements Local Differential Privacy (LDP) to protect
against model inversion attacks by ensuring weights sent to the global
server do not leak individual network flow data.

The engine applies:
    1. Gradient/weight clipping to bound sensitivity
    2. Calibrated Gaussian noise injection for privacy guarantees

Classes:
    DifferentialPrivacyEngine: Applies DP mechanisms to model weights.

Example:
    >>> dp_engine = DifferentialPrivacyEngine()
    >>> private_weights = dp_engine.apply_dp(
    ...     weights=model_weights,
    ...     clip_norm=1.0,
    ...     noise_multiplier=0.1
    ... )
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DifferentialPrivacyEngine:
    """
    Differential Privacy Engine for protecting model weights.
    
    Implements Local Differential Privacy (LDP) by:
    1. Clipping weight L2 norms to bound sensitivity
    2. Adding calibrated Gaussian noise proportional to sensitivity
    
    This ensures that model weights shared with the federated server
    cannot be used to reconstruct individual training samples.
    
    Attributes:
        _clip_count: Number of times clipping was applied.
        _noise_applied_count: Number of DP applications.
        _random_state: Optional numpy RandomState for reproducibility.
    
    Example:
        >>> engine = DifferentialPrivacyEngine()
        >>> weights = [np.random.randn(100, 50), np.random.randn(50)]
        >>> 
        >>> # Apply DP with clipping and noise
        >>> private_weights = engine.apply_dp(
        ...     weights=weights,
        ...     clip_norm=1.0,
        ...     noise_multiplier=0.1
        ... )
        >>> 
        >>> # Verify clipping worked
        >>> for w in private_weights:
        ...     assert np.linalg.norm(w) <= 1.0 + epsilon  # With noise tolerance
    
    Privacy Guarantees:
        - Bounded sensitivity via L2 clipping
        - (ε, δ)-differential privacy with Gaussian mechanism
        - Noise calibrated to clip_norm * noise_multiplier
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the Differential Privacy Engine.
        
        Args:
            random_state: Optional seed for reproducible noise generation.
                         Use only for testing; production should use None.
        """
        self._clip_count = 0
        self._noise_applied_count = 0
        self._random_state = np.random.RandomState(random_state) if random_state else None
        
        logger.info(
            f"DifferentialPrivacyEngine initialized "
            f"(random_state={'fixed' if random_state else 'random'})"
        )
    
    @property
    def clip_count(self) -> int:
        """Number of weight arrays that were clipped."""
        return self._clip_count
    
    @property
    def noise_applied_count(self) -> int:
        """Number of times DP was applied."""
        return self._noise_applied_count
    
    def apply_dp(
        self,
        weights: List[np.ndarray],
        clip_norm: float,
        noise_multiplier: float,
    ) -> List[np.ndarray]:
        """
        Apply differential privacy to model weights.
        
        This method:
        1. Calculates the L2 norm of each weight array
        2. Clips weights exceeding the clip_norm threshold
        3. Generates Gaussian noise calibrated to (noise_multiplier * clip_norm)
        4. Adds the noise to clipped weights
        
        Args:
            weights: List of numpy arrays representing model weights.
            clip_norm: Maximum L2 norm for each weight array.
                      Weights with larger norms are scaled down.
            noise_multiplier: Multiplier for noise standard deviation.
                             Noise std = noise_multiplier * clip_norm.
        
        Returns:
            List of numpy arrays with DP applied, preserving original
            shapes and dtypes.
        
        Raises:
            ValueError: If clip_norm <= 0 or noise_multiplier < 0.
        
        Example:
            >>> weights = [np.array([3.0, 4.0])]  # L2 norm = 5.0
            >>> private = engine.apply_dp(weights, clip_norm=1.0, noise_multiplier=0.1)
            >>> # Weight is clipped to norm=1.0, then noise added
        
        Note:
            The clipping is applied per-array, not globally.
            This provides per-parameter privacy guarantees.
        """
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be non-negative, got {noise_multiplier}")
        
        self._noise_applied_count += 1
        private_weights = []
        
        for i, w in enumerate(weights):
            # Step 1: Calculate L2 norm
            l2_norm = np.linalg.norm(w)
            
            # Step 2: Clip if norm exceeds threshold
            if l2_norm > clip_norm:
                # Scale down to clip_norm
                scale_factor = clip_norm / l2_norm
                clipped_w = w * scale_factor
                self._clip_count += 1
                logger.debug(
                    f"Weight array {i}: L2 norm {l2_norm:.4f} -> clipped to {clip_norm}"
                )
            else:
                clipped_w = w.copy()
            
            # Step 3: Generate calibrated Gaussian noise
            noise_std = noise_multiplier * clip_norm
            
            if self._random_state is not None:
                noise = self._random_state.normal(0, noise_std, size=w.shape)
            else:
                noise = np.random.normal(0, noise_std, size=w.shape)
            
            # Step 4: Add noise to clipped weights
            private_w = clipped_w + noise
            
            # Preserve original dtype and ensure output is np.ndarray
            # (astype can return scalar for 0-d arrays)
            private_w = np.asarray(private_w, dtype=w.dtype)
            
            private_weights.append(private_w)
        
        logger.info(
            f"DP applied to {len(weights)} weight arrays "
            f"(clip_norm={clip_norm}, noise_multiplier={noise_multiplier})"
        )
        
        return private_weights
    
    def compute_l2_norms(self, weights: List[np.ndarray]) -> List[float]:
        """
        Compute L2 norms for all weight arrays.
        
        Utility method for debugging and analysis.
        
        Args:
            weights: List of numpy arrays.
        
        Returns:
            List of L2 norms for each weight array.
        """
        return [float(np.linalg.norm(w)) for w in weights]
    
    def estimate_privacy_budget(
        self,
        noise_multiplier: float,
        delta: float = 1e-5,
    ) -> float:
        """
        Estimate epsilon for (ε, δ)-differential privacy.
        
        Uses the Gaussian mechanism privacy guarantee.
        
        Args:
            noise_multiplier: The noise multiplier used in apply_dp.
            delta: Privacy parameter δ (probability of privacy breach).
        
        Returns:
            Estimated epsilon value.
        
        Note:
            This is a simplified estimate. For rigorous privacy
            accounting, use tools like Opacus or TensorFlow Privacy.
        """
        if noise_multiplier <= 0:
            return float('inf')
        
        # Simplified Gaussian mechanism: ε ≈ sqrt(2 * ln(1.25/δ)) / σ
        # where σ = noise_multiplier (assuming sensitivity = 1 after clipping)
        import math
        epsilon = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
        
        return epsilon
    
    def reset_counters(self) -> None:
        """Reset internal counters for new session."""
        self._clip_count = 0
        self._noise_applied_count = 0
        logger.debug("DP engine counters reset")
