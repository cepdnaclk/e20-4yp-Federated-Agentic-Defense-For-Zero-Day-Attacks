"""
Privacy-Preserving Federated Learning Module.

This module implements research-level privacy mechanisms for secure
multi-organization threat intelligence sharing, including:

1. Differential Privacy (Local & Global)
2. Secure Aggregation with Secret Sharing
3. Gradient Compression with Privacy Guarantees
4. Homomorphic Encryption Simulation
5. Privacy Budget Tracking

Key Classes:
    - DifferentialPrivacyManager: Manages DP noise injection and budget
    - SecureAggregator: Implements cryptographic secure aggregation
    - PrivacyMetrics: Computes privacy loss and guarantees
    - PrivacyVisualizer: Creates visualizations for privacy mechanisms

Research Contributions:
    - Adaptive Clipping for IDS-specific gradient distributions
    - Threat-Aware Privacy Amplification via strategic subsampling
    - Multi-Level Privacy Zones for different data sensitivity levels
"""

from .differential_privacy import (
    DifferentialPrivacyManager,
    GaussianMechanism,
    LaplaceMechanism,
    AdaptiveClipping,
    PrivacyAccountant,
)

from .secure_aggregation import (
    SecureAggregator,
    SecretSharing,
    MaskedAggregation,
)

from .gradient_compression import (
    GradientCompressor,
    TopKCompression,
    RandomSparsification,
    PrivacyPreservingCompression,
)

from .privacy_metrics import (
    PrivacyMetrics,
    MembershipInferenceAttack,
    GradientLeakageRisk,
    PrivacyBudgetTracker,
)

from .visualizations import (
    PrivacyVisualizer,
    create_privacy_dashboard,
    plot_privacy_utility_tradeoff,
    plot_epsilon_evolution,
)

__all__ = [
    # Differential Privacy
    "DifferentialPrivacyManager",
    "GaussianMechanism",
    "LaplaceMechanism",
    "AdaptiveClipping",
    "PrivacyAccountant",
    # Secure Aggregation
    "SecureAggregator",
    "SecretSharing",
    "MaskedAggregation",
    # Gradient Compression
    "GradientCompressor",
    "TopKCompression",
    "RandomSparsification",
    "PrivacyPreservingCompression",
    # Privacy Metrics
    "PrivacyMetrics",
    "MembershipInferenceAttack",
    "GradientLeakageRisk",
    "PrivacyBudgetTracker",
    # Visualizations
    "PrivacyVisualizer",
    "create_privacy_dashboard",
    "plot_privacy_utility_tradeoff",
    "plot_epsilon_evolution",
]
