"""
Federated Learning Module for Privacy-Preserving IDS Training.

This module provides federated learning capabilities using Flower (flwr)
to enable privacy-preserving training of Agent One (Autoencoder) and
Agent Two (XGBoost) across distributed network nodes.

Components:
    - NetworkDefenseClient: Flower client for local model training
    - NetworkDefenseStrategy: Custom FedAvg strategy with callbacks
    - IntegrationCoordinator: Bridge between FL updates and RAG system
    - SemanticThreatReport: Structured threat analysis output

Example:
    >>> # Start server
    >>> python -m federated.server --rounds 5 --min-clients 2
    
    >>> # Start client (in separate terminal)
    >>> from federated import NetworkDefenseClient
    >>> client = NetworkDefenseClient(autoencoder=model, train_data=(X, None))
    >>> fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    
    >>> # Or run simulation demo
    >>> python scripts/federated_demo.py --num-clients 3 --num-rounds 5
"""

from federated.client import NetworkDefenseClient, create_client_fn
from federated.server import (
    NetworkDefenseStrategy,
    start_federated_server,
    AggregationResult,
)
from federated.coordinator import (
    IntegrationCoordinator,
    SemanticThreatReport,
    ThreatSeverity,
    ATTACK_TO_MITRE_MAP,
)
from federated.utils import (
    autoencoder_weights_to_numpy,
    numpy_to_autoencoder_weights,
    xgboost_to_numpy,
    numpy_to_xgboost,
    get_combined_weights,
    split_combined_weights,
)

__all__ = [
    # Client
    "NetworkDefenseClient",
    "create_client_fn",
    # Server
    "NetworkDefenseStrategy",
    "start_federated_server",
    "AggregationResult",
    # Coordinator
    "IntegrationCoordinator",
    "SemanticThreatReport",
    "ThreatSeverity",
    "ATTACK_TO_MITRE_MAP",
    # Utils
    "autoencoder_weights_to_numpy",
    "numpy_to_autoencoder_weights",
    "xgboost_to_numpy",
    "numpy_to_xgboost",
    "get_combined_weights",
    "split_combined_weights",
]
