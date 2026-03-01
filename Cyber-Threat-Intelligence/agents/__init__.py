"""
Agents Module for Multi-Agent Intrusion Detection System.

This module provides autonomous agents for network intrusion detection,
each specialized for different detection strategies.

Agents:
    AgentOne: Autoencoder-based anomaly detection agent.
    AgentTwo: XGBoost classification with LLM reasoning for zero-day threats.
    AgentThree: RL-based network mitigation agent.
"""

from agents.agent_one import AgentOne
from agents.models.autoencoder import AnomalyAutoencoder
from agents.agent_two import AgentTwo, ThreatAnalysisResult
from agents.models.xgboost_classifier import ThreatClassifier, ClassificationResult
from agents.agent_three import AgentThree, MitigationDecision
from agents.environments.network_defense_env import (
    NetworkDefenseEnv,
    ThreatState,
    MitigationAction,
)

__all__ = [
    # Agent One
    "AgentOne",
    "AnomalyAutoencoder",
    # Agent Two
    "AgentTwo",
    "ThreatAnalysisResult",
    "ThreatClassifier",
    "ClassificationResult",
    # Agent Three
    "AgentThree",
    "MitigationDecision",
    "NetworkDefenseEnv",
    "ThreatState",
    "MitigationAction",
]

__version__ = "1.0.0"
