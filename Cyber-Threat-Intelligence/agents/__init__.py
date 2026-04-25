"""
Agents Module for Multi-Agent Intrusion Detection System.

This module provides autonomous agents for network intrusion detection,
each specialized for different detection strategies.

Agents:
    AgentOne: Autoencoder-based anomaly detection agent.
    AgentTwo: XGBoost-only classification agent.
    AgentThree: RAG/LLM-based action recommendation agent.
    AgentThreeRL: Legacy PPO/RL mitigation agent (kept, not default).
"""

from agents.agent_one import AgentOne
from agents.models.autoencoder import AnomalyAutoencoder
from agents.agent_two import AgentTwo, ThreatAnalysisResult
from agents.models.xgboost_classifier import ThreatClassifier, ClassificationResult
from agents.agent_three import AgentThree, ActionRecommendation
from agents.agent_three_rl import AgentThreeRL, MitigationDecision
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
    "ActionRecommendation",
    "AgentThreeRL",
    "MitigationDecision",
    "NetworkDefenseEnv",
    "ThreatState",
    "MitigationAction",
]

__version__ = "1.0.0"
