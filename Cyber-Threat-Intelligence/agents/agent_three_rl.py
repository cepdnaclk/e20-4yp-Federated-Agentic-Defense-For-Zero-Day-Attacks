"""
Agent Three (Legacy): Reinforcement Learning Agent for Network Mitigation.

This module contains the PPO-based RL agent that previously served as the
"third agent" in the pipeline.

Per April 2026 refactor:
- The live pipeline's 3rd agent is now a RAG/LLM-based action recommender
  (see `agents/agent_three.py`).
- This RL agent is kept for future work but is not wired into the default
  coordinator flow.

Usage:
    >>> from agents.agent_three_rl import AgentThreeRL
    >>> agent = AgentThreeRL.from_pretrained("models/agent_three")
    >>> decision = agent.take_action(threat_analysis_result)
    >>> print(decision.action_name)
"""

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np

from agents.environments.network_defense_env import (
    NetworkDefenseEnv,
    ThreatState,
    MitigationAction,
)

logger = logging.getLogger(__name__)


@dataclass
class MitigationDecision:
    """Represents a mitigation decision from the RL agent."""

    action: MitigationAction
    action_name: str
    description: str
    confidence: float
    action_probabilities: Dict[str, float]
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": int(self.action),
            "action_name": self.action_name,
            "description": self.description,
            "confidence": self.confidence,
            "action_probabilities": self.action_probabilities,
            "reasoning": self.reasoning,
        }


class AgentThreeRL:
    """PPO-based mitigation agent (legacy)."""

    ACTION_NAMES = {
        MitigationAction.DO_NOTHING: "Do Nothing",
        MitigationAction.ALERT_ADMIN: "Alert Admin",
        MitigationAction.BLOCK_IP: "Block IP",
        MitigationAction.ISOLATE_SUBNET: "Isolate Subnet",
    }

    CATEGORY_TO_INDEX = {
        "Normal": 0,
        "Fuzzers": 1,
        "Analysis": 2,
        "Backdoor": 3,
        "DoS": 4,
        "Exploits": 5,
        "Generic": 6,
        "Reconnaissance": 7,
        "Shellcode": 8,
        "Worms": 9,
        "Unknown/Zero-day": 0,
    }

    SEVERITY_MAP = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "critical": 3,
        "unknown": 1,
    }

    def __init__(
        self,
        model: Optional[Any] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        seed: int = 42,
    ):
        self._model = model
        self._is_trained = model is not None

        self._learning_rate = learning_rate
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_range = clip_range
        self._ent_coef = ent_coef
        self._vf_coef = vf_coef
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._seed = seed

        # Lazily constructed env (avoid heavy deps during import/tests)
        self._env: Optional[NetworkDefenseEnv] = None

        logger.info("AgentThreeRL initialized (trained=%s)", self._is_trained)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _ensure_env(self) -> NetworkDefenseEnv:
        if self._env is None:
            self._env = NetworkDefenseEnv()
        return self._env

    def train(
        self,
        total_timesteps: int = 100_000,
        env: Optional[NetworkDefenseEnv] = None,
        verbose: int = 1,
    ) -> None:
        """Trains the RL agent."""
        if env is None:
            env = self._ensure_env()

        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for AgentThreeRL. "
                "Install with: pip install stable-baselines3 gymnasium"
            ) from e

        if self._model is None:
            self._model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self._learning_rate,
                gamma=self._gamma,
                gae_lambda=self._gae_lambda,
                clip_range=self._clip_range,
                ent_coef=self._ent_coef,
                vf_coef=self._vf_coef,
                n_steps=self._n_steps,
                batch_size=self._batch_size,
                n_epochs=self._n_epochs,
                verbose=verbose,
                seed=self._seed,
            )

        self._model.learn(total_timesteps=total_timesteps)
        self._is_trained = True

    def take_action(
        self,
        threat_input: Union["ThreatAnalysisResult", ThreatState, np.ndarray, Dict[str, Any]],
        deterministic: bool = True,
    ) -> MitigationDecision:
        """Choose a mitigation action."""
        env = self._ensure_env()

        threat_state = None

        # ThreatState passed directly
        if isinstance(threat_input, ThreatState):
            threat_state = threat_input

        # Raw observation
        elif isinstance(threat_input, np.ndarray):
            obs = threat_input.astype(np.float32)
            if obs.ndim != 1:
                obs = obs.flatten()
            # Best-effort: assume already matches env observation
            threat_state = ThreatState.from_observation(obs)

        # Dict
        elif isinstance(threat_input, dict):
            threat_state = ThreatState.from_dict(threat_input)

        else:
            # Assume ThreatAnalysisResult-like
            try:
                from agents.environments.network_defense_env import ThreatState

                threat_state = ThreatState.from_agent_two_result(threat_input)
            except Exception:
                threat_state = env._generate_threat_state()  # type: ignore[attr-defined]

        obs = threat_state.to_observation()

        # If model not trained, fall back to rule-based
        if self._model is None:
            return self._fallback_policy(threat_state)

        try:
            action, _ = self._model.predict(obs, deterministic=deterministic)
            action_int = int(action)
        except Exception as e:
            logger.warning("RL predict failed, using fallback: %s", e)
            return self._fallback_policy(threat_state)

        action_enum = MitigationAction(action_int % len(MitigationAction))

        action_probs = self._get_action_probabilities(obs)
        confidence = float(action_probs.get(str(int(action_enum)), 0.7))

        return MitigationDecision(
            action=action_enum,
            action_name=self.ACTION_NAMES.get(action_enum, "Unknown"),
            description=self._describe_action(action_enum),
            confidence=confidence,
            action_probabilities=action_probs,
            reasoning="PPO policy decision",
        )

    def _get_action_probabilities(self, obs: np.ndarray) -> Dict[str, float]:
        if self._model is None:
            return {str(int(a)): 0.25 for a in MitigationAction}

        try:
            import torch

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            distribution = self._model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
            return {str(i): float(p) for i, p in enumerate(probs)}
        except Exception:
            return {str(int(a)): 0.25 for a in MitigationAction}

    def _describe_action(self, action: MitigationAction) -> str:
        descriptions = {
            MitigationAction.DO_NOTHING: "Monitor traffic; no immediate mitigation.",
            MitigationAction.ALERT_ADMIN: "Alert an administrator / SOC analyst.",
            MitigationAction.BLOCK_IP: "Block the suspected source IP at edge controls.",
            MitigationAction.ISOLATE_SUBNET: "Isolate affected subnet or host segment.",
        }
        return descriptions.get(action, "No description.")

    def _fallback_policy(self, threat_state: ThreatState) -> MitigationDecision:
        # Simple deterministic heuristic
        if threat_state.severity >= 3:
            action = MitigationAction.ISOLATE_SUBNET
        elif threat_state.severity >= 2:
            action = MitigationAction.BLOCK_IP
        elif threat_state.is_anomaly:
            action = MitigationAction.ALERT_ADMIN
        else:
            action = MitigationAction.DO_NOTHING

        probs = {str(int(a)): 0.0 for a in MitigationAction}
        probs[str(int(action))] = 1.0

        return MitigationDecision(
            action=action,
            action_name=self.ACTION_NAMES.get(action, "Unknown"),
            description=self._describe_action(action),
            confidence=0.6,
            action_probabilities=probs,
            reasoning="Fallback rule-based policy (model not trained)",
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save PPO model if exists
        if self._model is not None:
            self._model.save(str(path / "ppo_model"))

        config = {
            "learning_rate": self._learning_rate,
            "gamma": self._gamma,
            "gae_lambda": self._gae_lambda,
            "clip_range": self._clip_range,
            "ent_coef": self._ent_coef,
            "vf_coef": self._vf_coef,
            "n_steps": self._n_steps,
            "batch_size": self._batch_size,
            "n_epochs": self._n_epochs,
            "seed": self._seed,
        }

        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "AgentThreeRL":
        path = Path(path)

        config_path = path / "config.json"
        config: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        model = None
        ppo_path = path / "ppo_model.zip"
        if ppo_path.exists():
            try:
                from stable_baselines3 import PPO

                env = NetworkDefenseEnv()
                model = PPO.load(str(path / "ppo_model"), env=env)
            except Exception as e:
                logger.warning("Failed to load PPO model: %s", e)

        return cls(model=model, **config)
