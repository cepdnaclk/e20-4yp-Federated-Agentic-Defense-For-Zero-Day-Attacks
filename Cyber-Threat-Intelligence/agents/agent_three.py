"""
Agent Three: Reinforcement Learning Agent for Network Mitigation.

This module implements a PPO-based RL agent that learns optimal
mitigation strategies based on threat classifications from Agent Two.

The agent learns to balance:
    - Security: Blocking real threats effectively
    - Availability: Not disrupting normal network traffic
    - Proportionality: Matching response severity to threat severity

Usage:
    >>> agent = AgentThree.from_pretrained("models/agent_three")
    >>> action = agent.take_action(threat_analysis_result)
    >>> print(f"Recommended: {action.description}")
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
    """
    Represents a mitigation decision from Agent Three.
    
    Attributes:
        action: The mitigation action to take.
        action_name: Human-readable action name.
        description: Detailed description of the action.
        confidence: Agent's confidence in this decision.
        action_probabilities: Probabilities for all actions.
        reasoning: Optional reasoning for the decision.
    """
    action: MitigationAction
    action_name: str
    description: str
    confidence: float
    action_probabilities: Dict[str, float]
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts decision to dictionary."""
        return {
            "action": int(self.action),
            "action_name": self.action_name,
            "description": self.description,
            "confidence": self.confidence,
            "action_probabilities": self.action_probabilities,
            "reasoning": self.reasoning,
        }


class AgentThree:
    """
    Reinforcement Learning Agent for Network Mitigation Decisions.
    
    This agent uses PPO (Proximal Policy Optimization) from Stable Baselines 3
    to learn optimal mitigation strategies based on threat analyses from Agent Two.
    
    Features:
        - Learns from experience to improve mitigation decisions
        - Balances security vs. availability
        - Provides confidence scores for decisions
        - Supports both deterministic and stochastic action selection
    
    Example:
        >>> # Load pre-trained agent
        >>> agent = AgentThree.from_pretrained("models/agent_three")
        >>> 
        >>> # Get action for a threat
        >>> decision = agent.take_action(threat_result)
        >>> print(f"Action: {decision.action_name}")
        >>> print(f"Confidence: {decision.confidence:.2%}")
        >>> 
        >>> # Train a new agent
        >>> agent = AgentThree()
        >>> agent.train(total_timesteps=100000)
        >>> agent.save("models/agent_three")
    """
    
    # Action names for human-readable output
    ACTION_NAMES = {
        MitigationAction.DO_NOTHING: "Do Nothing",
        MitigationAction.ALERT_ADMIN: "Alert Admin",
        MitigationAction.BLOCK_IP: "Block IP",
        MitigationAction.ISOLATE_SUBNET: "Isolate Subnet",
    }
    
    # Category to index mapping (must match environment)
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
        "Unknown/Zero-day": 0,  # Treat as unknown -> index 0 but with low confidence
    }
    
    SEVERITY_MAP = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "critical": 3,
    }
    
    def __init__(
        self,
        env: Optional[NetworkDefenseEnv] = None,
        model: Optional[Any] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        Initializes Agent Three.
        
        Args:
            env: Optional custom environment. Creates default if None.
            model: Optional pre-trained PPO model.
            learning_rate: Learning rate for PPO.
            n_steps: Number of steps per update.
            batch_size: Minibatch size.
            n_epochs: Number of epochs per update.
            gamma: Discount factor.
            device: Device to use ('auto', 'cpu', 'cuda').
            verbose: Verbosity level.
        """
        self._env = env or NetworkDefenseEnv()
        self._model = model
        self._is_trained = model is not None
        
        # PPO hyperparameters
        self._learning_rate = learning_rate
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._device = device
        self._verbose = verbose
        
        # Training history
        self._training_history: List[Dict[str, Any]] = []
        
        logger.info("AgentThree initialized (trained=%s)", self._is_trained)
    
    def _ensure_model(self) -> None:
        """Ensures model is initialized."""
        if self._model is None:
            try:
                from stable_baselines3 import PPO
                
                self._model = PPO(
                    "MlpPolicy",
                    self._env,
                    learning_rate=self._learning_rate,
                    n_steps=self._n_steps,
                    batch_size=self._batch_size,
                    n_epochs=self._n_epochs,
                    gamma=self._gamma,
                    device=self._device,
                    verbose=self._verbose,
                )
                logger.info("PPO model initialized")
            except ImportError:
                raise ImportError(
                    "stable-baselines3 is required for AgentThree. "
                    "Install with: pip install stable-baselines3"
                )
    
    def train(
        self,
        total_timesteps: int = 100000,
        progress_bar: bool = True,
        callback: Optional[Any] = None,
        log_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Trains the RL agent.
        
        Args:
            total_timesteps: Total number of training timesteps.
            progress_bar: Whether to show progress bar.
            callback: Optional training callback.
            log_interval: Logging interval.
        
        Returns:
            Training statistics dictionary.
        """
        self._ensure_model()
        
        logger.info("Starting training for %d timesteps...", total_timesteps)
        
        # Train
        self._model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=callback,
            log_interval=log_interval,
        )
        
        self._is_trained = True
        
        # Get final statistics
        stats = self._env.get_statistics()
        
        self._training_history.append({
            "timesteps": total_timesteps,
            "final_stats": stats,
        })
        
        logger.info("Training complete. Final accuracy: %.2f%%", 
                   stats.get("accuracy", 0) * 100)
        
        return stats
    
    def take_action(
        self,
        threat_input: Union["ThreatAnalysisResult", ThreatState, np.ndarray, Dict[str, Any]],
        deterministic: bool = True,
    ) -> MitigationDecision:
        """
        Determines the mitigation action for a given threat.
        
        This is the main interface for using Agent Three. It accepts
        various input formats and returns a structured decision.
        
        Args:
            threat_input: Threat information in one of these formats:
                - ThreatAnalysisResult from Agent Two
                - ThreatState object
                - Raw observation vector (numpy array)
                - Dictionary with threat fields
            deterministic: If True, uses most likely action.
                          If False, samples from action distribution.
        
        Returns:
            MitigationDecision with action and confidence.
        
        Example:
            >>> result = agent_two.analyze_threat(features)
            >>> decision = agent_three.take_action(result)
            >>> if decision.action == MitigationAction.BLOCK_IP:
            ...     firewall.block(source_ip)
        """
        if not self._is_trained:
            logger.warning("Agent not trained - using fallback policy")
            return self._fallback_policy(threat_input)
        
        # Convert input to observation
        observation = self._to_observation(threat_input)
        
        # Get action and probabilities from model
        action, _state = self._model.predict(observation, deterministic=deterministic)
        action = int(action)
        
        # Get action probabilities
        action_probs = self._get_action_probabilities(observation)
        
        # Build decision
        mitigation_action = MitigationAction(action)
        
        decision = MitigationDecision(
            action=mitigation_action,
            action_name=self.ACTION_NAMES[mitigation_action],
            description=MitigationAction.get_description(action),
            confidence=action_probs[action],
            action_probabilities={
                self.ACTION_NAMES[MitigationAction(i)]: float(p)
                for i, p in enumerate(action_probs)
            },
            reasoning=self._generate_reasoning(observation, action, action_probs),
        )
        
        return decision
    
    def _to_observation(
        self,
        threat_input: Union[Any, ThreatState, np.ndarray, Dict[str, Any]],
    ) -> np.ndarray:
        """Converts various input formats to observation vector."""
        if isinstance(threat_input, np.ndarray):
            # Already an observation
            if threat_input.shape == (14,):
                return threat_input.astype(np.float32)
            else:
                raise ValueError(f"Expected shape (14,), got {threat_input.shape}")
        
        if isinstance(threat_input, ThreatState):
            return threat_input.to_observation()
        
        if isinstance(threat_input, dict):
            # Convert dict to ThreatState
            threat_state = ThreatState(
                category_index=self.CATEGORY_TO_INDEX.get(
                    threat_input.get("category", "Normal"), 0
                ),
                confidence=threat_input.get("confidence", 0.5),
                severity_level=self.SEVERITY_MAP.get(
                    threat_input.get("severity", "medium"), 1
                ),
                is_zero_day=threat_input.get("is_zero_day", False),
                is_actual_threat=threat_input.get("is_actual_threat", True),
                attack_category=threat_input.get("category", "Normal"),
            )
            return threat_state.to_observation()
        
        # Assume it's a ThreatAnalysisResult from Agent Two
        try:
            category = threat_input.classification.predicted_category
            threat_state = ThreatState(
                category_index=self.CATEGORY_TO_INDEX.get(category, 0),
                confidence=threat_input.classification.confidence,
                severity_level=self.SEVERITY_MAP.get(
                    threat_input.severity, 1
                ),
                is_zero_day=threat_input.is_zero_day,
                is_actual_threat=True,  # Assume it's a threat if passed to Agent 3
                attack_category=category,
            )
            return threat_state.to_observation()
        except AttributeError:
            raise ValueError(f"Unsupported input type: {type(threat_input)}")
    
    def _get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Gets action probabilities from the policy."""
        import torch
        
        obs_tensor = torch.as_tensor(observation).float().unsqueeze(0)
        
        with torch.no_grad():
            # Get action distribution from policy
            distribution = self._model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
        
        return probs
    
    def _generate_reasoning(
        self,
        observation: np.ndarray,
        action: int,
        action_probs: np.ndarray,
    ) -> str:
        """Generates human-readable reasoning for the decision."""
        # Decode observation
        category_onehot = observation[:10]
        confidence = observation[10]
        severity = observation[11] * 3  # Denormalize
        is_zero_day = observation[12] > 0.5
        is_high_risk = observation[13] > 0.5
        
        # Find predicted category
        category_idx = int(np.argmax(category_onehot))
        categories = list(self.CATEGORY_TO_INDEX.keys())
        category = categories[category_idx] if category_idx < len(categories) else "Unknown"
        
        severity_names = ["low", "medium", "high", "critical"]
        severity_name = severity_names[min(int(severity), 3)]
        
        lines = [
            f"Threat Assessment:",
            f"  - Category: {category} (confidence: {confidence:.1%})",
            f"  - Severity: {severity_name}",
            f"  - Zero-day: {'Yes' if is_zero_day else 'No'}",
            f"  - High-risk: {'Yes' if is_high_risk else 'No'}",
            "",
            f"Decision: {self.ACTION_NAMES[MitigationAction(action)]}",
            f"  - Confidence: {action_probs[action]:.1%}",
        ]
        
        # Add alternative actions if confidence is lower
        if action_probs[action] < 0.9:
            sorted_actions = np.argsort(action_probs)[::-1]
            lines.append("  - Alternatives considered:")
            for alt_action in sorted_actions[1:3]:
                if action_probs[alt_action] > 0.1:
                    lines.append(
                        f"    * {self.ACTION_NAMES[MitigationAction(alt_action)]}: "
                        f"{action_probs[alt_action]:.1%}"
                    )
        
        return "\n".join(lines)
    
    def _fallback_policy(
        self,
        threat_input: Any,
    ) -> MitigationDecision:
        """
        Rule-based fallback policy when model is not trained.
        
        Uses simple heuristics based on severity and confidence.
        """
        observation = self._to_observation(threat_input)
        
        # Decode key features
        confidence = observation[10]
        severity = observation[11] * 3
        is_zero_day = observation[12] > 0.5
        
        # Simple rule-based decision
        if confidence < 0.5:
            # Low confidence -> alert only
            action = MitigationAction.ALERT_ADMIN
        elif severity >= 2.5:
            # Critical/high severity -> isolate
            action = MitigationAction.ISOLATE_SUBNET
        elif severity >= 1.5:
            # Medium-high -> block
            action = MitigationAction.BLOCK_IP
        elif is_zero_day:
            # Zero-day -> at least alert
            action = MitigationAction.ALERT_ADMIN
        else:
            # Low severity, high confidence normal -> do nothing
            action = MitigationAction.DO_NOTHING
        
        # Create simple probability distribution
        action_probs = np.zeros(4)
        action_probs[action] = 0.9
        remaining = 0.1 / 3
        for i in range(4):
            if i != action:
                action_probs[i] = remaining
        
        return MitigationDecision(
            action=action,
            action_name=self.ACTION_NAMES[action],
            description=MitigationAction.get_description(action),
            confidence=0.9,
            action_probabilities={
                self.ACTION_NAMES[MitigationAction(i)]: float(p)
                for i, p in enumerate(action_probs)
            },
            reasoning="Using fallback rule-based policy (model not trained)",
        )
    
    def evaluate(
        self,
        n_episodes: int = 100,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates the agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate.
            deterministic: Whether to use deterministic actions.
        
        Returns:
            Evaluation statistics.
        """
        if not self._is_trained:
            logger.warning("Evaluating untrained agent")
        
        self._ensure_model()
        
        # Create fresh environment for evaluation
        eval_env = NetworkDefenseEnv()
        
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self._model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = eval_env.step(int(action))
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        stats = eval_env.get_statistics()
        stats["mean_episode_reward"] = np.mean(episode_rewards)
        stats["std_episode_reward"] = np.std(episode_rewards)
        stats["n_episodes"] = n_episodes
        
        return stats
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the agent to disk.
        
        Args:
            path: Directory to save the agent.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self._model is not None:
            self._model.save(str(path / "ppo_model"))
        
        # Save metadata
        metadata = {
            "is_trained": self._is_trained,
            "learning_rate": self._learning_rate,
            "n_steps": self._n_steps,
            "batch_size": self._batch_size,
            "n_epochs": self._n_epochs,
            "gamma": self._gamma,
            "training_history": self._training_history,
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("AgentThree saved to: %s", path)
    
    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "AgentThree":
        """
        Loads a pre-trained agent from disk.
        
        Args:
            path: Directory containing saved agent.
        
        Returns:
            Loaded AgentThree instance.
        """
        from stable_baselines3 import PPO
        
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create environment
        env = NetworkDefenseEnv()
        
        # Load model
        model = PPO.load(str(path / "ppo_model"), env=env)
        
        # Create agent
        agent = cls(
            env=env,
            model=model,
            learning_rate=metadata.get("learning_rate", 3e-4),
            n_steps=metadata.get("n_steps", 2048),
            batch_size=metadata.get("batch_size", 64),
            n_epochs=metadata.get("n_epochs", 10),
            gamma=metadata.get("gamma", 0.99),
        )
        
        agent._is_trained = metadata.get("is_trained", True)
        agent._training_history = metadata.get("training_history", [])
        
        logger.info("AgentThree loaded from: %s", path)
        
        return agent
    
    @property
    def is_trained(self) -> bool:
        """Returns whether the agent has been trained."""
        return self._is_trained
    
    @property
    def env(self) -> NetworkDefenseEnv:
        """Returns the environment."""
        return self._env
    
    def get_config(self) -> Dict[str, Any]:
        """Returns agent configuration."""
        return {
            "is_trained": self._is_trained,
            "learning_rate": self._learning_rate,
            "n_steps": self._n_steps,
            "batch_size": self._batch_size,
            "n_epochs": self._n_epochs,
            "gamma": self._gamma,
            "device": self._device,
            "training_history": self._training_history,
        }
