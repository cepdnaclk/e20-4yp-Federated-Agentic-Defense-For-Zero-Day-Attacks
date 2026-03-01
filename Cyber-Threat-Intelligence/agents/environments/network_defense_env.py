"""
Network Defense Environment for Reinforcement Learning.

This module implements a custom Gymnasium environment for training
RL agents to make network mitigation decisions based on threat
classifications from Agent Two.

The environment simulates a network defense scenario where the agent
must decide how to respond to detected threats, balancing between
security (blocking threats) and availability (not disrupting legitimate traffic).
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class MitigationAction(IntEnum):
    """
    Discrete mitigation actions available to the RL agent.
    
    Actions are ordered by severity/impact:
        - DO_NOTHING: No action, continue monitoring
        - ALERT_ADMIN: Send alert to security team
        - BLOCK_IP: Block the source IP address
        - ISOLATE_SUBNET: Isolate the affected subnet (most severe)
    """
    DO_NOTHING = 0
    ALERT_ADMIN = 1
    BLOCK_IP = 2
    ISOLATE_SUBNET = 3
    
    @classmethod
    def get_description(cls, action: int) -> str:
        """Returns human-readable description of action."""
        descriptions = {
            cls.DO_NOTHING: "No action - continue monitoring",
            cls.ALERT_ADMIN: "Alert security administrator",
            cls.BLOCK_IP: "Block source IP address",
            cls.ISOLATE_SUBNET: "Isolate affected subnet",
        }
        return descriptions.get(action, "Unknown action")


@dataclass
class ThreatState:
    """
    Represents the state of a detected threat from Agent Two.
    
    This is the observation that the RL agent receives and must
    make mitigation decisions based on.
    
    Attributes:
        category_index: Index of predicted attack category (0-10).
        confidence: Classification confidence (0.0-1.0).
        severity_level: Encoded severity (0=low, 1=medium, 2=high, 3=critical).
        is_zero_day: Whether threat is classified as zero-day.
        is_actual_threat: Ground truth - whether this is a real attack.
        attack_category: String name of attack category.
        feature_vector: Optional raw feature vector.
    """
    category_index: int
    confidence: float
    severity_level: int
    is_zero_day: bool
    is_actual_threat: bool  # Ground truth for reward calculation
    attack_category: str = "Unknown"
    feature_vector: Optional[np.ndarray] = None
    
    def to_observation(self) -> np.ndarray:
        """
        Converts threat state to observation vector for RL agent.
        
        Returns:
            numpy array of shape (14,) containing:
            - category_onehot (10): One-hot encoded category
            - confidence (1): Classification confidence
            - severity_level (1): Encoded severity
            - is_zero_day (1): Zero-day flag
            - is_high_risk (1): Derived high-risk indicator
        """
        # One-hot encode category (10 categories)
        category_onehot = np.zeros(10, dtype=np.float32)
        if 0 <= self.category_index < 10:
            category_onehot[self.category_index] = 1.0
        
        # Derived features
        is_high_risk = float(self.severity_level >= 2 or self.is_zero_day)
        
        observation = np.concatenate([
            category_onehot,
            np.array([
                self.confidence,
                self.severity_level / 3.0,  # Normalize to [0, 1]
                float(self.is_zero_day),
                is_high_risk,
            ], dtype=np.float32)
        ])
        
        return observation
    
    @classmethod
    def from_agent_two_result(
        cls,
        result: Any,
        is_actual_threat: bool,
        category_to_index: Dict[str, int],
    ) -> "ThreatState":
        """
        Creates ThreatState from AgentTwo's ThreatAnalysisResult.
        
        Args:
            result: ThreatAnalysisResult from Agent Two.
            is_actual_threat: Ground truth label.
            category_to_index: Mapping from category names to indices.
        
        Returns:
            ThreatState instance.
        """
        severity_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        
        category = result.classification.predicted_category
        category_idx = category_to_index.get(category, 0)
        
        return cls(
            category_index=category_idx,
            confidence=result.classification.confidence,
            severity_level=severity_map.get(result.severity, 1),
            is_zero_day=result.is_zero_day,
            is_actual_threat=is_actual_threat,
            attack_category=category,
        )


class NetworkDefenseEnv(gym.Env):
    """
    Gymnasium environment for network defense decision-making.
    
    This environment simulates a network security scenario where an RL agent
    must decide how to respond to threats detected by Agent Two. The agent
    receives threat classifications and must choose appropriate mitigation
    actions while balancing security and availability.
    
    State Space:
        - 10-dim one-hot encoded attack category
        - 1-dim confidence score
        - 1-dim normalized severity level
        - 1-dim zero-day flag
        - 1-dim high-risk indicator
        Total: 14 dimensions, Box([0,1])
    
    Action Space:
        Discrete(4):
        - 0: Do Nothing
        - 1: Alert Admin
        - 2: Block IP
        - 3: Isolate Subnet
    
    Reward Design:
        - True Positive (correct action on real threat): +10 to +50
        - True Negative (do nothing on normal traffic): +5
        - False Positive (action on normal traffic): -20 to -100
        - False Negative (no action on real threat): -30 to -100
        - Action severity matching: Bonus/penalty based on proportionality
    
    Example:
        >>> env = NetworkDefenseEnv()
        >>> obs, info = env.reset()
        >>> action = agent.predict(obs)
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Attack categories matching Agent Two
    ATTACK_CATEGORIES = [
        "Normal",
        "Fuzzers",
        "Analysis", 
        "Backdoor",
        "DoS",
        "Exploits",
        "Generic",
        "Reconnaissance",
        "Shellcode",
        "Worms",
    ]
    
    # Category to index mapping
    CATEGORY_TO_INDEX = {cat: i for i, cat in enumerate(ATTACK_CATEGORIES)}
    
    # Severity levels for each category
    CATEGORY_SEVERITY = {
        "Normal": 0,        # low
        "Reconnaissance": 1, # medium
        "Fuzzers": 1,       # medium
        "Analysis": 1,      # medium
        "DoS": 2,           # high
        "Exploits": 2,      # high
        "Generic": 1,       # medium
        "Backdoor": 3,      # critical
        "Shellcode": 3,     # critical
        "Worms": 3,         # critical
    }
    
    # Recommended minimum action for each severity
    SEVERITY_MIN_ACTION = {
        0: MitigationAction.DO_NOTHING,    # low -> can do nothing
        1: MitigationAction.ALERT_ADMIN,   # medium -> at least alert
        2: MitigationAction.BLOCK_IP,      # high -> at least block
        3: MitigationAction.ISOLATE_SUBNET, # critical -> isolate
    }
    
    def __init__(
        self,
        threat_generator: Optional[Any] = None,
        max_steps: int = 100,
        attack_ratio: float = 0.7,
        render_mode: Optional[str] = None,
        # Reward parameters
        fp_penalty_base: float = -20.0,
        fn_penalty_base: float = -30.0,
        tp_reward_base: float = 10.0,
        tn_reward: float = 5.0,
        severity_multiplier: float = 2.0,
    ):
        """
        Initializes the NetworkDefenseEnv.
        
        Args:
            threat_generator: Optional generator for threat states.
                             If None, uses synthetic threat generation.
            max_steps: Maximum steps per episode.
            attack_ratio: Ratio of attacks vs normal traffic in synthetic mode.
            render_mode: Render mode ('human', 'ansi', or None).
            fp_penalty_base: Base penalty for false positives.
            fn_penalty_base: Base penalty for false negatives.
            tp_reward_base: Base reward for true positives.
            tn_reward: Reward for true negatives.
            severity_multiplier: Multiplier for severity-based adjustments.
        """
        super().__init__()
        
        self._threat_generator = threat_generator
        self._max_steps = max_steps
        self._attack_ratio = attack_ratio
        self.render_mode = render_mode
        
        # Reward parameters
        self._fp_penalty_base = fp_penalty_base
        self._fn_penalty_base = fn_penalty_base
        self._tp_reward_base = tp_reward_base
        self._tn_reward = tn_reward
        self._severity_multiplier = severity_multiplier
        
        # Define observation space (14 dimensions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )
        
        # Define action space (4 discrete actions)
        self.action_space = spaces.Discrete(4)
        
        # Episode state
        self._current_step = 0
        self._current_threat: Optional[ThreatState] = None
        self._episode_rewards: List[float] = []
        self._episode_actions: List[int] = []
        self._episode_outcomes: List[str] = []
        
        # Statistics
        self._total_episodes = 0
        self._total_true_positives = 0
        self._total_false_positives = 0
        self._total_true_negatives = 0
        self._total_false_negatives = 0
        
        # Random generator for reproducibility
        self._np_random: Optional[np.random.Generator] = None
        
        logger.info("NetworkDefenseEnv initialized: max_steps=%d, attack_ratio=%.2f",
                   max_steps, attack_ratio)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
        
        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()
        
        self._current_step = 0
        self._episode_rewards = []
        self._episode_actions = []
        self._episode_outcomes = []
        self._total_episodes += 1
        
        # Generate first threat
        self._current_threat = self._generate_threat()
        
        observation = self._current_threat.to_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.
        
        Args:
            action: Mitigation action (0-3).
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self._current_threat is not None, "Must call reset() first"
        
        self._current_step += 1
        
        # Calculate reward
        reward, outcome = self._calculate_reward(action, self._current_threat)
        
        self._episode_rewards.append(reward)
        self._episode_actions.append(action)
        self._episode_outcomes.append(outcome)
        
        # Update statistics
        self._update_statistics(outcome)
        
        # Check termination
        terminated = False  # Episodes don't terminate early
        truncated = self._current_step >= self._max_steps
        
        # Generate next threat
        if not truncated:
            self._current_threat = self._generate_threat()
        
        observation = self._current_threat.to_observation()
        info = self._get_info()
        info["reward"] = reward
        info["outcome"] = outcome
        info["action_taken"] = MitigationAction.get_description(action)
        
        return observation, reward, terminated, truncated, info
    
    def _generate_threat(self) -> ThreatState:
        """Generates a threat state (synthetic or from generator)."""
        if self._threat_generator is not None:
            return next(self._threat_generator)
        
        return self._generate_synthetic_threat()
    
    def _generate_synthetic_threat(self) -> ThreatState:
        """Generates a synthetic threat for training."""
        # Decide if this is an actual attack or normal traffic
        is_attack = self._np_random.random() < self._attack_ratio
        
        if is_attack:
            # Generate an attack
            # Weight towards more common attack types
            attack_weights = [0.0, 0.10, 0.05, 0.08, 0.15, 0.25, 0.20, 0.08, 0.05, 0.04]
            category_idx = self._np_random.choice(10, p=attack_weights)
        else:
            # Normal traffic
            category_idx = 0
        
        category = self.ATTACK_CATEGORIES[category_idx]
        severity = self.CATEGORY_SEVERITY[category]
        
        # Simulate classification confidence
        # Higher confidence for clearer cases, lower for edge cases
        if is_attack:
            # Attacks have varied confidence
            base_conf = self._np_random.uniform(0.5, 0.99)
            # Critical threats often have higher confidence
            if severity >= 2:
                base_conf = min(base_conf + 0.1, 0.99)
        else:
            # Normal traffic usually high confidence
            base_conf = self._np_random.uniform(0.7, 0.99)
        
        # Add some noise to make it realistic
        confidence = np.clip(base_conf + self._np_random.normal(0, 0.05), 0.3, 0.99)
        
        # Determine if classified as zero-day
        # Low confidence attacks might be flagged as zero-day
        is_zero_day = is_attack and confidence < 0.5
        
        return ThreatState(
            category_index=category_idx,
            confidence=confidence,
            severity_level=severity,
            is_zero_day=is_zero_day,
            is_actual_threat=is_attack,
            attack_category=category,
        )
    
    def _calculate_reward(
        self,
        action: int,
        threat: ThreatState,
    ) -> Tuple[float, str]:
        """
        Calculates reward based on action and threat ground truth.
        
        Reward design principles:
        1. Heavily penalize false positives (disrupting normal traffic)
        2. Penalize false negatives (missing real threats)
        3. Reward appropriate responses to threats
        4. Bonus for proportional responses (not over/under-reacting)
        
        Args:
            action: Action taken by agent.
            threat: Current threat state with ground truth.
        
        Returns:
            Tuple of (reward, outcome_string).
        """
        is_threat = threat.is_actual_threat
        took_action = action > MitigationAction.DO_NOTHING
        severity = threat.severity_level
        
        if is_threat:
            if took_action:
                # TRUE POSITIVE: Correctly responded to threat
                outcome = "true_positive"
                
                # Base reward scaled by severity
                reward = self._tp_reward_base * (1 + severity * self._severity_multiplier)
                
                # Check action proportionality
                min_action = self.SEVERITY_MIN_ACTION[severity]
                
                if action >= min_action:
                    # Appropriate or stronger response
                    reward += 10.0
                    if action == min_action:
                        # Perfectly proportional
                        reward += 5.0
                    elif action > min_action:
                        # Over-reaction (still okay but slight penalty)
                        reward -= (action - min_action) * 2.0
                else:
                    # Under-reaction (took action but not enough)
                    reward -= (min_action - action) * 5.0
                
            else:
                # FALSE NEGATIVE: Missed a real threat
                outcome = "false_negative"
                # Penalty scaled by threat severity
                reward = self._fn_penalty_base * (1 + severity * self._severity_multiplier)
                
        else:  # Normal traffic
            if took_action:
                # FALSE POSITIVE: Disrupted normal traffic
                outcome = "false_positive"
                # Penalty scaled by action severity
                # More severe actions = more disruption = bigger penalty
                action_severity_penalty = [0, 1, 2, 3][action]
                reward = self._fp_penalty_base * (1 + action_severity_penalty)
                
            else:
                # TRUE NEGATIVE: Correctly ignored normal traffic
                outcome = "true_negative"
                reward = self._tn_reward
        
        return float(reward), outcome
    
    def _update_statistics(self, outcome: str) -> None:
        """Updates running statistics."""
        if outcome == "true_positive":
            self._total_true_positives += 1
        elif outcome == "false_positive":
            self._total_false_positives += 1
        elif outcome == "true_negative":
            self._total_true_negatives += 1
        elif outcome == "false_negative":
            self._total_false_negatives += 1
    
    def _get_info(self) -> Dict[str, Any]:
        """Returns info dict for current state."""
        return {
            "step": self._current_step,
            "episode_number": self._total_episodes,
            "current_threat": {
                "category": self._current_threat.attack_category,
                "confidence": self._current_threat.confidence,
                "severity": self._current_threat.severity_level,
                "is_zero_day": self._current_threat.is_zero_day,
                "is_actual_threat": self._current_threat.is_actual_threat,
            } if self._current_threat else None,
            "episode_stats": {
                "mean_reward": np.mean(self._episode_rewards) if self._episode_rewards else 0.0,
                "total_reward": sum(self._episode_rewards),
                "actions_taken": len(self._episode_actions),
            },
            "global_stats": self.get_statistics(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns global statistics across all episodes."""
        total = (self._total_true_positives + self._total_false_positives +
                self._total_true_negatives + self._total_false_negatives)
        
        if total == 0:
            return {"total_decisions": 0}
        
        # Calculate metrics
        precision = (self._total_true_positives / 
                    (self._total_true_positives + self._total_false_positives + 1e-10))
        recall = (self._total_true_positives /
                 (self._total_true_positives + self._total_false_negatives + 1e-10))
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = ((self._total_true_positives + self._total_true_negatives) / total)
        
        return {
            "total_decisions": total,
            "true_positives": self._total_true_positives,
            "false_positives": self._total_false_positives,
            "true_negatives": self._total_true_negatives,
            "false_negatives": self._total_false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": self._total_false_positives / total,
        }
    
    def render(self) -> Optional[str]:
        """Renders the current state."""
        if self.render_mode is None:
            return None
        
        if self._current_threat is None:
            return "No active threat"
        
        lines = [
            "=" * 50,
            "NETWORK DEFENSE STATUS",
            "=" * 50,
            f"Step: {self._current_step}/{self._max_steps}",
            f"Episode: {self._total_episodes}",
            "-" * 50,
            "Current Threat:",
            f"  Category: {self._current_threat.attack_category}",
            f"  Confidence: {self._current_threat.confidence:.2%}",
            f"  Severity: {['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][self._current_threat.severity_level]}",
            f"  Zero-day: {'Yes' if self._current_threat.is_zero_day else 'No'}",
            f"  Actual threat: {'Yes' if self._current_threat.is_actual_threat else 'No'}",
            "-" * 50,
        ]
        
        if self._episode_actions:
            last_action = self._episode_actions[-1]
            last_reward = self._episode_rewards[-1]
            last_outcome = self._episode_outcomes[-1]
            lines.extend([
                f"Last Action: {MitigationAction.get_description(last_action)}",
                f"Last Reward: {last_reward:.2f}",
                f"Last Outcome: {last_outcome}",
            ])
        
        stats = self.get_statistics()
        if stats["total_decisions"] > 0:
            lines.extend([
                "-" * 50,
                "Performance:",
                f"  Accuracy: {stats['accuracy']:.2%}",
                f"  Precision: {stats['precision']:.2%}",
                f"  Recall: {stats['recall']:.2%}",
                f"  F1 Score: {stats['f1_score']:.2%}",
            ])
        
        lines.append("=" * 50)
        
        output = "\n".join(lines)
        
        if self.render_mode == "human":
            print(output)
        
        return output
    
    def close(self) -> None:
        """Cleanup when environment is closed."""
        pass


def make_threat_generator_from_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    classifier: Any,
    category_to_index: Dict[str, int],
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    """
    Creates a threat generator from a dataset and classifier.
    
    This allows training the RL agent on real data classified by Agent Two.
    
    Args:
        features: Feature array from dataset.
        labels: Ground truth labels (attack category names).
        classifier: Trained ThreatClassifier from Agent Two.
        category_to_index: Category name to index mapping.
        shuffle: Whether to shuffle the data.
        seed: Random seed.
    
    Yields:
        ThreatState objects.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    
    if shuffle:
        rng.shuffle(indices)
    
    severity_map = {
        "Normal": 0, "Reconnaissance": 1, "Fuzzers": 1, "Analysis": 1,
        "DoS": 2, "Exploits": 2, "Generic": 1,
        "Backdoor": 3, "Shellcode": 3, "Worms": 3,
    }
    
    i = 0
    while True:
        idx = indices[i % len(indices)]
        
        # Get classification from Agent Two's classifier
        result = classifier.predict(features[idx])
        
        # Get ground truth
        actual_label = labels[idx]
        is_actual_threat = actual_label != "Normal"
        
        yield ThreatState(
            category_index=category_to_index.get(result.predicted_category, 0),
            confidence=result.confidence,
            severity_level=severity_map.get(result.predicted_category, 1),
            is_zero_day=result.is_zero_day,
            is_actual_threat=is_actual_threat,
            attack_category=result.predicted_category,
            feature_vector=features[idx],
        )
        
        i += 1
        if i >= len(indices) and shuffle:
            rng.shuffle(indices)
            i = 0
