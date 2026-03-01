#!/usr/bin/env python
"""
Training script for Agent Three (RL-based Network Mitigation Agent).

This script trains a PPO agent to make optimal mitigation decisions
based on threat classifications from Agent Two.

Usage:
    python -m agents.train_rl --timesteps 100000 --save-path models/agent_three
    
    # With custom threat distribution
    python -m agents.train_rl --threat-ratio 0.3 --timesteps 50000
    
    # Resume training from checkpoint
    python -m agents.train_rl --load-path models/agent_three --timesteps 50000
"""

import argparse
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_environment(
    attack_ratio: float = 0.4,
    max_steps: int = 1000,
) -> "NetworkDefenseEnv":
    """
    Sets up the training environment.
    
    Args:
        attack_ratio: Ratio of attacks vs normal traffic (0-1).
        max_steps: Maximum steps per episode.
    
    Returns:
        Configured NetworkDefenseEnv instance.
    """
    from agents.environments.network_defense_env import NetworkDefenseEnv
    
    env = NetworkDefenseEnv(
        attack_ratio=attack_ratio,
        max_steps=max_steps,
    )
    
    logger.info(
        "Environment created: attack_ratio=%.2f, max_steps=%d",
        attack_ratio, max_steps
    )
    
    return env


def create_agent(
    env: "NetworkDefenseEnv",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    device: str = "auto",
    verbose: int = 1,
    load_path: Optional[str] = None,
) -> "AgentThree":
    """
    Creates or loads the RL agent.
    
    Args:
        env: Training environment.
        learning_rate: Learning rate.
        n_steps: Steps per update.
        batch_size: Minibatch size.
        n_epochs: Epochs per update.
        gamma: Discount factor.
        ent_coef: Entropy coefficient for exploration.
        clip_range: PPO clipping range.
        device: Device to use.
        verbose: Verbosity level.
        load_path: Path to load existing model from.
    
    Returns:
        Configured AgentThree instance.
    """
    from agents.agent_three import AgentThree
    
    if load_path and Path(load_path).exists():
        logger.info("Loading existing agent from: %s", load_path)
        agent = AgentThree.from_pretrained(load_path)
        # Update environment
        agent._env = env
        return agent
    
    logger.info("Creating new AgentThree with PPO")
    agent = AgentThree(
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        device=device,
        verbose=verbose,
    )
    
    return agent


def train_agent(
    agent: "AgentThree",
    total_timesteps: int = 100000,
    save_freq: int = 10000,
    save_path: str = "models/agent_three",
    eval_freq: int = 5000,
    eval_episodes: int = 20,
    progress_bar: bool = True,
) -> dict:
    """
    Trains the agent with periodic checkpoints and evaluation.
    
    Args:
        agent: AgentThree instance to train.
        total_timesteps: Total training timesteps.
        save_freq: Checkpoint save frequency.
        save_path: Path to save checkpoints.
        eval_freq: Evaluation frequency.
        eval_episodes: Episodes per evaluation.
        progress_bar: Show progress bar.
    
    Returns:
        Training results dictionary.
    """
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    from agents.environments.network_defense_env import NetworkDefenseEnv
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_path / "checkpoints"),
        name_prefix="ppo_checkpoint",
        verbose=1,
    )
    
    # Create evaluation environment
    eval_env = NetworkDefenseEnv(
        attack_ratio=agent.env._attack_ratio,
        max_steps=agent.env._max_steps,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    logger.info("Starting training for %d timesteps...", total_timesteps)
    start_time = datetime.now()
    
    stats = agent.train(
        total_timesteps=total_timesteps,
        progress_bar=progress_bar,
        callback=callbacks,
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save final model
    agent.save(save_path / "final_model")
    
    # Run final evaluation
    logger.info("Running final evaluation...")
    eval_stats = agent.evaluate(n_episodes=100, deterministic=True)
    
    results = {
        "training_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "training_stats": stats,
        "final_evaluation": eval_stats,
        "save_path": str(save_path),
    }
    
    # Save results
    with open(save_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def print_results(results: dict) -> None:
    """Prints training results summary."""
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    print(f"\nTraining Duration: {results['training_time_seconds']:.1f} seconds")
    print(f"Total Timesteps: {results['training_timesteps']:,}")
    
    eval_stats = results.get("final_evaluation", {})
    print("\nFinal Evaluation Results:")
    print(f"  - Mean Episode Reward: {eval_stats.get('mean_episode_reward', 0):.2f}")
    print(f"  - Accuracy: {eval_stats.get('accuracy', 0) * 100:.1f}%")
    print(f"  - Precision: {eval_stats.get('precision', 0) * 100:.1f}%")
    print(f"  - Recall (TPR): {eval_stats.get('recall', 0) * 100:.1f}%")
    print(f"  - F1 Score: {eval_stats.get('f1_score', 0) * 100:.1f}%")
    print(f"  - False Positive Rate: {eval_stats.get('false_positive_rate', 0) * 100:.1f}%")
    
    print(f"\nModel saved to: {results['save_path']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Agent Three (RL-based Network Mitigation Agent)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs per update",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per update",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    
    # Environment parameters
    parser.add_argument(
        "--attack-ratio",
        type=float,
        default=0.4,
        help="Ratio of attacks vs normal traffic (0-1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    
    # I/O parameters
    parser.add_argument(
        "--save-path", "-o",
        type=str,
        default="models/agent_three",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--load-path", "-l",
        type=str,
        default=None,
        help="Path to load existing model for continued training",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Checkpoint save frequency",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency",
    )
    
    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info("Random seed set to: %d", args.seed)
    
    try:
        # Setup environment
        env = setup_environment(
            attack_ratio=args.attack_ratio,
            max_steps=args.max_steps,
        )
        
        # Create agent
        agent = create_agent(
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            device=args.device,
            verbose=args.verbose,
            load_path=args.load_path,
        )
        
        # Train
        results = train_agent(
            agent=agent,
            total_timesteps=args.timesteps,
            save_freq=args.save_freq,
            save_path=args.save_path,
            eval_freq=args.eval_freq,
            progress_bar=not args.no_progress,
        )
        
        # Print results
        print_results(results)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
