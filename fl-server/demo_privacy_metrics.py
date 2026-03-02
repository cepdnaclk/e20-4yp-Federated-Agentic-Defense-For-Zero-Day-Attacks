"""
Demo Script for Privacy Metrics

This script simulates federation rounds to demonstrate the privacy metrics
collection and visualization capabilities.

Usage:
    python demo_privacy_metrics.py [--rounds 10] [--agents 5]
"""

import argparse
import os
import sys
import random
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy.privacy_metrics import PrivacyMetricsCollector
from privacy.privacy_analyzer import PrivacyAnalyzer


def generate_random_weights(num_layers: int = 5, layer_sizes: list = None) -> list:
    """Generate random model weights"""
    if layer_sizes is None:
        layer_sizes = [64, 32, 16, 32, 64]
    
    weights = []
    for i in range(num_layers):
        if i == 0:
            shape = (layer_sizes[i], 10)  # Input layer
        elif i == num_layers - 1:
            shape = (layer_sizes[i-1], layer_sizes[i])  # Output layer
        else:
            shape = (layer_sizes[i-1], layer_sizes[i])
        
        # Generate weights with some structure (not purely random)
        w = np.random.randn(*shape).astype(np.float32)
        # Add some sparsity
        mask = np.random.random(shape) > 0.3
        w *= mask
        weights.append(w * 0.1)  # Scale down
    
    return weights


def generate_embeddings(num_samples: int, dim: int = 16) -> tuple:
    """Generate random embeddings and reconstruction errors"""
    embeddings = np.random.randn(num_samples, dim).astype(np.float32)
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Generate reconstruction errors (higher = more anomalous)
    recon_errors = np.abs(np.random.randn(num_samples).astype(np.float32)) + 0.5
    
    return embeddings, recon_errors


def simulate_federation(
    num_rounds: int = 10,
    num_agents: int = 5,
    samples_per_agent: int = 100,
    log_path: str = "./privacy_logs"
):
    """Simulate federated learning rounds with privacy metrics collection"""
    
    print(f"\n{'='*60}")
    print("FEDERATED LEARNING PRIVACY METRICS DEMO")
    print(f"{'='*60}")
    print(f"Simulating {num_rounds} rounds with {num_agents} agents")
    print(f"Samples per agent: {samples_per_agent}")
    print(f"Log path: {log_path}")
    print()
    
    # Initialize collector
    collector = PrivacyMetricsCollector(
        log_path=log_path,
        target_epsilon=10.0,
        target_delta=1e-5,
        noise_multiplier=1.0,
        clip_norm=1.0
    )
    
    agent_ids = [f"agent_{i+1}" for i in range(num_agents)]
    
    for round_id in range(1, num_rounds + 1):
        print(f"\n--- Round {round_id}/{num_rounds} ---")
        
        # Start the round
        collector.start_round(round_id)
        
        # Simulate each agent's participation
        # Not all agents participate in every round
        participating_agents = random.sample(
            agent_ids,
            k=random.randint(max(2, num_agents // 2), num_agents)
        )
        
        aggregated_weights = []
        
        for agent_id in participating_agents:
            # Generate random model weights for this agent
            weights = generate_random_weights()
            sample_count = random.randint(
                samples_per_agent // 2,
                samples_per_agent * 2
            )
            
            # Calculate approximate bytes (simplified)
            raw_bytes = sum(w.nbytes for w in weights)
            
            # Record the agent's update
            collector.record_agent_update(
                agent_id=agent_id,
                weights=weights,
                sample_count=sample_count,
                raw_bytes=raw_bytes
            )
            
            # Generate and record signatures/embeddings
            num_signatures = random.randint(5, 20)
            embeddings, recon_errors = generate_embeddings(num_signatures)
            collector.record_signatures(agent_id, embeddings, recon_errors)
            
            print(f"  {agent_id}: {sample_count} samples, {num_signatures} signatures")
            
            # Accumulate weights for aggregation
            if not aggregated_weights:
                aggregated_weights = [w.copy() for w in weights]
            else:
                for i, w in enumerate(weights):
                    aggregated_weights[i] += w
        
        # Average the aggregated weights
        num_participants = len(participating_agents)
        aggregated_weights = [w / num_participants for w in aggregated_weights]
        
        # Simulate zero-day candidates found
        zero_days = random.randint(0, 2)
        
        # End the round and collect metrics
        metrics = collector.end_round(
            aggregated_weights=aggregated_weights,
            zero_day_count=zero_days
        )
        
        print(f"  Privacy: ε={metrics.epsilon:.4f}, exposure_risk={metrics.information_exposure_risk:.4f}")
        print(f"  Participants: {metrics.participating_agents}, Zero-days: {metrics.zero_day_candidates_found}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    
    summary = collector.get_privacy_summary()
    print(f"\nTotal rounds: {summary['total_rounds']}")
    print(f"Cumulative ε: {summary['cumulative_epsilon']:.4f}")
    print(f"Privacy budget consumed: {summary['privacy_budget_consumed']}")
    print(f"Average exposure risk: {summary['avg_exposure_risk']:.4f}")
    print(f"Average gradient similarity: {summary['avg_gradient_similarity']:.4f}")
    print(f"Total signatures shared: {summary['total_signatures_shared']}")
    
    return collector


def main():
    parser = argparse.ArgumentParser(
        description="Demo privacy metrics for federated learning"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=10,
        help="Number of federation rounds to simulate (default: 10)"
    )
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=5,
        help="Number of agents (default: 5)"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=100,
        help="Average samples per agent per round (default: 100)"
    )
    parser.add_argument(
        "--log-path", "-l",
        default="./privacy_logs",
        help="Path to store privacy logs (default: ./privacy_logs)"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations after simulation"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate text report after simulation"
    )

    args = parser.parse_args()

    # Run simulation
    collector = simulate_federation(
        num_rounds=args.rounds,
        num_agents=args.agents,
        samples_per_agent=args.samples,
        log_path=args.log_path
    )

    # Generate visualizations if requested
    if args.visualize or args.report:
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        analyzer = PrivacyAnalyzer(log_path=args.log_path)
        analyzer.load_metrics(collector.get_all_metrics())
        
        if args.visualize:
            analyzer.plot_privacy_dashboard(save=True, show=True)
        
        if args.report:
            report = analyzer.generate_report(
                output_file=os.path.join(args.log_path, "visualizations", "privacy_report.txt")
            )
            print("\n" + report)
        
        print(f"\nOutput saved to: {analyzer.output_path}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
