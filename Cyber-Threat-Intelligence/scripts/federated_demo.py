"""
Federated Learning Quick Start Demo.

This script provides a simple way to test the federated learning setup
using Flower's built-in simulation capabilities (no separate processes needed).

Usage:
    python scripts/federated_demo.py [--num-clients 5] [--num-rounds 3]

Example:
    python scripts/federated_demo.py --num-clients 3 --num-rounds 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters

from agents.models.autoencoder import AnomalyAutoencoder
from federated.client import NetworkDefenseClient, create_client_fn
from federated.server import NetworkDefenseStrategy, get_fit_config, get_evaluate_config
from federated.coordinator import IntegrationCoordinator, SemanticThreatReport
from federated.utils import autoencoder_weights_to_numpy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_client_data(client_id: str, num_samples: int = 500):
    """
    Generate synthetic training/validation data for a client.
    
    In production, each client would load its own local network traffic data.
    """
    # Simulate different data distributions per client
    seed = hash(client_id) % 10000
    np.random.seed(seed)
    
    # Training data
    X_train = np.random.randn(num_samples, 40).astype(np.float32)
    # Add some client-specific bias to simulate heterogeneous data
    X_train += np.random.randn(40) * 0.1
    
    # Validation data
    X_val = np.random.randn(num_samples // 5, 40).astype(np.float32)
    X_val += np.random.randn(40) * 0.1
    
    return (X_train, None), (X_val, None)


def client_fn(client_id: str) -> NetworkDefenseClient:
    """
    Create a NetworkDefenseClient for simulation.
    
    This function is called by Flower simulation to create client instances.
    """
    # Create autoencoder model
    model = AnomalyAutoencoder(input_dim=40, latent_dim=8, hidden_dims=[32, 16])
    
    # Generate client-specific data
    train_data, val_data = generate_client_data(client_id)
    
    return NetworkDefenseClient(
        autoencoder=model,
        train_data=train_data,
        val_data=val_data,
        training_config={
            "autoencoder_epochs": 3,
            "autoencoder_batch_size": 128,
            "autoencoder_lr": 0.001,
        },
        client_id=client_id,
    )


def run_simulation(num_clients: int = 5, num_rounds: int = 3):
    """
    Run federated learning simulation.
    
    This uses Flower's virtual client simulation, which runs all clients
    in a single process for easy testing.
    """
    logger.info(f"Starting simulation: {num_clients} clients, {num_rounds} rounds")
    
    # Create initial model and get parameters
    initial_model = AnomalyAutoencoder(input_dim=40, latent_dim=8, hidden_dims=[32, 16])
    initial_weights = autoencoder_weights_to_numpy(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)
    
    # Results tracking
    round_results = []
    
    def on_aggregate(result):
        """Callback after each aggregation round."""
        round_results.append({
            "round": result.round_number,
            "clients": result.num_clients,
            "loss": result.aggregated_loss,
        })
        logger.info(
            f"Round {result.round_number} complete: "
            f"loss={result.aggregated_loss:.6f}, clients={result.num_clients}"
        )
    
    # Create strategy
    strategy = NetworkDefenseStrategy(
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=0.5,  # Evaluate with 50% of clients
        min_fit_clients=max(2, num_clients // 2),
        min_evaluate_clients=1,
        min_available_clients=max(2, num_clients // 2),
        initial_parameters=initial_parameters,
        on_fit_config_fn=get_fit_config,
        on_evaluate_config_fn=get_evaluate_config,
        on_aggregate_callback=on_aggregate,
    )
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},  # Limit resources per client
    )
    
    return history, round_results


def demo_integration_coordinator():
    """
    Demonstrate the IntegrationCoordinator with mock agents.
    """
    from unittest.mock import Mock
    from types import SimpleNamespace
    
    logger.info("Demonstrating IntegrationCoordinator...")
    
    # Create mock agents
    mock_ae = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    
    mock_agent_one = Mock()
    mock_agent_one.detect = Mock(return_value=True)
    mock_agent_one.reconstruction_error = 0.05
    mock_agent_one.threshold = 0.0396
    mock_agent_one.model = mock_ae
    
    mock_agent_two = Mock()
    mock_agent_two.classify = Mock(return_value={"category": "DoS", "confidence": 0.85})
    
    mock_agent_three = Mock()
    mock_agent_three.recommend_actions = Mock(return_value=SimpleNamespace(
        primary_action="Rate-limit affected services",
        confidence=0.88,
        recommended_actions=[
            "Rate-limit affected services",
            "Block top offending source IPs",
            "Enable DDoS protection rules",
        ],
        threat_summary="Detected likely DoS behavior with high confidence.",
        cve_references=[],
        model="mock-llm",
    ))
    
    # Create coordinator
    coordinator = IntegrationCoordinator(
        agent_one=mock_agent_one,
        agent_two=mock_agent_two,
        agent_three=mock_agent_three,
    )
    
    # Process a sample
    sample = np.random.randn(40)
    report = coordinator.process_network_sample(
        sample,
        sample_id="demo_sample_001",
        include_rag_enrichment=False,
    )
    
    # Display report
    print("\n" + "=" * 60)
    print("Sample Threat Report")
    print("=" * 60)
    print(report.to_markdown())
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Demo for Network Defense IDS"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of federated clients (default: 5)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of training rounds (default: 3)",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip FL simulation, only run coordinator demo",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Network Defense Federated Learning Demo")
    print("=" * 60 + "\n")
    
    # Run federated simulation
    if not args.skip_simulation:
        try:
            history, results = run_simulation(
                num_clients=args.num_clients,
                num_rounds=args.num_rounds,
            )
            
            print("\n" + "=" * 60)
            print("Federated Learning Results")
            print("=" * 60)
            print(f"Rounds completed: {len(results)}")
            for r in results:
                print(f"  Round {r['round']}: loss={r['loss']:.6f}, clients={r['clients']}")
            
            if history.losses_distributed:
                final_loss = history.losses_distributed[-1][1]
                print(f"\nFinal distributed loss: {final_loss:.6f}")
                
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            print(f"\nSimulation failed: {e}")
            print("This might be due to missing Flower dependencies.")
            print("Install with: pip install flwr[simulation]")
    
    # Demonstrate coordinator
    print("\n")
    report = demo_integration_coordinator()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo run federated learning with separate processes:")
    print("  1. Start server: python -m federated.server --rounds 10")
    print("  2. Start clients: Use the scripts in scripts/ folder")
    print("\nTo run tests:")
    print("  pytest tests/test_federated_bridge.py -v")


if __name__ == "__main__":
    main()
