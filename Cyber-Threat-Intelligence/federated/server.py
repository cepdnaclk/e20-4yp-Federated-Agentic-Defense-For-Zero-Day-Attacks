"""
Federated Learning Server for Network Defense IDS.

This module implements the Flower server component that orchestrates
federated training across multiple network nodes. It uses FedAvg
(Federated Averaging) for model aggregation.

Usage:
    python -m federated.server --rounds 10 --min_clients 2

Classes:
    NetworkDefenseStrategy: Custom FedAvg with enhanced callbacks.

Functions:
    start_federated_server: Launch the federation server.
"""

import logging
import argparse
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    NDArrays,
    FitRes,
    EvaluateRes,
    logger as fl_logger,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of a federated aggregation round."""
    round_number: int
    num_clients: int
    aggregated_loss: float
    metrics: Dict[str, float]
    timestamp: str


class NetworkDefenseStrategy(FedAvg):
    """
    Custom Federated Averaging strategy for IDS training.
    
    Extends FedAvg with:
    - Per-round metric logging
    - Custom aggregation callbacks
    - Support for mixed model types (PyTorch + XGBoost)
    - Graceful handling of client failures
    
    Attributes:
        save_path: Directory to save aggregated models.
        round_results: History of aggregation results.
        on_aggregate_callback: Optional callback after each round.
    
    Example:
        >>> strategy = NetworkDefenseStrategy(
        ...     min_fit_clients=2,
        ...     min_available_clients=3,
        ...     save_path="./federated_checkpoints",
        ... )
        >>> fl.server.start_server(strategy=strategy)
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        save_path: Optional[str] = None,
        on_aggregate_callback: Optional[Callable[[AggregationResult], None]] = None,
    ):
        """
        Initialize Network Defense federated strategy.
        
        Args:
            fraction_fit: Fraction of clients to train per round.
            fraction_evaluate: Fraction of clients to evaluate per round.
            min_fit_clients: Minimum clients required for training.
            min_evaluate_clients: Minimum clients required for evaluation.
            min_available_clients: Minimum clients to start a round.
            evaluate_fn: Optional server-side evaluation function.
            on_fit_config_fn: Function to configure fit per round.
            on_evaluate_config_fn: Function to configure evaluate per round.
            accept_failures: Whether to continue if some clients fail.
            initial_parameters: Initial global model parameters.
            save_path: Directory to save model checkpoints.
            on_aggregate_callback: Callback invoked after aggregation.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        
        self.save_path = Path(save_path) if save_path else None
        self.on_aggregate_callback = on_aggregate_callback
        self.round_results: List[AggregationResult] = []
        
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"NetworkDefenseStrategy initialized: "
            f"min_fit={min_fit_clients}, min_eval={min_evaluate_clients}"
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client training results using FedAvg.
        
        This method is called after all clients complete their training.
        It averages the weights and logs metrics.
        
        Args:
            server_round: Current round number.
            results: List of (client, fit_result) tuples.
            failures: List of failed clients or exceptions.
        
        Returns:
            Tuple of (aggregated_parameters, metrics_dict).
        """
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}
        
        # Log client participation
        logger.info(
            f"Round {server_round}: Aggregating {len(results)} clients "
            f"({len(failures)} failures)"
        )
        
        # Call parent FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Collect per-client metrics
        client_metrics = {}
        total_samples = 0
        total_ae_loss = 0.0
        
        for client, fit_res in results:
            client_id = fit_res.metrics.get("client_id", "unknown")
            ae_loss = fit_res.metrics.get("autoencoder_loss", 0.0)
            num_samples = fit_res.num_examples
            
            client_metrics[client_id] = {
                "ae_loss": ae_loss,
                "samples": num_samples,
            }
            
            total_samples += num_samples
            total_ae_loss += ae_loss * num_samples
        
        # Weighted average loss
        avg_loss = total_ae_loss / total_samples if total_samples > 0 else 0.0
        
        # Build aggregation result
        result = AggregationResult(
            round_number=server_round,
            num_clients=len(results),
            aggregated_loss=avg_loss,
            metrics=client_metrics,
            timestamp=datetime.now().isoformat(),
        )
        self.round_results.append(result)
        
        # Save checkpoint if configured
        if self.save_path and aggregated_params:
            self._save_checkpoint(server_round, aggregated_params)
        
        # Invoke callback if provided
        if self.on_aggregate_callback:
            try:
                self.on_aggregate_callback(result)
            except Exception as e:
                logger.error(f"Aggregation callback error: {e}")
        
        logger.info(
            f"Round {server_round}: Aggregated loss = {avg_loss:.6f}, "
            f"total_samples = {total_samples}"
        )
        
        return aggregated_params, {"aggregated_loss": avg_loss, **metrics}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate client evaluation results.
        
        Args:
            server_round: Current round number.
            results: List of (client, eval_result) tuples.
            failures: List of failed evaluations.
        
        Returns:
            Tuple of (aggregated_loss, metrics_dict).
        """
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results")
            return None, {}
        
        # Call parent aggregation
        aggregated_loss, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Log evaluation summary
        loss_str = f"{aggregated_loss:.6f}" if aggregated_loss is not None else "N/A"
        logger.info(
            f"Round {server_round}: Evaluation complete. Aggregated loss: {loss_str}"
        )
        
        return aggregated_loss, metrics
    
    def _save_checkpoint(
        self, server_round: int, parameters: Parameters
    ) -> None:
        """Save aggregated parameters to disk."""
        checkpoint_file = self.save_path / f"round_{server_round}_params.npz"
        
        # Decode parameters to numpy arrays
        weights = fl.common.parameters_to_ndarrays(parameters)
        
        # Save as compressed numpy file
        np.savez_compressed(checkpoint_file, *[w for w in weights])
        
        logger.info(f"Saved checkpoint: {checkpoint_file}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Return aggregation history as list of dicts."""
        return [
            {
                "round": r.round_number,
                "clients": r.num_clients,
                "loss": r.aggregated_loss,
                "timestamp": r.timestamp,
            }
            for r in self.round_results
        ]


def get_fit_config(server_round: int) -> Dict[str, Scalar]:
    """
    Generate per-round training configuration.
    
    Args:
        server_round: Current round number.
    
    Returns:
        Configuration dict sent to clients.
    """
    config = {
        "server_round": server_round,
        "local_epochs": 3 if server_round < 5 else 5,  # More epochs later
        "batch_size": 256,
        "learning_rate": 0.001 * (0.95 ** server_round),  # LR decay
    }
    return config


def get_evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """
    Generate per-round evaluation configuration.
    
    Args:
        server_round: Current round number.
    
    Returns:
        Configuration dict sent to clients.
    """
    return {
        "server_round": server_round,
        "evaluate_all": server_round % 5 == 0,  # Full eval every 5 rounds
    }


def start_federated_server(
    server_address: str = "[::]:8080",
    num_rounds: int = 10,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    save_path: Optional[str] = None,
    initial_parameters: Optional[NDArrays] = None,
    on_aggregate_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Start the Network Defense federated learning server.
    
    This function launches a Flower server configured for IDS model
    aggregation. It blocks until all rounds complete or timeout.
    
    Args:
        server_address: Address to bind server (default: all interfaces:8080).
        num_rounds: Number of federated learning rounds.
        min_fit_clients: Minimum clients for training per round.
        min_available_clients: Minimum clients to start training.
        save_path: Directory for model checkpoints.
        initial_parameters: Initial model weights (or None for random).
        on_aggregate_callback: Callback after each aggregation.
    
    Returns:
        Dict containing training history and final metrics.
    
    Example:
        >>> # Start server with 10 rounds, minimum 3 clients
        >>> history = start_federated_server(
        ...     num_rounds=10,
        ...     min_fit_clients=3,
        ...     save_path="./checkpoints",
        ... )
        >>> print(f"Final loss: {history['final_loss']}")
    """
    logger.info(
        f"Starting federated server: address={server_address}, "
        f"rounds={num_rounds}, min_clients={min_fit_clients}"
    )
    
    # Convert initial parameters if provided
    init_params = None
    if initial_parameters is not None:
        init_params = fl.common.ndarrays_to_parameters(initial_parameters)
    
    # Create strategy
    strategy = NetworkDefenseStrategy(
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=get_fit_config,
        on_evaluate_config_fn=get_evaluate_config,
        initial_parameters=init_params,
        save_path=save_path,
        on_aggregate_callback=on_aggregate_callback,
    )
    
    # Start server
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    # Build result summary
    result = {
        "num_rounds": num_rounds,
        "history": strategy.get_history(),
        "losses_distributed": history.losses_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_distributed": history.metrics_distributed,
        "final_loss": (
            history.losses_distributed[-1][1]
            if history.losses_distributed else None
        ),
    }
    
    logger.info(f"Federated training complete: {num_rounds} rounds")
    
    return result


def main():
    """Command-line entry point for federated server."""
    parser = argparse.ArgumentParser(
        description="Network Defense Federated Learning Server"
    )
    parser.add_argument(
        "--address",
        type=str,
        default="[::]:8080",
        help="Server address (default: [::]:8080)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated rounds (default: 10)",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum clients per round (default: 2)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Start server
    result = start_federated_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        save_path=args.save_path,
    )
    
    print(f"\n{'=' * 60}")
    print("Federated Training Complete")
    print(f"{'=' * 60}")
    print(f"Rounds completed: {result['num_rounds']}")
    print(f"Final loss: {result['final_loss']}")
    print(f"History: {len(result['history'])} entries")


if __name__ == "__main__":
    main()
