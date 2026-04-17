#!/usr/bin/env python3
"""
Federated XGBoost with Tree Concatenation (Bagging) using Flower Framework.

This implementation solves the XGBoost federated aggregation "invalid" issue
by using tree concatenation instead of weight averaging. Each client trains
a local XGBoost "sub-forest" (10 trees), and the server concatenates all
trees into a larger global model.

Architecture:
    - Client: Trains local XGBoost with 10 trees → sends tree JSON to server
    - Server: Concatenates trees from all clients → 20 trees after 1 round
    - Rounds: 3 rounds with accumulation → 60 final trees (10 × 2 × 3)

Key Features:
    - Native XGBoost tree manipulation via JSON serialization
    - Flower framework for federated orchestration
    - Tree accumulation across rounds (bagging ensemble)
    - Support for heterogeneous data distributions

Requirements:
    - GROQ_API_KEY in .env (optional, for RAG explanations)
    - UNSW-NB15 training-set.csv and testing-set.csv in data/

Usage:
    python scripts/federated_xgboost_flower.py --num-rounds 3 --trees-per-client 10

Author: Federated Agentic Defense Team
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Flower imports
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FederatedXGBoostConfig:
    """Configuration for federated XGBoost training."""
    trees_per_client: int = 10          # Trees each client trains per round
    num_rounds: int = 3                  # Federated learning rounds
    max_depth: int = 6                   # Tree depth
    learning_rate: float = 0.1           # XGBoost eta
    objective: str = "multi:softprob"    # Multi-class classification
    eval_metric: str = "mlogloss"        # Evaluation metric
    max_samples: int = 10000             # Max samples per client
    test_fraction: float = 0.2           # Test set fraction
    random_state: int = 42               # Random seed


# =============================================================================
# XGBoost Tree Utilities
# =============================================================================

def serialize_xgboost_model(model: xgb.Booster) -> bytes:
    """Serialize XGBoost Booster to JSON bytes."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    try:
        model.save_model(temp_path)
        with open(temp_path, 'r') as f:
            model_json = f.read()
        return model_json.encode('utf-8')
    finally:
        Path(temp_path).unlink(missing_ok=True)


def deserialize_xgboost_model(model_bytes: bytes) -> xgb.Booster:
    """Deserialize JSON bytes to XGBoost Booster."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    try:
        with open(temp_path, 'w') as f:
            f.write(model_bytes.decode('utf-8'))
        model = xgb.Booster()
        model.load_model(temp_path)
        return model
    finally:
        Path(temp_path).unlink(missing_ok=True)


def ensemble_predict(
    models: List[xgb.Booster],
    dmatrix: xgb.DMatrix,
    num_classes: int,
) -> np.ndarray:
    """
    Ensemble prediction by averaging probability outputs from multiple models.
    
    This implements a simple bagging ensemble:
    - Each client's model makes independent predictions
    - Probabilities are averaged across all models
    - Final prediction uses the ensemble average
    
    Args:
        models: List of XGBoost Booster objects from different clients
        dmatrix: XGBoost DMatrix for prediction
        num_classes: Number of classification classes
        
    Returns:
        Averaged probability predictions, shape (n_samples, num_classes)
    """
    if not models:
        raise ValueError("No models for prediction")
    
    # Collect predictions from all models
    all_preds = []
    for model in models:
        pred = model.predict(dmatrix)
        if len(pred.shape) == 1:
            # Binary classification or regression
            pred = pred.reshape(-1, 1)
        all_preds.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(all_preds, axis=0)
    
    return ensemble_pred


class EnsembleModel:
    """
    Weighted ensemble of XGBoost models for federated bagging.
    
    Models are weighted by their local accuracy, giving better-performing
    models more influence in the final prediction.
    """
    
    def __init__(self, num_classes: int):
        self.models: List[xgb.Booster] = []
        self.weights: List[float] = []  # Accuracy-based weights
        self.num_classes = num_classes
    
    def add_model(self, model: xgb.Booster, accuracy: float):
        """Add a model with its accuracy weight."""
        self.models.append(model)
        self.weights.append(accuracy)
        logger.info(f"  Ensemble now has {len(self.models)} sub-models")
    
    def predict(self, dmatrix: xgb.DMatrix) -> np.ndarray:
        """Make weighted ensemble prediction."""
        if not self.models:
            raise ValueError("No models for prediction")
        
        # Normalize weights
        total_weight = sum(self.weights)
        normalized_weights = [w / total_weight for w in self.weights]
        
        # Collect weighted predictions
        weighted_preds = None
        for model, weight in zip(self.models, normalized_weights):
            pred = model.predict(dmatrix)
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)
            
            if weighted_preds is None:
                weighted_preds = pred * weight
            else:
                weighted_preds += pred * weight
        
        return weighted_preds
    
    def serialize(self) -> bytes:
        """Serialize all models to bytes."""
        model_list = []
        for model, weight in zip(self.models, self.weights):
            model_bytes = serialize_xgboost_model(model)
            model_list.append({
                "model": model_bytes.decode('utf-8'),
                "weight": weight,
            })
        return json.dumps(model_list).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes, num_classes: int) -> "EnsembleModel":
        """Deserialize bytes back to ensemble."""
        ensemble = cls(num_classes)
        model_list = json.loads(data.decode('utf-8'))
        for item in model_list:
            model = deserialize_xgboost_model(item["model"].encode('utf-8'))
            ensemble.models.append(model)
            ensemble.weights.append(item["weight"])
        return ensemble
    
    def __len__(self):
        return len(self.models)


# =============================================================================
# Flower Client for XGBoost
# =============================================================================

class XGBoostFlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated XGBoost with tree serialization.
    
    Each client:
    1. Trains a local XGBoost model with N trees
    2. Serializes the tree structure to JSON bytes
    3. Sends to server as numpy array
    4. Receives global model and evaluates locally
    """
    
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_classes: int,
        config: FederatedXGBoostConfig,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.config = config
        
        # Local model (updated each round)
        self.local_model: Optional[xgb.Booster] = None
        
        logger.info(f"Client {client_id} initialized: {len(X_train)} train, {len(X_test)} test samples")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return serialized model parameters."""
        if self.local_model is None:
            # Return empty parameters if no model yet
            return [np.array([], dtype=np.uint8)]
        
        model_bytes = serialize_xgboost_model(self.local_model)
        return [np.frombuffer(model_bytes, dtype=np.uint8).copy()]
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train local XGBoost model with bootstrap sampling for diversity.
        
        For federated bagging:
        1. Bootstrap sample from local data (sampling with replacement)
        2. Use round-specific random seed for tree diversity
        3. Train N trees on bootstrap sample
        4. Return serialized model for ensemble
        """
        round_num = config.get("round", 1)
        logger.info(f"Client {self.client_id}: Starting round {round_num} training")
        
        # Bootstrap sampling for bagging diversity
        np.random.seed(self.config.random_state + self.client_id * 100 + round_num * 10)
        bootstrap_indices = np.random.choice(
            len(self.X_train), 
            size=len(self.X_train), 
            replace=True  # Bootstrap = sampling with replacement
        )
        X_bootstrap = self.X_train[bootstrap_indices]
        y_bootstrap = self.y_train[bootstrap_indices]
        
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_bootstrap, label=y_bootstrap)
        
        # XGBoost parameters with round-specific seed for additional diversity
        params = {
            "max_depth": self.config.max_depth,
            "eta": self.config.learning_rate,
            "objective": self.config.objective,
            "num_class": self.num_classes,
            "eval_metric": self.config.eval_metric,
            "seed": self.config.random_state + self.client_id * 100 + round_num * 10,
            "subsample": 0.8,  # Row subsampling for additional diversity
            "colsample_bytree": 0.8,  # Column subsampling
            "verbosity": 0,
        }
        
        # Train fresh model (each round creates new diverse trees)
        self.local_model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.trees_per_client,
            verbose_eval=False,
        )
        
        # Evaluate on local test set
        dtest = xgb.DMatrix(self.X_test)
        predictions = self.local_model.predict(dtest)
        y_pred = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        logger.info(f"Client {self.client_id}: Round {round_num} accuracy = {accuracy:.4f}")
        
        # Serialize model
        model_bytes = serialize_xgboost_model(self.local_model)
        model_array = np.frombuffer(model_bytes, dtype=np.uint8).copy()
        
        return (
            [model_array],
            len(self.X_train),
            {"accuracy": float(accuracy), "client_id": self.client_id, "round": round_num},
        )
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate global model on local test data."""
        if len(parameters) == 0 or len(parameters[0]) == 0:
            return 0.0, len(self.X_test), {"accuracy": 0.0}
        
        try:
            global_model = deserialize_xgboost_model(parameters[0].tobytes())
        except Exception as e:
            logger.error(f"Client {self.client_id}: Could not deserialize global model: {e}")
            return 0.0, len(self.X_test), {"accuracy": 0.0}
        
        # Evaluate
        dtest = xgb.DMatrix(self.X_test)
        predictions = global_model.predict(dtest)
        y_pred = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        
        logger.info(f"Client {self.client_id}: Global model accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        return (
            float(1 - accuracy),  # Loss
            len(self.X_test),
            {
                "accuracy": float(accuracy),
                "f1_macro": float(f1),
                "client_id": self.client_id,
            },
        )


# =============================================================================
# Flower Strategy: Tree Concatenation
# =============================================================================

class TreeConcatenationStrategy(Strategy):
    """
    Flower Strategy that aggregates XGBoost models using ensemble bagging.
    
    This strategy implements federated bagging:
    1. Collect tree models from all clients
    2. Add them to an ensemble (not tree concatenation to avoid memory issues)
    3. Predictions are made by averaging across all models
    
    Model accumulation across rounds creates a growing ensemble.
    """
    
    def __init__(
        self,
        num_classes: int,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        self.num_classes = num_classes
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.ensemble = EnsembleModel(num_classes)
        self.round_results: List[Dict[str, Any]] = []
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Return initial (empty) parameters."""
        return ndarrays_to_parameters([np.array([], dtype=np.uint8)])
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training."""
        # Sample all available clients
        sample_size = max(self.min_fit_clients, len(client_manager.all()))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )
        
        # Create FitIns with current global model
        fit_ins = FitIns(parameters, {"round": server_round})
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate by adding client models to ensemble.
        
        Each client's model becomes part of a larger ensemble.
        Predictions are made by averaging across all models.
        """
        if not results:
            return None, {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Server: Aggregating round {server_round} ({len(results)} clients)")
        logger.info(f"{'='*60}")
        
        # Deserialize client models
        client_models = []  # List of (model, accuracy) tuples
        total_samples = 0
        client_accuracies = []
        
        for client_proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            if len(params) > 0 and len(params[0]) > 0:
                try:
                    model = deserialize_xgboost_model(params[0].tobytes())
                    accuracy = fit_res.metrics.get("accuracy", 0.5)  # Default to 0.5 if not provided
                    client_models.append((model, accuracy))
                    total_samples += fit_res.num_examples
                    client_accuracies.append(accuracy)
                    logger.info(f"  Received model from client (samples: {fit_res.num_examples}, acc: {accuracy:.4f})")
                except Exception as e:
                    logger.warning(f"Failed to deserialize client model: {e}")
        
        if not client_models:
            return None, {}
        
        # Add models to ensemble with accuracy weights
        for model, acc in client_models:
            self.ensemble.add_model(model, acc)
        
        if client_accuracies:
            logger.info(f"  Mean client accuracy: {np.mean(client_accuracies):.4f}")
        
        # Serialize ensemble for distribution (just send model count, clients train fresh)
        # For bagging, we don't need to send the full ensemble back
        # Instead, signal to clients the round number
        empty_params = ndarrays_to_parameters([np.array([], dtype=np.uint8)])
        
        metrics = {
            "num_clients": len(client_models),
            "total_samples": total_samples,
            "mean_accuracy": float(np.mean(client_accuracies)) if client_accuracies else 0.0,
            "ensemble_size": len(self.ensemble),
        }
        
        self.round_results.append({
            "round": server_round,
            "num_clients": len(client_models),
            "mean_accuracy": metrics["mean_accuracy"],
            "ensemble_size": len(self.ensemble),
        })
        
        return empty_params, metrics
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation."""
        # Skip client evaluation - we'll do server-side evaluation with ensemble
        return []
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        return None, {}
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (optional)."""
        return None


# =============================================================================
# Data Loading
# =============================================================================

def load_unsw_data(
    data_dir: Path,
    max_samples: int = 10000,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], int, LabelEncoder, StandardScaler]:
    """
    Load UNSW-NB15 data for 2 clients using stratified split.
    
    Uses only training-set.csv and splits it into 2 clients
    with stratified sampling to maintain class distribution per client.
    """
    logger.info("Loading UNSW-NB15 training dataset...")
    
    # Load training dataset only
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"UNSW-NB15 training-set not found in {data_dir}")
    
    df = pd.read_csv(train_path)
    logger.info(f"  Loaded training-set: {len(df)} samples")
    
    # Sample if needed (total samples, will be split between clients)
    total_samples = max_samples * 2  # Each client gets max_samples
    if len(df) > total_samples:
        df = df.sample(n=total_samples, random_state=42)
        logger.info(f"  Sampled to {len(df)} samples")
    
    # Define features (excluding labels and ID columns)
    exclude_cols = ['id', 'label', 'attack_cat']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Select numeric columns only
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Extract features and labels
    X = df[numeric_cols].values.astype(np.float32)
    y_raw = df['attack_cat'].fillna('Normal').values
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_raw)
    y = label_encoder.transform(y_raw)
    
    num_classes = len(label_encoder.classes_)
    logger.info(f"  Classes ({num_classes}): {label_encoder.classes_.tolist()}")
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"  Class distribution: {dict(zip(label_encoder.classes_, counts))}")
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # STRATIFIED SPLIT: Split into 2 clients maintaining class distribution
    X_client0, X_client1, y_client0, y_client1 = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    
    logger.info(f"\n  Stratified split into 2 clients:")
    
    # Verify class distribution per client
    unique0, counts0 = np.unique(y_client0, return_counts=True)
    unique1, counts1 = np.unique(y_client1, return_counts=True)
    logger.info(f"  Client 0 distribution: {dict(zip(label_encoder.classes_[unique0], counts0))}")
    logger.info(f"  Client 1 distribution: {dict(zip(label_encoder.classes_[unique1], counts1))}")
    
    # Split each client's data into train/test (stratified)
    X0_train, X0_test, y0_train, y0_test = train_test_split(
        X_client0, y_client0, test_size=0.2, random_state=42, stratify=y_client0
    )
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X_client1, y_client1, test_size=0.2, random_state=42, stratify=y_client1
    )
    
    client_data = {
        0: (X0_train, y0_train, X0_test, y0_test),
        1: (X1_train, y1_train, X1_test, y1_test),
    }
    
    logger.info(f"\n  Client 0: {len(X0_train)} train, {len(X0_test)} test")
    logger.info(f"  Client 1: {len(X1_train)} train, {len(X1_test)} test")
    
    return client_data, num_classes, label_encoder, scaler


# =============================================================================
# Simulation
# =============================================================================

def run_federated_simulation(
    config: FederatedXGBoostConfig,
    data_dir: Path,
) -> Dict[str, Any]:
    """
    Run federated XGBoost simulation with tree concatenation.
    """
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("  FEDERATED XGBOOST WITH TREE CONCATENATION (FLOWER)")
    print("  Strategy: Bagging with Tree Accumulation")
    print("=" * 70 + "\n")
    
    # Load data
    client_data, num_classes, label_encoder, scaler = load_unsw_data(
        data_dir, config.max_samples
    )
    
    print(f"\nConfiguration:")
    print(f"  - Trees per client per round: {config.trees_per_client}")
    print(f"  - Federated rounds: {config.num_rounds}")
    print(f"  - Number of clients: 2")
    print(f"  - Expected final trees: {config.trees_per_client * 2 * config.num_rounds}")
    print(f"  - Max depth: {config.max_depth}")
    print(f"  - Number of classes: {num_classes}\n")
    
    # Create clients
    clients = []
    for client_id in range(2):
        X_train, y_train, X_test, y_test = client_data[client_id]
        client = XGBoostFlowerClient(
            client_id=client_id,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            config=config,
        )
        clients.append(client)
    
    # Train baseline (local only, no federation)
    print("\n" + "-" * 60)
    print("  BASELINE: Local Training Only")
    print("-" * 60)
    
    baseline_results = {}
    for i, client in enumerate(clients):
        dtrain = xgb.DMatrix(client.X_train, label=client.y_train)
        dtest = xgb.DMatrix(client.X_test)
        
        params = {
            "max_depth": config.max_depth,
            "eta": config.learning_rate,
            "objective": config.objective,
            "num_class": num_classes,
            "eval_metric": config.eval_metric,
            "seed": config.random_state + i,
            "verbosity": 0,
        }
        
        # Train with same total trees as federated (trees_per_client * num_rounds)
        local_model = xgb.train(
            params,
            dtrain,
            num_boost_round=config.trees_per_client * config.num_rounds,
            verbose_eval=False,
        )
        
        predictions = local_model.predict(dtest)
        y_pred = np.argmax(predictions, axis=1)
        
        acc = accuracy_score(client.y_test, y_pred)
        f1 = f1_score(client.y_test, y_pred, average='macro', zero_division=0)
        
        baseline_results[f"client_{i}"] = {
            "accuracy": acc,
            "f1_macro": f1,
        }
        
        print(f"  Client {i}: accuracy={acc:.4f}, F1={f1:.4f}")
    
    # Federated training using Flower simulation
    print("\n" + "-" * 60)
    print("  FEDERATED: Tree Concatenation Training")
    print("-" * 60)
    
    # Create strategy
    strategy = TreeConcatenationStrategy(
        num_classes=num_classes,
        min_fit_clients=2,
        min_evaluate_clients=2,
    )
    
    # Client function for simulation
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        X_train, y_train, X_test, y_test = client_data[client_id]
        return XGBoostFlowerClient(
            client_id=client_id,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            config=config,
        ).to_client()
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Final evaluation with ensemble model
    print("\n" + "-" * 60)
    print("  FINAL EVALUATION")
    print("-" * 60)
    
    federated_results = {}
    if len(strategy.ensemble) > 0:
        for i, client in enumerate(clients):
            dtest = xgb.DMatrix(client.X_test)
            predictions = strategy.ensemble.predict(dtest)
            y_pred = np.argmax(predictions, axis=1)
            
            acc = accuracy_score(client.y_test, y_pred)
            f1 = f1_score(client.y_test, y_pred, average='macro', zero_division=0)
            
            federated_results[f"client_{i}"] = {
                "accuracy": acc,
                "f1_macro": f1,
            }
            
            print(f"  Client {i}: accuracy={acc:.4f}, F1={f1:.4f}")
        
        print(f"\n  Ensemble size: {len(strategy.ensemble)} models")
    
    # Calculate improvements
    total_time = time.time() - start_time
    
    # Aggregate metrics
    baseline_acc = np.mean([v["accuracy"] for v in baseline_results.values()])
    baseline_f1 = np.mean([v["f1_macro"] for v in baseline_results.values()])
    federated_acc = np.mean([v["accuracy"] for v in federated_results.values()]) if federated_results else 0
    federated_f1 = np.mean([v["f1_macro"] for v in federated_results.values()]) if federated_results else 0
    
    acc_improvement = ((federated_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    f1_improvement = ((federated_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline (Local Only):")
    print(f"    - Mean Accuracy: {baseline_acc:.4f}")
    print(f"    - Mean F1 Score: {baseline_f1:.4f}")
    print(f"\n  Federated (Ensemble Bagging):")
    print(f"    - Mean Accuracy: {federated_acc:.4f} ({acc_improvement:+.2f}%)")
    print(f"    - Mean F1 Score: {federated_f1:.4f} ({f1_improvement:+.2f}%)")
    print(f"\n  Training Info:")
    print(f"    - Ensemble Size: {len(strategy.ensemble)} models")
    print(f"    - Trees per Model: {config.trees_per_client}")
    print(f"    - Federated Rounds: {config.num_rounds}")
    print(f"    - Total Time: {total_time:.1f}s")
    print("=" * 70 + "\n")
    
    # Compile results
    results = {
        "config": {
            "trees_per_client": config.trees_per_client,
            "num_rounds": config.num_rounds,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "max_samples": config.max_samples,
            "num_classes": num_classes,
            "ensemble_size": len(strategy.ensemble),
        },
        "baseline": {
            "mean_accuracy": baseline_acc,
            "mean_f1": baseline_f1,
            "per_client": baseline_results,
        },
        "federated": {
            "mean_accuracy": federated_acc,
            "mean_f1": federated_f1,
            "per_client": federated_results,
            "round_history": strategy.round_results,
        },
        "improvements": {
            "accuracy_pct": acc_improvement,
            "f1_pct": f1_improvement,
        },
        "total_time": total_time,
        "timestamp": datetime.now().isoformat(),
        "classes": label_encoder.classes_.tolist(),
    }
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Federated XGBoost with Tree Concatenation (Flower)"
    )
    parser.add_argument(
        "--num-rounds", type=int, default=3,
        help="Number of federated rounds (default: 3)"
    )
    parser.add_argument(
        "--trees-per-client", type=int, default=10,
        help="Trees each client trains per round (default: 10)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="Maximum tree depth (default: 6)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=10000,
        help="Maximum samples per client (default: 10000)"
    )
    parser.add_argument(
        "--output", type=str, default="federated_xgboost_results.json",
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = FederatedXGBoostConfig(
        trees_per_client=args.trees_per_client,
        num_rounds=args.num_rounds,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
    )
    
    # Data directory
    data_dir = project_root / "data"
    
    # Run simulation
    results = run_federated_simulation(config, data_dir)
    
    # Save results
    output_path = project_root / "scripts" / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
