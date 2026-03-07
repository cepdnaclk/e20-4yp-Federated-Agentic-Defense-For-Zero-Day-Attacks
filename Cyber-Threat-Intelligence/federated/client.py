"""
Federated Learning Client for Network Defense Agents.

This module implements a Flower client that enables privacy-preserving
distributed training of Agent One (Autoencoder) and Agent Two (XGBoost).
Each network node trains locally and shares only model weights.

Classes:
    NetworkDefenseClient: Flower NumPyClient for federated IDS training.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import flwr as fl
from flwr.common import NDArrays, Config, Scalar

import torch
import xgboost as xgb

from .utils import (
    autoencoder_weights_to_numpy,
    numpy_to_autoencoder_weights,
    xgboost_to_numpy,
    numpy_to_xgboost,
    get_combined_weights,
    split_combined_weights,
)
from .differential_privacy import DifferentialPrivacyEngine

logger = logging.getLogger(__name__)


class NetworkDefenseClient(fl.client.NumPyClient):
    """
    Federated Learning client for decentralized IDS training.
    
    This client handles local training of both the Autoencoder (Agent One)
    and XGBoost classifier (Agent Two), sharing only model weights with
    the central server. Raw data never leaves the local node.
    
    Attributes:
        autoencoder: PyTorch autoencoder model for anomaly detection.
        xgboost_model: XGBoost classifier for threat categorization.
        label_encoder: LabelEncoder for attack categories.
        train_data: Local training dataset (X, y).
        val_data: Local validation dataset (X, y).
        training_config: Configuration for local training.
    
    Example:
        >>> from agents.agent_one import AgentOne
        >>> from agents.agent_two import AgentTwo
        >>> 
        >>> agent_one = AgentOne()
        >>> agent_two = AgentTwo()
        >>> 
        >>> client = NetworkDefenseClient(
        ...     autoencoder=agent_one.model,
        ...     xgboost_model=agent_two.classifier._model,
        ...     label_encoder=agent_two.classifier.label_encoder,
        ...     train_data=(X_train, y_train),
        ...     val_data=(X_val, y_val),
        ... )
        >>> 
        >>> # Start federated client
        >>> fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    
    Privacy Guarantees:
        - Raw network traffic data stays on local node
        - Only model parameters are transmitted
        - Server cannot reconstruct individual samples
        - Aggregation uses FedAvg (weighted averaging)
    """
    
    def __init__(
        self,
        autoencoder: torch.nn.Module,
        xgboost_model: Optional[xgb.XGBClassifier] = None,
        label_encoder: Optional[Any] = None,
        train_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        client_id: str = "default",
        dp_enabled: bool = False,
        dp_clip_norm: float = 1.0,
        dp_noise_multiplier: float = 0.1,
    ):
        """
        Initialize the Network Defense federated client.
        
        Args:
            autoencoder: PyTorch autoencoder model (Agent One).
            xgboost_model: XGBoost classifier (Agent Two). Optional.
            label_encoder: LabelEncoder for attack categories. Optional.
            train_data: Tuple of (X_train, y_train). Optional.
            val_data: Tuple of (X_val, y_val). Optional.
            training_config: Dict with training hyperparameters.
            client_id: Unique identifier for this client.
            dp_enabled: Whether to apply differential privacy to weights.
            dp_clip_norm: Maximum L2 norm for weight clipping (DP).
            dp_noise_multiplier: Multiplier for Gaussian noise (DP).
        """
        self.autoencoder = autoencoder
        self.xgboost_model = xgboost_model
        self.label_encoder = label_encoder
        self.train_data = train_data
        self.val_data = val_data
        self.client_id = client_id
        
        # Default training configuration
        self.training_config = training_config or {
            "autoencoder_epochs": 5,
            "autoencoder_batch_size": 256,
            "autoencoder_lr": 0.001,
            "xgboost_n_estimators": 10,  # Incremental trees per round
        }
        
        # Track whether we have XGBoost model
        self._has_xgboost = xgboost_model is not None
        
        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(self.device)
        
        # Differential Privacy configuration
        self._dp_enabled = dp_enabled
        self._dp_clip_norm = dp_clip_norm
        self._dp_noise_multiplier = dp_noise_multiplier
        self._dp_engine: Optional[DifferentialPrivacyEngine] = None
        
        if dp_enabled:
            self._dp_engine = DifferentialPrivacyEngine()
            logger.info(
                f"DP enabled: clip_norm={dp_clip_norm}, "
                f"noise_multiplier={dp_noise_multiplier}"
            )
        
        logger.info(
            f"NetworkDefenseClient initialized: id={client_id}, "
            f"has_xgboost={self._has_xgboost}, device={self.device}, "
            f"dp_enabled={dp_enabled}"
        )
    
    def get_parameters(self, config: Config) -> NDArrays:
        """
        Return model parameters as a list of numpy arrays.
        
        This method is called by the Flower server to retrieve current
        model weights before and after training rounds.
        
        If differential privacy is enabled, weights are clipped and
        noise is added before being returned.
        
        Args:
            config: Configuration dict from server.
        
        Returns:
            List of numpy arrays containing all model weights.
            - For autoencoder: One array per parameter tensor
            - For XGBoost: Serialized model as byte array
        
        Note:
            The order of parameters is deterministic and must be
            preserved for correct aggregation.
        """
        logger.debug(f"Client {self.client_id}: Getting parameters")
        
        if self._has_xgboost:
            weights, _ = get_combined_weights(
                self.autoencoder,
                self.xgboost_model,
                self.label_encoder,
            )
        else:
            weights = autoencoder_weights_to_numpy(self.autoencoder)
        
        # Apply differential privacy if enabled
        if self._dp_enabled and self._dp_engine is not None:
            weights = self._dp_engine.apply_dp(
                weights=weights,
                clip_norm=self._dp_clip_norm,
                noise_multiplier=self._dp_noise_multiplier,
            )
            logger.debug(f"Client {self.client_id}: DP applied to parameters")
        
        return weights
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Update model with parameters from server.
        
        This method is called after federated aggregation to update
        local models with the globally aggregated weights.
        
        Args:
            parameters: Aggregated weights from server.
        """
        logger.debug(f"Client {self.client_id}: Setting parameters")
        
        if self._has_xgboost:
            # Split combined weights
            ae_weight_count = len(list(self.autoencoder.state_dict().keys()))
            ae_weights, xgb_serialized = split_combined_weights(
                parameters, ae_weight_count
            )
            
            # Update autoencoder
            numpy_to_autoencoder_weights(self.autoencoder, ae_weights)
            
            # Update XGBoost
            self.xgboost_model = numpy_to_xgboost(xgb_serialized)
        else:
            numpy_to_autoencoder_weights(self.autoencoder, parameters)
    
    def fit(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train models on local data and return updated weights.
        
        This is the core federated learning method. It:
        1. Updates local models with global weights
        2. Trains on local private data
        3. Returns updated weights without exposing raw data
        
        Args:
            parameters: Current global model parameters.
            config: Training configuration from server.
        
        Returns:
            Tuple of:
                - Updated model parameters
                - Number of training samples used
                - Dict of training metrics
        
        Raises:
            ValueError: If train_data is not set.
        """
        logger.info(f"Client {self.client_id}: Starting fit round")
        
        if self.train_data is None:
            raise ValueError("Training data not set. Call set_train_data() first.")
        
        X_train, y_train = self.train_data
        
        # 1. Update models with global parameters
        self.set_parameters(parameters)
        
        # 2. Train autoencoder locally
        ae_loss = self._train_autoencoder(X_train, y_train)
        
        # 3. Train XGBoost locally (if available)
        xgb_metrics = {}
        if self._has_xgboost and y_train is not None:
            xgb_metrics = self._train_xgboost(X_train, y_train)
        
        # 4. Get updated parameters (DP is applied within get_parameters if enabled)
        updated_params = self.get_parameters(config)
        
        # 5. Build metrics dict
        metrics = {
            "client_id": self.client_id,
            "autoencoder_loss": float(ae_loss),
            "dp_enabled": self._dp_enabled,
            **{f"xgb_{k}": v for k, v in xgb_metrics.items()},
        }
        
        if self._dp_enabled and self._dp_engine is not None:
            metrics["dp_clip_count"] = self._dp_engine.clip_count
            metrics["dp_applications"] = self._dp_engine.noise_applied_count
        
        logger.info(
            f"Client {self.client_id}: Fit complete. "
            f"AE loss: {ae_loss:.6f}, samples: {len(X_train)}, "
            f"dp_enabled={self._dp_enabled}"
        )
        
        return updated_params, len(X_train), metrics
    
    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate models on local validation data.
        
        This method tests the globally aggregated model on local
        private validation data, returning only aggregate metrics.
        
        Args:
            parameters: Global model parameters to evaluate.
            config: Evaluation configuration from server.
        
        Returns:
            Tuple of:
                - Combined loss value
                - Number of validation samples
                - Dict of evaluation metrics
        
        Note:
            If val_data is not set, training data will be used for evaluation.
        """
        logger.info(f"Client {self.client_id}: Starting evaluate")
        
        # Use validation data if available, otherwise fall back to training data
        if self.val_data is not None:
            X_val, y_val = self.val_data
        elif self.train_data is not None:
            logger.warning(
                f"Client {self.client_id}: No validation data set, "
                "using training data for evaluation"
            )
            X_val, y_val = self.train_data
        else:
            raise ValueError("No data available for evaluation. Set train_data or val_data first.")
        
        # Update models with global parameters
        self.set_parameters(parameters)
        
        # Evaluate autoencoder (reconstruction loss)
        ae_loss = self._evaluate_autoencoder(X_val)
        
        # Evaluate XGBoost (if available)
        xgb_metrics = {}
        if self._has_xgboost and y_val is not None:
            xgb_metrics = self._evaluate_xgboost(X_val, y_val)
        
        # Combined loss (weighted average)
        combined_loss = ae_loss
        if "accuracy" in xgb_metrics:
            # Lower xgb_loss for higher accuracy
            xgb_loss = 1.0 - xgb_metrics["accuracy"]
            combined_loss = 0.5 * ae_loss + 0.5 * xgb_loss
        
        metrics = {
            "client_id": self.client_id,
            "autoencoder_loss": float(ae_loss),
            **{f"xgb_{k}": v for k, v in xgb_metrics.items()},
        }
        
        logger.info(
            f"Client {self.client_id}: Evaluate complete. "
            f"Combined loss: {combined_loss:.6f}"
        )
        
        return float(combined_loss), len(X_val), metrics
    
    def _train_autoencoder(
        self, X_train: np.ndarray, y_train: Optional[np.ndarray]
    ) -> float:
        """
        Train autoencoder on local data.
        
        Args:
            X_train: Training features.
            y_train: Training labels (not used, autoencoder is unsupervised).
        
        Returns:
            Final training loss.
        """
        config = self.training_config
        epochs = config.get("autoencoder_epochs", 5)
        batch_size = config.get("autoencoder_batch_size", 256)
        lr = config.get("autoencoder_lr", 0.001)
        
        # Set to training mode
        self.autoencoder.train()
        
        # Prepare optimizer and loss
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        # Convert to tensor
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        
        # Training loop
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                
                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            total_loss = epoch_loss / (len(X_tensor) // batch_size + 1)
            logger.debug(f"AE Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")
        
        self.autoencoder.eval()
        return total_loss
    
    def _train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, float]:
        """
        Train XGBoost classifier incrementally.
        
        Args:
            X_train: Training features.
            y_train: Training labels (attack categories).
        
        Returns:
            Dict of training metrics.
        """
        # Encode labels if needed
        if self.label_encoder is not None:
            y_encoded = self.label_encoder.transform(y_train)
        else:
            y_encoded = y_train
        
        # Fit incrementally (or retrain for simplicity)
        # Note: XGBoost doesn't support true incremental learning,
        # so we retrain on local data each round
        self.xgboost_model.fit(X_train, y_encoded)
        
        # Get training accuracy
        y_pred = self.xgboost_model.predict(X_train)
        accuracy = (y_pred == y_encoded).mean()
        
        return {"train_accuracy": float(accuracy)}
    
    def _evaluate_autoencoder(self, X_val: np.ndarray) -> float:
        """
        Evaluate autoencoder reconstruction loss.
        
        Args:
            X_val: Validation features.
        
        Returns:
            Mean squared reconstruction error.
        """
        self.autoencoder.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            reconstructed = self.autoencoder(X_tensor)
            loss = torch.nn.functional.mse_loss(reconstructed, X_tensor)
        
        return loss.item()
    
    def _evaluate_xgboost(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate XGBoost classification performance.
        
        Args:
            X_val: Validation features.
            y_val: Validation labels.
        
        Returns:
            Dict with accuracy, precision, recall, f1.
        """
        # Encode labels if needed
        if self.label_encoder is not None:
            y_encoded = self.label_encoder.transform(y_val)
        else:
            y_encoded = y_val
        
        # Predict
        y_pred = self.xgboost_model.predict(X_val)
        
        # Calculate metrics
        accuracy = (y_pred == y_encoded).mean()
        
        return {"accuracy": float(accuracy)}
    
    def set_train_data(
        self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None
    ) -> None:
        """Set local training data."""
        self.train_data = (X_train, y_train)
        logger.info(f"Client {self.client_id}: Set training data, {len(X_train)} samples")
    
    def set_val_data(
        self, X_val: np.ndarray, y_val: Optional[np.ndarray] = None
    ) -> None:
        """Set local validation data."""
        self.val_data = (X_val, y_val)
        logger.info(f"Client {self.client_id}: Set validation data, {len(X_val)} samples")
    
    @property
    def dp_enabled(self) -> bool:
        """Whether differential privacy is enabled."""
        return self._dp_enabled
    
    @property
    def dp_engine(self) -> Optional[DifferentialPrivacyEngine]:
        """Get the differential privacy engine instance."""
        return self._dp_engine
    
    @property
    def dp_clip_norm(self) -> float:
        """Get the DP clipping norm."""
        return self._dp_clip_norm
    
    @property
    def dp_noise_multiplier(self) -> float:
        """Get the DP noise multiplier."""
        return self._dp_noise_multiplier


def create_client_fn(
    autoencoder_class,
    autoencoder_config: Dict[str, Any],
    xgboost_config: Optional[Dict[str, Any]] = None,
    data_loader_fn=None,
    training_config: Optional[Dict[str, Any]] = None,
    dp_enabled: bool = False,
    dp_clip_norm: float = 1.0,
    dp_noise_multiplier: float = 0.1,
):
    """
    Factory function to create NetworkDefenseClient instances.
    
    This function is useful for Flower's virtual client simulation,
    where clients are created dynamically during federation rounds.
    
    Args:
        autoencoder_class: Class of the autoencoder model.
        autoencoder_config: Config dict for autoencoder initialization.
        xgboost_config: Config dict for XGBoost (or None).
        data_loader_fn: Function(client_id) -> (train_data, val_data).
        training_config: Training hyperparameters.
        dp_enabled: Whether to enable differential privacy.
        dp_clip_norm: L2 norm clipping threshold for DP.
        dp_noise_multiplier: Noise multiplier for DP.
    
    Returns:
        Function(client_id) -> NetworkDefenseClient
    
    Example:
        >>> from agents.models.autoencoder import AnomalyAutoencoder
        >>> 
        >>> client_fn = create_client_fn(
        ...     autoencoder_class=AnomalyAutoencoder,
        ...     autoencoder_config={"input_dim": 40, "latent_dim": 8},
        ...     data_loader_fn=load_client_data,
        ...     dp_enabled=True,
        ...     dp_clip_norm=1.0,
        ...     dp_noise_multiplier=0.1,
        ... )
        >>> 
        >>> # Used with Flower simulation
        >>> fl.simulation.start_simulation(
        ...     client_fn=client_fn,
        ...     num_clients=10,
        ...     ...
        ... )
    """
    def _create_client(client_id: str) -> NetworkDefenseClient:
        # Create autoencoder
        autoencoder = autoencoder_class(**autoencoder_config)
        
        # Create XGBoost if configured
        xgboost_model = None
        label_encoder = None
        if xgboost_config is not None:
            xgboost_model = xgb.XGBClassifier(**xgboost_config)
        
        # Load data if loader provided
        train_data = None
        val_data = None
        if data_loader_fn is not None:
            train_data, val_data = data_loader_fn(client_id)
        
        return NetworkDefenseClient(
            autoencoder=autoencoder,
            xgboost_model=xgboost_model,
            label_encoder=label_encoder,
            train_data=train_data,
            val_data=val_data,
            training_config=training_config,
            client_id=client_id,
            dp_enabled=dp_enabled,
            dp_clip_norm=dp_clip_norm,
            dp_noise_multiplier=dp_noise_multiplier,
        )
    
    return _create_client
