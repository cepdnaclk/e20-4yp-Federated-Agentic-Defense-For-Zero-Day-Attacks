"""
Autoencoder model for anomaly detection in network traffic.

This module defines a PyTorch autoencoder architecture optimized for
tabular network data (UNSW-NB15 dataset). The model learns to reconstruct
normal network traffic patterns, and anomalies are detected based on
high reconstruction errors.

Architecture Design:
    - Symmetric encoder-decoder with bottleneck
    - Batch normalization for training stability
    - Dropout for regularization
    - LeakyReLU activation to prevent dead neurons
    - Configurable layer dimensions for different data scales
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class AnomalyAutoencoder(nn.Module):
    """
    Autoencoder neural network for anomaly detection in network traffic.
    
    This autoencoder is designed for tabular network flow data. It learns
    a compressed representation of normal traffic patterns. During inference,
    anomalous traffic produces higher reconstruction errors since the model
    hasn't learned their patterns.
    
    Architecture:
        Encoder: input_dim -> hidden_dims[0] -> ... -> latent_dim
        Decoder: latent_dim -> hidden_dims[-1] -> ... -> input_dim
    
    Attributes:
        input_dim: Dimension of input features.
        latent_dim: Dimension of the bottleneck layer.
        hidden_dims: Dimensions of hidden layers.
        dropout_rate: Dropout probability for regularization.
        encoder: Encoder network (nn.Sequential).
        decoder: Decoder network (nn.Sequential).
    
    Example:
        >>> # Create model for UNSW-NB15 (40 features after preprocessing)
        >>> model = AnomalyAutoencoder(
        ...     input_dim=40,
        ...     latent_dim=8,
        ...     hidden_dims=[32, 16],
        ...     dropout_rate=0.2,
        ... )
        >>> 
        >>> # Forward pass
        >>> x = torch.randn(64, 40)  # Batch of 64 samples
        >>> reconstructed = model(x)
        >>> 
        >>> # Get reconstruction error
        >>> error = model.reconstruction_error(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        """
        Initializes the Autoencoder model.
        
        Args:
            input_dim: Number of input features.
            latent_dim: Dimension of the latent/bottleneck layer.
                       Smaller values force more compression.
            hidden_dims: List of hidden layer dimensions for encoder.
                        Decoder uses reversed order. Default: [32, 16].
            dropout_rate: Dropout probability (0.0 to 1.0).
            use_batch_norm: Whether to use batch normalization.
        
        Raises:
            ValueError: If input_dim < 1 or latent_dim < 1.
        """
        super().__init__()
        
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 16]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            "AnomalyAutoencoder created: input=%d, latent=%d, hidden=%s",
            input_dim, latent_dim, self.hidden_dims
        )
    
    def _build_encoder(self) -> nn.Sequential:
        """
        Builds the encoder network.
        
        The encoder compresses input features into a low-dimensional
        latent representation through progressively smaller layers.
        
        Returns:
            nn.Sequential encoder network.
        """
        layers: List[nn.Module] = []
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Bottleneck layer (no activation - linear projection to latent space)
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """
        Builds the decoder network.
        
        The decoder reconstructs input from the latent representation,
        mirroring the encoder architecture in reverse.
        
        Returns:
            nn.Sequential decoder network.
        """
        layers: List[nn.Module] = []
        
        # Start from latent dimension
        prev_dim = self.latent_dim
        
        # Hidden layers (reversed order)
        for hidden_dim in reversed(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (linear - reconstructing normalized features)
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        Initializes network weights using Xavier/Glorot initialization.
        
        This initialization helps maintain gradient flow and is well-suited
        for networks with tanh or sigmoid activations, and works reasonably
        well with ReLU variants.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        
        Example:
            >>> x = torch.randn(32, 40)
            >>> reconstructed = model(x)
            >>> assert reconstructed.shape == x.shape
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Latent tensor of shape (batch_size, latent_dim).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim).
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        return self.decoder(z)
    
    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        Computes reconstruction error (MSE) for input samples.
        
        Higher reconstruction error indicates potential anomaly,
        as the model struggles to reconstruct patterns it hasn't
        learned during training on normal data.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            reduction: Error reduction method:
                      'none': Returns per-sample errors (batch_size,)
                      'mean': Returns scalar mean error
                      'sum': Returns scalar sum of errors
        
        Returns:
            Reconstruction error tensor.
        
        Example:
            >>> errors = model.reconstruction_error(x_test, reduction="none")
            >>> anomalies = errors > threshold
        """
        self.eval()  # Ensure dropout is disabled
        with torch.no_grad():
            reconstructed = self.forward(x)
            
            # Per-sample MSE (mean over features)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            
            if reduction == "mean":
                return mse.mean()
            elif reduction == "sum":
                return mse.sum()
            else:  # "none"
                return mse
    
    def get_config(self) -> Dict[str, Any]:
        """
        Returns model configuration dictionary.
        
        Returns:
            Dictionary with model hyperparameters.
        """
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }
    
    def save(self, filepath: str) -> None:
        """
        Saves model weights and configuration.
        
        Args:
            filepath: Path to save the model (.pth file).
        
        Example:
            >>> model.save("models/autoencoder.pth")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.get_config(),
        }
        
        torch.save(checkpoint, filepath)
        logger.info("Model saved to: %s", filepath)
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "AnomalyAutoencoder":
        """
        Loads model from checkpoint file.
        
        Args:
            filepath: Path to the saved model file.
            device: Device to load model to ('cpu', 'cuda'). 
                   Auto-detects if None.
        
        Returns:
            Loaded AnomalyAutoencoder instance.
        
        Example:
            >>> model = AnomalyAutoencoder.load("models/autoencoder.pth")
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model from saved config
        config = checkpoint["config"]
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        logger.info("Model loaded from: %s (device: %s)", filepath, device)
        return model
    
    def summary(self) -> str:
        """
        Returns a summary string of the model architecture.
        
        Returns:
            Human-readable model summary.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        lines = [
            "=" * 60,
            "AnomalyAutoencoder Summary",
            "=" * 60,
            f"Input dimension:    {self.input_dim}",
            f"Latent dimension:   {self.latent_dim}",
            f"Hidden dimensions:  {self.hidden_dims}",
            f"Dropout rate:       {self.dropout_rate}",
            f"Batch normalization: {self.use_batch_norm}",
            "-" * 60,
            f"Total parameters:   {total_params:,}",
            f"Trainable params:   {trainable_params:,}",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Returns string representation of the model."""
        return (
            f"AnomalyAutoencoder(input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, hidden_dims={self.hidden_dims})"
        )
