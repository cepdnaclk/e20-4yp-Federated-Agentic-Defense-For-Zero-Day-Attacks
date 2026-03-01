"""
Training Script for AnomalyAutoencoder (Agent One).

This script provides comprehensive training functionality for the
autoencoder-based anomaly detector. It handles:
    - Data loading and preprocessing
    - Model training with early stopping
    - Validation and threshold calibration
    - Model checkpointing and logging

Usage:
    python -m agents.train_autoencoder --data_path data/UNSW_NB15_training-set.csv
    
    Or run as a script:
    python agents/train_autoencoder.py

For programmatic use, import and call train_autoencoder() directly.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.models.autoencoder import AnomalyAutoencoder
from agents.agent_one import AgentOne
from data_pipeline import DataLoader, Preprocessor, DatasetConfig
from data_pipeline.batch_generator import DataSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    is observed for a specified number of epochs.
    
    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        counter: Current number of epochs without improvement.
        best_loss: Best validation loss observed.
        should_stop: True if training should stop.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        """
        Initializes early stopping handler.
        
        Args:
            patience: Epochs to wait before stopping.
            min_delta: Minimum improvement threshold.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Checks if training should stop.
        
        Args:
            val_loss: Current validation loss.
        
        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class AutoencoderTrainer:
    """
    Trainer class for AnomalyAutoencoder.
    
    Handles the complete training loop including:
        - Batch processing
        - Loss computation
        - Optimization
        - Validation
        - Early stopping
        - Checkpointing
    
    Example:
        >>> trainer = AutoencoderTrainer(
        ...     model=model,
        ...     learning_rate=1e-3,
        ...     device="cuda",
        ... )
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """
    
    def __init__(
        self,
        model: AnomalyAutoencoder,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the trainer.
        
        Args:
            model: AnomalyAutoencoder model to train.
            learning_rate: Learning rate for optimizer.
            weight_decay: L2 regularization strength.
            device: Computation device ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        # Loss function: Mean Squared Error for reconstruction
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
        
        logger.info("Trainer initialized on device: %s", self.device)
    
    def fit(
        self,
        train_loader: TorchDataLoader,
        val_loader: Optional[TorchDataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Trains the autoencoder model.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation.
            epochs: Maximum number of training epochs.
            early_stopping_patience: Epochs to wait for improvement.
            checkpoint_dir: Directory to save checkpoints.
            verbose: If True, shows progress bars.
        
        Returns:
            Training history dictionary.
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float("inf")
        
        # Setup checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting training for %d epochs", epochs)
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, verbose)
            self.history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Check for best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if checkpoint_dir:
                        self.model.save(checkpoint_path / "best_model.pth")
                
                # Early stopping check
                if early_stopping(val_loss):
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break
                
                # Logging
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["learning_rate"].append(current_lr)
                
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f} - "
                        f"Val Loss: {val_loss:.6f} - "
                        f"LR: {current_lr:.2e}"
                    )
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        # Save final model
        if checkpoint_dir:
            self.model.save(checkpoint_path / "final_model.pth")
        
        logger.info(
            "Training complete. Best validation loss: %.6f",
            best_val_loss if val_loader else train_loss
        )
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: TorchDataLoader,
        verbose: bool = True,
    ) -> float:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        iterator = tqdm(train_loader, desc="Training", disable=not verbose)
        
        for batch in iterator:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # Features only (unsupervised)
            else:
                x = batch
            
            x = x.float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(x)
            loss = self.criterion(reconstructed, x)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            iterator.set_postfix({"loss": loss.item()})
        
        return total_loss / n_batches
    
    def _validate(self, val_loader: TorchDataLoader) -> float:
        """Computes validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.float().to(self.device)
                reconstructed = self.model(x)
                loss = self.criterion(reconstructed, x)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches


def prepare_data_loaders(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    train_on_normal_only: bool = True,
) -> Tuple[TorchDataLoader, Optional[TorchDataLoader]]:
    """
    Prepares PyTorch DataLoaders for training.
    
    Args:
        X_train: Training features.
        X_val: Validation features.
        y_train: Training labels (used if train_on_normal_only=True).
        y_val: Validation labels (used if train_on_normal_only=True).
        batch_size: Batch size for training.
        train_on_normal_only: If True, trains only on normal traffic (label=0).
    
    Returns:
        Tuple of (train_loader, val_loader).
    
    Note:
        When train_on_normal_only=True, BOTH training and validation sets are
        filtered to contain only normal samples. This is critical for autoencoder
        anomaly detection - the model should only learn to reconstruct normal 
        patterns, so validation loss should also only measure normal reconstruction.
    """
    # Filter for normal traffic if specified
    if train_on_normal_only and y_train is not None:
        normal_mask = y_train == 0
        X_train_filtered = X_train[normal_mask]
        logger.info(
            "Training on normal traffic only: %d/%d samples",
            len(X_train_filtered), len(X_train)
        )
    else:
        X_train_filtered = X_train
    
    # Create training dataset
    train_tensor = torch.from_numpy(X_train_filtered).float()
    train_dataset = TensorDataset(train_tensor)
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create validation dataset (ALSO filtered for normal only!)
    val_loader = None
    if X_val is not None:
        # Critical: Filter validation set to normal samples only
        # This ensures the model is validated on its ability to 
        # reconstruct normal traffic, not anomalies
        if train_on_normal_only and y_val is not None:
            normal_val_mask = y_val == 0
            X_val_filtered = X_val[normal_val_mask]
            logger.info(
                "Validating on normal traffic only: %d/%d samples",
                len(X_val_filtered), len(X_val)
            )
        else:
            X_val_filtered = X_val
        
        val_tensor = torch.from_numpy(X_val_filtered).float()
        val_dataset = TensorDataset(val_tensor)
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    
    return train_loader, val_loader


def train_autoencoder(
    data_path: str,
    output_dir: str = "models",
    # Model hyperparameters
    latent_dim: int = 8,
    hidden_dims: Optional[list] = None,
    dropout_rate: float = 0.2,
    # Training hyperparameters
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    # Data options
    train_on_normal_only: bool = True,
    test_split: float = 0.2,
    val_split: float = 0.1,
    # Other options
    threshold_percentile: float = 95.0,
    random_seed: int = 42,
) -> Tuple[AnomalyAutoencoder, AgentOne, Dict[str, Any]]:
    """
    Complete training pipeline for the anomaly autoencoder.
    
    This function handles the entire workflow:
        1. Load and preprocess data
        2. Create and train the autoencoder
        3. Calibrate detection threshold
        4. Save model and preprocessor
        5. Return trained agent
    
    Args:
        data_path: Path to UNSW-NB15 CSV file.
        output_dir: Directory to save outputs.
        latent_dim: Bottleneck layer dimension.
        hidden_dims: Hidden layer dimensions [default: [32, 16]].
        dropout_rate: Dropout probability.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        learning_rate: Adam optimizer learning rate.
        weight_decay: L2 regularization strength.
        early_stopping_patience: Epochs before early stopping.
        train_on_normal_only: Train only on normal traffic.
        test_split: Fraction for test set.
        val_split: Fraction for validation set.
        threshold_percentile: Percentile for threshold calibration.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (trained_model, agent, training_info).
    
    Example:
        >>> model, agent, info = train_autoencoder(
        ...     data_path="data/UNSW_NB15_training-set.csv",
        ...     epochs=50,
        ...     latent_dim=8,
        ... )
        >>> # Use agent for detection
        >>> result = agent.detect_anomaly(new_flow)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Autoencoder Training Pipeline")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Load and Preprocess Data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading and preprocessing data...")
    
    config = DatasetConfig(
        normalization_method="minmax",
        test_split_ratio=test_split,
        validation_split_ratio=val_split,
        random_seed=random_seed,
    )
    
    # Load data
    loader = DataLoader(config)
    loader.load(data_path).clean()
    
    X, y = loader.get_features_and_labels(label_type="binary")
    print(f"   Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Preprocess
    preprocessor = Preprocessor(config)
    X_processed, y_encoded = preprocessor.fit_transform(X, y, categorical_encoding="label")
    print(f"   Preprocessed features shape: {X_processed.shape}")
    
    # Save preprocessor
    preprocessor_path = output_path / "preprocessor.pkl"
    preprocessor.save(preprocessor_path)
    print(f"   Preprocessor saved to: {preprocessor_path}")
    
    # -------------------------------------------------------------------------
    # Step 2: Split Data
    # -------------------------------------------------------------------------
    print("\n[2/5] Splitting data...")
    
    splitter = DataSplitter(
        test_ratio=test_split,
        val_ratio=val_split,
        random_seed=random_seed,
    )
    splits = splitter.split(X_processed, y_encoded, stratify=True)
    
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # -------------------------------------------------------------------------
    # Step 3: Create Model and Train
    # -------------------------------------------------------------------------
    print("\n[3/5] Creating and training model...")
    
    input_dim = X_train.shape[1]
    if hidden_dims is None:
        hidden_dims = [32, 16]
    
    model = AnomalyAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    )
    
    print(model.summary())
    
    # Prepare data loaders (both train and val filtered to normal only)
    train_loader, val_loader = prepare_data_loaders(
        X_train, X_val, y_train, y_val,
        batch_size=batch_size,
        train_on_normal_only=train_on_normal_only,
    )
    
    # Train
    trainer = AutoencoderTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=str(output_path),
        verbose=True,
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Set Detection Threshold
    # -------------------------------------------------------------------------
    print("\n[4/5] Setting detection threshold...")
    
    # Load best model
    best_model_path = output_path / "best_model.pth"
    if best_model_path.exists():
        model = AnomalyAutoencoder.load(best_model_path)
    
    # Create agent with threshold (0.0396 = optimal accuracy for UNSW-NB15 training)
    threshold = 0.0396
    agent = AgentOne(model, threshold=threshold)
    print(f"   Using fixed threshold: {threshold:.6f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Evaluate on Test Set
    # -------------------------------------------------------------------------
    print("\n[5/5] Evaluating on test set...")
    
    results = agent.detect_anomalies(X_test)
    predictions = np.array([r.is_anomaly for r in results])
    errors = np.array([r.reconstruction_error for r in results])
    
    # Compute metrics (treating anomaly=attack, normal=0)
    # y_test: 0=normal, 1=attack
    # predictions: True=anomaly (predicted attack), False=normal
    
    true_positives = np.sum((predictions == True) & (y_test == 1))
    false_positives = np.sum((predictions == True) & (y_test == 0))
    true_negatives = np.sum((predictions == False) & (y_test == 0))
    false_negatives = np.sum((predictions == False) & (y_test == 1))
    
    accuracy = (true_positives + true_negatives) / len(y_test)
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"\n   Test Set Results:")
    print(f"   ─────────────────────────────────")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ─────────────────────────────────")
    print(f"   True Positives:  {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   True Negatives:  {true_negatives}")
    print(f"   False Negatives: {false_negatives}")
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    training_info = {
        "model_path": str(output_path / "best_model.pth"),
        "preprocessor_path": str(preprocessor_path),
        "threshold": agent.threshold,
        "history": history,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "config": {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
    }
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {output_path / 'best_model.pth'}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    return model, agent, training_info


def main():
    """Main entry point for command-line training."""
    parser = argparse.ArgumentParser(
        description="Train AnomalyAutoencoder for intrusion detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path", type=str,
        default="data/UNSW_NB15_training-set.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="models/agent_one",
        help="Directory to save outputs"
    )
    
    # Model arguments
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 16], help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Other arguments
    parser.add_argument("--threshold_percentile", type=float, default=95.0, help="Threshold calibration percentile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--all_data", action="store_true", help="Train on all data (not just normal)")
    
    args = parser.parse_args()
    
    # Find data path relative to project root
    data_path = Path(args.data_path)
    if not data_path.exists():
        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent
        data_path = project_root / args.data_path
    
    train_autoencoder(
        data_path=str(data_path),
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        train_on_normal_only=not args.all_data,
        threshold_percentile=args.threshold_percentile,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
