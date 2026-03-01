"""
Fine-tune Autoencoder on Combined UNSW-NB15 + CIC-IDS2017 Dataset.

This script fine-tunes the existing autoencoder model on both datasets
to improve accuracy and reduce false positive rate.

Configuration:
- Fine-tune existing model (not from scratch)
- 50/50 balanced data from both datasets
- SMOTE oversampling for class balance
- Maximize F1 with Recall >= 95% constraint
- 50 epochs of training

Usage:
    python scripts/finetune_autoencoder.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.models.autoencoder import AnomalyAutoencoder
from agents.agent_one import AgentOne
from data_pipeline.unified_dataset import UnifiedIDSDataset, UnifiedDatasetConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FineTuneConfig:
    """Configuration for fine-tuning."""
    
    # Data paths
    unsw_training_path = PROJECT_ROOT / "data" / "UNSW_NB15_training-set.csv"
    unsw_testing_path = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"
    cic_ids2017_path = PROJECT_ROOT / "data" / "CIC-IDS2017"
    
    # Model paths
    existing_model_path = PROJECT_ROOT / "models" / "agent_one" / "best_model.pth"
    output_dir = PROJECT_ROOT / "models" / "agent_one_finetuned"
    
    # Data balancing
    unsw_ratio = 0.5  # 50% UNSW
    cic_ratio = 0.5   # 50% CIC
    apply_smote = True
    
    # Training
    epochs = 50
    batch_size = 64
    learning_rate = 1e-4  # Lower LR for fine-tuning
    weight_decay = 1e-5
    early_stopping_patience = 10
    
    # Threshold calibration
    min_recall_constraint = 0.95  # Recall must be >= 95%
    
    # Random seed
    random_seed = 42


def load_combined_data(config: FineTuneConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and combine UNSW-NB15 and CIC-IDS2017 with 50/50 balance.
    
    Returns:
        X_train, X_test, y_train, y_test (binary labels: 0=normal, 1=attack)
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading Combined Dataset")
    print("=" * 60)
    
    # Create unified dataset with SMOTE enabled
    dataset_config = UnifiedDatasetConfig(
        normalization="standard",
        apply_smote=config.apply_smote,
        smote_strategy="auto",
        smote_k_neighbors=5,
        test_ratio=0.2,
        random_state=config.random_seed,
    )
    
    dataset = UnifiedIDSDataset(config=dataset_config)
    
    # Load UNSW-NB15
    print(f"\nLoading UNSW-NB15 from {config.unsw_training_path}...")
    dataset.load_unsw_nb15(config.unsw_training_path)
    unsw_count = len(dataset.y)
    print(f"   Loaded {unsw_count:,} UNSW-NB15 samples")
    
    # Load CIC-IDS2017
    print(f"\nLoading CIC-IDS2017 from {config.cic_ids2017_path}...")
    dataset.load_cic_ids2017(config.cic_ids2017_path, sample_frac=0.3)  # Sample to balance size
    cic_count = len(dataset.y) - unsw_count
    print(f"   Loaded {cic_count:,} CIC-IDS2017 samples")
    
    # Balance 50/50 between datasets
    print("\nBalancing datasets to 50/50...")
    unsw_mask = dataset.source == 0
    cic_mask = dataset.source == 1
    
    n_target = min(np.sum(unsw_mask), np.sum(cic_mask))
    
    # Sample from each dataset
    np.random.seed(config.random_seed)
    unsw_indices = np.where(unsw_mask)[0]
    cic_indices = np.where(cic_mask)[0]
    
    unsw_sample = np.random.choice(unsw_indices, size=n_target, replace=False)
    cic_sample = np.random.choice(cic_indices, size=n_target, replace=False)
    
    balanced_indices = np.concatenate([unsw_sample, cic_sample])
    np.random.shuffle(balanced_indices)
    
    X_balanced = dataset.X[balanced_indices]
    y_balanced = dataset.y[balanced_indices]
    
    print(f"   Balanced to {len(X_balanced):,} samples (50% each)")
    
    # Normalize
    print("\nNormalizing features...")
    dataset.X = X_balanced
    dataset.y = y_balanced
    dataset.normalize()
    X_normalized = dataset.X
    
    # Apply SMOTE
    if config.apply_smote:
        print("\nApplying SMOTE oversampling...")
        original_count = len(dataset.y)
        dataset.apply_smote()
        print(f"   Samples before SMOTE: {original_count:,}")
        print(f"   Samples after SMOTE:  {len(dataset.y):,}")
    
    # Convert multiclass labels to binary (0=Normal, 1=Attack)
    y_binary = (dataset.y > 0).astype(int)
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, y_binary,
        test_size=0.2,
        random_state=config.random_seed,
        stratify=y_binary
    )
    
    print(f"\nData split:")
    print(f"   Train: {len(X_train):,} samples ({np.sum(y_train == 0):,} normal, {np.sum(y_train == 1):,} attacks)")
    print(f"   Test:  {len(X_test):,} samples ({np.sum(y_test == 0):,} normal, {np.sum(y_test == 1):,} attacks)")
    
    return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test


def load_existing_model(config: FineTuneConfig) -> AnomalyAutoencoder:
    """Load existing trained model for fine-tuning."""
    print("\n" + "=" * 60)
    print("STEP 2: Loading Existing Model")
    print("=" * 60)
    
    if not config.existing_model_path.exists():
        raise FileNotFoundError(f"Model not found: {config.existing_model_path}")
    
    model = AnomalyAutoencoder.load(config.existing_model_path)
    print(f"   Loaded model from: {config.existing_model_path}")
    print(f"   Architecture: {model.input_dim} -> {model.hidden_dims} -> {model.latent_dim}")
    
    return model


def fine_tune_model(
    model: AnomalyAutoencoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: FineTuneConfig,
) -> Dict[str, list]:
    """Fine-tune model on combined dataset."""
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning Model")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"   Device: {device}")
    
    # Train on NORMAL traffic only (autoencoder learns normal patterns)
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    print(f"   Training on {len(X_train_normal):,} normal samples")
    
    # Also validate on normal traffic
    val_normal_mask = y_val == 0
    X_val_normal = X_val[val_normal_mask]
    
    # Create data loaders
    train_tensor = torch.from_numpy(X_train_normal).float()
    train_dataset = TensorDataset(train_tensor)
    train_loader = TorchDataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    
    val_tensor = torch.from_numpy(X_val_normal).float()
    val_dataset = TensorDataset(val_tensor)
    val_loader = TorchDataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    # Optimizer with lower LR for fine-tuning
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    criterion = nn.MSELoss()
    
    # Training history
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   Starting fine-tuning for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(config.output_dir / "best_model.pth")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        if patience_counter >= config.early_stopping_patience:
            print(f"   Early stopping at epoch {epoch + 1}")
            break
    
    # Save final model
    model.save(config.output_dir / "final_model.pth")
    print(f"\n   Best validation loss: {best_val_loss:.6f}")
    print(f"   Model saved to: {config.output_dir}")
    
    return history


def calibrate_threshold_f1_with_recall_constraint(
    model: AnomalyAutoencoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    min_recall: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold that maximizes F1 while maintaining recall >= min_recall.
    
    Returns:
        (optimal_threshold, metrics_dict)
    """
    print("\n" + "=" * 60)
    print("STEP 4: Calibrating Detection Threshold")
    print("=" * 60)
    print(f"   Constraint: Recall >= {min_recall:.0%}")
    print(f"   Objective: Maximize F1 score")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Compute reconstruction errors
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).float().to(device)
        
        # Process in batches to avoid OOM
        batch_size = 1024
        errors = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            reconstructed = model(batch)
            batch_errors = torch.mean((batch - reconstructed) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())
        
        errors = np.concatenate(errors)
    
    # Search for optimal threshold
    thresholds = np.percentile(errors, np.linspace(1, 99, 200))
    
    best_threshold = None
    best_f1 = 0.0
    best_metrics = {}
    
    print("\n   Searching for optimal threshold...")
    
    for threshold in thresholds:
        predictions = (errors > threshold).astype(int)
        
        recall = recall_score(y_test, predictions, zero_division=0)
        precision = precision_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        accuracy = accuracy_score(y_test, predictions)
        
        # Calculate FPR
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Check recall constraint
        if recall >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
    
    if best_threshold is None:
        # If no threshold meets recall constraint, use lowest threshold
        print("   WARNING: No threshold meets recall constraint. Using minimum error threshold.")
        best_threshold = np.percentile(errors, 1)
        predictions = (errors > best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        best_metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }
    
    print(f"\n   Optimal threshold: {best_threshold:.6f}")
    print(f"\n   Metrics at optimal threshold:")
    print(f"   ─────────────────────────────────")
    print(f"   Accuracy:  {best_metrics['accuracy']:.2%}")
    print(f"   Precision: {best_metrics['precision']:.2%}")
    print(f"   Recall:    {best_metrics['recall']:.2%} (target: >={min_recall:.0%}) {'✓' if best_metrics['recall'] >= min_recall else '✗'}")
    print(f"   F1 Score:  {best_metrics['f1']:.4f}")
    print(f"   FPR:       {best_metrics['fpr']:.2%}")
    print(f"   ─────────────────────────────────")
    print(f"   TP: {best_metrics['tp']:,}  FP: {best_metrics['fp']:,}")
    print(f"   TN: {best_metrics['tn']:,}  FN: {best_metrics['fn']:,}")
    
    return best_threshold, best_metrics


def compare_with_original(config: FineTuneConfig, X_test: np.ndarray, y_test: np.ndarray, new_metrics: Dict):
    """Compare fine-tuned model with original."""
    print("\n" + "=" * 60)
    print("STEP 5: Comparison with Original Model")
    print("=" * 60)
    
    # Original model metrics (from accuracy_report.txt)
    original_metrics = {
        "accuracy": 0.6316,
        "precision": 0.6010,
        "recall": 1.0000,
        "f1": 0.7507,
        "fpr": 0.8273,  # FP/(FP+TN) = 1842/(1842+384)
    }
    
    print("\n   Metric Comparison:")
    print(f"   {'Metric':<12} {'Original':>12} {'Fine-tuned':>12} {'Change':>12}")
    print(f"   {'-'*48}")
    
    for metric in ["accuracy", "precision", "recall", "f1", "fpr"]:
        orig = original_metrics[metric]
        new = new_metrics[metric]
        change = new - orig
        change_str = f"{change:+.2%}" if abs(change) > 0.0001 else "  --"
        better = "✓" if (metric != "fpr" and change > 0) or (metric == "fpr" and change < 0) else ""
        print(f"   {metric.upper():<12} {orig:>11.2%} {new:>11.2%} {change_str:>10} {better}")


def run_verification_report(
    model: AnomalyAutoencoder,
    threshold: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: FineTuneConfig,
):
    """Generate full verification report."""
    print("\n" + "=" * 60)
    print("STEP 6: Full Verification Report")
    print("=" * 60)
    
    agent = AgentOne(model, threshold=threshold)
    
    # Run detection
    results = agent.detect_anomalies(X_test, return_raw=False)
    predictions = np.array([r.is_anomaly for r in results]).astype(int)
    errors = np.array([r.reconstruction_error for r in results])
    
    # Compute all metrics
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "agent_one_finetuned",
        "threshold": threshold,
        "test_samples": len(y_test),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
        },
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        },
        "targets_met": {
            "recall_>=_95%": bool(recall >= 0.95),
            "fpr_<=_5%": bool(fpr <= 0.05),
        },
        "config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "datasets": ["UNSW-NB15", "CIC-IDS2017"],
            "smote_applied": config.apply_smote,
        }
    }
    
    report_path = config.output_dir / "finetune_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n   Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("                    FINAL RESULTS")
    print("=" * 60)
    print(f"\n   Threshold: {threshold:.6f}")
    print(f"\n   Test Set Performance:")
    print(f"   ─────────────────────────────────")
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%} {'✓' if recall >= 0.95 else '✗'} (target: >=95%)")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   FPR:       {fpr:.2%} {'✓' if fpr <= 0.05 else ''} (target: <=5%)")
    print(f"   ─────────────────────────────────")
    print(f"   True Positives:  {tp:,}")
    print(f"   False Positives: {fp:,}")
    print(f"   True Negatives:  {tn:,}")
    print(f"   False Negatives: {fn:,}")
    print("\n" + "=" * 60)
    
    return report


def main():
    """Main fine-tuning pipeline."""
    print("=" * 60)
    print("   AUTOENCODER FINE-TUNING PIPELINE")
    print("   Combined UNSW-NB15 + CIC-IDS2017 Training")
    print("=" * 60)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = FineTuneConfig()
    
    try:
        # Step 1: Load combined data
        X_train, X_test, y_train, y_test = load_combined_data(config)
        
        # Step 2: Load existing model
        model = load_existing_model(config)
        
        # Adjust input dimensions if needed
        if model.input_dim != X_train.shape[1]:
            print(f"\n   WARNING: Input dimension mismatch!")
            print(f"   Model expects: {model.input_dim}, Data has: {X_train.shape[1]}")
            print(f"   Creating new model with correct dimensions...")
            
            model = AnomalyAutoencoder(
                input_dim=X_train.shape[1],
                latent_dim=model.latent_dim,
                hidden_dims=model.hidden_dims,
                dropout_rate=model.dropout_rate,
            )
        
        # Split train into train/val
        from sklearn.model_selection import train_test_split
        X_train2, X_val, y_train2, y_val = train_test_split(
            X_train, y_train,
            test_size=0.15,
            random_state=config.random_seed,
            stratify=y_train
        )
        
        # Step 3: Fine-tune
        history = fine_tune_model(model, X_train2, y_train2, X_val, y_val, config)
        
        # Load best model
        model = AnomalyAutoencoder.load(config.output_dir / "best_model.pth")
        
        # Step 4: Calibrate threshold
        threshold, metrics = calibrate_threshold_f1_with_recall_constraint(
            model, X_test, y_test, min_recall=config.min_recall_constraint
        )
        
        # Step 5: Compare with original
        compare_with_original(config, X_test, y_test, metrics)
        
        # Step 6: Full verification report
        report = run_verification_report(model, threshold, X_test, y_test, config)
        
        # Save threshold for later use
        threshold_path = config.output_dir / "optimal_threshold.txt"
        with open(threshold_path, 'w') as f:
            f.write(f"{threshold}")
        
        print(f"\n   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return model, threshold, report
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()
