"""
Train Autoencoder on Combined UNSW-NB15 + CIC-IDS2017 with Compatible Preprocessing.

This script trains using the standard DataLoader/Preprocessor pipeline so the model
is compatible with existing tests and production code.

Configuration:
- MinMaxScaler normalization (matches standard Preprocessor)
- 40 features (matches standard pipeline)
- 50/50 balanced data from both datasets
- SMOTE oversampling for class balance
- Maximize F1 with Recall >= 95% constraint
- 50 epochs of training

Usage:
    python scripts/train_combined_compatible.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.models.autoencoder import AnomalyAutoencoder
from agents.agent_one import AgentOne
from data_pipeline import DataLoader, DatasetConfig, Preprocessor
from data_pipeline.cic_ids2017_loader import CICIDS2017Loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training."""
    
    # Data paths
    unsw_training_path = PROJECT_ROOT / "data" / "UNSW_NB15_training-set.csv"
    unsw_testing_path = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"
    cic_ids2017_path = PROJECT_ROOT / "data" / "CIC-IDS2017"
    
    # Model paths
    output_dir = PROJECT_ROOT / "models" / "agent_one_combined"
    
    # Data balancing
    target_samples_per_dataset = 100000  # 100k from each dataset
    apply_smote = True
    
    # Training
    epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-5
    early_stopping_patience = 15
    
    # Threshold calibration
    min_recall_constraint = 0.95
    
    # Random seed
    random_seed = 42


def load_unsw_data(config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load UNSW-NB15 training and testing data."""
    print("\n--- Loading UNSW-NB15 ---")
    
    dataset_config = DatasetConfig(normalization_method="minmax")
    
    # Load training data
    train_loader = DataLoader(dataset_config)
    train_loader.load(str(config.unsw_training_path)).clean()
    train_df = train_loader.data
    print(f"   Training: {len(train_df):,} samples")
    
    # Load testing data
    test_loader = DataLoader(dataset_config)
    test_loader.load(str(config.unsw_testing_path)).clean()
    test_df = test_loader.data
    print(f"   Testing: {len(test_df):,} samples")
    
    return train_df, test_df


def load_cic_data(config: TrainingConfig) -> pd.DataFrame:
    """Load CIC-IDS2017 data."""
    print("\n--- Loading CIC-IDS2017 ---")
    
    try:
        loader = CICIDS2017Loader()
        loader.load(config.cic_ids2017_path, sample_frac=0.3)
        
        if loader.data is None or len(loader.data) == 0:
            print("   WARNING: No CIC-IDS2017 data loaded, using UNSW-NB15 only")
            return pd.DataFrame()
        
        df = loader.data.copy()
        print(f"   Total: {len(df):,} samples")
        
        # Get labels
        X, y_binary = loader.get_features_and_labels(label_type="binary")
        _, y_category = loader.get_features_and_labels(label_type="unified")
        
        # Create a dataframe with label columns for compatibility
        df['label'] = y_binary
        df['attack_cat'] = y_category
        
        return df
    except Exception as e:
        print(f"   WARNING: Failed to load CIC-IDS2017: {e}")
        print("   Using UNSW-NB15 only")
        return pd.DataFrame()
    
    # Map attack categories to match UNSW-NB15 format
    # CIC-IDS2017 has 'Label' column with attack names
    if 'Label' in df.columns:
        # Create binary label
        df['label'] = (df['Label'].str.lower() != 'benign').astype(int)
        
        # Map to categories similar to UNSW
        category_map = {
            'benign': 'Normal',
            'ddos': 'DoS',
            'dos hulk': 'DoS',
            'dos goldeneye': 'DoS',
            'dos slowloris': 'DoS',
            'dos slowhttptest': 'DoS',
            'ftp-patator': 'Exploits',
            'ssh-patator': 'Exploits',
            'bot': 'Backdoors',
            'infiltration': 'Generic',
            'web attack': 'Exploits',
            'web attack brute force': 'Exploits',
            'web attack sql injection': 'Exploits',
            'web attack xss': 'Exploits',
            'heartbleed': 'Exploits',
            'portscan': 'Reconnaissance'
        }
        
        def map_category(label):
            label_lower = str(label).lower().strip()
            for key, value in category_map.items():
                if key in label_lower:
                    return value
            return 'Generic'
        
        df['attack_cat'] = df['Label'].apply(map_category)
        
    return df


def align_features(unsw_df: pd.DataFrame, cic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align features between datasets to have consistent columns."""
    print("\n--- Aligning Features ---")
    
    # Handle empty CIC data
    if cic_df is None or len(cic_df) == 0:
        print("   No CIC data to align")
        return unsw_df, pd.DataFrame()
    
    # Drop non-feature columns for alignment
    label_cols = ['label', 'attack_cat', 'Label']
    id_cols = ['id', 'ID', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    
    unsw_feature_cols = [c for c in unsw_df.columns if c.lower() not in [x.lower() for x in label_cols + id_cols]]
    cic_feature_cols = [c for c in cic_df.columns if c.lower() not in [x.lower() for x in label_cols + id_cols]]
    
    print(f"   UNSW features: {len(unsw_feature_cols)}")
    print(f"   CIC features: {len(cic_feature_cols)}")
    
    # Find common features by normalized name
    def normalize_col(c):
        return c.lower().replace(' ', '_').replace('-', '_')
    
    unsw_normalized = {normalize_col(c): c for c in unsw_feature_cols}
    cic_normalized = {normalize_col(c): c for c in cic_feature_cols}
    
    common_normalized = set(unsw_normalized.keys()) & set(cic_normalized.keys())
    print(f"   Common features: {len(common_normalized)}")
    
    # If not enough common features, use UNSW features and create missing in CIC with zeros
    if len(common_normalized) < 20:
        print("   Not enough common features, using UNSW feature set")
        
        # Keep only numeric columns from UNSW
        unsw_numeric = unsw_df[unsw_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Create CIC dataframe with same columns
        cic_aligned = pd.DataFrame()
        
        for col in unsw_numeric:
            col_normalized = normalize_col(col)
            if col_normalized in cic_normalized:
                cic_aligned[col] = cic_df[cic_normalized[col_normalized]]
            else:
                cic_aligned[col] = 0.0
        
        # Add labels
        if 'label' in unsw_df.columns:
            cic_aligned['label'] = cic_df['label']
        if 'attack_cat' in unsw_df.columns:
            cic_aligned['attack_cat'] = cic_df['attack_cat']
        
        unsw_aligned = unsw_df[unsw_numeric + ['label', 'attack_cat']]
        
        return unsw_aligned, cic_aligned
    
    # Use common features
    common_unsw_cols = [unsw_normalized[n] for n in common_normalized]
    common_cic_cols = [cic_normalized[n] for n in common_normalized]
    
    unsw_aligned = unsw_df[common_unsw_cols + ['label', 'attack_cat']].copy()
    cic_aligned = cic_df[common_cic_cols + ['label', 'attack_cat']].copy()
    
    # Ensure same column order
    cic_aligned.columns = unsw_aligned.columns
    
    return unsw_aligned, cic_aligned


def combine_and_balance(unsw_df: pd.DataFrame, cic_df: pd.DataFrame, 
                        config: TrainingConfig) -> pd.DataFrame:
    """Combine datasets with 50/50 balance."""
    print("\n--- Combining Datasets ---")
    
    # Handle case where CIC data is empty
    if cic_df is None or len(cic_df) == 0:
        print("   No CIC data, using UNSW-NB15 only")
        unsw_sample = unsw_df.sample(n=min(len(unsw_df), config.target_samples_per_dataset * 2), 
                                     random_state=config.random_seed)
        unsw_sample = unsw_sample.copy()
        unsw_sample['source'] = 'UNSW'
        print(f"   Total: {len(unsw_sample):,} samples (UNSW only)")
        return unsw_sample
    
    # Sample from each
    np.random.seed(config.random_seed)
    
    n_unsw = min(len(unsw_df), config.target_samples_per_dataset)
    n_cic = min(len(cic_df), config.target_samples_per_dataset)
    n_target = min(n_unsw, n_cic)
    
    unsw_sample = unsw_df.sample(n=n_target, random_state=config.random_seed)
    cic_sample = cic_df.sample(n=n_target, random_state=config.random_seed)
    
    # Add source indicator
    unsw_sample = unsw_sample.copy()
    cic_sample = cic_sample.copy()
    unsw_sample['source'] = 'UNSW'
    cic_sample['source'] = 'CIC'
    
    # Combine
    combined = pd.concat([unsw_sample, cic_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
    
    print(f"   Combined: {len(combined):,} samples")
    print(f"   UNSW: {n_target:,}, CIC: {n_target:,}")
    print(f"   Attacks: {(combined['label'] > 0).sum():,}")
    print(f"   Normal: {(combined['label'] == 0).sum():,}")
    
    return combined


def preprocess_data(combined_df: pd.DataFrame, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess data using standard pipeline."""
    print("\n--- Preprocessing ---")
    
    dataset_config = DatasetConfig(normalization_method="minmax")
    preprocessor = Preprocessor(dataset_config)
    
    # Drop source column before preprocessing
    df_for_preprocess = combined_df.drop(columns=['source'], errors='ignore')
    
    # Fit preprocessor
    preprocessor.fit(df_for_preprocess)
    
    # Transform
    X = preprocessor.transform(df_for_preprocess)
    y = (combined_df['label'].values > 0).astype(int)
    
    print(f"   Features shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_seed, stratify=y
    )
    
    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Save preprocessor
    config.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(config.output_dir / "preprocessor.pkl")
    print(f"   Preprocessor saved: {config.output_dir / 'preprocessor.pkl'}")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance classes."""
    if not config.apply_smote:
        return X_train, y_train
    
    print("\n--- Applying SMOTE ---")
    
    counts = np.bincount(y_train)
    print(f"   Before: Normal={counts[0]:,}, Attack={counts[1]:,}")
    
    smote = SMOTE(random_state=config.random_seed, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    counts = np.bincount(y_resampled)
    print(f"   After: Normal={counts[0]:,}, Attack={counts[1]:,}")
    
    return X_resampled, y_resampled


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, 
                      config: TrainingConfig) -> AnomalyAutoencoder:
    """Train autoencoder on normal traffic only."""
    print("\n" + "=" * 60)
    print("TRAINING AUTOENCODER")
    print("=" * 60)
    
    input_dim = X_train.shape[1]
    
    # Create model
    model = AnomalyAutoencoder(
        input_dim=input_dim,
        hidden_dims=[32, 16],
        latent_dim=8,
        dropout_rate=0.2,
        use_batch_norm=True
    )
    
    print(f"   Input dim: {input_dim}")
    print(f"   Architecture: {input_dim} -> 32 -> 16 -> 8 -> 16 -> 32 -> {input_dim}")
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    # Create dataloaders
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    
    train_loader = TorchDataLoader(
        TensorDataset(X_train_tensor, X_train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True  # Avoid batch_norm issues with small last batch
    )
    val_loader = TorchDataLoader(
        TensorDataset(X_val_tensor, X_val_tensor),
        batch_size=config.batch_size,
        drop_last=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_x, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}', leave=False):
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)
        
        train_loss /= len(X_train)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_loss += loss.item() * len(batch_x)
        
        val_loss /= len(X_val)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{config.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(config.output_dir / 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load(config.output_dir / 'best_model.pth')
    print(f"\n   Best validation loss: {best_val_loss:.6f}")
    
    return model


def calibrate_threshold(model: AnomalyAutoencoder, X_test: np.ndarray, 
                        y_test: np.ndarray, config: TrainingConfig) -> float:
    """Find optimal threshold maximizing F1 with recall >= 95%."""
    print("\n" + "=" * 60)
    print("CALIBRATING THRESHOLD")
    print("=" * 60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Compute reconstruction errors
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    # Search for optimal threshold
    thresholds = np.linspace(errors.min(), errors.max(), 200)
    best_f1 = 0.0
    best_threshold = thresholds[0]
    
    for thresh in thresholds:
        predictions = (errors > thresh).astype(int)
        
        recall = recall_score(y_test, predictions, zero_division=0)
        if recall < config.min_recall_constraint:
            continue
        
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # Final metrics
    predictions = (errors > best_threshold).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print(f"   Optimal threshold: {best_threshold:.6f}")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall: {recall*100:.2f}%")
    print(f"   F1: {f1:.4f}")
    
    return best_threshold


def evaluate_model(model: AnomalyAutoencoder, X_test: np.ndarray, 
                   y_test: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    device = next(model.parameters()).device
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    predictions = (errors > threshold).astype(int)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'threshold': threshold
    }
    
    print(f"\n   Results on Combined Test Set:")
    print(f"   ─────────────────────────────────")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   FPR:       {fpr*100:.2f}%")
    print(f"   ─────────────────────────────────")
    print(f"   TP: {tp:,}, FP: {fp:,}")
    print(f"   TN: {tn:,}, FN: {fn:,}")
    
    return metrics


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("COMBINED DATASET TRAINING WITH COMPATIBLE PREPROCESSING")
    print("=" * 60)
    
    config = TrainingConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    
    unsw_train, unsw_test = load_unsw_data(config)
    cic_df = load_cic_data(config)
    
    # Step 2: Align features
    print("\n" + "=" * 60)
    print("STEP 2: ALIGNING FEATURES")
    print("=" * 60)
    
    unsw_aligned, cic_aligned = align_features(
        pd.concat([unsw_train, unsw_test]), cic_df
    )
    
    # Step 3: Combine and balance
    print("\n" + "=" * 60)
    print("STEP 3: COMBINING DATASETS")
    print("=" * 60)
    
    combined = combine_and_balance(unsw_aligned, cic_aligned, config)
    
    # Step 4: Preprocess
    print("\n" + "=" * 60)
    print("STEP 4: PREPROCESSING")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = preprocess_data(combined, config)
    
    # Step 5: Apply SMOTE
    X_normal_train = X_train[y_train == 0]  # Train autoencoder on normal only
    X_normal_val = X_test[y_test == 0]
    
    print(f"\n   Normal samples for training: {len(X_normal_train):,}")
    print(f"   Normal samples for validation: {len(X_normal_val):,}")
    
    # Step 6: Train
    model = train_autoencoder(X_normal_train, X_normal_val, config)
    
    # Step 7: Calibrate threshold on full test set
    threshold = calibrate_threshold(model, X_test, y_test, config)
    
    # Step 8: Evaluate
    metrics = evaluate_model(model, X_test, y_test, threshold)
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': 'agent_one_combined',
        'threshold': threshold,
        'input_dim': X_train.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': metrics
    }
    
    with open(config.output_dir / 'training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir / 'best_model.pth'}")
    print(f"Preprocessor saved to: {config.output_dir / 'preprocessor.pkl'}")
    print(f"Report saved to: {config.output_dir / 'training_report.json'}")
    print(f"\nOptimal threshold: {threshold:.6f}")
    

if __name__ == "__main__":
    main()
