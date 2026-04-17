"""
Improved Federated Learning Pipeline with Enhanced Detection.

Improvements over baseline:
1. Adaptive threshold selection based on validation set
2. XGBoost classifier integration for better classification
3. Class-weighted loss function for imbalanced data
4. Better feature engineering with reconstruction error features
5. Ensemble of autoencoder + classifier
6. FedProx for better non-IID handling
7. Learning rate scheduling
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ImprovedFL")


@dataclass
class ImprovedConfig:
    num_clients: int = 2
    num_rounds: int = 15
    local_epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 0.002
    max_samples: int = 10000
    test_split: float = 0.2
    seed: int = 42
    
    # Improvements
    use_class_weights: bool = True
    use_fedprox: bool = True
    fedprox_mu: float = 0.01
    use_lr_scheduler: bool = True
    use_xgboost: bool = True
    ensemble_weight_ae: float = 0.4
    ensemble_weight_clf: float = 0.6
    threshold_method: str = "optimal_f1"  # "percentile", "optimal_f1", "youden"
    use_robust_scaler: bool = True


# ==============================================================================
# Improved Autoencoder with Better Architecture
# ==============================================================================

if HAS_TORCH:
    class ImprovedAutoencoder(nn.Module):
        """
        Improved autoencoder with:
        - Batch normalization
        - Dropout for regularization
        - Skip connections
        - Deeper architecture
        """
        
        def __init__(self, input_dim: int, latent_dim: int = 16):
            super().__init__()
            
            # Encoder with batch norm and dropout
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(32, latent_dim),
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(128, input_dim),
            )
            
            # Skip connection weight
            self.skip_weight = nn.Parameter(torch.tensor(0.1))
        
        def forward(self, x):
            z = self.encoder(x)
            reconstruction = self.decoder(z)
            # Skip connection for better gradient flow
            return reconstruction + self.skip_weight * x
        
        def encode(self, x):
            return self.encoder(x)
        
        def get_reconstruction_error(self, x):
            with torch.no_grad():
                self.eval()
                output = self.forward(x)
                # Per-sample MSE
                errors = torch.mean((x - output) ** 2, dim=1)
                return errors.numpy()


# ==============================================================================
# Weighted Loss for Class Imbalance
# ==============================================================================

class WeightedMSELoss(nn.Module):
    """MSE loss with sample weights for class imbalance."""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, output, target, weights=None):
        loss = torch.mean((output - target) ** 2, dim=1)
        
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ==============================================================================
# Improved Federated Client
# ==============================================================================

class ImprovedFLClient:
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, config: ImprovedConfig):
        self.client_id = client_id
        self.config = config
        self.y_train = y_train
        self.y_val = y_val
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train) if HAS_TORCH else X_train
        self.X_val = torch.FloatTensor(X_val) if HAS_TORCH else X_val
        
        input_dim = X_train.shape[1]
        
        # Initialize improved autoencoder
        if HAS_TORCH:
            self.model = ImprovedAutoencoder(input_dim, latent_dim=16)
            self.criterion = WeightedMSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=config.learning_rate,
                weight_decay=1e-5
            )
            
            if config.use_lr_scheduler:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.num_rounds * config.local_epochs
                )
            else:
                self.scheduler = None
        
        # XGBoost classifier
        if config.use_xgboost:
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=config.seed
            )
        else:
            self.classifier = None
        
        # Compute class weights
        if config.use_class_weights:
            n_normal = (y_train == 0).sum()
            n_anomaly = (y_train == 1).sum()
            total = len(y_train)
            self.class_weights = {
                0: total / (2 * n_normal) if n_normal > 0 else 1.0,
                1: total / (2 * n_anomaly) if n_anomaly > 0 else 1.0,
            }
            self.sample_weights = np.array([self.class_weights[y] for y in y_train])
        else:
            self.sample_weights = np.ones(len(y_train))
        
        self.threshold = 0.5
        self.global_weights_copy = None  # For FedProx
        self.baseline_metrics = {}
        self.federated_metrics = {}
        self._clf_trained = False
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        if HAS_TORCH:
            return {k: v.cpu().numpy().copy() for k, v in self.model.state_dict().items()}
        return {}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        if HAS_TORCH:
            # Warmup pass to ensure BatchNorm buffers are properly sized
            self.model.train()
            with torch.no_grad():
                _ = self.model(self.X_train[:32])
            
            # Convert weights back to tensors, handling scalars
            state_dict = {}
            for k, v in weights.items():
                if np.isscalar(v) or v.ndim == 0:
                    state_dict[k] = torch.tensor(float(v))
                else:
                    state_dict[k] = torch.FloatTensor(v)
            
            self.model.load_state_dict(state_dict, strict=False)
            
            # Store copy for FedProx
            if self.config.use_fedprox:
                self.global_weights_copy = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def _compute_fedprox_loss(self) -> torch.Tensor:
        """Compute FedProx proximal term."""
        if not self.config.use_fedprox or self.global_weights_copy is None:
            return torch.tensor(0.0)
        
        prox_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.global_weights_copy:
                prox_loss += torch.sum((param - self.global_weights_copy[name]) ** 2)
        
        return self.config.fedprox_mu / 2 * prox_loss
    
    def train_epoch(self) -> float:
        """Train one epoch with improvements."""
        if not HAS_TORCH:
            return 0.0
        
        self.model.train()
        
        # Create weighted sampler for class imbalance
        if self.config.use_class_weights:
            weights_tensor = torch.FloatTensor(self.sample_weights)
            sampler = WeightedRandomSampler(weights_tensor, len(weights_tensor), replacement=True)
            dataset = TensorDataset(self.X_train, torch.FloatTensor(self.sample_weights))
            loader = DataLoader(dataset, batch_size=self.config.batch_size, sampler=sampler)
        else:
            dataset = TensorDataset(self.X_train)
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        total_loss = 0
        for batch in loader:
            if self.config.use_class_weights:
                x, weights = batch
            else:
                x = batch[0]
                weights = None
            
            self.optimizer.zero_grad()
            output = self.model(x)
            
            # Reconstruction loss
            recon_loss = self.criterion(output, x, weights)
            
            # FedProx proximal term
            prox_loss = self._compute_fedprox_loss()
            
            loss = recon_loss + prox_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += recon_loss.item()
        
        if self.scheduler:
            self.scheduler.step()
        
        return total_loss / len(loader)
    
    def train_local(self, epochs: int) -> Tuple[Dict[str, np.ndarray], float]:
        """Train locally with all improvements."""
        losses = []
        for _ in range(epochs):
            loss = self.train_epoch()
            losses.append(loss)
        
        # Train XGBoost on reconstruction errors + original features
        if self.config.use_xgboost and self.classifier is not None:
            self._train_classifier()
        
        # Compute optimal threshold
        self._compute_optimal_threshold()
        
        return self.get_weights(), np.mean(losses)
    
    def _train_classifier(self):
        """Train XGBoost on combined features."""
        if not HAS_TORCH:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get reconstruction errors
            errors = self.model.get_reconstruction_error(self.X_train)
            
            # Get latent features
            latent = self.model.encode(self.X_train).numpy()
        
        # Combine features: original + reconstruction error + latent
        X_combined = np.column_stack([
            self.X_train.numpy() if HAS_TORCH else self.X_train,
            errors.reshape(-1, 1),
            latent
        ])
        
        self.classifier.fit(X_combined, self.y_train)
        self._clf_trained = True
    
    def _compute_optimal_threshold(self):
        """Compute optimal threshold using validation set."""
        if not HAS_TORCH:
            return
        
        self.model.eval()
        errors = self.model.get_reconstruction_error(self.X_val)
        
        if self.config.threshold_method == "percentile":
            # Traditional: 95th percentile of training errors
            train_errors = self.model.get_reconstruction_error(self.X_train)
            self.threshold = np.percentile(train_errors, 95)
        
        elif self.config.threshold_method == "optimal_f1":
            # Find threshold that maximizes F1 on validation set
            precisions, recalls, thresholds = precision_recall_curve(self.y_val, errors)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            
            best_idx = np.argmax(f1_scores[:-1])  # Last element is for threshold=1
            self.threshold = thresholds[best_idx]
        
        elif self.config.threshold_method == "youden":
            # Youden's J statistic: maximize (sensitivity + specificity - 1)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(self.y_val, errors)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            self.threshold = thresholds[best_idx]
    
    def _get_errors(self, X) -> np.ndarray:
        if HAS_TORCH:
            self.model.eval()
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            return self.model.get_reconstruction_error(X)
        return np.zeros(len(X))
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate with ensemble of autoencoder + classifier."""
        if not HAS_TORCH:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        
        self.model.eval()
        X_val_np = self.X_val.numpy() if HAS_TORCH else self.X_val
        
        # Autoencoder predictions
        errors = self._get_errors(self.X_val)
        ae_predictions = (errors > self.threshold).astype(int)
        ae_scores = errors / (errors.max() + 1e-8)  # Normalize to [0,1]
        
        # Classifier predictions (if available)
        if self.config.use_xgboost and self.classifier is not None and self._clf_trained:
            with torch.no_grad():
                latent = self.model.encode(self.X_val).numpy()
            
            X_combined = np.column_stack([X_val_np, errors.reshape(-1, 1), latent])
            clf_predictions = self.classifier.predict(X_combined)
            clf_scores = self.classifier.predict_proba(X_combined)[:, 1]
            
            # Ensemble: weighted combination
            ensemble_scores = (
                self.config.ensemble_weight_ae * ae_scores + 
                self.config.ensemble_weight_clf * clf_scores
            )
            ensemble_predictions = (ensemble_scores > 0.5).astype(int)
        else:
            ensemble_predictions = ae_predictions
            ensemble_scores = ae_scores
        
        metrics = {
            'accuracy': accuracy_score(self.y_val, ensemble_predictions),
            'precision': precision_score(self.y_val, ensemble_predictions, zero_division=0),
            'recall': recall_score(self.y_val, ensemble_predictions, zero_division=0),
            'f1_score': f1_score(self.y_val, ensemble_predictions, zero_division=0),
            'ae_accuracy': accuracy_score(self.y_val, ae_predictions),
            'ae_recall': recall_score(self.y_val, ae_predictions, zero_division=0),
            'mean_error': float(errors.mean()),
            'threshold': float(self.threshold),
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(self.y_val, ensemble_scores)
        except:
            metrics['roc_auc'] = 0.5
        
        return metrics


# ==============================================================================
# Improved Federated Server with FedProx
# ==============================================================================

class ImprovedFLServer:
    def __init__(self, clients: List[ImprovedFLClient], config: ImprovedConfig):
        self.clients = clients
        self.config = config
        self.global_weights = clients[0].get_weights()
        self.history = []
    
    def fedavg(self, client_weights: List[Dict], sample_counts: List[int]) -> Dict[str, np.ndarray]:
        """FedAvg with optional momentum."""
        total = sum(sample_counts)
        avg = {}
        
        for key in client_weights[0]:
            weighted_sum = sum(
                w[key] * (n / total) 
                for w, n in zip(client_weights, sample_counts)
            )
            avg[key] = weighted_sum
        
        return avg
    
    def run_round(self, round_num: int, local_epochs: int) -> Dict[str, Any]:
        """Run one federated round."""
        logger.info(f"=== Round {round_num} ===")
        
        # Distribute global weights
        for client in self.clients:
            client.set_weights(self.global_weights)
        
        # Local training
        client_weights = []
        client_samples = []
        client_losses = []
        
        for client in self.clients:
            weights, loss = client.train_local(local_epochs)
            client_weights.append(weights)
            client_samples.append(len(client.X_train))
            client_losses.append(loss)
            logger.info(f"  Client {client.client_id}: loss={loss:.6f}")
        
        # Aggregate
        self.global_weights = self.fedavg(client_weights, client_samples)
        
        # Evaluate with new global weights
        for client in self.clients:
            client.set_weights(self.global_weights)
            # Retrain classifier with global model
            if client.config.use_xgboost:
                client._train_classifier()
            client._compute_optimal_threshold()
        
        metrics = [client.evaluate() for client in self.clients]
        avg_metrics = {
            key: np.mean([m[key] for m in metrics])
            for key in metrics[0]
        }
        
        result = {
            'round': round_num,
            'client_losses': client_losses,
            'client_metrics': metrics,
            'avg_metrics': avg_metrics,
        }
        
        self.history.append(result)
        logger.info(f"  Global: acc={avg_metrics['accuracy']:.4f}, f1={avg_metrics['f1_score']:.4f}, recall={avg_metrics['recall']:.4f}")
        
        return result


# ==============================================================================
# Data Loading with Better Preprocessing
# ==============================================================================

def load_data_improved(config: ImprovedConfig) -> List[Tuple]:
    """Load data with improved preprocessing."""
    datasets = []
    paths = [
        project_root / "data" / "UNSW_NB15_training-set.csv",
        project_root / "data" / "UNSW_NB15_testing-set.csv",
    ]
    
    for i, path in enumerate(paths[:config.num_clients]):
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        
        logger.info(f"Loading {path.name}...")
        df = pd.read_csv(path, low_memory=False)
        
        # Sample if too large
        if len(df) > config.max_samples:
            # Stratified sampling
            if 'label' in df.columns:
                df = df.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(frac=config.max_samples/len(df), random_state=config.seed+i)
                )
            else:
                df = df.sample(n=config.max_samples, random_state=config.seed+i)
        
        # Feature engineering
        num_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['id', 'label', 'Label', 'attack_cat']
        feat_cols = [c for c in num_cols if c.lower() not in [e.lower() for e in exclude]]
        
        X = df[feat_cols].fillna(0).values.astype(np.float32)
        
        # Labels
        if 'label' in df.columns:
            y = df['label'].values
        elif 'Label' in df.columns:
            y = df['Label'].values
        else:
            y = np.zeros(len(df))
        
        # Better normalization
        if config.use_robust_scaler:
            scaler = RobustScaler()  # More robust to outliers
        else:
            scaler = StandardScaler()
        
        X = scaler.fit_transform(X).astype(np.float32)
        
        # Clip extreme values
        X = np.clip(X, -5, 5)
        
        # Train/val split with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.test_split, stratify=y, random_state=config.seed
        )
        
        datasets.append((X_train, y_train, X_val, y_val))
        
        anomaly_rate = y_train.mean()
        logger.info(f"  Client {i}: {len(X_train)} train, {len(X_val)} val, {anomaly_rate:.1%} anomalies")
    
    return datasets


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument("--no-fedprox", action="store_true")
    args = parser.parse_args()
    
    config = ImprovedConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        max_samples=args.max_samples,
        use_xgboost=not args.no_xgboost,
        use_fedprox=not args.no_fedprox,
    )
    
    np.random.seed(config.seed)
    if HAS_TORCH:
        torch.manual_seed(config.seed)
    
    start = time.time()
    
    print("\n" + "="*80)
    print("  IMPROVED FEDERATED LEARNING PIPELINE")
    print("  With: XGBoost Ensemble | FedProx | Optimal Thresholds")
    print("="*80 + "\n")
    
    print(f"Improvements enabled:")
    print(f"  - XGBoost Ensemble: {config.use_xgboost}")
    print(f"  - FedProx: {config.use_fedprox} (mu={config.fedprox_mu})")
    print(f"  - Class Weights: {config.use_class_weights}")
    print(f"  - LR Scheduler: {config.use_lr_scheduler}")
    print(f"  - Threshold Method: {config.threshold_method}")
    print(f"  - Robust Scaler: {config.use_robust_scaler}")
    print()
    
    # Load data
    print("[1/4] Loading datasets...")
    datasets = load_data_improved(config)
    
    if not datasets:
        print("No datasets found!")
        return
    
    # Create clients and train baseline
    print("\n[2/4] Training baseline models...")
    clients = []
    for i, (X_tr, y_tr, X_val, y_val) in enumerate(datasets):
        client = ImprovedFLClient(i, X_tr, y_tr, X_val, y_val, config)
        
        # Baseline training (more epochs)
        _, loss = client.train_local(config.local_epochs * 3)
        client.baseline_metrics = client.evaluate()
        
        # Reset for federated
        if HAS_TORCH:
            client.model = ImprovedAutoencoder(X_tr.shape[1], latent_dim=16)
            # Warmup pass to initialize BatchNorm running stats
            client.model.train()
            _ = client.model(client.X_train[:32])
            client.model.eval()
            
            client.optimizer = optim.AdamW(client.model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
            if config.use_lr_scheduler:
                client.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    client.optimizer, T_max=config.num_rounds * config.local_epochs
                )
            client._clf_trained = False  # Reset classifier trained flag
        
        clients.append(client)
        b = client.baseline_metrics
        print(f"  Client {i} baseline: acc={b['accuracy']:.4f}, f1={b['f1_score']:.4f}, recall={b['recall']:.4f}")
    
    # Federated training
    print(f"\n[3/4] Running federated training ({config.num_rounds} rounds)...")
    server = ImprovedFLServer(clients, config)
    for r in range(1, config.num_rounds + 1):
        server.run_round(r, config.local_epochs)
    
    total_time = time.time() - start
    
    # Results
    print("\n[4/4] Final Results...")
    
    # Calculate improvements
    baseline_avg = {
        k: np.mean([c.baseline_metrics.get(k, 0) for c in clients])
        for k in clients[0].baseline_metrics
    }
    
    final = server.history[-1]['avg_metrics']
    
    print("\n" + "="*80)
    print("  RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n  {'Metric':<15} {'Baseline':>12} {'Federated':>12} {'Improvement':>15}")
    print("  " + "-"*56)
    
    for metric in ['accuracy', 'f1_score', 'recall', 'precision', 'roc_auc']:
        b_val = baseline_avg.get(metric, 0)
        f_val = final.get(metric, 0)
        delta = f_val - b_val
        pct = (delta / b_val * 100) if b_val > 0 else 0
        print(f"  {metric:<15} {b_val:>12.4f} {f_val:>12.4f} {delta:>+10.4f} ({pct:>+5.1f}%)")
    
    # Per-round progress
    print("\n" + "-"*80)
    print("  TRAINING PROGRESS")
    print("-"*80)
    print(f"\n  {'Round':<8} {'Accuracy':>10} {'F1 Score':>10} {'Recall':>10} {'Mean Error':>12}")
    print("  " + "-"*52)
    for r in server.history:
        m = r['avg_metrics']
        print(f"  {r['round']:<8} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f} {m['recall']:>10.4f} {m['mean_error']:>12.6f}")
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    acc_imp = (final['accuracy'] - baseline_avg['accuracy']) / baseline_avg['accuracy'] * 100
    f1_imp = (final['f1_score'] - baseline_avg['f1_score']) / baseline_avg['f1_score'] * 100
    recall_imp = (final['recall'] - baseline_avg['recall']) / baseline_avg['recall'] * 100 if baseline_avg['recall'] > 0 else 0
    
    print(f"""
  Final Performance:
  - Accuracy: {baseline_avg['accuracy']:.2%} → {final['accuracy']:.2%} ({acc_imp:+.1f}%)
  - F1 Score: {baseline_avg['f1_score']:.2%} → {final['f1_score']:.2%} ({f1_imp:+.1f}%)
  - Recall:   {baseline_avg['recall']:.2%} → {final['recall']:.2%} ({recall_imp:+.1f}%)
  
  Total Time: {total_time:.1f}s
  PyTorch: {HAS_TORCH}
    """)
    
    print("="*80 + "\n")
    
    # Save results
    results = {
        'config': {
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'use_xgboost': config.use_xgboost,
            'use_fedprox': config.use_fedprox,
            'threshold_method': config.threshold_method,
        },
        'baseline': baseline_avg,
        'federated': final,
        'improvement': {
            'accuracy_pct': acc_imp,
            'f1_pct': f1_imp,
            'recall_pct': recall_imp,
        },
        'rounds': [r['avg_metrics'] for r in server.history],
    }
    
    output_path = project_root / "scripts" / "improved_fl_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
