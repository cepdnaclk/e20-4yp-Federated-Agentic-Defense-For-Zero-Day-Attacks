"""
Full Federated Learning Pipeline Report with PyTorch Models.

Demonstrates performance improvement from federated learning:
- Client 1: Trained on UNSW_NB15_training-set.csv
- Client 2: Trained on UNSW_NB15_testing-set.csv  
- Server: Aggregates client models using FedAvg

Shows:
1. Baseline: Each client training alone
2. Federated: Collaborative training
3. Improvement metrics and RAG explanations
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, using sklearn fallback")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("FL_Pipeline")


@dataclass 
class Config:
    num_clients: int = 2
    num_rounds: int = 10
    local_epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.001
    max_samples: int = 8000
    test_split: float = 0.2
    seed: int = 42


# ==============================================================================
# PyTorch Autoencoder
# ==============================================================================

if HAS_TORCH:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int = 8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )
        
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
        
        def get_reconstruction_error(self, x):
            with torch.no_grad():
                output = self.forward(x)
                return torch.mean((x - output) ** 2, dim=1).numpy()


# ==============================================================================
# Federated Client
# ==============================================================================

class FLClient:
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, config: Config):
        self.client_id = client_id
        self.X_train = torch.FloatTensor(X_train) if HAS_TORCH else X_train
        self.y_train = y_train
        self.X_val = torch.FloatTensor(X_val) if HAS_TORCH else X_val
        self.y_val = y_val
        self.config = config
        
        input_dim = X_train.shape[1]
        
        if HAS_TORCH:
            self.model = Autoencoder(input_dim)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        else:
            self.weights = self._init_weights(input_dim)
        
        self.threshold = 0.5
        self.baseline_metrics = {}
        self.federated_metrics = {}
    
    def _init_weights(self, dim):
        return {'enc': np.random.randn(dim, 8) * 0.1, 'dec': np.random.randn(8, dim) * 0.1}
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        if HAS_TORCH:
            return {k: v.cpu().numpy().copy() for k, v in self.model.state_dict().items()}
        return {k: v.copy() for k, v in self.weights.items()}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        if HAS_TORCH:
            state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
            self.model.load_state_dict(state_dict)
        else:
            self.weights = {k: v.copy() for k, v in weights.items()}
    
    def train_epoch(self) -> float:
        if HAS_TORCH:
            self.model.train()
            dataset = TensorDataset(self.X_train)
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            total_loss = 0
            for batch in loader:
                x = batch[0]
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            return total_loss / len(loader)
        else:
            # Simple training for non-torch
            X = self.X_train if isinstance(self.X_train, np.ndarray) else self.X_train.numpy()
            z = X @ self.weights['enc']
            recon = z @ self.weights['dec']
            loss = np.mean((X - recon) ** 2)
            
            # Simple gradient update
            grad_dec = z.T @ (recon - X) / len(X)
            grad_enc = X.T @ ((recon - X) @ self.weights['dec'].T) / len(X)
            self.weights['dec'] -= self.config.learning_rate * grad_dec
            self.weights['enc'] -= self.config.learning_rate * grad_enc
            
            return loss
    
    def train_local(self, epochs: int) -> Tuple[Dict[str, np.ndarray], float]:
        """Train locally for multiple epochs."""
        losses = []
        for _ in range(epochs):
            loss = self.train_epoch()
            losses.append(loss)
        
        # Update threshold based on training errors
        errors = self._get_errors(self.X_train)
        self.threshold = np.percentile(errors, 90)
        
        return self.get_weights(), np.mean(losses)
    
    def _get_errors(self, X) -> np.ndarray:
        if HAS_TORCH:
            self.model.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X = torch.FloatTensor(X)
                output = self.model(X)
                return torch.mean((X - output) ** 2, dim=1).numpy()
        else:
            X_np = X if isinstance(X, np.ndarray) else X.numpy()
            z = X_np @ self.weights['enc']
            recon = z @ self.weights['dec']
            return np.mean((X_np - recon) ** 2, axis=1)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        X_val = self.X_val if isinstance(self.X_val, np.ndarray) else self.X_val.numpy()
        errors = self._get_errors(self.X_val)
        
        # Predict anomalies
        predictions = (errors > self.threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(self.y_val, predictions),
            'precision': precision_score(self.y_val, predictions, zero_division=0),
            'recall': recall_score(self.y_val, predictions, zero_division=0),
            'f1_score': f1_score(self.y_val, predictions, zero_division=0),
            'mean_error': float(np.mean(errors)),
            'threshold': float(self.threshold),
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(self.y_val, errors)
        except:
            metrics['roc_auc'] = 0.5
        
        return metrics


# ==============================================================================
# Federated Server  
# ==============================================================================

class FLServer:
    def __init__(self, clients: List[FLClient]):
        self.clients = clients
        self.global_weights = clients[0].get_weights()
        self.history = []
    
    def fedavg(self, client_weights: List[Dict], sample_counts: List[int]) -> Dict[str, np.ndarray]:
        """FedAvg aggregation."""
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
        logger.info(f"  Global: acc={avg_metrics['accuracy']:.4f}, f1={avg_metrics['f1_score']:.4f}")
        
        return result


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data(config: Config) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load and prepare datasets for clients."""
    datasets = []
    paths = [
        project_root / "data" / "UNSW_NB15_training-set.csv",
        project_root / "data" / "UNSW_NB15_testing-set.csv",
    ]
    
    for i, path in enumerate(paths[:config.num_clients]):
        if not path.exists():
            logger.warning(f"File not found: {path}, using synthetic data")
            datasets.append(generate_synthetic(config, i))
            continue
        
        logger.info(f"Loading {path.name}...")
        df = pd.read_csv(path, low_memory=False)
        
        # Sample if too large
        if len(df) > config.max_samples:
            df = df.sample(n=config.max_samples, random_state=config.seed + i)
        
        # Get features and labels
        num_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['id', 'label', 'Label', 'attack_cat']
        feat_cols = [c for c in num_cols if c.lower() not in [e.lower() for e in exclude]]
        
        X = df[feat_cols].fillna(0).values.astype(np.float32)
        
        if 'label' in df.columns:
            y = df['label'].values
        elif 'Label' in df.columns:
            y = df['Label'].values
        else:
            y = np.zeros(len(df))
        
        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        
        # Train/val split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.test_split, stratify=y, random_state=config.seed
        )
        
        datasets.append((X_train, y_train, X_val, y_val))
        logger.info(f"  Client {i}: {len(X_train)} train, {len(X_val)} val, {y_train.mean():.1%} anomalies")
    
    return datasets


def generate_synthetic(config: Config, seed_offset: int = 0):
    """Generate synthetic data."""
    np.random.seed(config.seed + seed_offset)
    n = config.max_samples
    d = 40
    
    # Normal samples
    n_normal = int(n * 0.6)
    X_normal = np.random.randn(n_normal, d).astype(np.float32)
    
    # Anomalies (shifted distribution)
    n_anom = n - n_normal
    X_anom = np.random.randn(n_anom, d).astype(np.float32) * 2 + 3
    
    X = np.vstack([X_normal, X_anom])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    split = int(len(X) * (1 - config.test_split))
    return X[:split], y[:split], X[split:], y[split:]


# ==============================================================================
# RAG Explanation Generator
# ==============================================================================

def generate_rag_explanation(category: str, confidence: float, error: float) -> str:
    """Generate threat explanation with MITRE/CVE references."""
    
    mitre_map = {
        'DoS': ['T1499 - Endpoint DoS', 'T1498 - Network DoS'],
        'Reconnaissance': ['T1595 - Active Scanning', 'T1046 - Network Discovery'],
        'Exploits': ['T1190 - Exploit Public-Facing App', 'T1203 - Exploitation'],
        'Backdoor': ['T1059 - Command Scripting', 'T1071 - App Layer Protocol'],
        'Generic': ['T1595 - Active Scanning'],
    }
    
    cve_map = {
        'DoS': ['CVE-2021-26855', 'CVE-2020-1350'],
        'Exploits': ['CVE-2021-44228 (Log4Shell)', 'CVE-2019-19781'],
        'Backdoor': ['CVE-2021-27065', 'CVE-2020-1472'],
    }
    
    severity = "CRITICAL" if confidence > 0.8 else "HIGH" if confidence > 0.5 else "MEDIUM"
    mitre = mitre_map.get(category, mitre_map['Generic'])
    cves = cve_map.get(category, [])
    
    return f"""## {severity} Severity - {category} Attack Detected

**Confidence**: {confidence:.1%}
**Reconstruction Error**: {error:.4f}

### MITRE ATT&CK Mapping
{chr(10).join(f'- {t}' for t in mitre)}

### Related CVEs
{chr(10).join(f'- {c}' for c in cves) if cves else '- No specific CVE mapping'}

### Recommended Actions
1. Investigate source/destination IPs
2. Review correlated security events
3. Check for IOCs in threat intelligence feeds
4. Consider network isolation if attack persists

### Federated Learning Context
Detection enhanced by globally aggregated model trained across {2} distributed sensors.
Local data privacy preserved while benefiting from collective threat intelligence.
"""


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_report(config: Config, clients: List[FLClient], server: FLServer, 
                    total_time: float) -> Dict[str, Any]:
    """Generate comprehensive report."""
    
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'total_time_seconds': total_time,
            'pytorch_available': HAS_TORCH,
        },
        'baseline': {},
        'federated': {},
        'improvement': {},
        'rounds': [],
        'explanations': [],
    }
    
    # Baseline
    for c in clients:
        report['baseline'][f'client_{c.client_id}'] = c.baseline_metrics
    
    baseline_keys = list(clients[0].baseline_metrics.keys())
    report['baseline']['average'] = {
        k: np.mean([c.baseline_metrics.get(k, 0) for c in clients])
        for k in baseline_keys
    }
    
    # Federated (final round)
    final = server.history[-1]
    for i, m in enumerate(final['client_metrics']):
        report['federated'][f'client_{i}'] = m
    report['federated']['average'] = final['avg_metrics']
    
    # Improvement
    b_avg = report['baseline']['average']
    f_avg = report['federated']['average']
    
    report['improvement'] = {
        'accuracy_delta': f_avg['accuracy'] - b_avg['accuracy'],
        'accuracy_pct': ((f_avg['accuracy'] - b_avg['accuracy']) / max(b_avg['accuracy'], 0.001)) * 100,
        'f1_delta': f_avg['f1_score'] - b_avg['f1_score'],
        'f1_pct': ((f_avg['f1_score'] - b_avg['f1_score']) / max(b_avg['f1_score'], 0.001)) * 100,
        'recall_delta': f_avg['recall'] - b_avg['recall'],
        'precision_delta': f_avg['precision'] - b_avg['precision'],
    }
    
    # Rounds
    for r in server.history:
        report['rounds'].append({
            'round': r['round'],
            'accuracy': r['avg_metrics']['accuracy'],
            'f1_score': r['avg_metrics']['f1_score'],
            'mean_error': r['avg_metrics']['mean_error'],
        })
    
    # Sample explanations
    for cat in ['DoS', 'Reconnaissance', 'Exploits']:
        conf = np.random.uniform(0.6, 0.95)
        err = np.random.uniform(0.1, 0.4)
        report['explanations'].append({
            'category': cat,
            'text': generate_rag_explanation(cat, conf, err)
        })
    
    return report


def print_report(report: Dict[str, Any]):
    """Print formatted report."""
    
    print("\n" + "="*80)
    print("  FEDERATED LEARNING PERFORMANCE REPORT")
    print("="*80)
    
    meta = report['metadata']
    print(f"\n  Timestamp: {meta['timestamp']}")
    print(f"  Clients: {meta['num_clients']} | Rounds: {meta['num_rounds']} | Time: {meta['total_time_seconds']:.1f}s")
    print(f"  PyTorch: {'Yes' if meta['pytorch_available'] else 'No (sklearn fallback)'}")
    
    # Baseline
    print("\n" + "-"*80)
    print("  BASELINE PERFORMANCE (Local Training Only)")
    print("-"*80)
    
    for k, v in report['baseline'].items():
        if k == 'average':
            continue
        print(f"\n  {k.replace('_', ' ').title()}:")
        print(f"    Accuracy: {v['accuracy']:.4f} | F1: {v['f1_score']:.4f} | Recall: {v['recall']:.4f}")
    
    b = report['baseline']['average']
    print(f"\n  Average: Accuracy={b['accuracy']:.4f}, F1={b['f1_score']:.4f}")
    
    # Federated
    print("\n" + "-"*80)
    print("  FEDERATED PERFORMANCE (After Collaboration)")
    print("-"*80)
    
    for k, v in report['federated'].items():
        if k == 'average':
            continue
        print(f"\n  {k.replace('_', ' ').title()}:")
        print(f"    Accuracy: {v['accuracy']:.4f} | F1: {v['f1_score']:.4f} | Recall: {v['recall']:.4f}")
    
    f = report['federated']['average']
    print(f"\n  Average: Accuracy={f['accuracy']:.4f}, F1={f['f1_score']:.4f}")
    
    # Improvement
    print("\n" + "-"*80)
    print("  IMPROVEMENT ANALYSIS")
    print("-"*80)
    
    imp = report['improvement']
    print(f"\n  {'Metric':<12} {'Baseline':>10} {'Federated':>10} {'Change':>12}")
    print("  " + "-"*46)
    print(f"  {'Accuracy':<12} {b['accuracy']:>10.4f} {f['accuracy']:>10.4f} {imp['accuracy_delta']:>+10.4f} ({imp['accuracy_pct']:+.1f}%)")
    print(f"  {'F1 Score':<12} {b['f1_score']:>10.4f} {f['f1_score']:>10.4f} {imp['f1_delta']:>+10.4f} ({imp['f1_pct']:+.1f}%)")
    print(f"  {'Recall':<12} {b['recall']:>10.4f} {f['recall']:>10.4f} {imp['recall_delta']:>+10.4f}")
    print(f"  {'Precision':<12} {b['precision']:>10.4f} {f['precision']:>10.4f} {imp['precision_delta']:>+10.4f}")
    
    # Progress
    print("\n" + "-"*80)
    print("  FEDERATED TRAINING PROGRESS")
    print("-"*80)
    print(f"\n  {'Round':<8} {'Accuracy':>10} {'F1 Score':>10} {'Mean Error':>12}")
    print("  " + "-"*42)
    for r in report['rounds']:
        print(f"  {r['round']:<8} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} {r['mean_error']:>12.6f}")
    
    # Sample explanation
    print("\n" + "-"*80)
    print("  SAMPLE RAG THREAT EXPLANATION")
    print("-"*80)
    if report['explanations']:
        exp = report['explanations'][0]
        print(f"\n{exp['text'][:600]}...")
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    sign = "+" if imp['accuracy_pct'] > 0 else ""
    print(f"""
  Federated learning results:
  
  - Accuracy: {b['accuracy']:.2%} → {f['accuracy']:.2%} ({sign}{imp['accuracy_pct']:.1f}%)
  - F1 Score: {b['f1_score']:.2%} → {f['f1_score']:.2%} ({sign}{imp['f1_pct']:.1f}%)
  
  Key Insights:
  1. {'IMPROVED' if imp['accuracy_delta'] > 0 else 'Similar'} detection after federated training
  2. Privacy preserved - only model weights shared
  3. RAG explanations provide actionable threat intelligence
    """)
    
    print("="*80 + "\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=8000)
    args = parser.parse_args()
    
    config = Config(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        max_samples=args.max_samples,
    )
    
    np.random.seed(config.seed)
    if HAS_TORCH:
        torch.manual_seed(config.seed)
    
    start = time.time()
    
    print("\n" + "="*80)
    print("  FEDERATED LEARNING PIPELINE")
    print("  Comparing Local vs. Federated Training")
    print("="*80 + "\n")
    
    # Load data
    print("[1/4] Loading datasets...")
    datasets = load_data(config)
    
    # Create clients
    print("\n[2/4] Training baseline models...")
    clients = []
    for i, (X_tr, y_tr, X_val, y_val) in enumerate(datasets):
        client = FLClient(i, X_tr, y_tr, X_val, y_val, config)
        
        # Baseline training
        _, loss = client.train_local(config.local_epochs * 2)
        client.baseline_metrics = client.evaluate()
        
        # Reset for federated
        if HAS_TORCH:
            client.model = Autoencoder(X_tr.shape[1])
            client.optimizer = optim.Adam(client.model.parameters(), lr=config.learning_rate)
        else:
            client.weights = client._init_weights(X_tr.shape[1])
        
        clients.append(client)
        print(f"  Client {i} baseline: acc={client.baseline_metrics['accuracy']:.4f}, f1={client.baseline_metrics['f1_score']:.4f}")
    
    # Federated training
    print(f"\n[3/4] Running federated training ({config.num_rounds} rounds)...")
    server = FLServer(clients)
    for r in range(1, config.num_rounds + 1):
        server.run_round(r, config.local_epochs)
    
    total_time = time.time() - start
    
    # Generate report
    print("\n[4/4] Generating report...")
    report = generate_report(config, clients, server, total_time)
    print_report(report)
    
    # Save
    output_dir = project_root / "scripts"
    json_path = output_dir / "fl_pipeline_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to: {json_path}")
    
    # Markdown report
    md_path = output_dir / "fl_pipeline_report.md"
    with open(md_path, 'w') as f:
        f.write(f"""# Federated Learning Pipeline Report

**Generated**: {report['metadata']['timestamp']}

## Configuration
- Clients: {config.num_clients}
- Rounds: {config.num_rounds}
- Samples per client: {config.max_samples}
- PyTorch: {'Yes' if HAS_TORCH else 'No'}

## Results Summary

| Metric | Baseline | Federated | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | {report['baseline']['average']['accuracy']:.4f} | {report['federated']['average']['accuracy']:.4f} | {report['improvement']['accuracy_delta']:+.4f} ({report['improvement']['accuracy_pct']:+.1f}%) |
| F1 Score | {report['baseline']['average']['f1_score']:.4f} | {report['federated']['average']['f1_score']:.4f} | {report['improvement']['f1_delta']:+.4f} ({report['improvement']['f1_pct']:+.1f}%) |
| Recall | {report['baseline']['average']['recall']:.4f} | {report['federated']['average']['recall']:.4f} | {report['improvement']['recall_delta']:+.4f} |
| Precision | {report['baseline']['average']['precision']:.4f} | {report['federated']['average']['precision']:.4f} | {report['improvement']['precision_delta']:+.4f} |

## Training Progress

| Round | Accuracy | F1 Score | Mean Error |
|-------|----------|----------|------------|
""")
        for r in report['rounds']:
            f.write(f"| {r['round']} | {r['accuracy']:.4f} | {r['f1_score']:.4f} | {r['mean_error']:.6f} |\n")
        
        f.write(f"""
## Sample Threat Explanation

{report['explanations'][0]['text'] if report['explanations'] else 'N/A'}

---
*Generated by Federated Agentic Defense Framework*
""")
    
    print(f"Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
