"""
Full Agent Federated Learning Simulation with Groq LLM.

This script demonstrates the complete pipeline:
1. Agent One: Autoencoder-based anomaly detection
2. Agent Two: XGBoost classification + RAG with Groq LLM
3. Federated Learning: 2 distributed clients with UNSW-NB15 data
4. RAG Explanations: Using Groq's free LLM API for threat intelligence

The simulation shows:
- Baseline performance (local training only)
- Federated performance (after collaboration)
- Real LLM-generated threat explanations
- Accuracy improvements from federation

Usage:
    python scripts/full_agent_groq_simulation.py --num-rounds 10
    
Requirements:
    - GROQ_API_KEY in .env file
    - UNSW-NB15 dataset in data/ folder
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("FullAgentFL")

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - using simplified autoencoder")


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class SimulationConfig:
    """Configuration for full agent FL simulation."""
    num_clients: int = 2
    num_rounds: int = 10
    local_epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.002
    max_samples: int = 10000
    test_split: float = 0.2
    seed: int = 42
    
    # Thresholds
    anomaly_percentile: float = 90
    zero_day_threshold: float = 0.4
    
    # LLM Settings
    use_groq: bool = True
    groq_model: str = "llama-3.3-70b-versatile"
    max_rag_samples: int = 5  # Number of samples to explain
    

# ==============================================================================
# Autoencoder Model for Agent One
# ==============================================================================

if HAS_TORCH:
    class Autoencoder(nn.Module):
        """Autoencoder for anomaly detection (Agent One)."""
        
        def __init__(self, input_dim: int, latent_dim: int = 16):
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
                self.eval()
                output = self.forward(x)
                errors = torch.mean((x - output) ** 2, dim=1)
                return errors.numpy()


# ==============================================================================
# Groq LLM Integration
# ==============================================================================

class GroqThreatAnalyzer:
    """
    RAG-based threat analysis using Groq LLM.
    
    Generates human-readable threat explanations with:
    - MITRE ATT&CK mappings
    - CVE references
    - Recommended actions
    - Federated learning context
    """
    
    SYSTEM_PROMPT = """You are a cybersecurity expert specializing in network intrusion detection. 
Analyze the detected threat and provide a structured security report.

Your analysis should include:
1. Threat classification and confidence
2. MITRE ATT&CK technique mapping
3. Related CVEs if applicable
4. Severity assessment
5. Recommended mitigation actions
6. Key indicators that support the classification

Be concise but thorough. Use technical terminology appropriately."""

    ATTACK_CONTEXT = {
        "DoS": "Denial of Service attacks attempt to overwhelm network resources, causing service disruption. Related MITRE: T1498, T1499.",
        "Reconnaissance": "Network scanning and enumeration to identify potential targets. Related MITRE: T1595, T1592.",
        "Exploits": "Active exploitation of known vulnerabilities. Related MITRE: T1190, T1203.",
        "Backdoor": "Persistent unauthorized access mechanisms. Related MITRE: T1071, T1105.",
        "Fuzzers": "Automated testing/fuzzing to find vulnerabilities. Related MITRE: T1046.",
        "Generic": "General anomalous traffic patterns not matching specific categories.",
        "Analysis": "Suspicious analysis or data collection activity. Related MITRE: T1040, T1083.",
        "Shellcode": "Code injection or shellcode execution attempts. Related MITRE: T1059.",
        "Worms": "Self-propagating malicious code. Related MITRE: T1570.",
        "Normal": "Legitimate network traffic - no threat detected.",
    }
    
    def __init__(self, use_groq: bool = True, model: str = "llama-3.3-70b-versatile"):
        """Initialize the threat analyzer."""
        self.use_groq = use_groq
        self.model = model
        self._client = None
        
        if use_groq:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not found - falling back to mock responses")
                self.use_groq = False
            else:
                try:
                    from groq import Groq
                    self._client = Groq(api_key=api_key)
                    logger.info(f"Groq LLM initialized: {model}")
                except ImportError:
                    logger.warning("groq package not installed - falling back to mock")
                    self.use_groq = False
    
    def analyze_threat(
        self,
        features: np.ndarray,
        attack_category: str,
        classification_confidence: float,
        reconstruction_error: float,
        is_zero_day: bool,
        federated_round: int,
        num_clients: int,
    ) -> str:
        """
        Generate a comprehensive threat analysis.
        
        Args:
            features: Network flow features
            attack_category: Predicted attack type
            classification_confidence: Model confidence
            reconstruction_error: Autoencoder reconstruction error
            is_zero_day: Whether this is a potential zero-day
            federated_round: Current federated learning round
            num_clients: Number of federated clients
        
        Returns:
            Human-readable threat analysis
        """
        # Build the analysis prompt
        context = self.ATTACK_CONTEXT.get(attack_category, "Unknown attack pattern")
        
        prompt = f"""Analyze this network security threat:

## Detection Results
- **Attack Category**: {attack_category}
- **Classification Confidence**: {classification_confidence:.1%}
- **Anomaly Score**: {reconstruction_error:.4f}
- **Potential Zero-day**: {'Yes' if is_zero_day else 'No'}

## Attack Context
{context}

## Federated Learning Context
- Training data from {num_clients} distributed organizations
- Global model aggregated over {federated_round} federation rounds
- Cross-organizational threat intelligence enabled

## Key Feature Indicators
- Mean feature value: {np.mean(features):.4f}
- Max feature value: {np.max(features):.4f}
- Feature variance: {np.var(features):.4f}

Provide a structured threat analysis report with:
1. Threat Classification
2. MITRE ATT&CK Mapping
3. Severity Assessment  
4. Indicators of Compromise
5. Recommended Actions
"""
        
        if self.use_groq and self._client:
            return self._groq_generate(prompt)
        else:
            return self._mock_generate(attack_category, classification_confidence, 
                                       reconstruction_error, federated_round, num_clients)
    
    def _groq_generate(self, prompt: str) -> str:
        """Generate using Groq API."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _mock_generate(
        self,
        attack_category: str,
        confidence: float,
        error: float,
        federated_round: int,
        num_clients: int,
    ) -> str:
        """Generate mock response for testing."""
        severity = "CRITICAL" if attack_category in ["Backdoor", "Exploits", "Shellcode", "Worms"] else \
                   "HIGH" if attack_category in ["DoS"] else \
                   "MEDIUM" if attack_category in ["Reconnaissance", "Fuzzers", "Analysis"] else "LOW"
        
        mitre_map = {
            "DoS": ("T1498/T1499", "Network/Endpoint Denial of Service"),
            "Reconnaissance": ("T1595", "Active Scanning"),
            "Exploits": ("T1190", "Exploit Public-Facing Application"),
            "Backdoor": ("T1071", "Application Layer Protocol C2"),
            "Fuzzers": ("T1046", "Network Service Discovery"),
            "Analysis": ("T1040", "Network Sniffing"),
            "Shellcode": ("T1059", "Command and Scripting Interpreter"),
            "Worms": ("T1570", "Lateral Tool Transfer"),
            "Generic": ("Multiple", "Various techniques"),
            "Normal": ("N/A", "No threat"),
        }
        
        technique_id, technique_name = mitre_map.get(attack_category, ("Unknown", "Unknown"))
        
        return f"""## RAG-ENHANCED THREAT ANALYSIS

### 1. THREAT CLASSIFICATION
- **Attack Type**: {attack_category}
- **MITRE ATT&CK**: {technique_id} - {technique_name}
- **Confidence**: {confidence:.1%}

### 2. SEVERITY ASSESSMENT
- **Level**: {severity}
- **Anomaly Score**: {error:.4f}
- **Risk Factor**: {'Elevated - potential zero-day' if confidence < 0.7 else 'Known threat pattern'}

### 3. INDICATORS OF COMPROMISE
- Abnormal reconstruction error detected
- Traffic pattern matches {attack_category} signatures
- Cross-organizational correlation: Similar patterns detected

### 4. FEDERATED LEARNING INSIGHTS
- Global model trained on data from {num_clients} organizations
- Federation round: {federated_round}
- Model benefits from distributed threat intelligence

### 5. RECOMMENDED ACTIONS
1. {'IMMEDIATE: Block source IP and enable DDoS mitigation' if severity == 'CRITICAL' else 'MONITOR: Continue observation and logging'}
2. Review correlated security events
3. Update detection signatures based on new patterns
4. Share indicators with federated partners

---
*Generated: {datetime.now().isoformat()}*
*Model: {'Groq ' + self.model if self.use_groq else 'Mock LLM'}*
"""


# ==============================================================================
# Federated Learning Client
# ==============================================================================

class FederatedAgentClient:
    """
    Federated learning client with full agent pipeline.
    
    Combines:
    - Agent One: Autoencoder for anomaly detection
    - Agent Two: XGBoost for attack classification
    """
    
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_train_binary: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_val_binary: np.ndarray,
        config: SimulationConfig,
        label_encoder: LabelEncoder,
    ):
        """Initialize client with data and models."""
        self.client_id = client_id
        self.config = config
        self.label_encoder = label_encoder
        
        # Store labels
        self.y_train = y_train  # Multi-class
        self.y_train_binary = y_train_binary  # Binary (normal/anomaly)
        self.y_val = y_val
        self.y_val_binary = y_val_binary
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train) if HAS_TORCH else X_train
        self.X_val = torch.FloatTensor(X_val) if HAS_TORCH else X_val
        
        input_dim = X_train.shape[1]
        
        # Agent One: Autoencoder
        if HAS_TORCH:
            self.autoencoder = Autoencoder(input_dim, latent_dim=16)
            self.ae_optimizer = optim.Adam(
                self.autoencoder.parameters(),
                lr=config.learning_rate,
            )
            self.ae_criterion = nn.MSELoss()
        
        # Agent Two: XGBoost Classifier
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=config.seed,
        )
        self._clf_trained = False
        
        # Thresholds
        self.anomaly_threshold = 0.1
        
        # Metrics storage
        self.baseline_metrics = {}
        self.federated_metrics = {}
    
    def get_ae_weights(self) -> Dict[str, np.ndarray]:
        """Get autoencoder weights for federation."""
        if HAS_TORCH:
            return {k: v.cpu().numpy().copy() 
                    for k, v in self.autoencoder.state_dict().items()}
        return {}
    
    def set_ae_weights(self, weights: Dict[str, np.ndarray]):
        """Set autoencoder weights from server."""
        if HAS_TORCH:
            state_dict = {}
            for k, v in weights.items():
                if np.isscalar(v) or v.ndim == 0:
                    state_dict[k] = torch.tensor(float(v))
                else:
                    state_dict[k] = torch.FloatTensor(v)
            self.autoencoder.load_state_dict(state_dict)
    
    def train_autoencoder_epoch(self) -> float:
        """Train autoencoder for one epoch."""
        if not HAS_TORCH:
            return 0.0
        
        self.autoencoder.train()
        dataset = TensorDataset(self.X_train)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        total_loss = 0
        for batch in loader:
            x = batch[0]
            self.ae_optimizer.zero_grad()
            output = self.autoencoder(x)
            loss = self.ae_criterion(output, x)
            loss.backward()
            self.ae_optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def train_local(self, epochs: int) -> Tuple[Dict[str, np.ndarray], float]:
        """Train both agents locally."""
        # Train autoencoder
        losses = []
        for _ in range(epochs):
            loss = self.train_autoencoder_epoch()
            losses.append(loss)
        
        # Compute anomaly threshold
        self._compute_threshold()
        
        # Train XGBoost classifier
        self._train_classifier()
        
        return self.get_ae_weights(), np.mean(losses)
    
    def _compute_threshold(self):
        """Compute anomaly detection threshold."""
        if not HAS_TORCH:
            return
        
        self.autoencoder.eval()
        errors = self.autoencoder.get_reconstruction_error(self.X_train)
        self.anomaly_threshold = np.percentile(errors, self.config.anomaly_percentile)
    
    def _train_classifier(self):
        """Train XGBoost classifier."""
        if not HAS_TORCH:
            return
        
        self.autoencoder.eval()
        X_train_np = self.X_train.numpy() if HAS_TORCH else self.X_train
        
        # Only train on detected anomalies for better classification
        self.classifier.fit(X_train_np, self.y_train)
        self._clf_trained = True
    
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using autoencoder (Agent One)."""
        if not HAS_TORCH:
            return np.zeros(len(X))
        
        self.autoencoder.eval()
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X
        
        errors = self.autoencoder.get_reconstruction_error(X_tensor)
        return (errors > self.anomaly_threshold).astype(int)
    
    def classify_attacks(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify attack types using XGBoost (Agent Two)."""
        if not self._clf_trained:
            return np.zeros(len(X)), np.zeros(len(X))
        
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate full pipeline on validation set."""
        X_val_np = self.X_val.numpy() if HAS_TORCH else self.X_val
        
        # Agent One: Anomaly Detection
        anomaly_predictions = self.detect_anomalies(X_val_np)
        
        # Agent Two: Attack Classification
        class_predictions, confidences = self.classify_attacks(X_val_np)
        
        # Compute metrics
        metrics = {
            # Binary anomaly detection metrics
            'binary_accuracy': accuracy_score(self.y_val_binary, anomaly_predictions),
            'binary_precision': precision_score(self.y_val_binary, anomaly_predictions, zero_division=0),
            'binary_recall': recall_score(self.y_val_binary, anomaly_predictions, zero_division=0),
            'binary_f1': f1_score(self.y_val_binary, anomaly_predictions, zero_division=0),
            
            # Multi-class classification metrics
            'classification_accuracy': accuracy_score(self.y_val, class_predictions),
            'classification_f1_macro': f1_score(self.y_val, class_predictions, average='macro', zero_division=0),
            
            # Other
            'mean_confidence': float(np.mean(confidences)),
            'anomaly_threshold': float(self.anomaly_threshold),
        }
        
        try:
            metrics['binary_auc'] = roc_auc_score(
                self.y_val_binary, 
                self.autoencoder.get_reconstruction_error(self.X_val) if HAS_TORCH else np.zeros(len(self.y_val))
            )
        except:
            metrics['binary_auc'] = 0.5
        
        return metrics
    
    def get_sample_for_explanation(self) -> Tuple[np.ndarray, str, float, float]:
        """Get a sample for RAG explanation."""
        if not HAS_TORCH:
            return None, "Unknown", 0.0, 0.0
        
        X_val_np = self.X_val.numpy()
        
        # Find an anomalous sample
        errors = self.autoencoder.get_reconstruction_error(self.X_val)
        anomaly_indices = np.where(errors > self.anomaly_threshold)[0]
        
        if len(anomaly_indices) == 0:
            # Use highest error sample
            idx = np.argmax(errors)
        else:
            # Random anomaly
            idx = np.random.choice(anomaly_indices)
        
        features = X_val_np[idx]
        error = errors[idx]
        
        # Get classification
        pred, conf = self.classify_attacks(features.reshape(1, -1))
        category = self.label_encoder.inverse_transform([int(pred[0])])[0]
        
        return features, category, float(conf[0]), float(error)


# ==============================================================================
# Federated Learning Server
# ==============================================================================

class FederatedAgentServer:
    """Federated server for aggregating model weights."""
    
    def __init__(self, clients: List[FederatedAgentClient], config: SimulationConfig):
        """Initialize server with clients."""
        self.clients = clients
        self.config = config
        self.global_weights = clients[0].get_ae_weights()
        self.history = []
    
    def fedavg(self, client_weights: List[Dict], sample_counts: List[int]) -> Dict[str, np.ndarray]:
        """Federated averaging of model weights."""
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
        """Run one federated learning round."""
        logger.info(f"=== Round {round_num} ===")
        
        # Distribute global weights
        for client in self.clients:
            client.set_ae_weights(self.global_weights)
        
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
        
        # Aggregate weights
        self.global_weights = self.fedavg(client_weights, client_samples)
        
        # Update clients with global weights and evaluate
        for client in self.clients:
            client.set_ae_weights(self.global_weights)
            client._compute_threshold()
            client._train_classifier()
        
        # Evaluate all clients
        metrics = [client.evaluate() for client in self.clients]
        avg_metrics = {
            key: np.mean([m[key] for m in metrics])
            for key in metrics[0]
        }
        
        result = {
            'round': round_num,
            'client_losses': client_losses,
            'avg_metrics': avg_metrics,
        }
        
        self.history.append(result)
        
        logger.info(
            f"  Global: binary_acc={avg_metrics['binary_accuracy']:.4f}, "
            f"clf_acc={avg_metrics['classification_accuracy']:.4f}"
        )
        
        return result


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data(config: SimulationConfig) -> Tuple[List[Tuple], LabelEncoder]:
    """Load UNSW-NB15 data for 2 clients."""
    datasets = []
    label_encoder = LabelEncoder()
    
    paths = [
        project_root / "data" / "UNSW_NB15_training-set.csv",
        project_root / "data" / "UNSW_NB15_testing-set.csv",
    ]
    
    all_categories = []
    
    # First pass: collect all attack categories
    for path in paths[:config.num_clients]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if 'attack_cat' in df.columns:
                all_categories.extend(df['attack_cat'].fillna('Normal').unique())
    
    # Fit label encoder on all categories
    label_encoder.fit(list(set(all_categories)))
    
    # Second pass: load data
    for i, path in enumerate(paths[:config.num_clients]):
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        
        logger.info(f"Loading {path.name}...")
        df = pd.read_csv(path, low_memory=False)
        
        # Sample if too large
        if len(df) > config.max_samples:
            df = df.sample(n=config.max_samples, random_state=config.seed + i)
        
        # Get feature columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['id', 'label', 'Label', 'attack_cat']
        feat_cols = [c for c in num_cols if c.lower() not in [e.lower() for e in exclude]]
        
        X = df[feat_cols].fillna(0).values.astype(np.float32)
        
        # Binary labels (normal=0, attack=1)
        if 'label' in df.columns:
            y_binary = df['label'].values
        elif 'Label' in df.columns:
            y_binary = df['Label'].values
        else:
            y_binary = np.zeros(len(df))
        
        # Multi-class labels
        if 'attack_cat' in df.columns:
            y_multiclass = label_encoder.transform(df['attack_cat'].fillna('Normal'))
        else:
            y_multiclass = np.zeros(len(df))
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        
        # Train/val split
        X_train, X_val, y_train, y_val, y_train_bin, y_val_bin = train_test_split(
            X, y_multiclass, y_binary,
            test_size=config.test_split,
            stratify=y_binary,
            random_state=config.seed,
        )
        
        datasets.append((X_train, y_train, y_train_bin, X_val, y_val, y_val_bin))
        
        anomaly_rate = y_train_bin.mean()
        logger.info(f"  Client {i}: {len(X_train)} train, {len(X_val)} val, {anomaly_rate:.1%} attacks")
    
    return datasets, label_encoder


# ==============================================================================
# Main Simulation
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Agent FL Simulation with Groq")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--no-groq", action="store_true", help="Use mock LLM instead of Groq")
    args = parser.parse_args()
    
    config = SimulationConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        max_samples=args.max_samples,
        use_groq=not args.no_groq,
    )
    
    np.random.seed(config.seed)
    if HAS_TORCH:
        torch.manual_seed(config.seed)
    
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("  FULL AGENT FEDERATED LEARNING SIMULATION")
    print("  Agent One: Autoencoder | Agent Two: XGBoost + Groq RAG")
    print("=" * 80 + "\n")
    
    # Initialize threat analyzer with Groq
    print("[1/6] Initializing RAG system...")
    analyzer = GroqThreatAnalyzer(
        use_groq=config.use_groq,
        model=config.groq_model,
    )
    
    # Load data
    print("\n[2/6] Loading UNSW-NB15 datasets...")
    datasets, label_encoder = load_data(config)
    
    if not datasets:
        print("No datasets found!")
        return
    
    # Create clients
    print("\n[3/6] Creating federated clients and training baselines...")
    clients = []
    
    for i, (X_tr, y_tr, y_tr_bin, X_val, y_val, y_val_bin) in enumerate(datasets):
        client = FederatedAgentClient(
            client_id=i,
            X_train=X_tr,
            y_train=y_tr,
            y_train_binary=y_tr_bin,
            X_val=X_val,
            y_val=y_val,
            y_val_binary=y_val_bin,
            config=config,
            label_encoder=label_encoder,
        )
        
        # Train baseline (local only)
        _, loss = client.train_local(config.local_epochs * 3)
        client.baseline_metrics = client.evaluate()
        
        # Reset model for federated training
        if HAS_TORCH:
            input_dim = X_tr.shape[1]
            client.autoencoder = Autoencoder(input_dim, latent_dim=16)
            client.ae_optimizer = optim.Adam(
                client.autoencoder.parameters(),
                lr=config.learning_rate,
            )
            client._clf_trained = False
        
        clients.append(client)
        
        b = client.baseline_metrics
        print(f"  Client {i} baseline: binary_acc={b['binary_accuracy']:.4f}, "
              f"clf_acc={b['classification_accuracy']:.4f}")
    
    # Federated training
    print(f"\n[4/6] Running federated training ({config.num_rounds} rounds)...")
    server = FederatedAgentServer(clients, config)
    
    for r in range(1, config.num_rounds + 1):
        server.run_round(r, config.local_epochs)
    
    # Store federated metrics
    for client in clients:
        client.federated_metrics = client.evaluate()
    
    # Calculate averages
    baseline_avg = {
        k: np.mean([c.baseline_metrics.get(k, 0) for c in clients])
        for k in clients[0].baseline_metrics
    }
    
    federated_avg = {
        k: np.mean([c.federated_metrics.get(k, 0) for c in clients])
        for k in clients[0].federated_metrics
    }
    
    # Generate RAG explanations
    print(f"\n[5/6] Generating RAG threat explanations ({config.max_rag_samples} samples)...")
    explanations = []
    
    for i in range(config.max_rag_samples):
        # Get sample from random client
        client = clients[i % len(clients)]
        features, category, confidence, error = client.get_sample_for_explanation()
        
        if features is None:
            continue
        
        print(f"  Analyzing sample {i+1}: {category} (confidence: {confidence:.1%})")
        
        explanation = analyzer.analyze_threat(
            features=features,
            attack_category=category,
            classification_confidence=confidence,
            reconstruction_error=error,
            is_zero_day=confidence < config.zero_day_threshold,
            federated_round=config.num_rounds,
            num_clients=config.num_clients,
        )
        
        explanations.append({
            'sample_id': i,
            'category': category,
            'confidence': confidence,
            'error': error,
            'explanation': explanation,
        })
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n### Baseline Performance (Local Training Only)")
    print(f"  Binary Accuracy:         {baseline_avg['binary_accuracy']:.4f}")
    print(f"  Classification Accuracy: {baseline_avg['classification_accuracy']:.4f}")
    print(f"  Binary F1:               {baseline_avg['binary_f1']:.4f}")
    
    print("\n### Federated Performance (After Collaboration)")
    print(f"  Binary Accuracy:         {federated_avg['binary_accuracy']:.4f}")
    print(f"  Classification Accuracy: {federated_avg['classification_accuracy']:.4f}")
    print(f"  Binary F1:               {federated_avg['binary_f1']:.4f}")
    
    # Calculate improvements
    bin_acc_imp = (federated_avg['binary_accuracy'] - baseline_avg['binary_accuracy']) / \
                  baseline_avg['binary_accuracy'] * 100 if baseline_avg['binary_accuracy'] > 0 else 0
    clf_acc_imp = (federated_avg['classification_accuracy'] - baseline_avg['classification_accuracy']) / \
                  baseline_avg['classification_accuracy'] * 100 if baseline_avg['classification_accuracy'] > 0 else 0
    
    print("\n### Improvements")
    print(f"  Binary Accuracy:         {bin_acc_imp:+.1f}%")
    print(f"  Classification Accuracy: {clf_acc_imp:+.1f}%")
    
    # Print sample explanation
    if explanations:
        print("\n" + "-" * 80)
        print("  SAMPLE RAG THREAT EXPLANATION")
        print("-" * 80)
        print(explanations[0]['explanation'])
    
    print("\n" + "=" * 80)
    print(f"  Total Time: {total_time:.1f}s")
    print(f"  LLM: {'Groq ' + config.groq_model if analyzer.use_groq else 'Mock'}")
    print("=" * 80 + "\n")
    
    # Save results
    results = {
        'config': {
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'local_epochs': config.local_epochs,
            'max_samples': config.max_samples,
            'use_groq': analyzer.use_groq,
        },
        'baseline': baseline_avg,
        'federated': federated_avg,
        'improvements': {
            'binary_accuracy_pct': bin_acc_imp,
            'classification_accuracy_pct': clf_acc_imp,
        },
        'training_history': [
            {'round': r['round'], 'metrics': r['avg_metrics']}
            for r in server.history
        ],
        'explanations': explanations,
        'total_time': total_time,
    }
    
    # Save JSON results
    output_path = project_root / "scripts" / "full_agent_groq_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    
    # Save markdown report
    md_path = project_root / "scripts" / "full_agent_groq_report.md"
    with open(md_path, 'w') as f:
        f.write(f"""# Full Agent Federated Learning Report with Groq LLM

## Configuration
- **Clients**: {config.num_clients}
- **Rounds**: {config.num_rounds}
- **Local Epochs**: {config.local_epochs}
- **Max Samples/Client**: {config.max_samples}
- **LLM**: {'Groq ' + config.groq_model if analyzer.use_groq else 'Mock'}

## Pipeline Architecture

1. **Agent One (Autoencoder)**: Anomaly detection via reconstruction error
2. **Agent Two (XGBoost)**: Multi-class attack classification (10 categories)
3. **RAG System**: Groq LLM for threat intelligence and explanations

## Results Summary

### Baseline (Local Training Only)
| Metric | Value |
|--------|-------|
| Binary Accuracy | {baseline_avg['binary_accuracy']:.4f} |
| Classification Accuracy | {baseline_avg['classification_accuracy']:.4f} |
| Binary F1 | {baseline_avg['binary_f1']:.4f} |

### Federated (After Collaboration)
| Metric | Value |
|--------|-------|
| Binary Accuracy | {federated_avg['binary_accuracy']:.4f} |
| Classification Accuracy | {federated_avg['classification_accuracy']:.4f} |
| Binary F1 | {federated_avg['binary_f1']:.4f} |

### Improvements
| Metric | Baseline | Federated | Change |
|--------|----------|-----------|--------|
| Binary Accuracy | {baseline_avg['binary_accuracy']:.4f} | {federated_avg['binary_accuracy']:.4f} | {bin_acc_imp:+.1f}% |
| Classification Accuracy | {baseline_avg['classification_accuracy']:.4f} | {federated_avg['classification_accuracy']:.4f} | {clf_acc_imp:+.1f}% |

## Training Progress

| Round | Binary Acc | Classification Acc |
|-------|------------|-------------------|
""")
        for r in server.history:
            m = r['avg_metrics']
            f.write(f"| {r['round']} | {m['binary_accuracy']:.4f} | {m['classification_accuracy']:.4f} |\n")
        
        f.write(f"""
## Sample RAG Explanation

{explanations[0]['explanation'] if explanations else 'No explanations generated'}

---
*Generated: {datetime.now().isoformat()}*
*Total Time: {total_time:.1f}s*
""")
    
    print(f"Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
