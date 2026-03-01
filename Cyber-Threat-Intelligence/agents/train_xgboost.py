"""
Training Script for Agent Two's XGBoost Classifier.

This script handles the complete training pipeline for the threat
classification model:
    1. Load and preprocess UNSW-NB15 data
    2. Train optimized XGBoost classifier
    3. Evaluate and save model
    4. Optionally initialize vector DB with attack knowledge

Usage:
    python -m agents.train_xgboost --data-path data/UNSW_NB15_training-set.csv

Arguments:
    --data-path: Path to UNSW-NB15 CSV file
    --output-dir: Directory to save trained model
    --epochs: Number of boosting rounds (default: 200)
    --learning-rate: XGBoost learning rate (default: 0.1)
    --max-depth: Maximum tree depth (default: 8)
    --zero-day-threshold: Confidence threshold for zero-day (default: 0.5)
    --init-vector-db: Initialize vector DB with attack knowledge
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(
    data_path: str,
    test_split: float = 0.2,
    val_split: float = 0.1,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Loads and preprocesses UNSW-NB15 data for multi-class classification.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    from data_pipeline import DataLoader, DatasetConfig, Preprocessor
    from data_pipeline.batch_generator import DataSplitter
    
    print("[1/4] Loading data...")
    
    config = DatasetConfig(
        normalization_method="minmax",
        test_split_ratio=test_split,
        validation_split_ratio=val_split,
        random_seed=random_seed,
    )
    
    # Load data
    loader = DataLoader(config)
    loader.load(data_path).clean()
    
    # Get features and ATTACK CATEGORY labels (not binary)
    X, y = loader.get_features_and_labels(label_type="multiclass")
    print(f"   Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"   Attack categories: {np.unique(y)}")
    
    # Clean category labels (handle NaN/empty)
    y = np.array([str(label).strip() if label and str(label).strip() else "Normal" for label in y])
    
    print("\n[2/4] Preprocessing features...")
    
    # Save feature names before preprocessing
    feature_names = list(X.columns)
    
    # Preprocess (y=None returns only X_processed)
    preprocessor = Preprocessor(config)
    X_processed = preprocessor.fit_transform(X, y=None, categorical_encoding="label")
    print(f"   Preprocessed features shape: {X_processed.shape}")
    
    print("\n[3/4] Splitting data...")
    
    # Split data
    splitter = DataSplitter(
        test_ratio=test_split,
        val_ratio=val_split,
        random_seed=random_seed,
    )
    splits = splitter.split(X_processed, y, stratify=True)
    
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n   Class distribution (train):")
    for cat, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"     {cat}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    n_estimators: int = 200,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    zero_day_threshold: float = 0.5,
    use_gpu: bool = False,
) -> "ThreatClassifier":
    """Trains the XGBoost classifier."""
    from agents.models.xgboost_classifier import ThreatClassifier, summary
    
    print("\n[4/4] Training XGBoost classifier...")
    
    classifier = ThreatClassifier(
        zero_day_threshold=zero_day_threshold,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_gpu=use_gpu,
    )
    
    print(summary(classifier))
    
    # Train
    classifier.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        feature_names=feature_names,
        early_stopping_rounds=20,
        verbose=True,
    )
    
    return classifier


def evaluate_classifier(
    classifier: "ThreatClassifier",
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluates the classifier on test data."""
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    results = classifier.evaluate(X_test, y_test)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print("\nPer-class Metrics:")
    print("-" * 50)
    
    for cat, metrics in sorted(
        results['per_class_metrics'].items(),
        key=lambda x: x[1].get('support', 0),
        reverse=True
    ):
        if metrics['support'] > 0:
            print(f"  {cat:15s}: P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
                  f"Support={metrics['support']}")
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    return results


def initialize_attack_knowledge_base(output_dir: Path) -> None:
    """Initializes the vector database with attack knowledge."""
    print("\n" + "=" * 60)
    print("Initializing Attack Knowledge Base")
    print("=" * 60)
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from agents.interfaces import FAISSVectorDB
    except ImportError:
        print("   Skipping - langchain or sentence-transformers not installed")
        return
    
    # Attack knowledge documents
    attack_documents = [
        # DoS/DDoS
        {
            "text": """Denial of Service (DoS) Attack: A DoS attack aims to make a service unavailable by overwhelming it with traffic. Key indicators include: extremely high packet rates, abnormal connection patterns, repeated SYN packets without completing handshake (SYN flood), amplification through DNS/NTP/SSDP reflectors. Mitigation: rate limiting, traffic filtering, blackholing, CDN protection.""",
            "metadata": {"category": "DoS", "severity": "high"}
        },
        {
            "text": """SYN Flood Attack: A type of DoS where attacker sends many SYN packets but never completes TCP handshake. Signs: high number of half-open connections, source IPs may be spoofed, server resources exhausted. Look for: high spkts with low completed connections, abnormal sttl values, state=SYN or REQ.""",
            "metadata": {"category": "DoS", "severity": "high"}
        },
        # Reconnaissance
        {
            "text": """Port Scanning Attack: Reconnaissance technique to discover open ports and services. Types: TCP SYN scan (half-open), TCP connect scan, UDP scan, XMAS scan, NULL scan. Indicators: many connections to different ports from same source, short duration connections, specific flag combinations. Precursor to targeted attacks.""",
            "metadata": {"category": "Reconnaissance", "severity": "medium"}
        },
        {
            "text": """Network Reconnaissance: Attackers gather information about network topology, services, and vulnerabilities. Methods: ping sweeps, traceroute, DNS enumeration, banner grabbing. Look for: sequential IP connections, multiple service probes, unusual DNS queries. Defense: minimize information disclosure, use IDS.""",
            "metadata": {"category": "Reconnaissance", "severity": "medium"}
        },
        # Exploits
        {
            "text": """Exploit Attack: Leverages software vulnerabilities to gain unauthorized access. Common types: buffer overflow, SQL injection, command injection, path traversal, deserialization attacks. Network indicators: unusual payloads, shellcode signatures, abnormal service behavior, unexpected response sizes.""",
            "metadata": {"category": "Exploits", "severity": "high"}
        },
        {
            "text": """Web Application Exploit: Attacks targeting web services. SQL injection visible through unusual query parameters, XSS through script tags in requests, path traversal via ../ sequences. Look for: unusual response_body_len, high ct_flw_http_mthd, abnormal trans_depth. May lead to data theft or RCE.""",
            "metadata": {"category": "Exploits", "severity": "high"}
        },
        # Backdoors
        {
            "text": """Backdoor Attack: Malicious code providing persistent unauthorized access. Often installed after initial compromise. Signs: unexpected outbound connections (C2), unusual ports, encrypted traffic to unknown hosts, periodic beaconing patterns. Critical severity - indicates successful compromise.""",
            "metadata": {"category": "Backdoors", "severity": "critical"}
        },
        {
            "text": """Command and Control (C2) Traffic: Communication between compromised host and attacker server. Patterns: regular interval connections (beaconing), DNS tunneling, HTTP/HTTPS to unusual domains, encoded payloads. Detection: baseline traffic analysis, DNS monitoring, TLS inspection.""",
            "metadata": {"category": "Backdoors", "severity": "critical"}
        },
        # Shellcode
        {
            "text": """Shellcode Attack: Machine code injected to execute commands on target. Typically delivered via exploits. Indicators: NOP sleds in payloads, unusual byte patterns, execution of /bin/sh or cmd.exe, memory corruption signs. Often precedes full system compromise.""",
            "metadata": {"category": "Shellcode", "severity": "critical"}
        },
        # Worms
        {
            "text": """Worm Attack: Self-propagating malware spreading across networks. Signs: one infected host scanning/attacking others, rapid increase in similar traffic patterns, exploitation of same vulnerability across hosts. Famous examples: WannaCry, NotPetya, Conficker. Requires immediate network isolation.""",
            "metadata": {"category": "Worms", "severity": "critical"}
        },
        # Fuzzers
        {
            "text": """Fuzzing Attack: Sending malformed or random data to find vulnerabilities. Identified by: high volume of requests, random/unusual parameter values, incremental variations in payloads, application errors/crashes. May precede exploit development.""",
            "metadata": {"category": "Fuzzers", "severity": "medium"}
        },
        # Analysis
        {
            "text": """Traffic Analysis Attack: Gathering intelligence from network traffic patterns without accessing content. Methods: timing analysis, volume analysis, correlation attacks. Even encrypted traffic can reveal: communication partners, message sizes, timing patterns. Defense: traffic padding, mixing networks.""",
            "metadata": {"category": "Analysis", "severity": "medium"}
        },
        # Generic
        {
            "text": """Generic Network Attack: Attacks not fitting specific categories. May involve multiple techniques or novel methods. Signs: anomalous traffic patterns, unusual protocol behavior, suspicious timing sequences. Requires detailed analysis and human investigation to classify properly.""",
            "metadata": {"category": "Generic", "severity": "medium"}
        },
        # Zero-day indicators
        {
            "text": """Zero-Day Attack Indicators: Novel attacks exploiting unknown vulnerabilities. Warning signs: traffic patterns don't match known signatures, low confidence from ML models, unusual service/port combinations, suspicious but unclassifiable behavior. Requires careful analysis and may indicate APT activity.""",
            "metadata": {"category": "Unknown/Zero-day", "severity": "high"}
        },
    ]
    
    print(f"   Loading {len(attack_documents)} attack knowledge documents...")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and populate vector DB
    kb_path = output_dir / "knowledge_base"
    db = FAISSVectorDB(embeddings, persist_directory=str(kb_path))
    
    texts = [doc["text"] for doc in attack_documents]
    metadatas = [doc["metadata"] for doc in attack_documents]
    
    db.add_documents(texts, metadatas)
    db.persist()
    
    print(f"   Knowledge base saved to: {kb_path}")
    print(f"   Total documents: {db.count}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Agent Two's XGBoost classifier"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/UNSW_NB15_training-set.csv",
        help="Path to UNSW-NB15 CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/agent_two",
        help="Directory to save model",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument(
        "--zero-day-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for zero-day detection",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for training",
    )
    parser.add_argument(
        "--init-vector-db",
        action="store_true",
        help="Initialize vector DB with attack knowledge",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XGBoost Classifier Training Pipeline")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    (X_train, y_train, X_val, y_val, 
     X_test, y_test, feature_names) = load_and_preprocess_data(args.data_path)
    
    # Train classifier
    classifier = train_classifier(
        X_train, y_train,
        X_val, y_val,
        feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        zero_day_threshold=args.zero_day_threshold,
        use_gpu=args.use_gpu,
    )
    
    # Evaluate
    results = evaluate_classifier(classifier, X_test, y_test)
    
    # Save model
    classifier_path = output_path / "classifier"
    classifier.save(classifier_path)
    print(f"\n   Classifier saved to: {classifier_path}")
    
    # Save feature names
    with open(output_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    
    # Save evaluation results
    with open(output_path / "evaluation_results.json", "w") as f:
        # Remove non-serializable items
        results_serializable = {
            k: v for k, v in results.items() 
            if k != 'classification_report'
        }
        json.dump(results_serializable, f, indent=2)
    
    # Initialize vector DB if requested
    if args.init_vector_db:
        initialize_attack_knowledge_base(output_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {output_path}")
    print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
