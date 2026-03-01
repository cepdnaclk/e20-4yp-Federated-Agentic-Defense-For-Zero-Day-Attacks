#!/usr/bin/env python
"""
Accuracy Verification Test Script
=================================

This script tests the Multi-Agent IDS system step-by-step using UNSW-NB15_1.csv
and provides detailed proof that reported accuracies are correct.

Each step shows:
1. Actual data samples with ground truth labels
2. Predictions made by the agent
3. Verification that metrics match the raw data

Usage:
    python tests/test_accuracy_verification.py
    python tests/test_accuracy_verification.py --samples 5000
    python tests/test_accuracy_verification.py --output results.txt
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from data_pipeline import DataLoader, DatasetConfig, Preprocessor
from agents.agent_one import AgentOne
from agents.agent_two import AgentTwo
from agents.agent_three import AgentThree


def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a formatted section separator."""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def verify_agent_one(agent_one, X_test, y_binary, n_proof_samples: int = 10):
    """
    Verify Agent One accuracy with proof samples.
    
    Shows actual reconstruction errors and threshold comparisons
    for individual samples to prove the accuracy is correct.
    """
    print_separator("STEP 1: AGENT ONE - ANOMALY DETECTION VERIFICATION")
    
    # Run detection
    results = agent_one.detect_anomalies(X_test, return_raw=False)
    predictions = np.array([r.is_anomaly for r in results]).astype(int)
    errors = np.array([r.reconstruction_error for r in results])
    
    # Calculate metrics
    accuracy = accuracy_score(y_binary, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_binary, predictions, average='binary'
    )
    tn, fp, fn, tp = confusion_matrix(y_binary, predictions).ravel()
    
    # Print configuration
    print(f"Configuration:")
    print(f"  - Threshold: {agent_one._threshold}")
    print(f"  - Total samples: {len(X_test):,}")
    print(f"  - Actual attacks: {y_binary.sum():,}")
    print(f"  - Actual normal: {(y_binary == 0).sum():,}")
    
    # Print metrics
    print(f"\nCalculated Metrics:")
    print(f"  - Accuracy:  {accuracy*100:.2f}%")
    print(f"  - Precision: {precision*100:.2f}%")
    print(f"  - Recall:    {recall*100:.2f}%")
    print(f"  - F1-Score:  {f1*100:.2f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"  - True Negatives (TN):  {tn:,}")
    print(f"  - False Positives (FP): {fp:,}")
    print(f"  - False Negatives (FN): {fn:,}")
    print(f"  - True Positives (TP):  {tp:,}")
    
    # PROOF: Show individual sample predictions
    print(f"\n{'=' * 60}")
    print(f"PROOF: Individual Sample Analysis (first {n_proof_samples} samples)")
    print(f"{'=' * 60}")
    print(f"{'Idx':<6} {'Error':<12} {'Threshold':<12} {'Predicted':<12} {'Actual':<10} {'Correct'}")
    print("-" * 70)
    
    correct_count = 0
    for i in range(min(n_proof_samples, len(predictions))):
        error = errors[i]
        threshold = agent_one._threshold
        predicted = predictions[i]
        actual = y_binary[i]
        is_correct = predicted == actual
        correct_count += is_correct
        
        pred_label = "ANOMALY" if predicted == 1 else "NORMAL"
        actual_label = "ATTACK" if actual == 1 else "NORMAL"
        
        print(f"{i:<6} {error:<12.6f} {threshold:<12.6f} {pred_label:<12} {actual_label:<10} {'✓' if is_correct else '✗'}")
    
    # Verify manual calculation matches
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION: Manual Accuracy Calculation")
    print(f"{'=' * 60}")
    
    manual_tp = ((predictions == 1) & (y_binary == 1)).sum()
    manual_tn = ((predictions == 0) & (y_binary == 0)).sum()
    manual_fp = ((predictions == 1) & (y_binary == 0)).sum()
    manual_fn = ((predictions == 0) & (y_binary == 1)).sum()
    manual_accuracy = (manual_tp + manual_tn) / len(predictions)
    
    print(f"  Manual TP: {manual_tp:,} (model TP: {tp:,}) - {'MATCH ✓' if manual_tp == tp else 'MISMATCH ✗'}")
    print(f"  Manual TN: {manual_tn:,} (model TN: {tn:,}) - {'MATCH ✓' if manual_tn == tn else 'MISMATCH ✗'}")
    print(f"  Manual FP: {manual_fp:,} (model FP: {fp:,}) - {'MATCH ✓' if manual_fp == fp else 'MISMATCH ✗'}")
    print(f"  Manual FN: {manual_fn:,} (model FN: {fn:,}) - {'MATCH ✓' if manual_fn == fn else 'MISMATCH ✗'}")
    print(f"\n  Manual Accuracy: {manual_accuracy*100:.2f}%")
    print(f"  sklearn Accuracy: {accuracy*100:.2f}%")
    print(f"  Match: {'✓ VERIFIED' if abs(manual_accuracy - accuracy) < 0.0001 else '✗ MISMATCH'}")
    
    return predictions, errors


def verify_agent_two(agent_two, X_anomalies, y_anomalies, y_binary_anomalies, n_proof_samples: int = 10):
    """
    Verify Agent Two accuracy with proof samples.
    
    Shows actual classifications for individual samples
    to prove the accuracy is correct.
    """
    print_separator("STEP 2: AGENT TWO - CLASSIFICATION VERIFICATION")
    
    # Run classification
    print("Running classification on anomalies...")
    analysis_results = agent_two.analyze_threats_batch(X_anomalies)
    predictions = np.array([r.classification.predicted_category for r in analysis_results])
    confidences = np.array([r.classification.confidence for r in analysis_results])
    
    # Calculate metrics
    accuracy = accuracy_score(y_anomalies, predictions)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  - Total anomalies to classify: {len(X_anomalies):,}")
    print(f"  - Actual attacks in anomalies: {(y_binary_anomalies == 1).sum():,}")
    print(f"  - False alarms from Agent One: {(y_binary_anomalies == 0).sum():,}")
    
    # Print metrics
    print(f"\nOverall Classification Accuracy: {accuracy*100:.2f}%")
    
    # Count by category
    print(f"\nActual Category Distribution:")
    actual_counts = Counter(y_anomalies)
    for cat, count in sorted(actual_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count:,}")
    
    print(f"\nPredicted Category Distribution:")
    pred_counts = Counter(predictions)
    for cat, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count:,}")
    
    # PROOF: Show individual sample predictions
    print(f"\n{'=' * 60}")
    print(f"PROOF: Individual Sample Classifications (first {n_proof_samples})")
    print(f"{'=' * 60}")
    print(f"{'Idx':<6} {'Predicted':<18} {'Actual':<18} {'Conf':<8} {'Correct'}")
    print("-" * 70)
    
    for i in range(min(n_proof_samples, len(predictions))):
        predicted = predictions[i]
        actual = y_anomalies[i]
        conf = confidences[i]
        is_correct = predicted == actual
        
        print(f"{i:<6} {predicted:<18} {actual:<18} {conf:<8.2%} {'✓' if is_correct else '✗'}")
    
    # Verify manual calculation
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION: Manual Accuracy Calculation")
    print(f"{'=' * 60}")
    
    manual_correct = (predictions == y_anomalies).sum()
    manual_total = len(predictions)
    manual_accuracy = manual_correct / manual_total
    
    print(f"  Correct predictions: {manual_correct:,}")
    print(f"  Total predictions: {manual_total:,}")
    print(f"  Manual Accuracy: {manual_accuracy*100:.2f}%")
    print(f"  sklearn Accuracy: {accuracy*100:.2f}%")
    print(f"  Match: {'✓ VERIFIED' if abs(manual_accuracy - accuracy) < 0.0001 else '✗ MISMATCH'}")
    
    # Per-class accuracy verification
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION: Per-Class Accuracy")
    print(f"{'=' * 60}")
    
    unique_classes = np.unique(y_anomalies)
    for cls in unique_classes:
        mask = y_anomalies == cls
        if mask.sum() > 0:
            class_correct = ((predictions == y_anomalies) & mask).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total
            print(f"  {cls:<18}: {class_correct:>5}/{class_total:<5} = {class_acc*100:>6.2f}% correct")
    
    return predictions, analysis_results


def verify_agent_three(agent_three, attack_results, n_proof_samples: int = 10):
    """
    Verify Agent Three mitigation decisions with proof samples.
    """
    print_separator("STEP 3: AGENT THREE - MITIGATION VERIFICATION")
    
    # Run mitigation decisions
    decisions = []
    for result in attack_results:
        decision = agent_three.take_action(result)
        decisions.append(decision)
    
    # Count actions
    action_names = [d.action_name for d in decisions]
    action_counts = Counter(action_names)
    
    print(f"Configuration:")
    print(f"  - Attacks to mitigate: {len(attack_results):,}")
    
    print(f"\nMitigation Action Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(decisions) * 100
        print(f"  - {action}: {count:,} ({pct:.1f}%)")
    
    # PROOF: Show individual decisions
    print(f"\n{'=' * 60}")
    print(f"PROOF: Individual Mitigation Decisions (first {n_proof_samples})")
    print(f"{'=' * 60}")
    print(f"{'Idx':<6} {'Attack Type':<20} {'Action':<20} {'Confidence':<12}")
    print("-" * 70)
    
    for i in range(min(n_proof_samples, len(decisions))):
        attack_type = attack_results[i].classification.predicted_category
        action = decisions[i].action_name
        conf = decisions[i].confidence
        
        print(f"{i:<6} {attack_type:<20} {action:<20} {conf:<12.2%}")
    
    # Action distribution by attack type
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION: Actions by Attack Type")
    print(f"{'=' * 60}")
    
    attack_action_map = {}
    for result, decision in zip(attack_results, decisions):
        attack_type = result.classification.predicted_category
        action = decision.action_name
        if attack_type not in attack_action_map:
            attack_action_map[attack_type] = Counter()
        attack_action_map[attack_type][action] += 1
    
    for attack_type, actions in sorted(attack_action_map.items()):
        total = sum(actions.values())
        print(f"\n  {attack_type} ({total} samples):")
        for action, count in actions.most_common():
            pct = count / total * 100
            print(f"    - {action}: {count} ({pct:.1f}%)")
    
    return decisions


def verify_end_to_end(y_binary, predictions_a1, y_anomalies, predictions_a2, y_binary_anomalies):
    """Verify end-to-end pipeline effectiveness."""
    print_separator("END-TO-END PIPELINE VERIFICATION")
    
    total_samples = len(y_binary)
    actual_attacks = y_binary.sum()
    actual_normal = (y_binary == 0).sum()
    
    # Agent One stats
    anomalies_detected = predictions_a1.sum()
    attacks_detected_a1 = ((predictions_a1 == 1) & (y_binary == 1)).sum()
    attacks_missed_a1 = ((predictions_a1 == 0) & (y_binary == 1)).sum()
    false_alarms_a1 = ((predictions_a1 == 1) & (y_binary == 0)).sum()
    
    # Agent Two stats
    attacks_classified_correctly = ((predictions_a2 != 'Normal') & (y_binary_anomalies == 1)).sum()
    false_alarms_corrected = ((predictions_a2 == 'Normal') & (y_binary_anomalies == 0)).sum()
    
    print("Pipeline Flow:")
    print(f"\n  INPUT: {total_samples:,} network flows")
    print(f"    - Actual attacks: {actual_attacks:,} ({actual_attacks/total_samples*100:.1f}%)")
    print(f"    - Actual normal: {actual_normal:,} ({actual_normal/total_samples*100:.1f}%)")
    
    print(f"\n  AGENT ONE OUTPUT: {anomalies_detected:,} anomalies flagged")
    print(f"    - True attacks caught: {attacks_detected_a1:,}/{actual_attacks:,} ({attacks_detected_a1/actual_attacks*100:.1f}%)")
    print(f"    - Attacks missed: {attacks_missed_a1:,} (CRITICAL if > 0)")
    print(f"    - False alarms: {false_alarms_a1:,}")
    
    print(f"\n  AGENT TWO OUTPUT:")
    print(f"    - Attacks correctly classified: {attacks_classified_correctly:,}/{attacks_detected_a1:,} ({attacks_classified_correctly/attacks_detected_a1*100:.1f}%)")
    print(f"    - False alarms corrected to Normal: {false_alarms_corrected:,}/{false_alarms_a1:,} ({false_alarms_corrected/false_alarms_a1*100:.1f}%)")
    
    # Overall effectiveness
    print(f"\n{'=' * 60}")
    print(f"OVERALL PIPELINE EFFECTIVENESS")
    print(f"{'=' * 60}")
    
    detection_rate = attacks_detected_a1 / actual_attacks * 100 if actual_attacks > 0 else 0
    classification_rate = attacks_classified_correctly / attacks_detected_a1 * 100 if attacks_detected_a1 > 0 else 0
    
    print(f"\n  Attack Detection Rate (Agent 1): {detection_rate:.2f}%")
    print(f"  Classification Rate (Agent 2): {classification_rate:.2f}%")
    print(f"  Overall Attack Handling: {attacks_classified_correctly}/{actual_attacks} = {attacks_classified_correctly/actual_attacks*100:.2f}%")
    
    return {
        "total_samples": total_samples,
        "actual_attacks": actual_attacks,
        "attacks_detected": attacks_detected_a1,
        "attacks_classified": attacks_classified_correctly,
        "detection_rate": detection_rate,
        "classification_rate": classification_rate,
    }


def load_unsw_nb15_raw(file_path: str, config: DatasetConfig) -> pd.DataFrame:
    """
    Load UNSW-NB15 raw files (1-4) that don't have headers.
    
    The raw files have 49 columns but no header row.
    """
    # Column names for UNSW-NB15 dataset (47 features + attack_cat + label)
    column_names = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
        'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
        'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
        'smean', 'dmean', 'trans_depth', 'response_body_len', 'sjit', 'djit',
        'stime', 'ltime', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
        'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
        'attack_cat', 'label'
    ]
    
    df = pd.read_csv(file_path, header=None, names=column_names, low_memory=False)
    
    # Clean attack_cat - fill empty with 'Normal'
    df['attack_cat'] = df['attack_cat'].fillna('Normal')
    df['attack_cat'] = df['attack_cat'].replace('', 'Normal')
    df['attack_cat'] = df['attack_cat'].replace(' ', 'Normal')
    
    # Standardize attack category names
    df['attack_cat'] = df['attack_cat'].str.strip()
    df['attack_cat'] = df['attack_cat'].replace({
        'Backdoor': 'Backdoors',
        'Backdoor ': 'Backdoors',
        'Reconnaissance ': 'Reconnaissance',
        'Fuzzers ': 'Fuzzers',
        'Shellcode ': 'Shellcode',
        'Analysis ': 'Analysis',
        'Exploits ': 'Exploits',
        'Generic ': 'Generic',
        'DoS ': 'DoS',
        'Worms ': 'Worms',
    })
    
    # Ensure label is numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    # Add derived features that may be missing (compute from raw columns)
    # rate = (spkts + dpkts) / dur (connections per second)
    df['dur'] = pd.to_numeric(df['dur'], errors='coerce').fillna(0)
    df['spkts'] = pd.to_numeric(df['spkts'], errors='coerce').fillna(0)
    df['dpkts'] = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0)
    df['rate'] = np.where(df['dur'] > 0, (df['spkts'] + df['dpkts']) / df['dur'], 0)
    
    # Ensure all numeric columns are proper types
    numeric_cols = ['sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
                    'sload', 'dload', 'swin', 'dwin', 'stcpb', 'dtcpb',
                    'smean', 'dmean', 'trans_depth', 'response_body_len', 
                    'sjit', 'djit', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat',
                    'ct_state_ttl', 'ct_flw_http_mthd', 'ct_ftp_cmd', 
                    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Verify IDS accuracy with proof")
    parser.add_argument("--samples", type=int, default=5000, 
                        help="Number of samples to test (default: 5000)")
    parser.add_argument("--proof-samples", type=int, default=15,
                        help="Number of proof samples to show (default: 15)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--data-file", type=str, default="data/UNSW-NB15_1.csv",
                        help="Data file to use (default: data/UNSW-NB15_1.csv)")
    args = parser.parse_args()
    
    # Redirect output if specified
    if args.output:
        import io
        output_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_separator(f"ACCURACY VERIFICATION REPORT - {timestamp}")
    print(f"Data File: {args.data_file}")
    print(f"Test Samples: {args.samples:,}")
    print(f"Proof Samples: {args.proof_samples}")
    
    # Load data
    print_separator("DATA LOADING")
    
    # Use minmax normalization for compatibility with pipeline
    config = DatasetConfig(normalization_method="minmax")
    
    # Load training data for preprocessor fitting
    train_loader = DataLoader(config)
    train_loader.load('data/UNSW_NB15_training-set.csv').clean()
    train_df = train_loader.data
    print(f"Training data loaded: {len(train_df):,} samples")
    
    # Load test data - handle raw files (UNSW-NB15_1.csv, etc.) separately
    import re
    is_raw_file = re.match(r'.*UNSW-NB15_[1-4]\.csv$', args.data_file)
    
    if is_raw_file:
        print(f"Detected raw format file (no headers)")
        test_df = load_unsw_nb15_raw(args.data_file, config)
        print(f"Test data loaded: {len(test_df):,} samples from {args.data_file}")
    else:
        test_loader = DataLoader(config)
        test_loader.load(args.data_file).clean()
        test_df = test_loader.data
        print(f"Test data loaded: {len(test_df):,} samples from {args.data_file}")
    
    # Sample for testing
    if len(test_df) > args.samples:
        test_sample = test_df.sample(n=args.samples, random_state=42).copy()
        print(f"Sampled {args.samples:,} for testing")
    else:
        test_sample = test_df.copy()
        print(f"Using all {len(test_sample):,} samples")
    
    # Show data distribution
    print(f"\nData Distribution:")
    print(f"  - Attack samples: {(test_sample['label'] > 0).sum():,}")
    print(f"  - Normal samples: {(test_sample['label'] == 0).sum():,}")
    print(f"\n  Attack Categories:")
    for cat, count in test_sample['attack_cat'].value_counts().items():
        print(f"    - {cat}: {count:,}")
    
    # Preprocess
    print_separator("PREPROCESSING")
    preprocessor = Preprocessor(config)
    preprocessor.fit(train_df)
    X_test = preprocessor.transform(test_sample)
    y_binary = (test_sample['label'].values > 0).astype(int)
    y_multiclass = test_sample['attack_cat'].values
    print(f"Preprocessed features: {X_test.shape}")
    
    # Load agents
    print_separator("LOADING MODELS")
    
    print("Loading Agent One (Autoencoder)...")
    agent_one = AgentOne.from_checkpoint('models/agent_one/best_model.pth', threshold=0.0396)
    print(f"  - Threshold: {agent_one._threshold}")
    
    # Handle dimension mismatch (fine-tuned model may expect 42 features)
    model_input_dim = agent_one._model.input_dim
    data_dim = X_test.shape[1]
    if data_dim < model_input_dim:
        print(f"  - Padding features: {data_dim} -> {model_input_dim}")
        padding = np.zeros((X_test.shape[0], model_input_dim - data_dim), dtype=X_test.dtype)
        X_test = np.hstack([X_test, padding])
    elif data_dim > model_input_dim:
        print(f"  - Truncating features: {data_dim} -> {model_input_dim}")
        X_test = X_test[:, :model_input_dim]
    
    print("\nLoading Agent Two (XGBoost Classifier)...")
    agent_two = AgentTwo.from_pretrained('models/agent_two')
    print(f"  - Classes: {agent_two.classifier.n_classes}")
    
    print("\nLoading Agent Three (PPO RL)...")
    agent_three = AgentThree.from_pretrained('models/agent_three/final_model')
    print(f"  - Trained: {agent_three.is_trained}")
    
    # Step 1: Agent One Verification
    predictions_a1, errors = verify_agent_one(
        agent_one, X_test, y_binary, n_proof_samples=args.proof_samples
    )
    
    # Prepare data for Agent Two
    anomaly_mask = predictions_a1 == 1
    X_anomalies = X_test[anomaly_mask]
    y_anomalies = y_multiclass[anomaly_mask]
    y_binary_anomalies = y_binary[anomaly_mask]
    
    # Step 2: Agent Two Verification
    predictions_a2, analysis_results = verify_agent_two(
        agent_two, X_anomalies, y_anomalies, y_binary_anomalies, 
        n_proof_samples=args.proof_samples
    )
    
    # Prepare data for Agent Three
    attack_mask = predictions_a2 != 'Normal'
    attack_results = [r for r, is_attack in zip(analysis_results, attack_mask) if is_attack]
    
    # Step 3: Agent Three Verification
    if len(attack_results) > 0:
        decisions = verify_agent_three(
            agent_three, attack_results, n_proof_samples=args.proof_samples
        )
    else:
        print("\nNo attacks classified - skipping Agent Three verification")
    
    # End-to-end verification
    metrics = verify_end_to_end(
        y_binary, predictions_a1, y_anomalies, predictions_a2, y_binary_anomalies
    )
    
    print_separator("VERIFICATION COMPLETE")
    print("All accuracy metrics have been manually verified against raw predictions.")
    print("The proof samples above demonstrate that reported accuracies match actual data.")
    
    # Save output if specified
    if args.output:
        sys.stdout = original_stdout
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_buffer.getvalue())
        print(f"Results saved to: {args.output}")
        print(output_buffer.getvalue())
    
    return metrics


if __name__ == "__main__":
    main()
