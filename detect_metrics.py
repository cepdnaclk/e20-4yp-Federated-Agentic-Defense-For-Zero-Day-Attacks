#!/usr/bin/env python3
"""
Federated Agentic Defense - Detection Metrics Comparison
Compares agent detection results with UNSW-NB15 ground truth labels
Cross-platform compatible (Windows/Linux/macOS)
"""

import json
import csv
import sys
import os
import argparse
from collections import Counter, defaultdict
import statistics

def extract_numeric_flow_ids(jsonl_file):
    """Extract all numeric flow_ids from system metrics"""
    flow_ids = []
    print(f"Reading system metrics from: {jsonl_file}")
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('event_type') == 'inference_result':
                        flow_id = data.get('flow_id')
                        if isinstance(flow_id, int) and flow_id > 0:
                            flow_ids.append(flow_id)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error on line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: System metrics file not found: {jsonl_file}")
        return []
    
    return flow_ids

def load_agent_predictions(jsonl_file, max_flow_id):
    """Load agent predictions from system metrics"""
    predictions = {}
    triage_data = {}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event_type = data.get('event_type')
                flow_id = data.get('flow_id')
                
                if not isinstance(flow_id, int) or flow_id <= 0 or flow_id > max_flow_id:
                    continue
                
                if event_type == 'inference_result':
                    predictions[flow_id] = {
                        'prediction': data.get('prediction', 0),
                        'anomaly_score': data.get('anomaly_score', 0.0),
                        'threshold_exceeded': data.get('threshold_exceeded', False),
                        'reconstruction_error': data.get('reconstruction_error', 0.0)
                    }
                elif event_type == 'triage_classification':
                    triage_data[flow_id] = {
                        'classification': data.get('classification', 'Unknown'),
                        'target_pipeline': data.get('target_pipeline', 'Unknown'),
                        'processing_time_ms': data.get('processing_time_ms', 0)
                    }
            except json.JSONDecodeError:
                continue
    
    # Merge triage data with predictions
    for flow_id in triage_data:
        if flow_id in predictions:
            predictions[flow_id].update(triage_data[flow_id])
    
    return predictions

def load_ground_truth(csv_file, max_flow_id):
    """Load ground truth labels from UNSW-NB15"""
    labels = {}
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            print(f"UNSW dataset columns: {len(header)} columns")
            
            for row_num, row in enumerate(reader, 2):  # Start from row 2 (after header)
                if len(row) >= 2:
                    try:
                        flow_id = int(row[0])
                        if 1 <= flow_id <= max_flow_id:
                            label = int(row[-1])  # Last column is label
                            attack_cat = row[-2] if len(row) > 2 else 'Unknown'
                            labels[flow_id] = {
                                'label': label,
                                'category': attack_cat.strip() if attack_cat else 'Unknown'
                            }
                        elif flow_id > max_flow_id:
                            break  # Stop reading beyond max_flow_id
                    except (ValueError, IndexError) as e:
                        continue
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {csv_file}")
        return {}
    
    return labels

def calculate_metrics(predictions, ground_truth):
    """Calculate comprehensive performance metrics"""
    tp = fp = tn = fn = 0
    matches = []
    anomaly_scores_benign = []
    anomaly_scores_attack = []
    
    for flow_id in predictions:
        if flow_id in ground_truth:
            agent_pred = predictions[flow_id]['prediction']
            true_label = ground_truth[flow_id]['label']
            anomaly_score = predictions[flow_id]['anomaly_score']
            
            matches.append((flow_id, agent_pred, true_label))
            
            # Collect scores for analysis
            if true_label == 0:
                anomaly_scores_benign.append(anomaly_score)
            else:
                anomaly_scores_attack.append(anomaly_score)
            
            # Confusion matrix
            if agent_pred == 1 and true_label == 1:
                tp += 1
            elif agent_pred == 1 and true_label == 0:
                fp += 1
            elif agent_pred == 0 and true_label == 1:
                fn += 1
            elif agent_pred == 0 and true_label == 0:
                tn += 1
    
    total = tp + fp + tn + fn
    if total == 0:
        return None, matches, {}, {}
    
    # Basic metrics
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'total': total,
        'matched_flows': len(matches),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': fpr,
        'specificity': specificity
    }
    
    # Score analysis
    score_stats = {
        'benign_scores': {
            'count': len(anomaly_scores_benign),
            'mean': statistics.mean(anomaly_scores_benign) if anomaly_scores_benign else 0,
            'median': statistics.median(anomaly_scores_benign) if anomaly_scores_benign else 0,
            'max': max(anomaly_scores_benign) if anomaly_scores_benign else 0
        },
        'attack_scores': {
            'count': len(anomaly_scores_attack),
            'mean': statistics.mean(anomaly_scores_attack) if anomaly_scores_attack else 0,
            'median': statistics.median(anomaly_scores_attack) if anomaly_scores_attack else 0,
            'max': max(anomaly_scores_attack) if anomaly_scores_attack else 0
        }
    }
    
    # Category analysis
    category_stats = analyze_by_category(predictions, ground_truth, matches)
    
    return metrics, matches, score_stats, category_stats

def analyze_by_category(predictions, ground_truth, matches):
    """Analyze performance by attack category"""
    category_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'total': 0})
    
    for flow_id, agent_pred, true_label in matches:
        category = ground_truth[flow_id]['category']
        category_results[category]['total'] += 1
        
        if agent_pred == 1 and true_label == 1:
            category_results[category]['tp'] += 1
        elif agent_pred == 1 and true_label == 0:
            category_results[category]['fp'] += 1
        elif agent_pred == 0 and true_label == 1:
            category_results[category]['fn'] += 1
        elif agent_pred == 0 and true_label == 0:
            category_results[category]['tn'] += 1
    
    # Calculate metrics per category
    for category in category_results:
        stats = category_results[category]
        total = stats['total']
        if total > 0:
            accuracy = (stats['tp'] + stats['tn']) / total
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            stats.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
    
    return dict(category_results)

def save_detailed_results(matches, predictions, ground_truth, output_dir):
    """Save detailed flow-by-flow results"""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'detailed_results.csv')
    
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Flow_ID', 'Agent_Prediction', 'Ground_Truth', 'Match', 
            'Anomaly_Score', 'Threshold_Exceeded', 'Attack_Category', 'Classification'
        ])
        
        for flow_id, agent_pred, true_label in matches:
            match_status = "CORRECT" if agent_pred == true_label else "INCORRECT"
            pred_data = predictions[flow_id]
            truth_data = ground_truth[flow_id]
            
            writer.writerow([
                flow_id, 
                agent_pred, 
                true_label, 
                match_status,
                f"{pred_data['anomaly_score']:.8f}",
                pred_data['threshold_exceeded'],
                truth_data['category'],
                pred_data.get('classification', 'N/A')
            ])
    
    return results_file

def print_results(metrics, score_stats, category_stats, fp_flows, fn_flows):
    """Print comprehensive results"""
    print("\n" + "="*70)
    print("           FEDERATED AGENTIC DEFENSE METRICS")
    print("="*70)
    print(f"Total Flows Analyzed: {metrics['total']}")
    print(f"Matched Flow IDs: {metrics['matched_flows']}")
    print(f"Data Coverage: {metrics['matched_flows']}/{metrics['total']} flows")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {metrics['tp']:4d}")
    print(f"  False Positives (FP): {metrics['fp']:4d}")
    print(f"  True Negatives (TN):  {metrics['tn']:4d}")
    print(f"  False Negatives (FN): {metrics['fn']:4d}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:             {metrics['accuracy']:.3f}")
    print(f"  Precision:            {metrics['precision']:.3f}")
    print(f"  Recall (Sensitivity): {metrics['recall']:.3f}")
    print(f"  F1-Score:             {metrics['f1_score']:.3f}")
    print(f"  Specificity:          {metrics['specificity']:.3f}")
    print(f"  False Positive Rate:  {metrics['false_positive_rate']:.3f}")
    
    print(f"\nAnomaly Score Analysis:")
    if score_stats['benign_scores']['count'] > 0:
        print(f"  Benign Traffic  ({score_stats['benign_scores']['count']:3d} flows):")
        print(f"    Mean Score: {score_stats['benign_scores']['mean']:.6f}")
        print(f"    Max Score:  {score_stats['benign_scores']['max']:.6f}")
    
    if score_stats['attack_scores']['count'] > 0:
        print(f"  Attack Traffic  ({score_stats['attack_scores']['count']:3d} flows):")
        print(f"    Mean Score: {score_stats['attack_scores']['mean']:.6f}")
        print(f"    Max Score:  {score_stats['attack_scores']['max']:.6f}")
    
    if category_stats:
        print(f"\nPerformance by Attack Category:")
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                print(f"  {category:15s}: Acc={stats['accuracy']:.3f}, "
                      f"Recall={stats['recall']:.3f}, F1={stats['f1_score']:.3f} "
                      f"({stats['total']} samples)")
    
    # Error analysis
    if fp_flows:
        fp_display = ', '.join(str(fid) for fid in fp_flows[:15])
        if len(fp_flows) > 15:
            fp_display += f'... (+{len(fp_flows)-15} more)'
        print(f"\nFalse Positive Flow IDs ({len(fp_flows)} total): {fp_display}")
    
    if fn_flows:
        fn_display = ', '.join(str(fid) for fid in fn_flows[:15])
        if len(fn_flows) > 15:
            fn_display += f'... (+{len(fn_flows)-15} more)'
        print(f"False Negative Flow IDs ({len(fn_flows)} total): {fn_display}")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Compare Federated Agentic Defense results with UNSW-NB15 ground truth'
    )
    parser.add_argument('--system-metrics', 
                       default='agentic-ids-local/src/logs/system_metrics.jsonl',
                       help='Path to system metrics JSONL file')
    parser.add_argument('--unsw-dataset', 
                       default='agentic-ids-local/src/data/UNSW_NB15/UNSW_NB15_training-set.csv',
                       help='Path to UNSW-NB15 dataset CSV file')
    parser.add_argument('--output-dir', 
                       default='detection_metrics',
                       help='Output directory for results')
    parser.add_argument('--max-flow-id', type=int,
                       help='Maximum flow ID to analyze (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    print("Federated Agentic Defense - Detection Metrics Analysis")
    print(f"System metrics: {args.system_metrics}")
    print(f"UNSW dataset: {args.unsw_dataset}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Extract maximum flow ID
    if args.max_flow_id:
        max_flow_id = args.max_flow_id
        print(f"Using specified maximum flow ID: {max_flow_id}")
    else:
        print("Auto-detecting maximum flow ID...")
        flow_ids = extract_numeric_flow_ids(args.system_metrics)
        if not flow_ids:
            print("Error: No numeric flow IDs found in system metrics!")
            return 1
        max_flow_id = max(flow_ids)
        print(f"Auto-detected maximum flow ID: {max_flow_id}")
    
    print(f"Will analyze flows 1 to {max_flow_id}")
    print()
    
    # Load data
    print("Loading agent predictions...")
    predictions = load_agent_predictions(args.system_metrics, max_flow_id)
    
    print("Loading ground truth labels...")
    ground_truth = load_ground_truth(args.unsw_dataset, max_flow_id)
    
    if not predictions:
        print("Error: No agent predictions loaded!")
        return 1
    if not ground_truth:
        print("Error: No ground truth labels loaded!")
        return 1
    
    print(f"Loaded {len(predictions)} agent predictions")
    print(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics, matches, score_stats, category_stats = calculate_metrics(predictions, ground_truth)
    
    if metrics is None:
        print("Error: No matching flow IDs found!")
        return 1
    
    # Save detailed results
    results_file = save_detailed_results(matches, predictions, ground_truth, args.output_dir)
    
    # Error analysis
    fp_flows = [fid for fid, agent_pred, true_label in matches if agent_pred == 1 and true_label == 0]
    fn_flows = [fid for fid, agent_pred, true_label in matches if agent_pred == 0 and true_label == 1]
    
    # Print comprehensive results
    print_results(metrics, score_stats, category_stats, fp_flows, fn_flows)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Analysis complete! Check {args.output_dir}/ for all outputs.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
