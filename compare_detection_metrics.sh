#!/bin/bash

# Federated Agentic Defense Detection Metrics Comparison Script
# Compares agent detection results with ground truth UNSW-NB15 labels
# Usage: ./compare_detection_metrics.sh

echo "=========================================="
echo "  Federated Agentic Defense Metrics"
echo "=========================================="
echo "Comparing agent detection results with UNSW-NB15 ground truth..."
echo ""

# Configuration
SYSTEM_METRICS="agentic-ids-local/src/logs/system_metrics.jsonl"
UNSW_TRAINING="agentic-ids-local/src/data/UNSW_NB15/UNSW_NB15_training-set.csv"
UNSW_TESTING="agentic-ids-local/src/data/UNSW_NB15/UNSW_NB15_testing-set.csv"
OUTPUT_DIR="metrics_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p $OUTPUT_DIR

# Extract maximum flow_id from system_metrics.jsonl (numeric only, skip UUIDs)
echo "Finding maximum flow_id in system metrics..."
MAX_FLOW_ID=$(grep '"inference_result"' $SYSTEM_METRICS | \
    jq -r '.flow_id' | \
    grep -E '^[0-9]+$' | \
    sort -n | \
    tail -1)

echo "Maximum numeric flow_id found: $MAX_FLOW_ID"
echo ""

# Extract agent predictions and create temporary files
echo "Extracting agent predictions..."
AGENT_PREDICTIONS_FILE="$OUTPUT_DIR/agent_predictions_$TIMESTAMP.csv"
GROUND_TRUTH_FILE="$OUTPUT_DIR/ground_truth_$TIMESTAMP.csv"

cat > $AGENT_PREDICTIONS_FILE << 'EOF'
flow_id,agent_prediction,anomaly_score,threshold_exceeded,classification
EOF

# Process system_metrics.jsonl for agent predictions
grep '"inference_result"' $SYSTEM_METRICS | \
    jq -r 'select(.flow_id | type == "number" and . > 0 and . <= '$MAX_FLOW_ID') | 
           [.flow_id, .prediction, .anomaly_score, .threshold_exceeded] | @csv' >> $AGENT_PREDICTIONS_FILE

# Get triage classifications
TRIAGE_FILE="$OUTPUT_DIR/triage_classifications_$TIMESTAMP.csv"
cat > $TRIAGE_FILE << 'EOF'
flow_id,classification,target_pipeline,reasoning_summary
EOF

grep '"triage_classification"' $SYSTEM_METRICS | \
    jq -r 'select(.flow_id | type == "number" and . > 0 and . <= '$MAX_FLOW_ID') | 
           [.flow_id, .classification, .target_pipeline, (.reasoning | split(".")[0])] | @csv' >> $TRIAGE_FILE

echo "Extracting ground truth from UNSW-NB15..."

# Extract ground truth labels from UNSW dataset
cat > $GROUND_TRUTH_FILE << 'EOF'
flow_id,ground_truth_label,attack_category
EOF

# Try training set first, then testing set if needed
if [ -f "$UNSW_TRAINING" ]; then
    echo "Using UNSW training set..."
    head -n $(($MAX_FLOW_ID + 1)) "$UNSW_TRAINING" | \
        tail -n +2 | \
        awk -F',' -v max_id="$MAX_FLOW_ID" 'NR <= max_id {print $1","$NF","$(NF-1)}' >> $GROUND_TRUTH_FILE
elif [ -f "$UNSW_TESTING" ]; then
    echo "Using UNSW testing set..."
    head -n $(($MAX_FLOW_ID + 1)) "$UNSW_TESTING" | \
        tail -n +2 | \
        awk -F',' -v max_id="$MAX_FLOW_ID" 'NR <= max_id {print $1","$NF","$(NF-1)}' >> $GROUND_TRUTH_FILE
else
    echo "ERROR: Neither UNSW training nor testing set found!"
    exit 1
fi

echo ""
echo "Creating comprehensive comparison analysis..."

# Create Python script for detailed metrics calculation
METRICS_SCRIPT="$OUTPUT_DIR/calculate_metrics.py"
cat > $METRICS_SCRIPT << 'PYTHON_EOF'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import json
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(agent_file, ground_truth_file, triage_file):
    """Load and merge agent predictions with ground truth"""
    try:
        agent_df = pd.read_csv(agent_file)
        truth_df = pd.read_csv(ground_truth_file)
        triage_df = pd.read_csv(triage_file)
        
        print(f"Loaded {len(agent_df)} agent predictions")
        print(f"Loaded {len(truth_df)} ground truth labels")
        print(f"Loaded {len(triage_df)} triage classifications")
        
        # Merge on flow_id
        merged = pd.merge(agent_df, truth_df, on='flow_id', how='inner')
        merged = pd.merge(merged, triage_df, on='flow_id', how='left')
        
        print(f"Successfully merged {len(merged)} records")
        return merged
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def calculate_metrics(df):
    """Calculate comprehensive detection metrics"""
    y_true = df['ground_truth_label'].astype(int)
    y_pred = df['agent_prediction'].astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'total_samples': len(df),
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'benign_class': {
            'precision': float(precision[0]) if len(precision) > 0 else 0,
            'recall': float(recall[0]) if len(recall) > 0 else 0,
            'f1_score': float(fscore[0]) if len(fscore) > 0 else 0,
            'support': int(support[0]) if len(support) > 0 else 0
        },
        'attack_class': {
            'precision': float(precision[1]) if len(precision) > 1 else 0,
            'recall': float(recall[1]) if len(recall) > 1 else 0,
            'f1_score': float(fscore[1]) if len(fscore) > 1 else 0,
            'support': int(support[1]) if len(support) > 1 else 0
        }
    }
    
    return metrics, cm

def analyze_triage_performance(df):
    """Analyze triage classification performance"""
    triage_analysis = {}
    
    if 'classification' in df.columns:
        # Map classifications to binary predictions
        df['triage_binary'] = df['classification'].apply(
            lambda x: 1 if 'SUSPICIOUS' in str(x) or 'ZERO-DAY' in str(x) else 0
        )
        
        # Classification distribution
        class_dist = df['classification'].value_counts().to_dict()
        triage_analysis['classification_distribution'] = class_dist
        
        # Performance by triage class
        for class_type in df['classification'].unique():
            if pd.notna(class_type):
                subset = df[df['classification'] == class_type]
                if len(subset) > 0:
                    accuracy = accuracy_score(subset['ground_truth_label'], subset['agent_prediction'])
                    triage_analysis[f'{class_type}_accuracy'] = float(accuracy)
    
    return triage_analysis

def generate_attack_category_analysis(df):
    """Analyze performance by attack category"""
    category_analysis = {}
    
    if 'attack_category' in df.columns:
        for category in df['attack_category'].unique():
            if pd.notna(category) and category != 'Normal':
                subset = df[df['attack_category'] == category]
                if len(subset) > 0:
                    y_true = subset['ground_truth_label']
                    y_pred = subset['agent_prediction']
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                    
                    category_analysis[category] = {
                        'samples': len(subset),
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(fscore)
                    }
    
    return category_analysis

def create_visualizations(df, cm, output_dir):
    """Create visualization plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix - Agent vs Ground Truth')
    plt.ylabel('Ground Truth')
    plt.xlabel('Agent Prediction')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Anomaly Score Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    benign_scores = df[df['ground_truth_label'] == 0]['anomaly_score']
    attack_scores = df[df['ground_truth_label'] == 1]['anomaly_score']
    
    plt.hist(benign_scores, bins=30, alpha=0.7, label='Benign', color='green')
    plt.hist(attack_scores, bins=30, alpha=0.7, label='Attack', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution by Ground Truth')
    plt.legend()
    
    # Classification Distribution
    plt.subplot(1, 2, 2)
    if 'classification' in df.columns:
        class_counts = df['classification'].value_counts()
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        plt.title('Triage Classification Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python calculate_metrics.py <agent_predictions.csv> <ground_truth.csv> <triage_classifications.csv>")
        sys.exit(1)
    
    agent_file, truth_file, triage_file = sys.argv[1:4]
    output_dir = os.path.dirname(agent_file)
    
    # Load and merge data
    df = load_data(agent_file, truth_file, triage_file)
    
    # Calculate metrics
    metrics, cm = calculate_metrics(df)
    triage_analysis = analyze_triage_performance(df)
    category_analysis = generate_attack_category_analysis(df)
    
    # Combine all results
    results = {
        'evaluation_summary': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_flows_analyzed': int(df['flow_id'].max()),
            'agent_predictions_count': len(df),
            'data_coverage': f"{len(df)}/{int(df['flow_id'].max())} flows ({100*len(df)/int(df['flow_id'].max()):.1f}%)"
        },
        'detection_metrics': metrics,
        'triage_analysis': triage_analysis,
        'attack_category_analysis': category_analysis,
        'detailed_errors': {
            'false_positives': df[(df['ground_truth_label'] == 0) & (df['agent_prediction'] == 1)]['flow_id'].tolist(),
            'false_negatives': df[(df['ground_truth_label'] == 1) & (df['agent_prediction'] == 0)]['flow_id'].tolist()
        }
    }
    
    # Save results
    results_file = f'{output_dir}/metrics_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    create_visualizations(df, cm, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("           DETECTION PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Flows Analyzed: {results['evaluation_summary']['total_flows_analyzed']}")
    print(f"Agent Predictions: {results['evaluation_summary']['agent_predictions_count']}")
    print(f"Data Coverage: {results['evaluation_summary']['data_coverage']}")
    print(f"\nOverall Accuracy: {metrics['accuracy']:.3f}")
    print(f"True Positives: {metrics['confusion_matrix']['true_positive']}")
    print(f"True Negatives: {metrics['confusion_matrix']['true_negative']}")
    print(f"False Positives: {metrics['confusion_matrix']['false_positive']}")
    print(f"False Negatives: {metrics['confusion_matrix']['false_negative']}")
    
    print(f"\nBenign Class Performance:")
    print(f"  Precision: {metrics['benign_class']['precision']:.3f}")
    print(f"  Recall: {metrics['benign_class']['recall']:.3f}")
    print(f"  F1-Score: {metrics['benign_class']['f1_score']:.3f}")
    
    print(f"\nAttack Class Performance:")
    print(f"  Precision: {metrics['attack_class']['precision']:.3f}")
    print(f"  Recall: {metrics['attack_class']['recall']:.3f}")
    print(f"  F1-Score: {metrics['attack_class']['f1_score']:.3f}")
    
    if category_analysis:
        print(f"\nPerformance by Attack Category:")
        for category, stats in category_analysis.items():
            print(f"  {category}: Acc={stats['accuracy']:.3f}, F1={stats['f1_score']:.3f} ({stats['samples']} samples)")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
PYTHON_EOF

# Make Python script executable
chmod +x $METRICS_SCRIPT

echo "Running comprehensive metrics analysis..."
python3 $METRICS_SCRIPT "$AGENT_PREDICTIONS_FILE" "$GROUND_TRUTH_FILE" "$TRIAGE_FILE"

# Create summary report
SUMMARY_REPORT="$OUTPUT_DIR/detection_summary_$TIMESTAMP.md"
cat > $SUMMARY_REPORT << 'REPORT_EOF'
# Federated Agentic Defense - Detection Performance Report

## Executive Summary
This report compares the performance of the federated agentic defense system against the UNSW-NB15 network intrusion detection dataset.

## Key Findings

### Detection Accuracy
- **Overall Accuracy**: [See JSON results]
- **True Positive Rate** (Attack Detection): [See JSON results]
- **False Positive Rate** (False Alarms): [See JSON results]

### Triage Performance
- **Suspicious Classifications**: [See JSON results]
- **Zero-Day Classifications**: [See JSON results]
- **Benign Classifications**: [See JSON results]

### Attack Category Analysis
Different attack types showed varying detection rates:
- **Reconnaissance**: [See JSON results]
- **DoS**: [See JSON results]
- **Exploits**: [See JSON results]
- **Other Categories**: [See JSON results]

## Recommendations
1. Review false positive cases for pattern analysis
2. Improve detection for categories with low recall
3. Tune anomaly score thresholds based on results
4. Consider additional features for zero-day detection

## Files Generated
- `metrics_results_*.json`: Detailed numerical results
- `confusion_matrix.png`: Performance visualization
- `analysis_plots.png`: Score distributions and classifications
- `agent_predictions_*.csv`: Raw agent predictions
- `ground_truth_*.csv`: UNSW-NB15 labels used

REPORT_EOF

echo ""
echo "Analysis complete! Results saved in: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
ls -la $OUTPUT_DIR/
echo ""
echo "Key outputs:"
echo "  - JSON metrics: $(ls $OUTPUT_DIR/metrics_results_*.json | head -1)"
echo "  - Summary report: $SUMMARY_REPORT"
echo "  - Visualizations: $OUTPUT_DIR/confusion_matrix.png, $OUTPUT_DIR/analysis_plots.png"
echo ""
echo "To view detailed results:"
echo "  cat $(ls $OUTPUT_DIR/metrics_results_*.json | head -1) | jq"