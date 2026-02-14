#!/bin/bash

# Simple Federated Agentic Defense Detection Metrics Script
# Basic comparison without external Python dependencies
# Usage: ./simple_metrics_comparison.sh

echo "=========================================="
echo "  Simple Detection Metrics Comparison"
echo "=========================================="

# Configuration
SYSTEM_METRICS="agentic-ids-local/src/logs/system_metrics.jsonl"
UNSW_TRAINING="agentic-ids-local/src/data/UNSW_NB15/UNSW_NB15_training-set.csv"
OUTPUT_DIR="simple_metrics"
mkdir -p $OUTPUT_DIR

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq first."
    exit 1
fi

# Extract maximum flow_id
echo "Finding maximum flow_id..."
MAX_FLOW_ID=$(grep '"inference_result"' $SYSTEM_METRICS | \
    jq -r '.flow_id' | \
    grep -E '^[0-9]+$' | \
    sort -n | \
    tail -1)

echo "Maximum numeric flow_id: $MAX_FLOW_ID"

# Extract agent predictions
echo "Extracting agent predictions..."
TEMP_AGENT="$OUTPUT_DIR/agent_temp.jsonl"
grep '"inference_result"' $SYSTEM_METRICS | \
    jq -c 'select(.flow_id | type == "number" and . > 0 and . <= '$MAX_FLOW_ID')' > $TEMP_AGENT

# Extract ground truth labels
echo "Extracting ground truth labels..."
TEMP_TRUTH="$OUTPUT_DIR/truth_temp.csv"
if [ -f "$UNSW_TRAINING" ]; then
    head -n $(($MAX_FLOW_ID + 1)) "$UNSW_TRAINING" | tail -n +2 > $TEMP_TRUTH
else
    echo "Error: UNSW training set not found at $UNSW_TRAINING"
    exit 1
fi

# Calculate basic metrics using simple tools
echo "Calculating metrics..."

# Count totals
TOTAL_AGENT=$(cat $TEMP_AGENT | wc -l)
TOTAL_TRUTH=$(cat $TEMP_TRUTH | wc -l)

echo "Agent predictions: $TOTAL_AGENT"
echo "Ground truth records: $TOTAL_TRUTH"

# Simple Python script for basic calculations
cat > "$OUTPUT_DIR/simple_calc.py" << 'EOF'
import json
import sys
import csv
from collections import Counter

# Read agent predictions
agent_predictions = {}
with open(sys.argv[1], 'r') as f:
    for line in f:
        data = json.loads(line)
        flow_id = data['flow_id']
        agent_predictions[flow_id] = {
            'prediction': data['prediction'],
            'anomaly_score': data['anomaly_score'],
            'threshold_exceeded': data['threshold_exceeded']
        }

# Read ground truth
ground_truth = {}
with open(sys.argv[2], 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            try:
                flow_id = int(row[0])
                label = int(row[-1])  # Last column is label
                ground_truth[flow_id] = label
            except (ValueError, IndexError):
                continue

# Calculate metrics for matching flow_ids
tp = fp = tn = fn = 0
matches = []

for flow_id in agent_predictions:
    if flow_id in ground_truth:
        agent_pred = agent_predictions[flow_id]['prediction']
        true_label = ground_truth[flow_id]
        
        matches.append((flow_id, agent_pred, true_label))
        
        if agent_pred == 1 and true_label == 1:
            tp += 1
        elif agent_pred == 1 and true_label == 0:
            fp += 1
        elif agent_pred == 0 and true_label == 1:
            fn += 1
        elif agent_pred == 0 and true_label == 0:
            tn += 1

total = tp + fp + tn + fn
if total > 0:
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("===== DETECTION PERFORMANCE METRICS =====")
    print(f"Total Flows Analyzed: {total}")
    print(f"Matched Flow IDs: {len(matches)}")
    print("")
    print("Confusion Matrix:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Negatives (FN): {fn}")
    print("")
    print("Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.3f} ({(tp+tn)}/{total})")
    print(f"  Precision: {precision:.3f} ({tp}/{tp+fp})")
    print(f"  Recall:    {recall:.3f} ({tp}/{tp+fn})")
    print(f"  F1-Score:  {f1:.3f}")
    print("")
    print("Detection Rates:")
    print(f"  Attack Detection Rate (Sensitivity): {recall:.3f}")
    print(f"  False Positive Rate: {fp/(fp+tn):.3f}" if (fp+tn) > 0 else "  False Positive Rate: N/A")
    print(f"  Specificity (True Negative Rate): {tn/(tn+fp):.3f}" if (tn+fp) > 0 else "  Specificity: N/A")
    
    # Anomaly score analysis
    agent_scores = [agent_predictions[fid]['anomaly_score'] for fid, _, _ in matches]
    true_attacks = [agent_predictions[fid]['anomaly_score'] for fid, _, true_label in matches if true_label == 1]
    true_benigns = [agent_predictions[fid]['anomaly_score'] for fid, _, true_label in matches if true_label == 0]
    
    print("")
    print("Anomaly Score Statistics:")
    print(f"  Overall Mean Score: {sum(agent_scores)/len(agent_scores):.6f}")
    if true_attacks:
        print(f"  Mean Score for Attacks: {sum(true_attacks)/len(true_attacks):.6f}")
    if true_benigns:
        print(f"  Mean Score for Benign: {sum(true_benigns)/len(true_benigns):.6f}")
    
    # Save detailed results
    with open('simple_metrics/detailed_results.txt', 'w') as out:
        out.write("Flow_ID,Agent_Prediction,Ground_Truth,Match,Anomaly_Score\n")
        for flow_id, agent_pred, true_label in matches:
            match_status = "CORRECT" if agent_pred == true_label else "INCORRECT"
            score = agent_predictions[flow_id]['anomaly_score']
            out.write(f"{flow_id},{agent_pred},{true_label},{match_status},{score}\n")
    
    print(f"\nDetailed results saved to: simple_metrics/detailed_results.txt")
    
    # Error analysis
    fp_flows = [fid for fid, agent_pred, true_label in matches if agent_pred == 1 and true_label == 0]
    fn_flows = [fid for fid, agent_pred, true_label in matches if agent_pred == 0 and true_label == 1]
    
    if fp_flows:
        print(f"\nFalse Positive Flow IDs: {fp_flows[:10]}{'...' if len(fp_flows) > 10 else ''}")
    if fn_flows:
        print(f"False Negative Flow IDs: {fn_flows[:10]}{'...' if len(fn_flows) > 10 else ''}")

else:
    print("No matching flow IDs found between agent predictions and ground truth!")

EOF

# Run the calculation
python3 "$OUTPUT_DIR/simple_calc.py" "$TEMP_AGENT" "$TEMP_TRUTH"

echo ""
echo "===========================================" 
echo "Simple metrics analysis complete!"
echo "Check $OUTPUT_DIR/ for detailed results"
echo ""

# Cleanup
rm -f $TEMP_AGENT $TEMP_TRUTH