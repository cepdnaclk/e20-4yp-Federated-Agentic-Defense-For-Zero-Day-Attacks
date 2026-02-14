# Detection Metrics Comparison Scripts

This directory contains scripts to compare the performance of the Federated Agentic Defense system against the ground truth UNSW-NB15 dataset.

## Available Scripts

### 1. `simple_metrics_comparison.sh` (Recommended for quick analysis)
- **Dependencies**: Only `jq` and `python3` (standard library)
- **Output**: Basic performance metrics, confusion matrix, detailed CSV
- **Runtime**: Fast (~30 seconds)

### 2. `compare_detection_metrics.sh` (Comprehensive analysis)
- **Dependencies**: Python packages (see requirements_metrics.txt)
- **Output**: JSON metrics, visualizations, comprehensive report
- **Runtime**: Slower (~2-3 minutes)

## Quick Start

### Option 1: Simple Analysis (Recommended)
```bash
# Run basic metrics comparison
./simple_metrics_comparison.sh
```

### Option 2: Full Analysis with Visualizations
```bash
# Install Python dependencies first
pip install -r requirements_metrics.txt

# Run comprehensive analysis
./compare_detection_metrics.sh
```

## Prerequisites

### Required:
- `jq` (JSON processor) - Install: `sudo apt install jq` (Linux) or `brew install jq` (macOS)
- `python3` with standard library
- Bash shell environment

### Optional (for full analysis):
- Python packages: pandas, numpy, scikit-learn, matplotlib, seaborn

## Input Files Required

The scripts expect these files to exist:
1. `agentic-ids-local/src/logs/system_metrics.jsonl` - Agent inference results
2. `agentic-ids-local/src/data/UNSW_NB15/UNSW_NB15_training-set.csv` - Ground truth labels

## Output Files

### Simple Analysis (`simple_metrics_comparison.sh`):
- `simple_metrics/detailed_results.txt` - Flow-by-flow comparison
- Console output with confusion matrix and performance metrics

### Full Analysis (`compare_detection_metrics.sh`):
- `metrics_results/metrics_results_TIMESTAMP.json` - Comprehensive metrics
- `metrics_results/confusion_matrix.png` - Visualization
- `metrics_results/analysis_plots.png` - Score distributions
- `metrics_results/detection_summary_TIMESTAMP.md` - Summary report

## Metrics Calculated

### Core Performance Metrics:
- **Accuracy**: Overall correct classifications
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (True negatives + False positives)

### Confusion Matrix:
- **True Positives (TP)**: Correctly detected attacks
- **False Positives (FP)**: Benign traffic flagged as attacks
- **True Negatives (TN)**: Correctly identified benign traffic
- **False Negatives (FN)**: Missed attacks

### Additional Analysis (Full version):
- Performance by attack category (DoS, Exploits, etc.)
- Triage classification effectiveness
- Anomaly score distributions
- Error pattern analysis

## Understanding the Results

### Good Performance Indicators:
- **High Accuracy** (>90%): Overall system effectiveness
- **High Recall** (>85%): Good attack detection rate
- **Low False Positive Rate** (<10%): Minimal false alarms
- **Balanced F1-Score** (>80%): Good overall balance

### Areas for Improvement:
- **Low Recall**: May miss attacks - tune sensitivity
- **High False Positives**: Too many false alarms - increase threshold
- **Category-specific issues**: Some attack types harder to detect

## Flow ID Mapping

The scripts automatically:
1. Find the maximum numeric flow_id in system_metrics.jsonl
2. Compare only flows up to this maximum (as specified)
3. Skip UUID-based flow_ids for consistency
4. Map agent predictions to corresponding UNSW-NB15 ground truth labels

## Troubleshooting

### Common Issues:

1. **"jq not found"**
   ```bash
   # Install jq
   sudo apt install jq    # Ubuntu/Debian
   brew install jq        # macOS
   ```

2. **"No matching flow IDs found"**
   - Check that system_metrics.jsonl contains inference_result events
   - Verify UNSW-NB15 dataset path is correct

3. **Python dependency errors**
   ```bash
   # For full analysis, install dependencies:
   pip install -r requirements_metrics.txt
   ```

4. **Permission denied**
   ```bash
   chmod +x *.sh
   ```

## Example Output

```
===== DETECTION PERFORMANCE METRICS =====
Total Flows Analyzed: 83
Matched Flow IDs: 75

Confusion Matrix:
  True Positives (TP):  42
  False Positives (FP): 8
  True Negatives (TN):  20
  False Negatives (FN): 5

Performance Metrics:
  Accuracy:  0.827 (62/75)
  Precision: 0.840 (42/50)
  Recall:    0.894 (42/47)
  F1-Score:  0.866

Detection Rates:
  Attack Detection Rate (Sensitivity): 0.894
  False Positive Rate: 0.286
  Specificity (True Negative Rate): 0.714
```

## Customization

### Modify Thresholds:
Edit the scripts to change anomaly score thresholds or classification criteria.

### Add Custom Metrics:
Extend the Python calculation scripts to include domain-specific metrics.

### Change Input Paths:
Update the file paths at the beginning of each script for different datasets.

## Support

For issues or questions:
1. Check that all input files exist and are readable
2. Verify dependencies are installed  
3. Run the simple version first to isolate complex dependency issues
4. Check console output for specific error messages