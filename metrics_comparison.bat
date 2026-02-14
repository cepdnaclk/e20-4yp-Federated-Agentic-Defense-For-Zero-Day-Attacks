@echo off
REM Windows-compatible Detection Metrics Comparison Script
REM Usage: metrics_comparison.bat

echo ==========================================
echo   Detection Metrics Comparison (Windows)
echo ==========================================

REM Configuration
set "SYSTEM_METRICS=agentic-ids-local\src\logs\system_metrics.jsonl"
set "UNSW_TRAINING=agentic-ids-local\src\data\UNSW_NB15\UNSW_NB15_training-set.csv"
set "OUTPUT_DIR=win_metrics"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 is required but not found.
    echo Please install Python 3 and add it to PATH.
    pause
    exit /b 1
)

echo Creating Windows-compatible metrics calculator...

REM Create Python script for complete analysis
(
echo import json
echo import csv
echo import sys
echo import os
echo from collections import Counter
echo import re
echo.
echo def extract_numeric_flow_ids^(jsonl_file^):
echo     """Extract all numeric flow_ids from system metrics"""
echo     flow_ids = []
echo     with open^(jsonl_file, 'r', encoding='utf-8'^) as f:
echo         for line in f:
echo             line = line.strip^(^)
echo             if not line:
echo                 continue
echo             try:
echo                 data = json.loads^(line^)
echo                 if data.get^('event_type'^) == 'inference_result':
echo                     flow_id = data.get^('flow_id'^)
echo                     if isinstance^(flow_id, int^) and flow_id ^> 0:
echo                         flow_ids.append^(flow_id^)
echo             except json.JSONDecodeError:
echo                 continue
echo     return flow_ids
echo.
echo def load_agent_predictions^(jsonl_file, max_flow_id^):
echo     """Load agent predictions from system metrics"""
echo     predictions = {}
echo     with open^(jsonl_file, 'r', encoding='utf-8'^) as f:
echo         for line in f:
echo             line = line.strip^(^)
echo             if not line:
echo                 continue
echo             try:
echo                 data = json.loads^(line^)
echo                 if data.get^('event_type'^) == 'inference_result':
echo                     flow_id = data.get^('flow_id'^)
echo                     if isinstance^(flow_id, int^) and 1 ^<= flow_id ^<= max_flow_id:
echo                         predictions[flow_id] = {
echo                             'prediction': data.get^('prediction', 0^),
echo                             'anomaly_score': data.get^('anomaly_score', 0.0^),
echo                             'threshold_exceeded': data.get^('threshold_exceeded', False^)
echo                         }
echo             except json.JSONDecodeError:
echo                 continue
echo     return predictions
echo.
echo def load_ground_truth^(csv_file, max_flow_id^):
echo     """Load ground truth labels from UNSW-NB15"""
echo     labels = {}
echo     try:
echo         with open^(csv_file, 'r', encoding='utf-8'^) as f:
echo             reader = csv.reader^(f^)
echo             next^(reader^)  # Skip header
echo             for row in reader:
echo                 if len^(row^) ^>= 2:
echo                     try:
echo                         flow_id = int^(row[0]^)
echo                         if 1 ^<= flow_id ^<= max_flow_id:
echo                             label = int^(row[-1]^)  # Last column is label
echo                             attack_cat = row[-2] if len^(row^) ^> 2 else 'Unknown'
echo                             labels[flow_id] = {'label': label, 'category': attack_cat}
echo                     except ^(ValueError, IndexError^):
echo                         continue
echo     except FileNotFoundError:
echo         print^(f"Error: Ground truth file not found: {csv_file}"^)
echo         return {}
echo     return labels
echo.
echo def calculate_metrics^(predictions, ground_truth^):
echo     """Calculate performance metrics"""
echo     tp = fp = tn = fn = 0
echo     matches = []
echo     
echo     for flow_id in predictions:
echo         if flow_id in ground_truth:
echo             agent_pred = predictions[flow_id]['prediction']
echo             true_label = ground_truth[flow_id]['label']
echo             
echo             matches.append^(^(flow_id, agent_pred, true_label^)^)
echo             
echo             if agent_pred == 1 and true_label == 1:
echo                 tp += 1
echo             elif agent_pred == 1 and true_label == 0:
echo                 fp += 1
echo             elif agent_pred == 0 and true_label == 1:
echo                 fn += 1
echo             elif agent_pred == 0 and true_label == 0:
echo                 tn += 1
echo     
echo     total = tp + fp + tn + fn
echo     if total == 0:
echo         return None, matches
echo     
echo     accuracy = ^(tp + tn^) / total
echo     precision = tp / ^(tp + fp^) if ^(tp + fp^) ^> 0 else 0
echo     recall = tp / ^(tp + fn^) if ^(tp + fn^) ^> 0 else 0
echo     f1 = 2 * ^(precision * recall^) / ^(precision + recall^) if ^(precision + recall^) ^> 0 else 0
echo     fpr = fp / ^(fp + tn^) if ^(fp + tn^) ^> 0 else 0
echo     specificity = tn / ^(tn + fp^) if ^(tn + fp^) ^> 0 else 0
echo     
echo     metrics = {
echo         'total': total,
echo         'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
echo         'accuracy': accuracy,
echo         'precision': precision,
echo         'recall': recall,
echo         'f1_score': f1,
echo         'fpr': fpr,
echo         'specificity': specificity
echo     }
echo     
echo     return metrics, matches
echo.
echo def main^(^):
echo     # File paths
echo     system_metrics = sys.argv[1] if len^(sys.argv^) ^> 1 else 'agentic-ids-local/src/logs/system_metrics.jsonl'
echo     unsw_file = sys.argv[2] if len^(sys.argv^) ^> 2 else 'agentic-ids-local/src/data/UNSW_NB15/UNSW_NB15_training-set.csv'
echo     output_dir = sys.argv[3] if len^(sys.argv^) ^> 3 else 'win_metrics'
echo     
echo     print^(f"System metrics: {system_metrics}"^)
echo     print^(f"UNSW dataset: {unsw_file}"^)
echo     print^(f"Output directory: {output_dir}"^)
echo     print^(^)
echo     
echo     # Extract maximum flow ID
echo     print^("Finding maximum flow ID..."^)
echo     flow_ids = extract_numeric_flow_ids^(system_metrics^)
echo     if not flow_ids:
echo         print^("Error: No numeric flow IDs found in system metrics!"^)
echo         return
echo     
echo     max_flow_id = max^(flow_ids^)
echo     print^(f"Maximum flow ID: {max_flow_id}"^)
echo     print^(f"Total agent predictions: {len^(flow_ids^)}"^)
echo     print^(^)
echo     
echo     # Load data
echo     print^("Loading agent predictions..."^)
echo     predictions = load_agent_predictions^(system_metrics, max_flow_id^)
echo     
echo     print^("Loading ground truth labels..."^)
echo     ground_truth = load_ground_truth^(unsw_file, max_flow_id^)
echo     
echo     if not predictions:
echo         print^("Error: No agent predictions loaded!"^)
echo         return
echo     if not ground_truth:
echo         print^("Error: No ground truth labels loaded!"^)
echo         return
echo     
echo     print^(f"Loaded {len^(predictions^)} agent predictions"^)
echo     print^(f"Loaded {len^(ground_truth^)} ground truth labels"^)
echo     print^(^)
echo     
echo     # Calculate metrics
echo     print^("Calculating performance metrics..."^)
echo     metrics, matches = calculate_metrics^(predictions, ground_truth^)
echo     
echo     if metrics is None:
echo         print^("Error: No matching flow IDs found!"^)
echo         return
echo     
echo     # Print results
echo     print^("===== DETECTION PERFORMANCE METRICS ====="^)
echo     print^(f"Total Flows Analyzed: {metrics['total']}"^)
echo     print^(f"Matched Flow IDs: {len^(matches^)}"^)
echo     print^(^)
echo     print^("Confusion Matrix:"^)
echo     print^(f"  True Positives ^(TP^):  {metrics['tp']}"^)
echo     print^(f"  False Positives ^(FP^): {metrics['fp']}"^)
echo     print^(f"  True Negatives ^(TN^):  {metrics['tn']}"^)
echo     print^(f"  False Negatives ^(FN^): {metrics['fn']}"^)
echo     print^(^)
echo     print^("Performance Metrics:"^)
echo     print^(f"  Accuracy:  {metrics['accuracy']:.3f}"^)
echo     print^(f"  Precision: {metrics['precision']:.3f}"^)
echo     print^(f"  Recall:    {metrics['recall']:.3f}"^)
echo     print^(f"  F1-Score:  {metrics['f1_score']:.3f}"^)
echo     print^(^)
echo     print^("Detection Rates:"^)
echo     print^(f"  Attack Detection Rate: {metrics['recall']:.3f}"^)
echo     print^(f"  False Positive Rate: {metrics['fpr']:.3f}"^)
echo     print^(f"  Specificity: {metrics['specificity']:.3f}"^)
echo     
echo     # Save detailed results
echo     os.makedirs^(output_dir, exist_ok=True^)
echo     results_file = os.path.join^(output_dir, 'detailed_results.csv'^)
echo     
echo     with open^(results_file, 'w', newline='', encoding='utf-8'^) as f:
echo         writer = csv.writer^(f^)
echo         writer.writerow^(['Flow_ID', 'Agent_Prediction', 'Ground_Truth', 'Match', 'Anomaly_Score']^)
echo         
echo         for flow_id, agent_pred, true_label in matches:
echo             match_status = "CORRECT" if agent_pred == true_label else "INCORRECT"
echo             score = predictions[flow_id]['anomaly_score']
echo             writer.writerow^([flow_id, agent_pred, true_label, match_status, f'{score:.6f}']^)
echo     
echo     print^(f"\nDetailed results saved to: {results_file}"^)
echo     
echo     # Error analysis
echo     fp_flows = [fid for fid, agent_pred, true_label in matches if agent_pred == 1 and true_label == 0]
echo     fn_flows = [fid for fid, agent_pred, true_label in matches if agent_pred == 0 and true_label == 1]
echo     
echo     if fp_flows:
echo         fp_display = ', '.join^(map^(str, fp_flows[:10]^)^)
echo         if len^(fp_flows^) ^> 10:
echo             fp_display += '...'
echo         print^(f"\nFalse Positive Flow IDs: {fp_display}"^)
echo     if fn_flows:
echo         fn_display = ', '.join^(map^(str, fn_flows[:10]^)^)
echo         if len^(fn_flows^) ^> 10:
echo             fn_display += '...'
echo         print^(f"False Negative Flow IDs: {fn_display}"^)
echo     
echo     print^(f"\nAnalysis complete! Check {output_dir}/ for detailed results."^)
echo.
echo if __name__ == "__main__":
echo     main^(^)
) > "%OUTPUT_DIR%\windows_metrics.py"

echo Running metrics analysis...
python "%OUTPUT_DIR%\windows_metrics.py" "%SYSTEM_METRICS%" "%UNSW_TRAINING%" "%OUTPUT_DIR%"

echo.
echo ==========================================
echo Windows metrics analysis complete!
echo Check %OUTPUT_DIR%\ for detailed results
echo ==========================================
pause