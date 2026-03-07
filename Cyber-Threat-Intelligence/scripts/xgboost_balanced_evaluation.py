"""
XGBoost with Class Imbalance Handling - UNSW-NB15 Dataset
=========================================================

Techniques to improve minority class F1 scores:
1. SMOTE (Synthetic Minority Over-sampling)
2. Class Weights (cost-sensitive learning)
3. SMOTE + Tomek Links (hybrid sampling)
4. Threshold Tuning (per-class optimal thresholds)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Installing imbalanced-learn...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"])
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    IMBLEARN_AVAILABLE = True

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_unsw_data(data_dir: Path):
    """Load UNSW-NB15 training and testing datasets."""
    print("\n" + "="*60)
    print("  LOADING UNSW-NB15 DATASET")
    print("="*60)
    
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"  Training samples: {len(df_train):,}")
    print(f"  Testing samples:  {len(df_test):,}")
    
    # Features
    exclude_cols = ['id', 'label', 'attack_cat']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = df_train[numeric_cols].values.astype(np.float32)
    X_test = df_test[numeric_cols].values.astype(np.float32)
    
    y_train_raw = df_train['attack_cat'].fillna('Normal').values
    y_test_raw = df_test['attack_cat'].fillna('Normal').values
    
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train_raw, y_test_raw])
    label_encoder.fit(all_labels)
    
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    print(f"  Classes: {label_encoder.classes_.tolist()}")
    
    # Show imbalance
    print("\n  Class Distribution (Training):")
    counter = Counter(y_train)
    max_count = max(counter.values())
    for cls_idx in sorted(counter.keys()):
        count = counter[cls_idx]
        ratio = max_count / count
        bar = "█" * int(50 * count / max_count)
        print(f"    {label_encoder.classes_[cls_idx]:15s}: {count:6d} (1:{ratio:.0f}) {bar}")
    
    # Handle NaN/Inf and standardize
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, label_encoder, numeric_cols


def compute_class_weights(y_train):
    """Compute balanced class weights."""
    counter = Counter(y_train)
    n_samples = len(y_train)
    n_classes = len(counter)
    
    weights = {}
    for cls, count in counter.items():
        weights[cls] = n_samples / (n_classes * count)
    
    # Normalize so minimum weight is 1
    min_weight = min(weights.values())
    weights = {k: v / min_weight for k, v in weights.items()}
    
    return weights


def compute_sample_weights(y_train, class_weights):
    """Convert class weights to per-sample weights."""
    return np.array([class_weights[y] for y in y_train])


def apply_smote(X_train, y_train, strategy='auto', k_neighbors=5):
    """Apply SMOTE oversampling."""
    print(f"\n  Applying SMOTE (k={k_neighbors})...")
    
    # Adjust k_neighbors for very small classes
    counter = Counter(y_train)
    min_samples = min(counter.values())
    k = min(k_neighbors, min_samples - 1)
    k = max(1, k)
    
    smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"  Before: {len(y_train):,} samples")
    print(f"  After:  {len(y_resampled):,} samples")
    
    return X_resampled, y_resampled


def apply_borderline_smote(X_train, y_train):
    """Apply Borderline-SMOTE (focus on decision boundary samples)."""
    print("\n  Applying Borderline-SMOTE...")
    
    counter = Counter(y_train)
    min_samples = min(counter.values())
    k = min(5, min_samples - 1)
    k = max(1, k)
    
    try:
        bsmote = BorderlineSMOTE(k_neighbors=k, random_state=42)
        X_resampled, y_resampled = bsmote.fit_resample(X_train, y_train)
        print(f"  Before: {len(y_train):,} samples")
        print(f"  After:  {len(y_resampled):,} samples")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"  Borderline-SMOTE failed: {e}, falling back to SMOTE")
        return apply_smote(X_train, y_train)


def apply_adasyn(X_train, y_train):
    """Apply ADASYN (Adaptive Synthetic Sampling)."""
    print("\n  Applying ADASYN...")
    
    counter = Counter(y_train)
    min_samples = min(counter.values())
    k = min(5, min_samples - 1)
    k = max(1, k)
    
    adasyn = ADASYN(n_neighbors=k, random_state=42)
    try:
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        print(f"  Before: {len(y_train):,} samples")
        print(f"  After:  {len(y_resampled):,} samples")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"  ADASYN failed: {e}")
        print("  Falling back to SMOTE...")
        return apply_smote(X_train, y_train)


def train_xgboost(X_train, y_train, num_classes, sample_weights=None, 
                  n_estimators=100, max_depth=6, scale_pos_weight=None):
    """Train XGBoost model with optional class weights."""
    
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': max_depth,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42,
        'verbosity': 0,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False
    )
    
    return model


def find_optimal_thresholds(model, X_val, y_val, num_classes):
    """Find optimal decision thresholds per class using validation data."""
    dval = xgb.DMatrix(X_val)
    y_prob = model.predict(dval)
    
    optimal_thresholds = []
    
    for cls in range(num_classes):
        best_thresh = 0.5
        best_f1 = 0
        
        # Try different thresholds
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_cls = (y_prob[:, cls] >= thresh).astype(int)
            y_true_cls = (y_val == cls).astype(int)
            
            if y_pred_cls.sum() > 0:
                f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        
        optimal_thresholds.append(best_thresh)
    
    return np.array(optimal_thresholds)


def predict_with_thresholds(model, X, thresholds):
    """Make predictions using per-class thresholds."""
    dmatrix = xgb.DMatrix(X)
    y_prob = model.predict(dmatrix)
    
    # Adjust probabilities by thresholds
    adjusted_prob = y_prob / thresholds
    y_pred = np.argmax(adjusted_prob, axis=1)
    
    return y_pred, y_prob


def evaluate_model(model, X_test, y_test, label_encoder, thresholds=None):
    """Evaluate model with optional threshold tuning."""
    dtest = xgb.DMatrix(X_test)
    y_prob = model.predict(dtest)
    
    if thresholds is not None:
        adjusted_prob = y_prob / thresholds
        y_pred = np.argmax(adjusted_prob, axis=1)
    else:
        y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'report': report,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def run_experiment(name, X_train, y_train, X_test, y_test, label_encoder, 
                   sample_weights=None, use_threshold_tuning=False):
    """Run a single experiment."""
    num_classes = len(label_encoder.classes_)
    
    print(f"\n  Training XGBoost ({name})...")
    start = datetime.now()
    model = train_xgboost(X_train, y_train, num_classes, sample_weights)
    train_time = (datetime.now() - start).total_seconds()
    
    thresholds = None
    if use_threshold_tuning:
        print("  Finding optimal thresholds...")
        # Use 20% of training data for threshold tuning
        n_val = int(len(X_train) * 0.2)
        thresholds = find_optimal_thresholds(
            model, X_train[:n_val], y_train[:n_val], num_classes
        )
        print(f"  Thresholds: {thresholds.round(2).tolist()}")
    
    results = evaluate_model(model, X_test, y_test, label_encoder, thresholds)
    results['train_time'] = train_time
    results['model'] = model
    results['thresholds'] = thresholds
    
    return results


def print_comparison_table(all_results, label_encoder):
    """Print comparison table of all methods."""
    print("\n" + "="*80)
    print("  COMPARISON: ALL METHODS")
    print("="*80)
    
    # Overall metrics
    print("\n  OVERALL METRICS:")
    print("-"*80)
    print(f"  {'Method':<30s} {'Accuracy':>12s} {'F1 (Macro)':>12s} {'F1 (Weighted)':>14s}")
    print("-"*80)
    
    for name, results in all_results.items():
        acc = results['accuracy'] * 100
        f1m = results['f1_macro'] * 100
        f1w = results['f1_weighted'] * 100
        print(f"  {name:<30s} {acc:>11.2f}% {f1m:>11.2f}% {f1w:>13.2f}%")
    
    # Per-class F1 scores
    print("\n\n  PER-CLASS F1 SCORES (%):")
    print("-"*120)
    header = f"  {'Class':<15s}"
    for name in all_results.keys():
        header += f" {name[:12]:>12s}"
    print(header)
    print("-"*120)
    
    for cls_name in label_encoder.classes_:
        row = f"  {cls_name:<15s}"
        for name, results in all_results.items():
            if cls_name in results['report']:
                f1 = results['report'][cls_name]['f1-score'] * 100
                row += f" {f1:>11.2f}%"
            else:
                row += f" {'N/A':>12s}"
        print(row)
    
    print("-"*120)


def plot_comparison(all_results, label_encoder, output_dir):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = label_encoder.classes_
    method_names = list(all_results.keys())
    
    # 1. Overall Metrics Comparison
    print("\n  [1/4] Overall Metrics Comparison...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(method_names))
    width = 0.25
    
    accuracies = [all_results[m]['accuracy'] * 100 for m in method_names]
    f1_macros = [all_results[m]['f1_macro'] * 100 for m in method_names]
    f1_weighted = [all_results[m]['f1_weighted'] * 100 for m in method_names]
    
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar(x, f1_macros, width, label='F1 (Macro)', color='#e74c3c')
    bars3 = ax.bar(x + width, f1_weighted, width, label='F1 (Weighted)', color='#2ecc71')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('XGBoost Class Imbalance Handling - Overall Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Class F1 Heatmap
    print("  [2/4] Per-Class F1 Heatmap...")
    f1_matrix = []
    for cls_name in class_names:
        row = []
        for method in method_names:
            if cls_name in all_results[method]['report']:
                row.append(all_results[method]['report'][cls_name]['f1-score'] * 100)
            else:
                row.append(0)
        f1_matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(method_names, rotation=30, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(method_names)):
            text = ax.text(j, i, f'{f1_matrix[i][j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Per-Class F1 Score (%) by Method', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='F1 Score (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_f1_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Minority Classes Focus
    print("  [3/4] Minority Classes Focus...")
    minority_classes = ['Analysis', 'Backdoor', 'DoS', 'Shellcode', 'Worms']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(minority_classes))
    width = 0.8 / len(method_names)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))
    
    for idx, (method, color) in enumerate(zip(method_names, colors)):
        f1_scores = []
        for cls in minority_classes:
            if cls in all_results[method]['report']:
                f1_scores.append(all_results[method]['report'][cls]['f1-score'] * 100)
            else:
                f1_scores.append(0)
        
        offset = (idx - len(method_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=method, color=color)
    
    ax.set_xlabel('Minority Class', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('F1 Scores on Minority Classes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(minority_classes)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'minority_classes_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Improvement from Baseline
    print("  [4/4] Improvement from Baseline...")
    baseline_f1 = {cls: all_results['Baseline']['report'].get(cls, {}).get('f1-score', 0) * 100 
                   for cls in class_names}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods_to_compare = [m for m in method_names if m != 'Baseline'][:4]
    
    for ax, method in zip(axes.flat, methods_to_compare):
        improvements = []
        for cls in class_names:
            baseline = baseline_f1[cls]
            current = all_results[method]['report'].get(cls, {}).get('f1-score', 0) * 100
            improvements.append(current - baseline)
        
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        ax.barh(class_names, improvements, color=colors)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('F1 Improvement (%)', fontsize=10)
        ax.set_title(f'{method} vs Baseline', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('F1 Score Improvement Over Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_from_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Plots saved to: {output_dir}")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("  XGBOOST CLASS IMBALANCE HANDLING - UNSW-NB15")
    print("  Comparing: Baseline, SMOTE, Class Weights, SMOTE+Tomek, Threshold Tuning")
    print("="*70)
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    output_dir = script_dir / "xgboost_balanced_results"
    
    # Load data
    X_train, y_train, X_test, y_test, label_encoder, feature_names = load_unsw_data(data_dir)
    
    num_classes = len(label_encoder.classes_)
    all_results = {}
    
    # ========================================
    # 1. BASELINE (No balancing)
    # ========================================
    print("\n" + "-"*60)
    print("  EXPERIMENT 1: BASELINE (No Balancing)")
    print("-"*60)
    
    results = run_experiment(
        "Baseline", X_train, y_train, X_test, y_test, label_encoder
    )
    all_results['Baseline'] = results
    print(f"  Accuracy: {results['accuracy']*100:.2f}%, F1 (macro): {results['f1_macro']*100:.2f}%")
    
    # ========================================
    # 2. CLASS WEIGHTS
    # ========================================
    print("\n" + "-"*60)
    print("  EXPERIMENT 2: CLASS WEIGHTS (Cost-Sensitive)")
    print("-"*60)
    
    class_weights = compute_class_weights(y_train)
    print("  Class weights computed:")
    for cls_idx, weight in sorted(class_weights.items()):
        print(f"    {label_encoder.classes_[cls_idx]:15s}: {weight:.2f}")
    
    sample_weights = compute_sample_weights(y_train, class_weights)
    
    results = run_experiment(
        "Class Weights", X_train, y_train, X_test, y_test, label_encoder,
        sample_weights=sample_weights
    )
    all_results['Class Weights'] = results
    print(f"  Accuracy: {results['accuracy']*100:.2f}%, F1 (macro): {results['f1_macro']*100:.2f}%")
    
    # ========================================
    # 3. SMOTE
    # ========================================
    print("\n" + "-"*60)
    print("  EXPERIMENT 3: SMOTE (Oversampling)")
    print("-"*60)
    
    X_smote, y_smote = apply_smote(X_train, y_train)
    
    results = run_experiment(
        "SMOTE", X_smote, y_smote, X_test, y_test, label_encoder
    )
    all_results['SMOTE'] = results
    print(f"  Accuracy: {results['accuracy']*100:.2f}%, F1 (macro): {results['f1_macro']*100:.2f}%")
    
    # ========================================
    # 4. BORDERLINE-SMOTE (Focus on boundary samples)
    # ========================================
    print("\n" + "-"*60)
    print("  EXPERIMENT 4: BORDERLINE-SMOTE (Boundary Focus)")
    print("-"*60)
    
    X_bsmote, y_bsmote = apply_borderline_smote(X_train, y_train)
    
    results = run_experiment(
        "Borderline-SMOTE", X_bsmote, y_bsmote, X_test, y_test, label_encoder
    )
    all_results['Borderline-SMOTE'] = results
    print(f"  Accuracy: {results['accuracy']*100:.2f}%, F1 (macro): {results['f1_macro']*100:.2f}%")
    
    # ========================================
    # 5. SMOTE + CLASS WEIGHTS
    # ========================================
    print("\n" + "-"*60)
    print("  EXPERIMENT 5: SMOTE + CLASS WEIGHTS (Combined)")
    print("-"*60)
    
    # Use lighter weights since SMOTE already balances
    light_weights = {k: max(1.0, v * 0.5) for k, v in class_weights.items()}
    sample_weights_smote = compute_sample_weights(y_smote, light_weights)
    
    results = run_experiment(
        "SMOTE+Weights", X_smote, y_smote, X_test, y_test, label_encoder,
        sample_weights=sample_weights_smote
    )
    all_results['SMOTE+Weights'] = results
    print(f"  Accuracy: {results['accuracy']*100:.2f}%, F1 (macro): {results['f1_macro']*100:.2f}%")
    
    # ========================================
    # 6. THRESHOLD TUNING
    # ========================================
    print("\n" + "-"*60)
    print("  EXPERIMENT 6: THRESHOLD TUNING (Per-Class Thresholds)")
    print("-"*60)
    
    results = run_experiment(
        "Threshold Tuning", X_train, y_train, X_test, y_test, label_encoder,
        use_threshold_tuning=True
    )
    all_results['Threshold Tuning'] = results
    print(f"  Accuracy: {results['accuracy']*100:.2f}%, F1 (macro): {results['f1_macro']*100:.2f}%")
    
    # ========================================
    # COMPARISON
    # ========================================
    print_comparison_table(all_results, label_encoder)
    
    # Generate plots
    print("\n" + "="*60)
    print("  GENERATING COMPARISON PLOTS")
    print("="*60)
    
    # Remove non-serializable items for plotting
    plot_results = {k: {kk: vv for kk, vv in v.items() if kk not in ['model', 'y_pred', 'y_prob', 'thresholds']} 
                    for k, v in all_results.items()}
    plot_comparison(all_results, label_encoder, output_dir)
    
    # Save results to JSON
    json_results = {}
    for method, results in all_results.items():
        json_results[method] = {
            'accuracy': float(results['accuracy']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted']),
            'train_time': float(results['train_time']),
            'per_class': {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))} 
                          for k, v in results['report'].items() if isinstance(v, dict)}
        }
    
    results_path = output_dir / 'balanced_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Final Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY - BEST METHODS FOR MINORITY CLASSES")
    print("="*70)
    
    minority_classes = ['Analysis', 'Backdoor', 'DoS', 'Shellcode', 'Worms']
    
    print(f"\n  {'Class':<15s} {'Best Method':<20s} {'Baseline F1':>12s} {'Best F1':>12s} {'Improvement':>12s}")
    print("-"*75)
    
    for cls in minority_classes:
        baseline_f1 = all_results['Baseline']['report'].get(cls, {}).get('f1-score', 0) * 100
        
        best_method = 'Baseline'
        best_f1 = baseline_f1
        
        for method, results in all_results.items():
            f1 = results['report'].get(cls, {}).get('f1-score', 0) * 100
            if f1 > best_f1:
                best_f1 = f1
                best_method = method
        
        improvement = best_f1 - baseline_f1
        print(f"  {cls:<15s} {best_method:<20s} {baseline_f1:>11.2f}% {best_f1:>11.2f}% {improvement:>+11.2f}%")
    
    print("\n" + "="*70)
    print(f"  Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
