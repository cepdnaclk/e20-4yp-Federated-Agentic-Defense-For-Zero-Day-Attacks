"""
XGBoost Baseline Evaluation on UNSW-NB15 Dataset
================================================

Single model training and evaluation with comprehensive metrics and visualizations.
No federated learning - just baseline XGBoost performance.
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
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_unsw_data(data_dir: Path):
    """Load UNSW-NB15 training and testing datasets."""
    print("\n" + "="*60)
    print("  LOADING UNSW-NB15 DATASET")
    print("="*60)
    
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"UNSW-NB15 datasets not found in {data_dir}")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"  Training samples: {len(df_train):,}")
    print(f"  Testing samples:  {len(df_test):,}")
    
    # Define features (excluding labels and ID columns)
    exclude_cols = ['id', 'label', 'attack_cat']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    
    # Select numeric columns only
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Features: {len(numeric_cols)}")
    
    # Extract features
    X_train = df_train[numeric_cols].values.astype(np.float32)
    X_test = df_test[numeric_cols].values.astype(np.float32)
    
    # Labels
    y_train_raw = df_train['attack_cat'].fillna('Normal').values
    y_test_raw = df_test['attack_cat'].fillna('Normal').values
    
    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train_raw, y_test_raw])
    label_encoder.fit(all_labels)
    
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    num_classes = len(label_encoder.classes_)
    print(f"  Classes ({num_classes}): {label_encoder.classes_.tolist()}")
    
    # Class distribution
    print("\n  Training set class distribution:")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    for cls_idx, count in zip(train_unique, train_counts):
        cls_name = label_encoder.classes_[cls_idx]
        pct = 100 * count / len(y_train)
        print(f"    {cls_name:15s}: {count:6d} ({pct:5.2f}%)")
    
    print("\n  Testing set class distribution:")
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    for cls_idx, count in zip(test_unique, test_counts):
        cls_name = label_encoder.classes_[cls_idx]
        pct = 100 * count / len(y_test)
        print(f"    {cls_name:15s}: {count:6d} ({pct:5.2f}%)")
    
    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, label_encoder, numeric_cols


def train_xgboost(X_train, y_train, num_classes, n_estimators=100, max_depth=6):
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("  TRAINING XGBOOST MODEL")
    print("="*60)
    print(f"  Estimators: {n_estimators}")
    print(f"  Max depth:  {max_depth}")
    print(f"  Classes:    {num_classes}")
    
    # XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': max_depth,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42,
        'verbosity': 1,
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Train
    print("\n  Training...")
    start_time = datetime.now()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False
    )
    
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"  Training time: {train_time:.2f}s")
    
    return model, train_time


def evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder):
    """Comprehensive model evaluation."""
    print("\n" + "="*60)
    print("  MODEL EVALUATION")
    print("="*60)
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train)
    dtest = xgb.DMatrix(X_test)
    
    # Predictions
    y_train_prob = model.predict(dtrain)
    y_test_prob = model.predict(dtest)
    
    y_train_pred = np.argmax(y_train_prob, axis=1)
    y_test_pred = np.argmax(y_test_prob, axis=1)
    
    # Training metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    
    # Testing metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print("\n  TRAINING SET METRICS:")
    print(f"    Accuracy:           {train_acc*100:.2f}%")
    print(f"    F1 Score (macro):   {train_f1_macro*100:.2f}%")
    print(f"    F1 Score (weighted):{train_f1_weighted*100:.2f}%")
    print(f"    Precision:          {train_precision*100:.2f}%")
    print(f"    Recall:             {train_recall*100:.2f}%")
    
    print("\n  TESTING SET METRICS:")
    print(f"    Accuracy:           {test_acc*100:.2f}%")
    print(f"    F1 Score (macro):   {test_f1_macro*100:.2f}%")
    print(f"    F1 Score (weighted):{test_f1_weighted*100:.2f}%")
    print(f"    Precision:          {test_precision*100:.2f}%")
    print(f"    Recall:             {test_recall*100:.2f}%")
    
    # Per-class metrics
    print("\n  PER-CLASS METRICS (Test Set):")
    print("-"*60)
    report = classification_report(
        y_test, y_test_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    
    print(f"  {'Class':<15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-"*60)
    for cls_name in label_encoder.classes_:
        if cls_name in report:
            metrics = report[cls_name]
            print(f"  {cls_name:<15s} {metrics['precision']*100:>9.2f}% {metrics['recall']*100:>9.2f}% {metrics['f1-score']*100:>9.2f}% {int(metrics['support']):>10d}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    metrics = {
        'train': {
            'accuracy': float(train_acc),
            'f1_macro': float(train_f1_macro),
            'f1_weighted': float(train_f1_weighted),
            'precision': float(train_precision),
            'recall': float(train_recall),
        },
        'test': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted),
            'precision': float(test_precision),
            'recall': float(test_recall),
        },
        'per_class': report,
        'confusion_matrix': cm.tolist(),
    }
    
    return metrics, y_test_pred, y_test_prob, cm


def plot_results(metrics, y_test, y_test_pred, y_test_prob, cm, label_encoder, model, feature_names, output_dir):
    """Generate comprehensive visualization plots."""
    print("\n" + "="*60)
    print("  GENERATING PLOTS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_
    
    # 1. Confusion Matrix Heatmap
    print("  [1/6] Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('XGBoost Confusion Matrix - UNSW-NB15', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Normalized Confusion Matrix
    print("  [2/6] Normalized Confusion Matrix...")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('XGBoost Normalized Confusion Matrix - UNSW-NB15', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Per-Class Metrics Bar Chart
    print("  [3/6] Per-Class Metrics...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for cls_name in class_names:
        if cls_name in metrics['per_class']:
            precision_scores.append(metrics['per_class'][cls_name]['precision'] * 100)
            recall_scores.append(metrics['per_class'][cls_name]['recall'] * 100)
            f1_scores.append(metrics['per_class'][cls_name]['f1-score'] * 100)
        else:
            precision_scores.append(0)
            recall_scores.append(0)
            f1_scores.append(0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_scores, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall_scores, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Attack Category', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('XGBoost Per-Class Performance - UNSW-NB15', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance
    print("  [4/6] Feature Importance...")
    importance_dict = model.get_score(importance_type='weight')
    
    if importance_dict:
        # Map feature names
        importance_df = pd.DataFrame([
            {'feature': f'f{i}' if f'f{i}' in importance_dict else feature_names[i] if i < len(feature_names) else f'f{i}', 
             'importance': importance_dict.get(f'f{i}', 0)}
            for i in range(len(feature_names))
        ])
        importance_df['feature'] = feature_names[:len(importance_df)]
        importance_df = importance_df.sort_values('importance', ascending=True).tail(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['feature'], importance_df['importance'], color='#9b59b6')
        ax.set_xlabel('Importance (Weight)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('XGBoost Top 20 Feature Importance - UNSW-NB15', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. ROC Curves (One-vs-Rest)
    print("  [5/6] ROC Curves...")
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        if y_test_bin[:, i].sum() > 0:  # Only plot if class exists
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{cls_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('XGBoost ROC Curves (One-vs-Rest) - UNSW-NB15', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Train vs Test Metrics Comparison
    print("  [6/6] Train vs Test Comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision', 'Recall']
    train_values = [
        metrics['train']['accuracy'] * 100,
        metrics['train']['f1_macro'] * 100,
        metrics['train']['f1_weighted'] * 100,
        metrics['train']['precision'] * 100,
        metrics['train']['recall'] * 100,
    ]
    test_values = [
        metrics['test']['accuracy'] * 100,
        metrics['test']['f1_macro'] * 100,
        metrics['test']['f1_weighted'] * 100,
        metrics['test']['precision'] * 100,
        metrics['test']['recall'] * 100,
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Training', color='#3498db')
    bars2 = ax.bar(x + width/2, test_values, width, label='Testing', color='#e74c3c')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('XGBoost Train vs Test Performance - UNSW-NB15', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'train_test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  All plots saved to: {output_dir}")
    return output_dir


def main():
    """Main function."""
    print("\n" + "="*70)
    print("  XGBOOST BASELINE EVALUATION - UNSW-NB15 DATASET")
    print("  Single Model Training and Testing")
    print("="*70)
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    output_dir = script_dir / "xgboost_baseline_results"
    
    # Configuration
    n_estimators = 100
    max_depth = 6
    
    # Load data
    X_train, y_train, X_test, y_test, label_encoder, feature_names = load_unsw_data(data_dir)
    
    num_classes = len(label_encoder.classes_)
    
    # Train model
    model, train_time = train_xgboost(X_train, y_train, num_classes, n_estimators, max_depth)
    
    # Evaluate
    metrics, y_test_pred, y_test_prob, cm = evaluate_model(
        model, X_train, y_train, X_test, y_test, label_encoder
    )
    
    # Plot results
    plot_dir = plot_results(
        metrics, y_test, y_test_pred, y_test_prob, cm,
        label_encoder, model, feature_names, output_dir
    )
    
    # Save metrics to JSON
    results = {
        'config': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'num_classes': num_classes,
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'features': len(feature_names),
        },
        'training_time_seconds': train_time,
        'metrics': {
            'train': metrics['train'],
            'test': metrics['test'],
        },
        'class_names': label_encoder.classes_.tolist(),
    }
    
    results_path = output_dir / 'xgboost_baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"""
  Configuration:
    - Estimators: {n_estimators}
    - Max Depth:  {max_depth}
    - Features:   {len(feature_names)}
    - Classes:    {num_classes}

  Training Set ({len(y_train):,} samples):
    - Accuracy:   {metrics['train']['accuracy']*100:.2f}%
    - F1 (macro): {metrics['train']['f1_macro']*100:.2f}%
    - Precision:  {metrics['train']['precision']*100:.2f}%
    - Recall:     {metrics['train']['recall']*100:.2f}%

  Testing Set ({len(y_test):,} samples):
    - Accuracy:   {metrics['test']['accuracy']*100:.2f}%
    - F1 (macro): {metrics['test']['f1_macro']*100:.2f}%
    - Precision:  {metrics['test']['precision']*100:.2f}%
    - Recall:     {metrics['test']['recall']*100:.2f}%

  Training Time: {train_time:.2f}s

  Results saved to:
    - Metrics: {results_path}
    - Plots:   {plot_dir}
""")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
