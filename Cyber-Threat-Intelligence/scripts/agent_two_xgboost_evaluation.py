"""
Agent Two XGBoost Model Evaluation on UNSW-NB15
================================================

Evaluates the existing Agent Two ThreatClassifier model
on UNSW-NB15 training and testing datasets.

Uses the pre-trained model from models/agent_two/classifier
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import data pipeline for proper preprocessing
from data_pipeline.config import DatasetConfig
from data_pipeline.data_loader import DataLoader
from data_pipeline.preprocessor import Preprocessor

# Import directly to avoid heavy torch imports
import xgboost as xgb
import joblib

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Attack categories for Agent Two
ATTACK_CATEGORIES = [
    "Normal", "Fuzzers", "Analysis", "Backdoor", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"
]


class SimpleClassifier:
    """Simple wrapper for loaded XGBoost model matching Agent Two interface."""
    
    def __init__(self, model, label_encoder, zero_day_threshold=0.5):
        self.model = model
        self.label_encoder = label_encoder
        self.zero_day_threshold = zero_day_threshold
        self.ATTACK_CATEGORIES = ATTACK_CATEGORIES
        self.n_classes = len(ATTACK_CATEGORIES)
        self._feature_names = None
    
    @classmethod
    def load(cls, model_dir):
        """Load model from directory."""
        model_dir = Path(model_dir)
        
        # Load metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load XGBoost model
        model = xgb.XGBClassifier()
        model.load_model(str(model_dir / "xgboost_model.json"))
        
        # Load label encoder
        label_encoder = joblib.load(model_dir / "label_encoder.pkl")
        
        instance = cls(
            model=model,
            label_encoder=label_encoder,
            zero_day_threshold=metadata.get("zero_day_threshold", 0.5)
        )
        instance._feature_names = metadata.get("feature_names")
        
        return instance
    
    def predict_batch(self, X):
        """Predict on batch of samples."""
        probs = self.model.predict_proba(X)
        results = []
        
        for i, prob in enumerate(probs):
            pred_idx = int(np.argmax(prob))
            confidence = float(prob[pred_idx])
            pred_cat = self.label_encoder.inverse_transform([pred_idx])[0]
            is_zero_day = confidence < self.zero_day_threshold
            
            prob_dict = {cat: float(p) for cat, p in zip(self.ATTACK_CATEGORIES, prob)}
            
            results.append({
                'category_id': pred_idx,
                'predicted_category': pred_cat if not is_zero_day else "Unknown/Zero-day",
                'confidence': confidence,
                'is_zero_day': is_zero_day,
                'all_probabilities': prob_dict,
            })
        
        return results
    
    def get_feature_importance(self, importance_type='gain', top_k=None):
        """Get feature importance."""
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        if self._feature_names:
            new_importance = {}
            for k, v in importance.items():
                if k.startswith('f'):
                    idx = int(k[1:])
                    if idx < len(self._feature_names):
                        new_importance[self._feature_names[idx]] = v
                else:
                    new_importance[k] = v
            importance = new_importance
        
        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        if top_k:
            sorted_imp = dict(list(sorted_imp.items())[:top_k])
        
        return sorted_imp


def load_unsw_data(data_dir: Path, _=None):
    """Load UNSW-NB15 data using the same preprocessing as training."""
    print("\n" + "="*60)
    print("  LOADING UNSW-NB15 DATASET")
    print("="*60)
    
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"
    
    # Use data pipeline (same as training)
    config = DatasetConfig()
    
    # Load and preprocess training data
    train_loader = DataLoader(config)
    train_loader.load(str(train_path)).clean()
    X_train_df, y_train_series = train_loader.get_features_and_labels(label_type="multiclass")
    
    # Load and preprocess test data
    test_loader = DataLoader(config)
    test_loader.load(str(test_path)).clean()
    X_test_df, y_test_series = test_loader.get_features_and_labels(label_type="multiclass")
    
    print(f"  Training samples: {len(X_train_df):,}")
    print(f"  Testing samples:  {len(X_test_df):,}")
    
    # Clean y values (handle NaN/empty)
    y_train_raw = np.array([str(label).strip() if label and str(label).strip() else "Normal" 
                           for label in y_train_series])
    y_test_raw = np.array([str(label).strip() if label and str(label).strip() else "Normal" 
                          for label in y_test_series])
    
    # Create preprocessor and fit on training data (same as training script!)
    preprocessor = Preprocessor(config)
    X_train = preprocessor.fit_transform(X_train_df, y=None, categorical_encoding="label")
    
    # Transform test data using fitted preprocessor
    X_test = preprocessor.transform(X_test_df)
    
    # Get actual feature list
    actual_features = preprocessor.numerical_features + preprocessor.categorical_features
    print(f"  Features used: {len(actual_features)}")
    
    # Show class distribution
    print("\n  Training class distribution:")
    train_counts = pd.Series(y_train_raw).value_counts()
    for cls, count in train_counts.items():
        pct = 100 * count / len(y_train_raw)
        print(f"    {cls:15s}: {count:6d} ({pct:5.2f}%)")
    
    print("\n  Testing class distribution:")
    test_counts = pd.Series(y_test_raw).value_counts()
    for cls, count in test_counts.items():
        pct = 100 * count / len(y_test_raw)
        print(f"    {cls:15s}: {count:6d} ({pct:5.2f}%)")
    
    return X_train, y_train_raw, X_test, y_test_raw, preprocessor


def evaluate_classifier(classifier: SimpleClassifier, X, y_raw, dataset_name: str):
    """Evaluate the classifier and return metrics."""
    print(f"\n  Evaluating on {dataset_name} set...")
    
    # Get predictions using batch predict
    results = classifier.predict_batch(X)
    
    # Extract predictions
    y_pred_raw = []
    confidences = []
    zero_day_flags = []
    
    for r in results:
        # Use the underlying predicted category (not "Unknown/Zero-day")
        pred_cat = classifier.label_encoder.inverse_transform([r['category_id']])[0]
        y_pred_raw.append(pred_cat)
        confidences.append(r['confidence'])
        zero_day_flags.append(r['is_zero_day'])
    
    y_pred_raw = np.array(y_pred_raw)
    confidences = np.array(confidences)
    
    # Align labels - make sure both use the same categories
    all_categories = classifier.ATTACK_CATEGORIES
    
    # Encode both actual and predicted
    label_encoder = LabelEncoder()
    label_encoder.fit(all_categories)
    
    # LabelEncoder sorts alphabetically - get sorted categories for reports
    sorted_categories = list(label_encoder.classes_)
    
    # Handle any labels not in categories
    y_true_filtered = np.array([y if y in all_categories else 'Normal' for y in y_raw])
    y_pred_filtered = np.array([y if y in all_categories else 'Normal' for y in y_pred_raw])
    
    y_true = label_encoder.transform(y_true_filtered)
    y_pred = label_encoder.transform(y_pred_filtered)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics - use sorted categories to match LabelEncoder order
    report = classification_report(
        y_true, y_pred,
        target_names=sorted_categories,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get probabilities for ROC
    y_probs = np.array([list(r['all_probabilities'].values()) for r in results])
    
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'report': report,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'confidences': confidences,
        'zero_day_count': sum(zero_day_flags),
        'mean_confidence': float(np.mean(confidences)),
        'sorted_categories': sorted_categories,  # For proper plot labeling
    }
    
    return metrics


def print_metrics(metrics, name):
    """Print formatted metrics."""
    print(f"\n  {name} SET METRICS:")
    print(f"    Accuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"    F1 Score (macro):   {metrics['f1_macro']*100:.2f}%")
    print(f"    F1 Score (weighted):{metrics['f1_weighted']*100:.2f}%")
    print(f"    Precision:          {metrics['precision']*100:.2f}%")
    print(f"    Recall:             {metrics['recall']*100:.2f}%")
    print(f"    Mean Confidence:    {metrics['mean_confidence']*100:.2f}%")
    print(f"    Zero-day Flags:     {metrics['zero_day_count']}")


def plot_results(train_metrics, test_metrics, classifier, output_dir):
    """Generate comprehensive visualization plots."""
    print("\n" + "="*60)
    print("  GENERATING PLOTS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use sorted categories from metrics (matches LabelEncoder order)
    class_names = test_metrics['sorted_categories']
    num_classes = len(class_names)
    
    # 1. Test Confusion Matrix
    print("  [1/7] Test Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(12, 10))
    cm = test_metrics['confusion_matrix']
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Agent Two XGBoost - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Normalized Confusion Matrix
    print("  [2/7] Normalized Confusion Matrix...")
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
    ax.set_title('Agent Two XGBoost - Normalized Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Per-Class Metrics Bar Chart
    print("  [3/7] Per-Class Metrics...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for cls_name in class_names:
        if cls_name in test_metrics['report']:
            precision_scores.append(test_metrics['report'][cls_name]['precision'] * 100)
            recall_scores.append(test_metrics['report'][cls_name]['recall'] * 100)
            f1_scores.append(test_metrics['report'][cls_name]['f1-score'] * 100)
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
    ax.set_title('Agent Two XGBoost - Per-Class Performance (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance
    print("  [4/7] Feature Importance...")
    try:
        importance = classifier.get_feature_importance(importance_type='gain', top_k=20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        features = list(importance.keys())
        values = list(importance.values())
        
        ax.barh(features, values, color='#9b59b6')
        ax.set_xlabel('Importance (Gain)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Agent Two XGBoost - Top 20 Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    Could not generate feature importance: {e}")
    
    # 5. ROC Curves
    print("  [5/7] ROC Curves...")
    y_test_bin = label_binarize(test_metrics['y_true'], classes=range(num_classes))
    y_probs = test_metrics['y_probs']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        if y_test_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{cls_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Agent Two XGBoost - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Train vs Test Comparison
    print("  [6/7] Train vs Test Comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision', 'Recall']
    train_values = [
        train_metrics['accuracy'] * 100,
        train_metrics['f1_macro'] * 100,
        train_metrics['f1_weighted'] * 100,
        train_metrics['precision'] * 100,
        train_metrics['recall'] * 100,
    ]
    test_values = [
        test_metrics['accuracy'] * 100,
        test_metrics['f1_macro'] * 100,
        test_metrics['f1_weighted'] * 100,
        test_metrics['precision'] * 100,
        test_metrics['recall'] * 100,
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Training', color='#3498db')
    bars2 = ax.bar(x + width/2, test_values, width, label='Testing', color='#e74c3c')
    
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
    ax.set_title('Agent Two XGBoost - Train vs Test Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'train_test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. Confidence Distribution
    print("  [7/7] Confidence Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training confidence
    axes[0].hist(train_metrics['confidences'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0.5, color='red', linestyle='--', label='Zero-day Threshold')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Training Set Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    
    # Testing confidence
    axes[1].hist(test_metrics['confidences'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0.5, color='red', linestyle='--', label='Zero-day Threshold')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Testing Set Confidence Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  All plots saved to: {output_dir}")
    return output_dir


def main():
    """Main function."""
    print("\n" + "="*70)
    print("  AGENT TWO XGBOOST MODEL EVALUATION")
    print("  Evaluating Pre-trained Model on UNSW-NB15 Dataset")
    print("="*70)
    
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    model_dir = project_dir / "models" / "agent_two" / "classifier"
    output_dir = script_dir / "agent_two_evaluation_results"
    
    # Load the pre-trained classifier
    print("\n" + "="*60)
    print("  LOADING PRE-TRAINED CLASSIFIER")
    print("="*60)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}")
    
    classifier = SimpleClassifier.load(model_dir)
    print(f"  Model loaded from: {model_dir}")
    print(f"  Zero-day threshold: {classifier.zero_day_threshold}")
    print(f"  Number of classes: {classifier.n_classes}")
    print(f"  Categories: {classifier.ATTACK_CATEGORIES}")
    
    # Load data using proper data pipeline preprocessing
    X_train, y_train_raw, X_test, y_test_raw, preprocessor = load_unsw_data(data_dir, None)
    feature_names = preprocessor.numerical_features + preprocessor.categorical_features
    print(f"  Model features: {len(feature_names)}")
    
    # Evaluate on training set
    print("\n" + "="*60)
    print("  EVALUATING ON TRAINING SET")
    print("="*60)
    
    start = datetime.now()
    train_metrics = evaluate_classifier(classifier, X_train, y_train_raw, "TRAINING")
    train_time = (datetime.now() - start).total_seconds()
    print_metrics(train_metrics, "TRAINING")
    print(f"    Evaluation time: {train_time:.2f}s")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("  EVALUATING ON TEST SET")
    print("="*60)
    
    start = datetime.now()
    test_metrics = evaluate_classifier(classifier, X_test, y_test_raw, "TESTING")
    test_time = (datetime.now() - start).total_seconds()
    print_metrics(test_metrics, "TESTING")
    print(f"    Evaluation time: {test_time:.2f}s")
    
    # Print per-class metrics table
    print("\n" + "="*60)
    print("  PER-CLASS METRICS (Test Set)")
    print("="*60)
    print(f"  {'Class':<15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-"*60)
    # Use sorted categories from metrics (matches LabelEncoder order)
    for cls_name in test_metrics['sorted_categories']:
        if cls_name in test_metrics['report']:
            m = test_metrics['report'][cls_name]
            print(f"  {cls_name:<15s} {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% {m['f1-score']*100:>9.2f}% {int(m['support']):>10d}")
    
    # Generate plots
    plot_dir = plot_results(train_metrics, test_metrics, classifier, output_dir)
    
    # Save results to JSON
    results = {
        'model_config': {
            'zero_day_threshold': classifier.zero_day_threshold,
            'n_classes': classifier.n_classes,
            'categories': classifier.ATTACK_CATEGORIES,
            'n_features': len(feature_names),
        },
        'training': {
            'samples': len(y_train_raw),
            'accuracy': float(train_metrics['accuracy']),
            'f1_macro': float(train_metrics['f1_macro']),
            'f1_weighted': float(train_metrics['f1_weighted']),
            'precision': float(train_metrics['precision']),
            'recall': float(train_metrics['recall']),
            'mean_confidence': float(train_metrics['mean_confidence']),
            'zero_day_count': int(train_metrics['zero_day_count']),
            'eval_time_seconds': train_time,
        },
        'testing': {
            'samples': len(y_test_raw),
            'accuracy': float(test_metrics['accuracy']),
            'f1_macro': float(test_metrics['f1_macro']),
            'f1_weighted': float(test_metrics['f1_weighted']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'mean_confidence': float(test_metrics['mean_confidence']),
            'zero_day_count': int(test_metrics['zero_day_count']),
            'eval_time_seconds': test_time,
        },
    }
    
    results_path = output_dir / 'agent_two_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY - AGENT TWO XGBOOST MODEL")
    print("="*70)
    print(f"""
  Model Configuration:
    - Zero-day Threshold: {classifier.zero_day_threshold}
    - Number of Classes:  {classifier.n_classes}
    - Number of Features: {len(feature_names)}

  Training Set ({len(y_train_raw):,} samples):
    - Accuracy:        {train_metrics['accuracy']*100:.2f}%
    - F1 (macro):      {train_metrics['f1_macro']*100:.2f}%
    - F1 (weighted):   {train_metrics['f1_weighted']*100:.2f}%
    - Mean Confidence: {train_metrics['mean_confidence']*100:.2f}%
    - Zero-day Flags:  {train_metrics['zero_day_count']:,}

  Testing Set ({len(y_test_raw):,} samples):
    - Accuracy:        {test_metrics['accuracy']*100:.2f}%
    - F1 (macro):      {test_metrics['f1_macro']*100:.2f}%
    - F1 (weighted):   {test_metrics['f1_weighted']*100:.2f}%
    - Mean Confidence: {test_metrics['mean_confidence']*100:.2f}%
    - Zero-day Flags:  {test_metrics['zero_day_count']:,}

  Results saved to:
    - Metrics: {results_path}
    - Plots:   {plot_dir}
""")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
