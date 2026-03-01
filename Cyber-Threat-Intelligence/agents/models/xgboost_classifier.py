"""
XGBoost Classifier for Multi-class Attack Classification.

This module provides an optimized XGBoost model for classifying network
traffic into specific attack categories from the UNSW-NB15 dataset.

The classifier distinguishes between:
    - Normal traffic
    - Known attack categories (Fuzzers, Analysis, Backdoors, DoS, etc.)
    - Unknown/Zero-day threats (low confidence predictions)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """
    Result of threat classification.
    
    Attributes:
        predicted_category: Predicted attack category name.
        category_id: Numeric ID of predicted category.
        confidence: Confidence score (probability) for prediction.
        is_zero_day: True if classified as unknown/zero-day threat.
        all_probabilities: Dict mapping category names to probabilities.
        features: Original input features (optional).
    """
    predicted_category: str
    category_id: int
    confidence: float
    is_zero_day: bool
    all_probabilities: Dict[str, float]
    features: Optional[np.ndarray] = None


class ThreatClassifier:
    """
    XGBoost-based threat classifier for UNSW-NB15 attack categories.
    
    This classifier categorizes network traffic anomalies into specific
    attack types or flags them as unknown/zero-day if confidence is low.
    
    Features:
        - Optimized XGBoost with GPU support (if available)
        - Confidence-based zero-day detection
        - Multi-class classification with probability outputs
        - Feature importance analysis
        - Model persistence (save/load)
    
    Attack Categories:
        - Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits,
          Generic, Reconnaissance, Shellcode, Worms
    
    Example:
        >>> classifier = ThreatClassifier()
        >>> classifier.fit(X_train, y_train)
        >>> result = classifier.predict(anomaly_features)
        >>> print(f"Category: {result.predicted_category}")
        >>> print(f"Zero-day: {result.is_zero_day}")
    """
    
    # Known attack categories in UNSW-NB15
    ATTACK_CATEGORIES = [
        "Normal",
        "Fuzzers",
        "Analysis", 
        "Backdoor",
        "DoS",
        "Exploits",
        "Generic",
        "Reconnaissance",
        "Shellcode",
        "Worms",
    ]
    
    def __init__(
        self,
        zero_day_threshold: float = 0.5,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        use_gpu: bool = False,
        random_state: int = 42,
    ):
        """
        Initializes the ThreatClassifier.
        
        Args:
            zero_day_threshold: Confidence threshold below which threats
                               are classified as unknown/zero-day.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            min_child_weight: Minimum sum of instance weight in child.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns per tree.
            reg_alpha: L1 regularization term.
            reg_lambda: L2 regularization term.
            use_gpu: Whether to use GPU acceleration.
            random_state: Random seed for reproducibility.
        """
        self._zero_day_threshold = zero_day_threshold
        self._random_state = random_state
        self._use_gpu = use_gpu
        
        # Label encoder for category names
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self.ATTACK_CATEGORIES)
        
        # XGBoost parameters
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "objective": "multi:softprob",
            "num_class": len(self.ATTACK_CATEGORIES),
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
        }
        
        # Add GPU parameters if requested
        if use_gpu:
            self._params["tree_method"] = "gpu_hist"
            self._params["predictor"] = "gpu_predictor"
        else:
            self._params["tree_method"] = "hist"
            self._params["n_jobs"] = -1
        
        self._model: Optional[xgb.XGBClassifier] = None
        self._is_fitted = False
        self._feature_names: Optional[List[str]] = None
        
        logger.info(
            "ThreatClassifier initialized: zero_day_threshold=%.2f, "
            "n_estimators=%d, max_depth=%d",
            zero_day_threshold, n_estimators, max_depth
        )
    
    @property
    def zero_day_threshold(self) -> float:
        """Returns the zero-day detection threshold."""
        return self._zero_day_threshold
    
    @zero_day_threshold.setter
    def zero_day_threshold(self, value: float) -> None:
        """Sets a new zero-day threshold."""
        if not 0 < value < 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {value}")
        self._zero_day_threshold = value
        logger.info("Zero-day threshold updated to: %.2f", value)
    
    @property
    def is_fitted(self) -> bool:
        """Returns whether the model has been trained."""
        return self._is_fitted
    
    @property
    def n_classes(self) -> int:
        """Returns the number of attack categories."""
        return len(self.ATTACK_CATEGORIES)
    
    def fit(
        self,
        X: np.ndarray,
        y: Union[np.ndarray, List[str]],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Union[np.ndarray, List[str]]] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True,
    ) -> "ThreatClassifier":
        """
        Trains the XGBoost classifier.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels (category names or encoded integers).
            X_val: Validation features for early stopping.
            y_val: Validation labels.
            feature_names: Names of input features for importance analysis.
            early_stopping_rounds: Rounds without improvement before stopping.
            verbose: Whether to print training progress.
        
        Returns:
            Self for method chaining.
        """
        # Encode labels if they are strings
        if isinstance(y[0], str):
            y_encoded = self._label_encoder.transform(y)
        else:
            y_encoded = np.asarray(y)
        
        # Store feature names
        self._feature_names = feature_names
        
        # Create model
        self._model = xgb.XGBClassifier(**self._params)
        
        # Prepare eval set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            if isinstance(y_val[0], str):
                y_val_encoded = self._label_encoder.transform(y_val)
            else:
                y_val_encoded = np.asarray(y_val)
            eval_set = [(X_val, y_val_encoded)]
        
        # Train model
        logger.info("Training ThreatClassifier on %d samples...", len(X))
        
        self._model.fit(
            X, y_encoded,
            eval_set=eval_set,
            verbose=verbose,
        )
        
        self._is_fitted = True
        logger.info("ThreatClassifier training complete")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_features: bool = False,
    ) -> ClassificationResult:
        """
        Classifies a single network flow.
        
        Args:
            X: Feature vector, shape (n_features,) or (1, n_features).
            return_features: If True, includes features in result.
        
        Returns:
            ClassificationResult with predicted category and confidence.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Ensure 2D input
        X = np.atleast_2d(X)
        
        # Get probabilities
        probabilities = self._model.predict_proba(X)[0]
        
        # Get predicted class
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        predicted_category = self._label_encoder.inverse_transform([predicted_idx])[0]
        
        # Check for zero-day (low confidence)
        is_zero_day = confidence < self._zero_day_threshold
        
        # Build probability dict
        prob_dict = {
            cat: float(prob) 
            for cat, prob in zip(self.ATTACK_CATEGORIES, probabilities)
        }
        
        return ClassificationResult(
            predicted_category=predicted_category if not is_zero_day else "Unknown/Zero-day",
            category_id=predicted_idx,
            confidence=confidence,
            is_zero_day=is_zero_day,
            all_probabilities=prob_dict,
            features=X[0] if return_features else None,
        )
    
    def predict_batch(
        self,
        X: np.ndarray,
        return_features: bool = False,
    ) -> List[ClassificationResult]:
        """
        Classifies multiple network flows.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features).
            return_features: If True, includes features in results.
        
        Returns:
            List of ClassificationResult objects.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get all probabilities
        probabilities = self._model.predict_proba(X)
        
        results = []
        for i, probs in enumerate(probabilities):
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])
            predicted_category = self._label_encoder.inverse_transform([predicted_idx])[0]
            
            is_zero_day = confidence < self._zero_day_threshold
            
            prob_dict = {
                cat: float(prob) 
                for cat, prob in zip(self.ATTACK_CATEGORIES, probs)
            }
            
            results.append(ClassificationResult(
                predicted_category=predicted_category if not is_zero_day else "Unknown/Zero-day",
                category_id=predicted_idx,
                confidence=confidence,
                is_zero_day=is_zero_day,
                all_probabilities=prob_dict,
                features=X[i] if return_features else None,
            ))
        
        return results
    
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Returns feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover').
            top_k: If specified, returns only top-k features.
        
        Returns:
            Dict mapping feature names to importance scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        importance = self._model.get_booster().get_score(
            importance_type=importance_type
        )
        
        # Use feature names if available
        if self._feature_names:
            importance = {
                self._feature_names[int(k[1:])] if k.startswith('f') else k: v
                for k, v in importance.items()
            }
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        if top_k:
            sorted_importance = dict(list(sorted_importance.items())[:top_k])
        
        return sorted_importance
    
    def evaluate(
        self,
        X: np.ndarray,
        y: Union[np.ndarray, List[str]],
    ) -> Dict[str, Any]:
        """
        Evaluates model performance on test data.
        
        Args:
            X: Test features.
            y: True labels.
        
        Returns:
            Dict with accuracy, per-class metrics, and confusion matrix.
        """
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, classification_report
        )
        
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Encode labels
        if isinstance(y[0], str):
            y_true = self._label_encoder.transform(y)
        else:
            y_true = np.asarray(y)
        
        # Get predictions
        y_pred = self._model.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Build results
        per_class = {}
        for i, cat in enumerate(self.ATTACK_CATEGORIES):
            if i < len(precision):
                per_class[cat] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]) if i < len(support) else 0,
                }
        
        return {
            "accuracy": float(accuracy),
            "per_class_metrics": per_class,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=self.ATTACK_CATEGORIES,
                zero_division=0,
            ),
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Returns model configuration."""
        return {
            "zero_day_threshold": self._zero_day_threshold,
            "params": self._params,
            "n_classes": self.n_classes,
            "attack_categories": self.ATTACK_CATEGORIES,
            "is_fitted": self._is_fitted,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the model to disk.
        
        Args:
            path: Directory path to save model files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        if self._model:
            self._model.save_model(str(path / "xgboost_model.json"))
        
        # Save metadata
        metadata = {
            "zero_day_threshold": self._zero_day_threshold,
            "params": self._params,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save label encoder
        joblib.dump(self._label_encoder, path / "label_encoder.pkl")
        
        logger.info("ThreatClassifier saved to: %s", path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ThreatClassifier":
        """
        Loads a model from disk.
        
        Args:
            path: Directory path containing saved model files.
        
        Returns:
            Loaded ThreatClassifier instance.
        """
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Filter out XGBoost-specific params not in __init__
        excluded_params = {
            "objective", "num_class", "eval_metric", 
            "use_label_encoder", "tree_method", "n_jobs"
        }
        
        # Create instance
        instance = cls(
            zero_day_threshold=metadata["zero_day_threshold"],
            **{k: v for k, v in metadata["params"].items() 
               if k not in excluded_params}
        )
        
        # Load XGBoost model
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(str(path / "xgboost_model.json"))
        
        # Load label encoder
        instance._label_encoder = joblib.load(path / "label_encoder.pkl")
        
        instance._feature_names = metadata.get("feature_names")
        instance._is_fitted = metadata.get("is_fitted", True)
        
        logger.info("ThreatClassifier loaded from: %s", path)
        
        return instance


def summary(classifier: ThreatClassifier) -> str:
    """Returns a summary string of the classifier."""
    config = classifier.get_config()
    lines = [
        "=" * 60,
        "ThreatClassifier Summary",
        "=" * 60,
        f"Zero-day threshold:  {config['zero_day_threshold']:.2f}",
        f"Number of classes:   {config['n_classes']}",
        f"Is fitted:           {config['is_fitted']}",
        "-" * 60,
        "Attack Categories:",
    ]
    for cat in config['attack_categories']:
        lines.append(f"  - {cat}")
    lines.append("=" * 60)
    return "\n".join(lines)
