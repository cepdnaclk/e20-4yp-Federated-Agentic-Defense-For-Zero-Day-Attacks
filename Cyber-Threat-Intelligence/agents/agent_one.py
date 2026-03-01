"""
Agent One: Autoencoder-based Anomaly Detection Agent.

This module provides the AgentOne class, which wraps the AnomalyAutoencoder
model and provides a clean interface for anomaly detection in network traffic.

The agent handles:
    - Model loading and device management
    - Preprocessing integration
    - Threshold-based anomaly classification
    - Batch and single-sample inference

Design Philosophy:
    - Separation of concerns: Model definition is in models/autoencoder.py
    - Clean API: Simple detect_anomaly() interface for downstream use
    - Configurability: Adjustable threshold without retraining
    - Production-ready: Handles device management and preprocessing
"""

import logging
from typing import Optional, Tuple, Union, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from agents.models.autoencoder import AnomalyAutoencoder

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    Result of anomaly detection for a network flow.
    
    Attributes:
        is_anomaly: True if the flow is classified as anomalous.
        reconstruction_error: Raw reconstruction error value.
        threshold: Threshold used for classification.
        confidence: Confidence score (0-1), higher = more likely anomaly.
        raw_input: Original input data (optional).
    """
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
    confidence: float
    raw_input: Optional[np.ndarray] = None


class AgentOne:
    """
    Autoencoder-based anomaly detection agent for network intrusion detection.
    
    AgentOne uses a trained autoencoder to detect anomalies in network traffic.
    The detection is based on reconstruction error: normal traffic is
    reconstructed accurately, while anomalous traffic produces high errors.
    
    The agent provides:
        - Single sample detection via detect_anomaly()
        - Batch detection via detect_anomalies()
        - Configurable threshold
        - Integration with data preprocessor
    
    Attributes:
        model: The underlying AnomalyAutoencoder model.
        threshold: Reconstruction error threshold for anomaly classification.
        device: Computation device ('cpu' or 'cuda').
        preprocessor: Optional data preprocessor for raw input handling.
    
    Example:
        >>> # Load trained model
        >>> agent = AgentOne.from_checkpoint(
        ...     model_path="models/autoencoder.pth",
        ...     threshold=0.1,
        ... )
        >>> 
        >>> # Detect anomaly in a single network flow
        >>> result = agent.detect_anomaly(network_flow)
        >>> if result.is_anomaly:
        ...     print(f"ALERT: Anomaly detected (error={result.reconstruction_error:.4f})")
        >>> 
        >>> # Batch detection
        >>> results = agent.detect_anomalies(batch_of_flows)
    """
    
    def __init__(
        self,
        model: AnomalyAutoencoder,
        threshold: float = 0.1,
        device: Optional[str] = None,
        preprocessor: Optional[Any] = None,
    ) -> None:
        """
        Initializes AgentOne with a trained model.
        
        Args:
            model: Trained AnomalyAutoencoder instance.
            threshold: Reconstruction error threshold. Flows with error
                      above this value are classified as anomalies.
                      Typical range: 0.01 - 0.5 depending on data scale.
            device: Computation device ('cpu', 'cuda'). Auto-detects if None.
            preprocessor: Optional Preprocessor instance for handling raw input.
        
        Raises:
            ValueError: If threshold is not positive.
        """
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        
        self._model = model
        self._threshold = threshold
        self._preprocessor = preprocessor
        
        # Setup device
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        
        # Move model to device and set to eval mode
        self._model.to(self._device)
        self._model.eval()
        
        # Statistics for adaptive thresholding
        self._error_history: List[float] = []
        self._max_history_size = 10000
        
        logger.info(
            "AgentOne initialized: threshold=%.4f, device=%s",
            self._threshold, self._device
        )
    
    @property
    def model(self) -> AnomalyAutoencoder:
        """Returns the underlying autoencoder model."""
        return self._model
    
    @property
    def threshold(self) -> float:
        """Returns the current anomaly detection threshold."""
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        """
        Sets a new anomaly detection threshold.
        
        Args:
            value: New threshold value (must be positive).
        
        Raises:
            ValueError: If value is not positive.
        """
        if value <= 0:
            raise ValueError(f"Threshold must be positive, got {value}")
        self._threshold = value
        logger.info("Threshold updated to: %.4f", value)
    
    @property
    def device(self) -> str:
        """Returns the computation device."""
        return self._device
    
    @classmethod
    def from_checkpoint(
        cls,
        model_path: Union[str, Path],
        threshold: float = 0.1,
        device: Optional[str] = None,
        preprocessor: Optional[Any] = None,
    ) -> "AgentOne":
        """
        Creates AgentOne from a saved model checkpoint.
        
        Args:
            model_path: Path to the saved model file (.pth).
            threshold: Anomaly detection threshold.
            device: Computation device.
            preprocessor: Optional data preprocessor.
        
        Returns:
            Initialized AgentOne instance.
        
        Example:
            >>> agent = AgentOne.from_checkpoint(
            ...     "models/autoencoder.pth",
            ...     threshold=0.05
            ... )
        """
        model = AnomalyAutoencoder.load(model_path, device=device)
        return cls(model, threshold=threshold, device=device, preprocessor=preprocessor)
    
    def detect_anomaly(
        self,
        network_flow: Union[np.ndarray, torch.Tensor],
        return_raw: bool = True,
    ) -> DetectionResult:
        """
        Detects if a single network flow is anomalous.
        
        This is the primary method for anomaly detection. It computes
        the reconstruction error for the input flow and compares it
        against the configured threshold.
        
        Args:
            network_flow: Network flow features as numpy array or tensor.
                         Shape: (n_features,) for single sample.
                         Must match model's input dimension after preprocessing.
            return_raw: If True, includes raw input in the result.
        
        Returns:
            DetectionResult containing:
                - is_anomaly: True if reconstruction error > threshold
                - reconstruction_error: Raw error value
                - threshold: Threshold used
                - confidence: How likely the flow is anomalous (0-1)
                - raw_input: Original input if return_raw=True
        
        Example:
            >>> flow = preprocessed_data[0]  # Single flow, shape (40,)
            >>> result = agent.detect_anomaly(flow)
            >>> print(f"Anomaly: {result.is_anomaly}, Error: {result.reconstruction_error:.4f}")
        """
        # Handle input preprocessing
        x = self._prepare_input(network_flow)
        
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Compute reconstruction error
        with torch.no_grad():
            error = self._model.reconstruction_error(x, reduction="none")
            error_value = error.item()
        
        # Update history for statistics
        self._update_error_history(error_value)
        
        # Classification
        is_anomaly = error_value > self._threshold
        
        # Compute confidence score
        confidence = self._compute_confidence(error_value)
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            reconstruction_error=error_value,
            threshold=self._threshold,
            confidence=confidence,
            raw_input=network_flow if return_raw else None,
        )
    
    def detect_anomalies(
        self,
        network_flows: Union[np.ndarray, torch.Tensor],
        return_raw: bool = False,
    ) -> List[DetectionResult]:
        """
        Detects anomalies in a batch of network flows.
        
        Efficient batch processing for multiple flows.
        
        Args:
            network_flows: Batch of network flows.
                          Shape: (batch_size, n_features).
            return_raw: If True, includes raw inputs in results.
        
        Returns:
            List of DetectionResult, one per input flow.
        
        Example:
            >>> flows = preprocessed_data[:100]  # 100 flows
            >>> results = agent.detect_anomalies(flows)
            >>> n_anomalies = sum(1 for r in results if r.is_anomaly)
            >>> print(f"Detected {n_anomalies} anomalies out of {len(results)}")
        """
        x = self._prepare_input(network_flows)
        
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Compute reconstruction errors for all samples
        with torch.no_grad():
            errors = self._model.reconstruction_error(x, reduction="none")
            errors_np = errors.cpu().numpy()
        
        # Build results
        results = []
        for i, error_value in enumerate(errors_np):
            is_anomaly = error_value > self._threshold
            confidence = self._compute_confidence(error_value)
            
            raw_input = None
            if return_raw:
                if isinstance(network_flows, np.ndarray):
                    raw_input = network_flows[i]
                else:
                    raw_input = network_flows[i].cpu().numpy()
            
            results.append(DetectionResult(
                is_anomaly=is_anomaly,
                reconstruction_error=float(error_value),
                threshold=self._threshold,
                confidence=confidence,
                raw_input=raw_input,
            ))
        
        return results
    
    def get_reconstruction_errors(
        self,
        network_flows: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Computes reconstruction errors without classification.
        
        Useful for threshold calibration and analysis.
        
        Args:
            network_flows: Batch of network flows.
        
        Returns:
            Array of reconstruction errors, shape (batch_size,).
        """
        x = self._prepare_input(network_flows)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            errors = self._model.reconstruction_error(x, reduction="none")
            return errors.cpu().numpy()
    
    def calibrate_threshold(
        self,
        normal_data: Union[np.ndarray, torch.Tensor],
        percentile: float = 95.0,
    ) -> float:
        """
        Calibrates threshold based on normal traffic data.
        
        Sets the threshold such that `percentile`% of normal traffic
        is correctly classified as normal. This helps minimize false
        positives while maintaining detection capability.
        
        Args:
            normal_data: Dataset of known normal network flows.
            percentile: Percentile of normal data errors to use as threshold.
                       Higher values (e.g., 99) reduce false positives but
                       may miss some anomalies.
        
        Returns:
            The calibrated threshold value.
        
        Example:
            >>> # Use validation set of normal traffic
            >>> new_threshold = agent.calibrate_threshold(X_normal, percentile=95)
            >>> print(f"New threshold: {new_threshold:.4f}")
        """
        errors = self.get_reconstruction_errors(normal_data)
        new_threshold = float(np.percentile(errors, percentile))
        
        self._threshold = new_threshold
        logger.info(
            "Threshold calibrated: %.4f (%.1f percentile of %d samples)",
            new_threshold, percentile, len(errors)
        )
        
        return new_threshold
    
    def optimize_threshold(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: np.ndarray,
        metric: str = "f1",
        n_thresholds: int = 100,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Finds the optimal threshold by maximizing a metric on labeled data.
        
        This method searches across different threshold values to find
        the one that maximizes the specified metric (F1, accuracy, etc.).
        Use this when you have a labeled validation/test set.
        
        Args:
            X: Features (both normal and anomaly samples).
            y: Labels (0=normal, 1=anomaly/attack).
            metric: Metric to optimize. Options:
                   - 'f1': F1 score (balanced precision-recall)
                   - 'accuracy': Overall accuracy
                   - 'recall': Maximize recall (may increase false positives)
                   - 'precision': Maximize precision (may decrease recall)
            n_thresholds: Number of threshold values to try.
        
        Returns:
            Tuple of (optimal_threshold, metrics_dict) where metrics_dict
            contains precision, recall, f1, and accuracy at optimal threshold.
        
        Example:
            >>> # Find best threshold using F1 score
            >>> best_threshold, metrics = agent.optimize_threshold(
            ...     X_val, y_val, metric='f1'
            ... )
            >>> print(f"Optimal threshold: {best_threshold:.4f}")
            >>> print(f"Metrics: {metrics}")
        """
        # Compute reconstruction errors for all samples
        errors = self.get_reconstruction_errors(X)
        y = np.asarray(y)
        
        # Create threshold candidates from min to max error
        min_error = errors.min()
        max_error = errors.max()
        thresholds = np.linspace(min_error, max_error, n_thresholds)
        
        best_threshold = self._threshold
        best_score = -1.0
        best_metrics = {}
        
        for thresh in thresholds:
            predictions = (errors > thresh).astype(int)
            
            # Compute metrics
            tp = np.sum((predictions == 1) & (y == 1))
            fp = np.sum((predictions == 1) & (y == 0))
            tn = np.sum((predictions == 0) & (y == 0))
            fn = np.sum((predictions == 0) & (y == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (tp + tn) / len(y)
            
            # Select score based on metric
            if metric == "f1":
                score = f1
            elif metric == "accuracy":
                score = accuracy
            elif metric == "recall":
                score = recall
            elif metric == "precision":
                score = precision
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = float(thresh)
                best_metrics = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "accuracy": float(accuracy),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                }
        
        # Update threshold
        self._threshold = best_threshold
        logger.info(
            "Threshold optimized: %.4f (best %s=%.4f)",
            best_threshold, metric, best_score
        )
        
        return best_threshold, best_metrics
    
    def find_constrained_threshold(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: np.ndarray,
        min_accuracy: float = 0.95,
        n_thresholds: int = 1000,
        verbose: bool = True,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Finds threshold with accuracy >= min_accuracy and minimum false negatives.
        
        This method searches for thresholds that meet the accuracy constraint,
        then among those, selects the one with the fewest false negatives
        (i.e., maximum recall for attacks).
        
        Args:
            X: Features (both normal and anomaly samples).
            y: Labels (0=normal, 1=anomaly/attack).
            min_accuracy: Minimum required accuracy (default: 0.95 = 95%).
            n_thresholds: Number of thresholds to search (higher = finer search).
            verbose: If True, prints search progress.
        
        Returns:
            Tuple of (best_threshold, results_dict) where results_dict contains:
                - 'found': bool, whether a valid threshold was found
                - 'best_metrics': metrics at optimal threshold (if found)
                - 'best_accuracy_achievable': max accuracy found
                - 'all_candidates': list of all thresholds meeting constraint
        
        Example:
            >>> threshold, results = agent.find_constrained_threshold(
            ...     X_val, y_val, min_accuracy=0.95
            ... )
            >>> if results['found']:
            ...     print(f"Found threshold: {threshold:.4f}")
            ... else:
            ...     print(f"Max achievable accuracy: {results['best_accuracy_achievable']:.4f}")
        """
        # Compute reconstruction errors for all samples
        errors = self.get_reconstruction_errors(X)
        y = np.asarray(y)
        
        # Create threshold candidates with fine granularity
        min_error = errors.min()
        max_error = errors.max()
        thresholds = np.linspace(min_error, max_error, n_thresholds)
        
        # Store all results for analysis
        all_results = []
        candidates = []  # Thresholds meeting accuracy constraint
        best_accuracy = 0.0
        best_accuracy_metrics = {}
        best_accuracy_threshold = None
        
        if verbose:
            print(f"\n   Searching {n_thresholds} thresholds...")
            print(f"   Error range: [{min_error:.6f}, {max_error:.6f}]")
        
        for thresh in thresholds:
            predictions = (errors > thresh).astype(int)
            
            # Compute metrics
            tp = np.sum((predictions == 1) & (y == 1))
            fp = np.sum((predictions == 1) & (y == 0))
            tn = np.sum((predictions == 0) & (y == 0))
            fn = np.sum((predictions == 0) & (y == 1))
            
            accuracy = (tp + tn) / len(y)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            result = {
                "threshold": float(thresh),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }
            all_results.append(result)
            
            # Track best accuracy seen
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_metrics = result.copy()
                best_accuracy_threshold = thresh
            
            # Check if meets constraint
            if accuracy >= min_accuracy:
                candidates.append(result)
        
        if verbose:
            print(f"   Best accuracy achievable: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            print(f"   Thresholds with accuracy >= {min_accuracy*100:.1f}%: {len(candidates)}")
        
        # Find best candidate (minimum false negatives among those meeting constraint)
        if candidates:
            # Sort by false negatives (ascending), then by accuracy (descending)
            candidates.sort(key=lambda x: (x["false_negatives"], -x["accuracy"]))
            best = candidates[0]
            
            self._threshold = best["threshold"]
            
            if verbose:
                print(f"\n   ✓ Found threshold meeting constraint!")
                print(f"   Threshold: {best['threshold']:.6f}")
                print(f"   Accuracy: {best['accuracy']:.4f}")
                print(f"   False Negatives: {best['false_negatives']}")
                print(f"   Recall: {best['recall']:.4f}")
                print(f"   Precision: {best['precision']:.4f}")
            
            logger.info(
                "Constrained threshold found: %.4f (accuracy=%.4f, FN=%d)",
                best["threshold"], best["accuracy"], best["false_negatives"]
            )
            
            return best["threshold"], {
                "found": True,
                "best_metrics": best,
                "best_accuracy_achievable": best_accuracy,
                "all_candidates": candidates,
            }
        else:
            if verbose:
                print(f"\n   ✗ No threshold achieves {min_accuracy*100:.1f}% accuracy")
                print(f"   Maximum accuracy achievable: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
                print(f"   At threshold: {best_accuracy_threshold:.6f}")
                print(f"\n   Recommendation: Lower min_accuracy or improve model")
            
            return None, {
                "found": False,
                "best_metrics": best_accuracy_metrics,
                "best_accuracy_achievable": best_accuracy,
                "all_candidates": [],
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns agent statistics and diagnostics.
        
        Returns:
            Dictionary containing:
                - threshold: Current threshold
                - device: Computation device
                - model_config: Model architecture config
                - error_stats: Statistics from error history
        """
        stats = {
            "threshold": self._threshold,
            "device": self._device,
            "model_config": self._model.get_config(),
        }
        
        if self._error_history:
            errors = np.array(self._error_history)
            stats["error_stats"] = {
                "count": len(errors),
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
                "median": float(np.median(errors)),
            }
        
        return stats
    
    def _prepare_input(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Prepares input data for model inference.
        
        Handles conversion from numpy arrays and device placement.
        
        Args:
            x: Input data as numpy array or tensor.
        
        Returns:
            Tensor on the correct device.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        
        return x.to(self._device)
    
    def _compute_confidence(self, error: float) -> float:
        """
        Computes confidence score for anomaly classification.
        
        Uses a sigmoid-like transformation to map error values
        to a 0-1 confidence score.
        
        Args:
            error: Reconstruction error value.
        
        Returns:
            Confidence score between 0 and 1.
        """
        # Scale error relative to threshold
        # At threshold, confidence = 0.5
        # Much higher than threshold -> confidence approaches 1
        # Much lower than threshold -> confidence approaches 0
        scaled = (error - self._threshold) / (self._threshold + 1e-8)
        confidence = 1.0 / (1.0 + np.exp(-5 * scaled))  # Sigmoid with steepness 5
        return float(confidence)
    
    def _update_error_history(self, error: float) -> None:
        """Updates error history for statistics tracking."""
        self._error_history.append(error)
        if len(self._error_history) > self._max_history_size:
            self._error_history = self._error_history[-self._max_history_size:]
    
    def __repr__(self) -> str:
        """Returns string representation of the agent."""
        return (
            f"AgentOne(threshold={self._threshold:.4f}, "
            f"device={self._device}, "
            f"model={self._model})"
        )
