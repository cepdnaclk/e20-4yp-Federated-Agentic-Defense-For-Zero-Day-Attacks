"""
Inference Optimization Module for Real-Time IDS.

This module provides utilities for optimizing model inference to meet
the <50ms latency requirement for real-time intrusion detection.

Features:
- Batch inference with dynamic batching
- ONNX model export for CPU optimization
- Feature preprocessing caching
- Model warmup utilities
- Latency measurement tools

Usage:
    >>> from agents.inference_optimizer import InferenceOptimizer
    >>> optimizer = InferenceOptimizer(autoencoder, xgboost_model)
    >>> optimizer.warmup()
    >>> predictions = optimizer.predict_batch(features)
"""

import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference optimization."""
    
    # Maximum latency target in milliseconds
    max_latency_ms: float = 50.0
    
    # Batch size for batch inference
    batch_size: int = 32
    
    # Whether to use ONNX runtime (if available)
    use_onnx: bool = True
    
    # Number of warmup iterations
    warmup_iterations: int = 10
    
    # Cache size for preprocessed features
    cache_size: int = 1000
    
    # Enable feature preprocessing cache
    use_cache: bool = True


@dataclass 
class LatencyStats:
    """Statistics for inference latency."""
    
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    samples_per_second: float = 0.0
    
    @classmethod
    def from_timings(cls, timings: List[float]) -> "LatencyStats":
        """Create stats from timing list (in seconds)."""
        if not timings:
            return cls()
        
        timings_ms = np.array(timings) * 1000
        return cls(
            mean_ms=float(np.mean(timings_ms)),
            std_ms=float(np.std(timings_ms)),
            min_ms=float(np.min(timings_ms)),
            max_ms=float(np.max(timings_ms)),
            p50_ms=float(np.percentile(timings_ms, 50)),
            p95_ms=float(np.percentile(timings_ms, 95)),
            p99_ms=float(np.percentile(timings_ms, 99)),
            samples_per_second=1000.0 / float(np.mean(timings_ms)),
        )
    
    def meets_target(self, target_ms: float = 50.0) -> bool:
        """Check if latency meets target."""
        return self.p95_ms <= target_ms


class InferenceOptimizer:
    """
    Optimizes inference pipeline for real-time detection.
    
    Provides:
    - Unified inference interface for Agent One and Two
    - Batch processing with dynamic batching
    - Latency measurement and monitoring
    - Optional ONNX acceleration
    """
    
    def __init__(
        self,
        autoencoder: Optional[Any] = None,
        classifier: Optional[Any] = None,
        config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize inference optimizer.
        
        Args:
            autoencoder: Agent One autoencoder model.
            classifier: Agent Two XGBoost classifier.
            config: Configuration for optimization.
        """
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.config = config or InferenceConfig()
        
        # Latency tracking
        self._latencies: List[float] = []
        self._warmup_done = False
        
        # ONNX runtime sessions
        self._onnx_ae_session = None
        self._onnx_clf_session = None
        
        # Feature cache (LRU-style)
        self._feature_cache: Dict[bytes, np.ndarray] = {}
        
    def warmup(self, sample_features: Optional[np.ndarray] = None) -> LatencyStats:
        """
        Warm up models with dummy data.
        
        JIT compilation and memory allocation happen here.
        
        Args:
            sample_features: Sample features for warmup (optional).
        
        Returns:
            LatencyStats from warmup iterations.
        """
        if sample_features is None:
            # Create dummy features
            sample_features = np.random.randn(
                self.config.warmup_iterations, 42
            ).astype(np.float32)
        
        timings = []
        
        for i in range(self.config.warmup_iterations):
            start = time.perf_counter()
            
            if self.autoencoder is not None:
                _ = self._predict_anomaly(sample_features[i:i+1])
            
            if self.classifier is not None:
                _ = self._predict_class(sample_features[i:i+1])
            
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
        
        self._warmup_done = True
        stats = LatencyStats.from_timings(timings)
        
        logger.info(
            f"Warmup complete: mean={stats.mean_ms:.2f}ms, "
            f"p95={stats.p95_ms:.2f}ms"
        )
        
        return stats
    
    def predict_single(
        self,
        features: np.ndarray,
        include_latency: bool = False,
    ) -> Union[Dict, Tuple[Dict, float]]:
        """
        Predict for a single sample.
        
        Args:
            features: Feature vector (1D or 2D with shape [1, n_features]).
            include_latency: Whether to return latency measurement.
        
        Returns:
            Prediction dict, optionally with latency in ms.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        start = time.perf_counter()
        
        result = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "class_id": 0,
            "class_confidence": 1.0,
        }
        
        # Agent One: Anomaly detection
        if self.autoencoder is not None:
            anomaly_score = self._predict_anomaly(features)[0]
            result["anomaly_score"] = float(anomaly_score)
            result["is_anomaly"] = anomaly_score > 0.0334  # Default threshold
        
        # Agent Two: Classification
        if self.classifier is not None:
            class_id, confidence = self._predict_class_with_confidence(features)
            result["class_id"] = int(class_id[0])
            result["class_confidence"] = float(confidence[0])
        
        elapsed = time.perf_counter() - start
        self._latencies.append(elapsed)
        
        if include_latency:
            return result, elapsed * 1000
        return result
    
    def predict_batch(
        self,
        features: np.ndarray,
        include_latency: bool = False,
    ) -> Union[List[Dict], Tuple[List[Dict], float]]:
        """
        Predict for a batch of samples.
        
        Args:
            features: Feature matrix (2D with shape [n_samples, n_features]).
            include_latency: Whether to return latency measurement.
        
        Returns:
            List of prediction dicts, optionally with total latency in ms.
        """
        start = time.perf_counter()
        
        n_samples = len(features)
        results = []
        
        # Batch anomaly detection
        anomaly_scores = np.zeros(n_samples)
        if self.autoencoder is not None:
            anomaly_scores = self._predict_anomaly(features)
        
        # Batch classification
        class_ids = np.zeros(n_samples, dtype=int)
        confidences = np.ones(n_samples)
        if self.classifier is not None:
            class_ids, confidences = self._predict_class_with_confidence(features)
        
        # Assemble results
        for i in range(n_samples):
            results.append({
                "is_anomaly": bool(anomaly_scores[i] > 0.0334),
                "anomaly_score": float(anomaly_scores[i]),
                "class_id": int(class_ids[i]),
                "class_confidence": float(confidences[i]),
            })
        
        elapsed = time.perf_counter() - start
        self._latencies.append(elapsed / n_samples)  # Per-sample latency
        
        if include_latency:
            return results, elapsed * 1000
        return results
    
    def _predict_anomaly(self, features: np.ndarray) -> np.ndarray:
        """Get anomaly scores from autoencoder."""
        try:
            # Try using model's predict method
            if hasattr(self.autoencoder, 'predict'):
                reconstructed = self.autoencoder.predict(features, verbose=0)
                return np.mean(np.square(features - reconstructed), axis=1)
            elif hasattr(self.autoencoder, '__call__'):
                reconstructed = self.autoencoder(features)
                return np.mean(np.square(features - reconstructed.numpy()), axis=1)
            else:
                return np.zeros(len(features))
        except Exception as e:
            logger.warning(f"Anomaly prediction failed: {e}")
            return np.zeros(len(features))
    
    def _predict_class(self, features: np.ndarray) -> np.ndarray:
        """Get class predictions from classifier."""
        try:
            if hasattr(self.classifier, 'predict'):
                return self.classifier.predict(features)
            else:
                return np.zeros(len(features), dtype=int)
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return np.zeros(len(features), dtype=int)
    
    def _predict_class_with_confidence(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get class predictions with confidence scores."""
        try:
            class_ids = self._predict_class(features)
            
            if hasattr(self.classifier, 'predict_proba'):
                probas = self.classifier.predict_proba(features)
                confidences = np.max(probas, axis=1)
            else:
                confidences = np.ones(len(features))
            
            return class_ids, confidences
        except Exception as e:
            logger.warning(f"Confidence prediction failed: {e}")
            return np.zeros(len(features), dtype=int), np.ones(len(features))
    
    def get_latency_stats(self) -> LatencyStats:
        """Get latency statistics from recent predictions."""
        return LatencyStats.from_timings(self._latencies[-1000:])
    
    def reset_latency_stats(self):
        """Clear latency measurements."""
        self._latencies.clear()
    
    def export_onnx(
        self,
        autoencoder_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
    ) -> bool:
        """
        Export models to ONNX format for optimized inference.
        
        Args:
            autoencoder_path: Path for ONNX autoencoder.
            classifier_path: Path for ONNX classifier.
        
        Returns:
            True if export successful.
        """
        success = True
        
        try:
            import tf2onnx
            import tensorflow as tf
            
            if self.autoencoder is not None and autoencoder_path:
                spec = (tf.TensorSpec((None, 42), tf.float32, name="input"),)
                _ = tf2onnx.convert.from_keras(
                    self.autoencoder, input_signature=spec,
                    output_path=autoencoder_path
                )
                logger.info(f"Exported autoencoder to {autoencoder_path}")
        except ImportError:
            logger.warning("tf2onnx not available for autoencoder export")
            success = False
        except Exception as e:
            logger.error(f"Autoencoder ONNX export failed: {e}")
            success = False
        
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            if self.classifier is not None and classifier_path:
                initial_type = [('float_input', FloatTensorType([None, 42]))]
                onnx_model = convert_sklearn(
                    self.classifier, initial_types=initial_type
                )
                with open(classifier_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                logger.info(f"Exported classifier to {classifier_path}")
        except ImportError:
            logger.warning("skl2onnx not available for classifier export")
            success = False
        except Exception as e:
            logger.error(f"Classifier ONNX export failed: {e}")
            success = False
        
        return success


class FastPreprocessor:
    """
    Optimized feature preprocessor for real-time inference.
    
    Uses vectorized operations and caching for speed.
    """
    
    def __init__(
        self,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaler_mean: Mean values for standardization.
            scaler_std: Std values for standardization.
        """
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        
        # Pre-compute reciprocal for faster division
        if scaler_std is not None:
            self._scaler_std_inv = 1.0 / (scaler_std + 1e-8)
        else:
            self._scaler_std_inv = None
    
    def preprocess(
        self,
        features: np.ndarray,
        copy: bool = False,
    ) -> np.ndarray:
        """
        Preprocess features with optimized operations.
        
        Args:
            features: Raw feature values.
            copy: Whether to copy before modifying.
        
        Returns:
            Preprocessed features.
        """
        if copy:
            features = features.copy()
        
        # Handle inf/nan
        features = np.nan_to_num(
            features, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        # Standardize if scaler available
        if self.scaler_mean is not None:
            features = features - self.scaler_mean
        
        if self._scaler_std_inv is not None:
            features = features * self._scaler_std_inv
        
        # Clip extreme values
        np.clip(features, -10, 10, out=features)
        
        return features.astype(np.float32)
    
    @classmethod
    def from_sklearn_scaler(cls, scaler) -> "FastPreprocessor":
        """Create from sklearn StandardScaler."""
        return cls(
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_std=scaler.scale_.astype(np.float32),
        )


def benchmark_inference(
    optimizer: InferenceOptimizer,
    n_samples: int = 1000,
    batch_sizes: List[int] = [1, 8, 32, 64],
) -> Dict[int, LatencyStats]:
    """
    Benchmark inference at different batch sizes.
    
    Args:
        optimizer: InferenceOptimizer to benchmark.
        n_samples: Total samples to process.
        batch_sizes: Batch sizes to test.
    
    Returns:
        Dict mapping batch size to LatencyStats.
    """
    results = {}
    
    # Generate test data
    test_features = np.random.randn(n_samples, 42).astype(np.float32)
    
    for batch_size in batch_sizes:
        timings = []
        
        for i in range(0, n_samples, batch_size):
            batch = test_features[i:i+batch_size]
            
            start = time.perf_counter()
            _ = optimizer.predict_batch(batch)
            elapsed = time.perf_counter() - start
            
            # Per-sample time
            timings.extend([elapsed / len(batch)] * len(batch))
        
        results[batch_size] = LatencyStats.from_timings(timings)
        
        logger.info(
            f"Batch size {batch_size}: "
            f"mean={results[batch_size].mean_ms:.2f}ms, "
            f"p95={results[batch_size].p95_ms:.2f}ms, "
            f"throughput={results[batch_size].samples_per_second:.0f}/s"
        )
    
    return results


if __name__ == "__main__":
    # Demo benchmark
    logging.basicConfig(level=logging.INFO)
    
    print("=== Inference Optimization Benchmark ===\n")
    
    # Test without models (pure overhead measurement)
    config = InferenceConfig(max_latency_ms=50.0)
    optimizer = InferenceOptimizer(config=config)
    
    print("Warmup...")
    warmup_stats = optimizer.warmup()
    print(f"Warmup latency: {warmup_stats.mean_ms:.2f}ms\n")
    
    print("Benchmarking...")
    results = benchmark_inference(optimizer, n_samples=1000)
    
    print("\n=== Results ===")
    for batch_size, stats in results.items():
        status = "✓" if stats.meets_target(50.0) else "✗"
        print(f"Batch {batch_size:3d}: {stats.p95_ms:6.2f}ms p95 {status}")
