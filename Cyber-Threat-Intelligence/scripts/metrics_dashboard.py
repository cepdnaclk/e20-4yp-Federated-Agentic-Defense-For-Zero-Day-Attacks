"""
Metrics Dashboard for Multi-Agent IDS System.

This module provides real-time monitoring and visualization of:
- Detection accuracy (recall, FPR, precision)
- Inference latency
- Class distribution
- Federated learning convergence
- Zero-day detection rates

Usage:
    >>> from scripts.metrics_dashboard import MetricsDashboard
    >>> dashboard = MetricsDashboard()
    >>> dashboard.update_predictions(y_true, y_pred, latencies)
    >>> dashboard.print_report()
    >>> dashboard.save_report("metrics_report.html")
"""

import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    
    class_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    support: int = 0
    false_positive_rate: float = 0.0


@dataclass
class OverallMetrics:
    """Overall system metrics."""
    
    # Detection metrics
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    
    # Target metrics
    attack_recall: float = 0.0  # Target: >95%
    false_positive_rate: float = 0.0  # Target: <5%
    
    # Latency metrics
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    
    # Sample counts
    total_samples: int = 0
    attack_samples: int = 0
    normal_samples: int = 0
    
    # Zero-day metrics
    zero_day_candidates: int = 0
    low_confidence_rate: float = 0.0
    
    # Timestamps
    timestamp: str = ""
    window_start: str = ""
    window_end: str = ""


@dataclass 
class FederatedMetrics:
    """Metrics for federated learning."""
    
    global_accuracy: float = 0.0
    client_accuracies: Dict[str, float] = field(default_factory=dict)
    accuracy_parity: float = 0.0  # Min/Max ratio
    communication_rounds: int = 0
    convergence_rate: float = 0.0


class MetricsDashboard:
    """
    Real-time metrics dashboard for IDS monitoring.
    
    Tracks and visualizes:
    - Per-class and overall detection metrics
    - Inference latency statistics
    - Zero-day detection candidates
    - Federated learning progress
    """
    
    # Category names matching unified taxonomy
    CATEGORY_NAMES = [
        "Normal", "DoS/DDoS", "Reconnaissance", "Exploits",
        "Brute_Force", "Malware", "Analysis"
    ]
    
    def __init__(
        self,
        window_size: int = 1000,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize dashboard.
        
        Args:
            window_size: Number of recent samples to track.
            confidence_threshold: Threshold for low-confidence detection.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # Rolling buffers
        self._y_true: List[int] = []
        self._y_pred: List[int] = []
        self._confidences: List[float] = []
        self._latencies: List[float] = []
        self._timestamps: List[float] = []
        
        # Federated learning tracking
        self._fed_rounds: List[Dict] = []
        
        # Start time
        self._start_time = time.time()
    
    def update_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        latencies_ms: Optional[np.ndarray] = None,
    ):
        """
        Update metrics with new predictions.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            confidences: Prediction confidence scores.
            latencies_ms: Inference latencies in milliseconds.
        """
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
        
        self._y_true.extend(y_true.tolist())
        self._y_pred.extend(y_pred.tolist())
        
        if confidences is not None:
            self._confidences.extend(np.atleast_1d(confidences).tolist())
        else:
            self._confidences.extend([1.0] * len(y_true))
        
        if latencies_ms is not None:
            self._latencies.extend(np.atleast_1d(latencies_ms).tolist())
        
        self._timestamps.extend([time.time()] * len(y_true))
        
        # Trim to window size
        if len(self._y_true) > self.window_size:
            self._y_true = self._y_true[-self.window_size:]
            self._y_pred = self._y_pred[-self.window_size:]
            self._confidences = self._confidences[-self.window_size:]
            self._latencies = self._latencies[-self.window_size:]
            self._timestamps = self._timestamps[-self.window_size:]
    
    def update_federated_round(
        self,
        round_num: int,
        global_accuracy: float,
        client_accuracies: Dict[str, float],
    ):
        """
        Update federated learning metrics.
        
        Args:
            round_num: Communication round number.
            global_accuracy: Accuracy of global model.
            client_accuracies: Per-client accuracies.
        """
        self._fed_rounds.append({
            "round": round_num,
            "global_accuracy": global_accuracy,
            "client_accuracies": client_accuracies,
            "timestamp": time.time(),
        })
    
    def get_class_metrics(self) -> List[ClassMetrics]:
        """Get per-class metrics."""
        if not self._y_true:
            return []
        
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        
        metrics = []
        
        for i, name in enumerate(self.CATEGORY_NAMES):
            # Calculate metrics for this class
            true_pos = np.sum((y_true == i) & (y_pred == i))
            false_pos = np.sum((y_true != i) & (y_pred == i))
            false_neg = np.sum((y_true == i) & (y_pred != i))
            true_neg = np.sum((y_true != i) & (y_pred != i))
            
            support = int(np.sum(y_true == i))
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0.0
            
            metrics.append(ClassMetrics(
                class_name=name,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=support,
                false_positive_rate=fpr,
            ))
        
        return metrics
    
    def get_overall_metrics(self) -> OverallMetrics:
        """Get overall system metrics."""
        if not self._y_true:
            return OverallMetrics(timestamp=datetime.now().isoformat())
        
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        confidences = np.array(self._confidences)
        
        # Accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Per-class metrics
        class_metrics = self.get_class_metrics()
        
        # Macro averages
        precisions = [m.precision for m in class_metrics if m.support > 0]
        recalls = [m.recall for m in class_metrics if m.support > 0]
        f1s = [m.f1_score for m in class_metrics if m.support > 0]
        
        macro_precision = np.mean(precisions) if precisions else 0.0
        macro_recall = np.mean(recalls) if recalls else 0.0
        macro_f1 = np.mean(f1s) if f1s else 0.0
        
        # Weighted F1
        supports = [m.support for m in class_metrics]
        if sum(supports) > 0:
            weighted_f1 = sum(m.f1_score * m.support for m in class_metrics) / sum(supports)
        else:
            weighted_f1 = 0.0
        
        # Attack recall (class 0 is Normal, rest are attacks)
        attack_mask = y_true > 0
        if np.sum(attack_mask) > 0:
            attack_recall = np.mean(y_pred[attack_mask] > 0)
        else:
            attack_recall = 0.0
        
        # False positive rate (Normal misclassified as attack)
        normal_mask = y_true == 0
        if np.sum(normal_mask) > 0:
            fpr = np.mean(y_pred[normal_mask] > 0)
        else:
            fpr = 0.0
        
        # Latency metrics
        if self._latencies:
            latencies = np.array(self._latencies)
            mean_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = 1000.0 / mean_latency if mean_latency > 0 else 0.0
        else:
            mean_latency = p95_latency = p99_latency = 0.0
            throughput = 0.0
        
        # Zero-day candidates (attacks with low confidence)
        attack_pred_mask = y_pred > 0
        if np.sum(attack_pred_mask) > 0:
            low_conf_attacks = np.sum(
                attack_pred_mask & (confidences < self.confidence_threshold)
            )
            low_conf_rate = low_conf_attacks / np.sum(attack_pred_mask)
        else:
            low_conf_attacks = 0
            low_conf_rate = 0.0
        
        return OverallMetrics(
            accuracy=float(accuracy),
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            weighted_f1=float(weighted_f1),
            attack_recall=float(attack_recall),
            false_positive_rate=float(fpr),
            mean_latency_ms=float(mean_latency),
            p95_latency_ms=float(p95_latency),
            p99_latency_ms=float(p99_latency),
            throughput_per_sec=float(throughput),
            total_samples=len(y_true),
            attack_samples=int(np.sum(y_true > 0)),
            normal_samples=int(np.sum(y_true == 0)),
            zero_day_candidates=int(low_conf_attacks),
            low_confidence_rate=float(low_conf_rate),
            timestamp=datetime.now().isoformat(),
            window_start=datetime.fromtimestamp(
                self._timestamps[0] if self._timestamps else time.time()
            ).isoformat(),
            window_end=datetime.fromtimestamp(
                self._timestamps[-1] if self._timestamps else time.time()
            ).isoformat(),
        )
    
    def get_federated_metrics(self) -> Optional[FederatedMetrics]:
        """Get federated learning metrics."""
        if not self._fed_rounds:
            return None
        
        latest = self._fed_rounds[-1]
        
        client_accs = list(latest["client_accuracies"].values())
        parity = min(client_accs) / max(client_accs) if client_accs and max(client_accs) > 0 else 0.0
        
        # Convergence rate (improvement over rounds)
        if len(self._fed_rounds) >= 2:
            first_acc = self._fed_rounds[0]["global_accuracy"]
            last_acc = self._fed_rounds[-1]["global_accuracy"]
            convergence = (last_acc - first_acc) / len(self._fed_rounds)
        else:
            convergence = 0.0
        
        return FederatedMetrics(
            global_accuracy=latest["global_accuracy"],
            client_accuracies=latest["client_accuracies"],
            accuracy_parity=parity,
            communication_rounds=len(self._fed_rounds),
            convergence_rate=convergence,
        )
    
    def check_targets(self) -> Dict[str, Tuple[bool, str]]:
        """
        Check if performance targets are met.
        
        Returns:
            Dict of target -> (met, message).
        """
        metrics = self.get_overall_metrics()
        
        return {
            "attack_recall": (
                metrics.attack_recall >= 0.95,
                f"Attack Recall: {metrics.attack_recall:.1%} (target: ≥95%)"
            ),
            "false_positive_rate": (
                metrics.false_positive_rate <= 0.05,
                f"FPR: {metrics.false_positive_rate:.1%} (target: ≤5%)"
            ),
            "latency": (
                metrics.p95_latency_ms <= 50.0 or metrics.p95_latency_ms == 0,
                f"P95 Latency: {metrics.p95_latency_ms:.1f}ms (target: ≤50ms)"
            ),
        }
    
    def print_report(self):
        """Print formatted metrics report."""
        overall = self.get_overall_metrics()
        class_metrics = self.get_class_metrics()
        targets = self.check_targets()
        
        print("\n" + "=" * 60)
        print("           MULTI-AGENT IDS METRICS DASHBOARD")
        print("=" * 60)
        print(f"Timestamp: {overall.timestamp}")
        print(f"Window: {overall.window_start} to {overall.window_end}")
        print(f"Samples: {overall.total_samples:,} ({overall.attack_samples:,} attacks)")
        print()
        
        # Target status
        print("-" * 60)
        print("TARGET STATUS")
        print("-" * 60)
        for target, (met, msg) in targets.items():
            status = "✓" if met else "✗"
            print(f"  [{status}] {msg}")
        print()
        
        # Overall metrics
        print("-" * 60)
        print("OVERALL METRICS")
        print("-" * 60)
        print(f"  Accuracy:         {overall.accuracy:.2%}")
        print(f"  Attack Recall:    {overall.attack_recall:.2%}")
        print(f"  FPR:              {overall.false_positive_rate:.2%}")
        print(f"  Macro F1:         {overall.macro_f1:.3f}")
        print(f"  Weighted F1:      {overall.weighted_f1:.3f}")
        print()
        
        # Latency metrics
        if overall.mean_latency_ms > 0:
            print("-" * 60)
            print("LATENCY METRICS")
            print("-" * 60)
            print(f"  Mean Latency:     {overall.mean_latency_ms:.2f}ms")
            print(f"  P95 Latency:      {overall.p95_latency_ms:.2f}ms")
            print(f"  P99 Latency:      {overall.p99_latency_ms:.2f}ms")
            print(f"  Throughput:       {overall.throughput_per_sec:.0f} samples/sec")
            print()
        
        # Zero-day detection
        if overall.zero_day_candidates > 0:
            print("-" * 60)
            print("ZERO-DAY DETECTION")
            print("-" * 60)
            print(f"  Low-Conf Attacks: {overall.zero_day_candidates}")
            print(f"  Low-Conf Rate:    {overall.low_confidence_rate:.1%}")
            print()
        
        # Per-class metrics
        print("-" * 60)
        print("PER-CLASS METRICS")
        print("-" * 60)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)
        for m in class_metrics:
            print(f"{m.class_name:<15} {m.precision:>10.3f} {m.recall:>10.3f} "
                  f"{m.f1_score:>10.3f} {m.support:>10,}")
        print()
        
        # Federated metrics
        fed_metrics = self.get_federated_metrics()
        if fed_metrics:
            print("-" * 60)
            print("FEDERATED LEARNING")
            print("-" * 60)
            print(f"  Rounds:           {fed_metrics.communication_rounds}")
            print(f"  Global Accuracy:  {fed_metrics.global_accuracy:.2%}")
            print(f"  Accuracy Parity:  {fed_metrics.accuracy_parity:.2%}")
            for client, acc in fed_metrics.client_accuracies.items():
                print(f"    {client}: {acc:.2%}")
            print()
        
        print("=" * 60)
    
    def save_report(self, filepath: str):
        """Save metrics report to file."""
        report = {
            "overall": asdict(self.get_overall_metrics()),
            "per_class": [asdict(m) for m in self.get_class_metrics()],
            "targets": {k: {"met": v[0], "message": v[1]} for k, v in self.check_targets().items()},
        }
        
        fed = self.get_federated_metrics()
        if fed:
            report["federated"] = asdict(fed)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
    
    def reset(self):
        """Reset all metrics."""
        self._y_true.clear()
        self._y_pred.clear()
        self._confidences.clear()
        self._latencies.clear()
        self._timestamps.clear()
        self._fed_rounds.clear()
        self._start_time = time.time()


def demo_dashboard():
    """Demonstrate dashboard with synthetic data."""
    np.random.seed(42)
    
    dashboard = MetricsDashboard(window_size=1000)
    
    # Generate synthetic predictions
    n_samples = 500
    
    # Ground truth: 70% Normal, 30% attacks
    y_true = np.random.choice(
        range(7),
        size=n_samples,
        p=[0.7, 0.1, 0.05, 0.05, 0.04, 0.04, 0.02]
    )
    
    # Predictions with ~85% accuracy, some confusion
    y_pred = y_true.copy()
    error_mask = np.random.random(n_samples) < 0.15
    y_pred[error_mask] = np.random.randint(0, 7, np.sum(error_mask))
    
    # Confidence scores (lower for errors)
    confidences = np.where(y_true == y_pred, 
                          np.random.uniform(0.7, 0.99, n_samples),
                          np.random.uniform(0.3, 0.6, n_samples))
    
    # Latencies (simulate ~10ms average)
    latencies = np.random.exponential(scale=10, size=n_samples)
    
    # Update dashboard
    dashboard.update_predictions(y_true, y_pred, confidences, latencies)
    
    # Add federated rounds
    for i in range(5):
        dashboard.update_federated_round(
            round_num=i+1,
            global_accuracy=0.75 + i * 0.03,
            client_accuracies={
                "Hospital": 0.73 + i * 0.03,
                "Bank": 0.77 + i * 0.03,
                "Telecom": 0.75 + i * 0.03,
            }
        )
    
    # Print report
    dashboard.print_report()
    
    # Save JSON report
    dashboard.save_report("metrics_report.json")
    print("\nReport saved to metrics_report.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_dashboard()
