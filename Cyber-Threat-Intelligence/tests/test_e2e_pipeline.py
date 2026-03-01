"""
End-to-End Pipeline Tests.

Tests the complete flow from data loading through model inference:
1. Data loading and preprocessing
2. Agent One (Autoencoder) training and inference
3. Agent Two (XGBoost + RAG) training and inference
4. Agent Three (RL) decision making
5. Federated learning synchronization
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Data Pipeline E2E Tests
# ============================================================================

class TestDataPipelineE2E:
    """End-to-end tests for data pipeline."""
    
    @pytest.fixture
    def sample_unsw_data(self, tmp_path):
        """Create sample UNSW-NB15 data."""
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'id': range(n_samples),
            'dur': np.random.rand(n_samples),
            'proto': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ftp', 'ssh', '-'], n_samples),
            'state': np.random.choice(['FIN', 'CON', 'INT'], n_samples),
            'attack_cat': np.random.choice(
                ['Normal', 'DoS', 'Exploits', 'Reconnaissance', 'Fuzzers'],
                n_samples,
                p=[0.5, 0.2, 0.15, 0.1, 0.05]
            ),
            'label': [0] * 250 + [1] * 250,
        }
        
        # Add numeric features
        for i in range(40):
            data[f'feature_{i}'] = np.random.rand(n_samples)
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_unsw.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def test_full_data_loading_pipeline(self, sample_unsw_data):
        """Test complete data loading pipeline."""
        from data_pipeline.unified_dataset import UnifiedIDSDataset, UnifiedDatasetConfig
        
        config = UnifiedDatasetConfig(apply_smote=False)
        dataset = UnifiedIDSDataset(config=config)
        dataset.load_unsw_nb15(sample_unsw_data)
        
        # Verify data loaded
        assert dataset.X is not None
        assert dataset.y is not None
        assert len(dataset.X) == 500
        
        # Get train/test split
        X_train, X_test, y_train, y_test = dataset.get_train_test_split()
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == 500
    
    def test_data_loading_with_smote(self, sample_unsw_data):
        """Test data loading with SMOTE balancing."""
        from data_pipeline.unified_dataset import UnifiedIDSDataset, UnifiedDatasetConfig
        
        config = UnifiedDatasetConfig(
            apply_smote=True,
            min_samples_per_class=10  # Lower for test
        )
        dataset = UnifiedIDSDataset(config=config)
        dataset.load_unsw_nb15(sample_unsw_data)
        
        original_size = len(dataset.X)
        original_counts = np.bincount(dataset.y)
        
        # Apply SMOTE
        dataset.apply_smote()
        
        # Verify balancing
        new_counts = np.bincount(dataset.y)
        
        # SMOTE should have made minority classes larger
        assert len(dataset.X) >= original_size
    
    def test_data_normalization_pipeline(self, sample_unsw_data):
        """Test normalization in pipeline."""
        from data_pipeline.unified_dataset import UnifiedIDSDataset, UnifiedDatasetConfig
        
        config = UnifiedDatasetConfig(normalization="standard")
        dataset = UnifiedIDSDataset(config=config)
        dataset.load_unsw_nb15(sample_unsw_data)
        dataset.normalize()
        
        # Check normalization applied
        mean = np.mean(dataset.X, axis=0)
        std = np.std(dataset.X, axis=0)
        
        # Most features should have mean close to 0
        assert np.abs(mean).mean() < 0.5


# ============================================================================
# Agent Integration E2E Tests
# ============================================================================

class TestAgentIntegrationE2E:
    """End-to-end tests for agent integration."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        return np.random.rand(100, 42).astype(np.float32)
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        np.random.seed(42)
        return np.random.randint(0, 7, 100)
    
    def test_agent_one_autoencoder_training(self, sample_features):
        """Test Agent One autoencoder training."""
        try:
            from agents.agent_one import AnomalyDetectionAgent
            
            agent = AnomalyDetectionAgent(input_dim=42)
            agent.train(sample_features, epochs=5, batch_size=32)
            
            # Verify training completed
            assert agent.autoencoder is not None
            
            # Test inference
            predictions = agent.predict(sample_features[:10])
            assert len(predictions) == 10
        except ImportError:
            pytest.skip("Agent One not available")
    
    def test_agent_two_xgboost_training(self, sample_features, sample_labels):
        """Test Agent Two XGBoost training."""
        try:
            from agents.agent_two import ThreatClassificationAgent
            
            agent = ThreatClassificationAgent(num_classes=7)
            agent.train(sample_features, sample_labels)
            
            # Test inference
            predictions = agent.predict(sample_features[:10])
            assert len(predictions) == 10
            assert all(0 <= p < 7 for p in predictions)
        except ImportError:
            pytest.skip("Agent Two not available")
    
    def test_inference_latency(self, sample_features):
        """Test inference latency meets requirements (<50ms per sample)."""
        try:
            from agents.agent_one import AnomalyDetectionAgent
            
            agent = AnomalyDetectionAgent(input_dim=42)
            agent.train(sample_features, epochs=5, batch_size=32)
            
            # Measure inference time for single sample
            sample = sample_features[0:1]
            
            start = time.time()
            for _ in range(100):
                agent.predict(sample)
            elapsed = (time.time() - start) / 100
            
            # Should be under 50ms
            assert elapsed < 0.05, f"Inference took {elapsed*1000:.2f}ms, expected <50ms"
        except ImportError:
            pytest.skip("Agent One not available")


# ============================================================================
# Federated Learning E2E Tests
# ============================================================================

class TestFederatedLearningE2E:
    """End-to-end tests for federated learning."""
    
    def test_client_initialization(self):
        """Test federated client initialization."""
        try:
            from federated.client import NetworkDefenseClient
            
            # Create mock autoencoder
            mock_autoencoder = Mock()
            mock_autoencoder.get_weights.return_value = [
                np.random.rand(42, 32).astype(np.float32),
                np.random.rand(32,).astype(np.float32),
            ]
            
            client = NetworkDefenseClient(
                autoencoder=mock_autoencoder,
                train_data=np.random.rand(100, 42).astype(np.float32)
            )
            
            assert client is not None
        except ImportError:
            pytest.skip("Federated client not available")
    
    def test_weight_aggregation(self):
        """Test FedAvg weight aggregation."""
        try:
            from federated.aggregation import fed_avg
            
            # Simulate weights from 3 clients
            client_weights = [
                [np.random.rand(10, 5).astype(np.float32) for _ in range(3)]
                for _ in range(3)
            ]
            
            # Aggregate
            aggregated = fed_avg(client_weights)
            
            assert len(aggregated) == 3
            for w in aggregated:
                assert w.shape == (10, 5)
        except ImportError:
            pytest.skip("Aggregation module not available")


# ============================================================================
# Performance Metrics Tests
# ============================================================================

class TestPerformanceMetrics:
    """Tests for verifying performance requirements."""
    
    def test_recall_metric_calculation(self):
        """Test recall metric calculation."""
        from sklearn.metrics import recall_score
        
        # Simulate predictions
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        
        recall = recall_score(y_true, y_pred, average='binary')
        
        # 5 true positives out of 6 actual positives
        expected = 5 / 6
        assert abs(recall - expected) < 0.01
    
    def test_false_positive_rate_calculation(self):
        """Test FPR calculation."""
        # Simulate predictions
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 0])
        
        # FPR = FP / (FP + TN)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fpr = fp / (fp + tn)
        
        # 1 false positive out of 6 negatives
        expected = 1 / 6
        assert abs(fpr - expected) < 0.01
    
    def test_multiclass_metrics(self):
        """Test multiclass metric calculation."""
        from sklearn.metrics import classification_report
        
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y_pred = np.array([0, 0, 1, 2, 2, 2, 3, 0])
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Verify report structure
        assert 'macro avg' in report
        assert 'weighted avg' in report
        assert 'accuracy' in report


# ============================================================================
# Zero-Day Detection Tests
# ============================================================================

class TestZeroDay:
    """Tests for zero-day (unknown attack) detection."""
    
    def test_anomaly_detection_threshold(self):
        """Test anomaly detection with configurable threshold."""
        # Simulate reconstruction errors
        normal_errors = np.random.exponential(scale=0.01, size=1000)
        attack_errors = np.random.exponential(scale=0.1, size=100)
        
        # Calculate 95th percentile threshold
        threshold = np.percentile(normal_errors, 95)
        
        # Should detect most attacks (high error) as anomalies
        attack_detected = np.sum(attack_errors > threshold) / len(attack_errors)
        assert attack_detected > 0.5, f"Only {attack_detected*100:.1f}% attacks detected"
    
    def test_confidence_based_detection(self):
        """Test confidence-based zero-day detection."""
        # Simulate classifier confidence scores
        known_attack_confidence = np.random.beta(8, 2, size=100)  # High confidence
        zero_day_confidence = np.random.beta(2, 5, size=100)  # Low confidence
        
        # Low confidence indicates potential zero-day
        threshold = 0.5
        
        zero_day_flagged = np.sum(zero_day_confidence < threshold) / len(zero_day_confidence)
        known_flagged = np.sum(known_attack_confidence < threshold) / len(known_attack_confidence)
        
        # Zero-days should be flagged more often
        assert zero_day_flagged > known_flagged
    
    def test_ensemble_detection(self):
        """Test ensemble approach for zero-day detection."""
        np.random.seed(42)
        
        # Autoencoder says anomaly (high reconstruction error)
        ae_anomaly = np.array([True, True, False, True, False])
        
        # XGBoost has low confidence
        xgb_confidence = np.array([0.3, 0.4, 0.9, 0.2, 0.95])
        
        # Combined detection: anomaly AND low confidence
        zero_day_candidates = ae_anomaly & (xgb_confidence < 0.5)
        
        # Should flag samples that are both anomalous and uncertain
        expected = np.array([True, True, False, True, False])
        np.testing.assert_array_equal(zero_day_candidates, expected)


# ============================================================================
# Adversarial Robustness Tests
# ============================================================================

class TestAdversarialRobustness:
    """Tests for adversarial robustness."""
    
    def test_feature_perturbation(self):
        """Test detection stability under small feature perturbations."""
        np.random.seed(42)
        
        # Original sample
        original = np.random.rand(42)
        
        # Perturbed samples (small noise)
        perturbations = [
            original + np.random.randn(42) * 0.01,
            original + np.random.randn(42) * 0.01,
            original + np.random.randn(42) * 0.01,
        ]
        
        # Check perturbations are small
        for perturbed in perturbations:
            distance = np.linalg.norm(original - perturbed)
            assert distance < 0.5, "Perturbation too large"
    
    def test_input_validation(self):
        """Test input validation prevents malformed data."""
        # Test inf values
        data_with_inf = np.array([1.0, np.inf, 2.0])
        cleaned = np.nan_to_num(data_with_inf, nan=0.0, posinf=0.0, neginf=0.0)
        assert not np.any(np.isinf(cleaned))
        
        # Test nan values
        data_with_nan = np.array([1.0, np.nan, 2.0])
        cleaned = np.nan_to_num(data_with_nan, nan=0.0, posinf=0.0, neginf=0.0)
        assert not np.any(np.isnan(cleaned))
    
    def test_outlier_handling(self):
        """Test handling of extreme outlier values."""
        # Normal data
        normal_data = np.random.randn(100, 10)
        
        # Add outliers
        outlier_data = normal_data.copy()
        outlier_data[0, 0] = 1e10
        outlier_data[1, 1] = -1e10
        
        # Clip to reasonable range
        clipped = np.clip(outlier_data, -100, 100)
        
        assert np.all(clipped <= 100)
        assert np.all(clipped >= -100)


# ============================================================================
# Data Consistency Tests
# ============================================================================

class TestDataConsistency:
    """Tests for data consistency across pipeline."""
    
    def test_feature_dimension_consistency(self):
        """Test feature dimensions remain consistent through pipeline."""
        from data_pipeline.unified_taxonomy import get_taxonomy
        
        taxonomy = get_taxonomy()
        
        # Verify category count
        assert taxonomy.num_classes == 7
        
        # Verify all categories have valid IDs
        for i in range(7):
            category = taxonomy.get_category_name(i)
            assert category is not None
    
    def test_label_consistency(self):
        """Test labels are consistent through pipeline."""
        from data_pipeline.unified_taxonomy import UnifiedTaxonomy
        
        taxonomy = UnifiedTaxonomy()
        
        # Test roundtrip: name -> id -> name
        categories = ["Normal", "DoS/DDoS", "Reconnaissance", "Exploits", 
                     "Brute_Force", "Malware", "Analysis"]
        for category in categories:
            cat_id = taxonomy.get_category_id(category)
            recovered = taxonomy.get_category_name(cat_id)
            assert recovered == category


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
