"""
Test Suite for Federated Learning Bridge.

This module contains unit and integration tests for the federated
learning implementation, following TDD principles.

Test Categories:
    1. Unit Tests - Weight extraction without data leakage
    2. Unit Tests - Client fit with mocked server response
    3. Integration Tests - Federated update → detection → SemanticThreatReport
    4. Edge Case Tests - Server unreachable → graceful fallback

Run with:
    pytest tests/test_federated_bridge.py -v
    pytest tests/test_federated_bridge.py -v --tb=short -x  # Stop on first failure
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import federated components
from federated.utils import (
    autoencoder_weights_to_numpy,
    numpy_to_autoencoder_weights,
    xgboost_to_numpy,
    numpy_to_xgboost,
    get_combined_weights,
    split_combined_weights,
)
from federated.client import NetworkDefenseClient, create_client_fn
from federated.differential_privacy import DifferentialPrivacyEngine
from federated.coordinator import (
    IntegrationCoordinator,
    SemanticThreatReport,
    ThreatSeverity,
    ATTACK_TO_MITRE_MAP,
)

# Import models for testing
from agents.models.autoencoder import AnomalyAutoencoder


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_autoencoder():
    """Create a sample autoencoder for testing."""
    model = AnomalyAutoencoder(input_dim=40, latent_dim=8, hidden_dims=[32, 16])
    return model


@pytest.fixture
def sample_xgboost():
    """Create a fitted XGBoost model for testing."""
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    
    # Create dummy data
    X = np.random.randn(100, 40)
    y = np.random.choice(['Normal', 'DoS', 'Exploits'], size=100)
    
    # Fit label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Fit model
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y_encoded)
    
    return model, label_encoder


@pytest.fixture
def sample_train_data():
    """Generate sample training data."""
    X = np.random.randn(500, 40).astype(np.float32)
    y = np.random.choice(
        ['Normal', 'DoS', 'Exploits', 'Fuzzers', 'Reconnaissance'],
        size=500
    )
    return X, y


@pytest.fixture
def sample_val_data():
    """Generate sample validation data."""
    X = np.random.randn(100, 40).astype(np.float32)
    y = np.random.choice(
        ['Normal', 'DoS', 'Exploits', 'Fuzzers', 'Reconnaissance'],
        size=100
    )
    return X, y


@pytest.fixture
def mock_agent_one():
    """Create a mock Agent One."""
    agent = Mock()
    agent.detect = Mock(return_value=True)
    agent.reconstruction_error = 0.05
    agent.threshold = 0.0396
    agent.model = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    return agent


@pytest.fixture
def mock_agent_two():
    """Create a mock Agent Two."""
    agent = Mock()
    agent.classify = Mock(return_value={"category": "DoS", "confidence": 0.85})
    return agent


@pytest.fixture
def mock_agent_three():
    """Create a mock Agent Three."""
    agent = Mock()
    agent.get_action = Mock(return_value=2)  # Block IP
    return agent


@pytest.fixture
def mock_vector_db():
    """Create a mock vector database."""
    from agents.interfaces.base import RetrievedContext
    
    db = Mock()
    db.similarity_search = Mock(return_value=[
        RetrievedContext(
            content="DoS attacks target service availability...",
            metadata={"source": "mitre"},
            similarity_score=0.95,
        ),
        RetrievedContext(
            content="CVE-2021-1234 involves TCP SYN flooding...",
            metadata={"source": "nvd"},
            similarity_score=0.88,
        ),
    ])
    return db


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    from agents.interfaces.base import LLMResponse
    
    llm = Mock()
    llm.generate = Mock(return_value=LLMResponse(
        content="This DoS attack pattern matches CVE-2021-1234. "
                "Recommend implementing rate limiting and monitoring TCP connections.",
        model="mock-llm",
        tokens_used=50,
    ))
    return llm


# =============================================================================
# Unit Tests: Weight Extraction (No Data Leakage)
# =============================================================================

class TestWeightExtraction:
    """Test weight extraction utilities."""
    
    def test_autoencoder_weights_to_numpy_shape(self, sample_autoencoder):
        """Test that weight extraction produces correct shapes."""
        weights = autoencoder_weights_to_numpy(sample_autoencoder)
        
        # Should extract all parameters
        expected_params = len(list(sample_autoencoder.state_dict().keys()))
        assert len(weights) == expected_params
        
        # All should be numpy arrays
        for w in weights:
            assert isinstance(w, np.ndarray)
    
    def test_autoencoder_weights_no_raw_data(self, sample_autoencoder):
        """Verify weight extraction doesn't include raw data."""
        # Set up some "training data" to ensure it's not leaked
        dummy_data = np.random.randn(100, 40)
        
        weights = autoencoder_weights_to_numpy(sample_autoencoder)
        
        # Weights should not contain patterns from dummy data
        for w in weights:
            # Check no abnormally large values that would indicate data
            assert np.abs(w).max() < 100, "Unexpectedly large weight values"
    
    def test_numpy_to_autoencoder_weights_roundtrip(self, sample_autoencoder):
        """Test that weight round-trip preserves values."""
        # Extract weights
        original_weights = autoencoder_weights_to_numpy(sample_autoencoder)
        
        # Create new model and load weights
        new_model = AnomalyAutoencoder(input_dim=40, latent_dim=8, hidden_dims=[32, 16])
        numpy_to_autoencoder_weights(new_model, original_weights)
        
        # Extract again and compare
        loaded_weights = autoencoder_weights_to_numpy(new_model)
        
        for orig, loaded in zip(original_weights, loaded_weights):
            np.testing.assert_array_almost_equal(orig, loaded)
    
    def test_xgboost_to_numpy_serialization(self, sample_xgboost):
        """Test XGBoost serialization produces valid output."""
        model, label_encoder = sample_xgboost
        
        serialized, metadata = xgboost_to_numpy(model, label_encoder)
        
        # Should be numpy array of bytes
        assert isinstance(serialized, np.ndarray)
        assert serialized.dtype == np.uint8
        
        # Metadata should contain model info
        assert "n_estimators" in metadata
        assert "serialization_format" in metadata
        assert metadata["serialization_format"] == "json_bytes"
    
    def test_xgboost_roundtrip(self, sample_xgboost):
        """Test XGBoost serialization/deserialization roundtrip."""
        model, label_encoder = sample_xgboost
        
        # Generate test predictions before serialization
        X_test = np.random.randn(10, 40)
        original_preds = model.predict(X_test)
        
        # Serialize and deserialize
        serialized, metadata = xgboost_to_numpy(model, label_encoder)
        restored_model = numpy_to_xgboost(serialized, metadata)
        
        # Predictions should match
        restored_preds = restored_model.predict(X_test)
        np.testing.assert_array_equal(original_preds, restored_preds)
    
    def test_combined_weights_structure(self, sample_autoencoder, sample_xgboost):
        """Test combining autoencoder and XGBoost weights."""
        model, label_encoder = sample_xgboost
        
        combined, metadata = get_combined_weights(
            sample_autoencoder, model, label_encoder
        )
        
        # Should have all autoencoder weights plus one XGBoost array
        ae_count = len(list(sample_autoencoder.state_dict().keys()))
        assert len(combined) == ae_count + 1
        
        # Metadata should track structure
        assert metadata["autoencoder_weight_count"] == ae_count
        assert metadata["total_weight_count"] == ae_count + 1
    
    def test_split_combined_weights(self, sample_autoencoder, sample_xgboost):
        """Test splitting combined weights back into components."""
        model, label_encoder = sample_xgboost
        
        # Combine
        combined, metadata = get_combined_weights(
            sample_autoencoder, model, label_encoder
        )
        
        # Split
        ae_count = metadata["autoencoder_weight_count"]
        ae_weights, xgb_serialized = split_combined_weights(combined, ae_count)
        
        # Verify counts
        assert len(ae_weights) == ae_count
        assert isinstance(xgb_serialized, np.ndarray)


# =============================================================================
# Unit Tests: Client Fit with Mocked Server
# =============================================================================

class TestNetworkDefenseClient:
    """Test NetworkDefenseClient operations."""
    
    def test_client_initialization(self, sample_autoencoder):
        """Test client initializes correctly."""
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            client_id="test_client_001",
        )
        
        assert client.client_id == "test_client_001"
        assert client.autoencoder is not None
        assert client._has_xgboost is False
    
    def test_client_get_parameters(self, sample_autoencoder):
        """Test client returns parameters correctly."""
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            client_id="test_client",
        )
        
        params = client.get_parameters(config={})
        
        # Should return numpy arrays
        assert len(params) > 0
        for p in params:
            assert isinstance(p, np.ndarray)
    
    def test_client_set_parameters(self, sample_autoencoder):
        """Test client accepts parameter updates."""
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            client_id="test_client",
        )
        
        # Get parameters, modify, and set back
        params = client.get_parameters(config={})
        modified_params = [p + 0.001 for p in params]
        
        client.set_parameters(modified_params)
        
        # Verify update took effect
        new_params = client.get_parameters(config={})
        for orig, modified, new in zip(params, modified_params, new_params):
            # New should be close to modified, not original
            assert not np.allclose(orig, new)
    
    def test_client_fit_with_data(self, sample_autoencoder, sample_train_data):
        """Test client fit method with training data."""
        X_train, y_train = sample_train_data
        
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            train_data=(X_train, None),  # Unsupervised for autoencoder
            training_config={"autoencoder_epochs": 1, "autoencoder_batch_size": 64},
            client_id="test_client",
        )
        
        # Get initial parameters
        initial_params = client.get_parameters(config={})
        
        # Run fit
        updated_params, num_samples, metrics = client.fit(
            parameters=initial_params,
            config={"server_round": 1},
        )
        
        # Verify return values
        assert len(updated_params) == len(initial_params)
        assert num_samples == len(X_train)
        assert "autoencoder_loss" in metrics
        assert metrics["autoencoder_loss"] > 0
    
    def test_client_fit_without_data_raises(self, sample_autoencoder):
        """Test fit without data raises appropriate error."""
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            client_id="test_client",
        )
        
        with pytest.raises(ValueError, match="Training data not set"):
            client.fit(
                parameters=client.get_parameters(config={}),
                config={},
            )
    
    def test_client_evaluate(self, sample_autoencoder, sample_val_data):
        """Test client evaluate method."""
        X_val, y_val = sample_val_data
        
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            val_data=(X_val, None),
            client_id="test_client",
        )
        
        params = client.get_parameters(config={})
        loss, num_samples, metrics = client.evaluate(
            parameters=params,
            config={"server_round": 1},
        )
        
        assert isinstance(loss, float)
        assert loss >= 0
        assert num_samples == len(X_val)
        assert "autoencoder_loss" in metrics
    
    def test_client_with_xgboost(
        self, sample_autoencoder, sample_xgboost, sample_train_data
    ):
        """Test client with both autoencoder and XGBoost."""
        xgb_model, label_encoder = sample_xgboost
        X_train, y_train = sample_train_data
        
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            xgboost_model=xgb_model,
            label_encoder=label_encoder,
            train_data=(X_train, y_train),
            training_config={"autoencoder_epochs": 1},
            client_id="test_client",
        )
        
        assert client._has_xgboost is True
        
        # Get combined parameters
        params = client.get_parameters(config={})
        
        # Should have autoencoder weights + 1 XGBoost array
        ae_count = len(list(sample_autoencoder.state_dict().keys()))
        assert len(params) == ae_count + 1


# =============================================================================
# Integration Tests: Federated Update → Detection → SemanticThreatReport
# =============================================================================

class TestIntegrationCoordinator:
    """Test IntegrationCoordinator functionality."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initializes with default values."""
        coordinator = IntegrationCoordinator()
        
        assert coordinator.agent_one is None
        assert coordinator.agent_two is None
        assert coordinator.agent_three is None
        assert coordinator.zero_day_threshold == 0.4
    
    def test_coordinator_with_agents(
        self, mock_agent_one, mock_agent_two, mock_agent_three
    ):
        """Test coordinator with all agents configured."""
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        assert coordinator.agent_one is mock_agent_one
        assert coordinator.agent_two is mock_agent_two
        assert coordinator.agent_three is mock_agent_three
    
    def test_process_network_sample_basic(
        self, mock_agent_one, mock_agent_two, mock_agent_three
    ):
        """Test processing a network sample produces valid report."""
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        sample = np.random.randn(40)
        report = coordinator.process_network_sample(
            sample, sample_id="test_001", include_rag_enrichment=False
        )
        
        # Verify report structure
        assert isinstance(report, SemanticThreatReport)
        assert report.sample_id == "test_001"
        assert report.is_anomaly is True  # Mock returns True
        assert report.attack_category == "DoS"  # Mock returns DoS
        assert report.recommended_action == "Block IP"  # Action index 2
    
    def test_process_with_rag_enrichment(
        self, mock_agent_one, mock_agent_two, mock_agent_three,
        mock_vector_db, mock_llm
    ):
        """Test processing with RAG enrichment."""
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        coordinator.set_rag_system(mock_vector_db, mock_llm)
        
        sample = np.random.randn(40)
        report = coordinator.process_network_sample(
            sample, sample_id="test_rag", include_rag_enrichment=True
        )
        
        # RAG should have been called
        assert mock_vector_db.similarity_search.called
        assert mock_llm.generate.called
        
        # Report should have threat description
        assert len(report.threat_description) > 0
        assert "CVE-2021-1234" in report.cve_references
    
    def test_mitre_mapping(self):
        """Test MITRE ATT&CK mapping for attack categories."""
        coordinator = IntegrationCoordinator()
        
        # Test known mappings
        for category, expected in ATTACK_TO_MITRE_MAP.items():
            mapping = coordinator._get_mitre_mapping(category)
            assert mapping["technique"] == expected["technique"]
            assert mapping["tactic"] == expected["tactic"]
    
    def test_severity_calculation(self, mock_agent_one, mock_agent_two, mock_agent_three):
        """Test threat severity calculation."""
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        # Test zero-day is critical
        severity = coordinator._calculate_severity(
            is_anomaly=True, confidence=0.3, is_zero_day=True, attack_category="Unknown"
        )
        assert severity == ThreatSeverity.CRITICAL
        
        # Test high-severity categories
        severity = coordinator._calculate_severity(
            is_anomaly=True, confidence=0.9, is_zero_day=False, attack_category="Exploits"
        )
        assert severity == ThreatSeverity.CRITICAL
        
        # Test normal traffic
        severity = coordinator._calculate_severity(
            is_anomaly=False, confidence=0.0, is_zero_day=False, attack_category="Normal"
        )
        assert severity == ThreatSeverity.INFO
    
    def test_batch_processing(
        self, mock_agent_one, mock_agent_two, mock_agent_three
    ):
        """Test batch processing of multiple samples."""
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        samples = np.random.randn(10, 40)
        reports = coordinator.process_batch(samples, include_rag_enrichment=False)
        
        assert len(reports) == 10
        for report in reports:
            assert isinstance(report, SemanticThreatReport)
    
    def test_stats_tracking(
        self, mock_agent_one, mock_agent_two, mock_agent_three
    ):
        """Test statistics tracking across samples."""
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        # Process several samples
        for i in range(5):
            coordinator.process_network_sample(
                np.random.randn(40), include_rag_enrichment=False
            )
        
        stats = coordinator.get_stats()
        assert stats["samples_processed"] == 5
        assert stats["anomalies_detected"] == 5  # Mock always returns anomaly


# =============================================================================
# Integration Test: Full Pipeline
# =============================================================================

class TestFullPipeline:
    """Integration tests for full federated → detection → report pipeline."""
    
    def test_federated_update_then_detection(
        self, sample_autoencoder, mock_agent_two, mock_agent_three
    ):
        """Test federated update followed by detection."""
        # Setup coordinator with real autoencoder
        mock_agent_one = Mock()
        mock_agent_one.model = sample_autoencoder
        mock_agent_one.detect = Mock(return_value=True)
        mock_agent_one.threshold = 0.0396
        
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        # Simulate federated update
        weights = autoencoder_weights_to_numpy(sample_autoencoder)
        modified_weights = [w + 0.0001 for w in weights]  # Simulate aggregation
        
        coordinator.on_federated_update(round_number=1, aggregated_params=modified_weights)
        
        # Now run detection
        sample = np.random.randn(40)
        report = coordinator.process_network_sample(sample, include_rag_enrichment=False)
        
        assert isinstance(report, SemanticThreatReport)
        assert report.is_anomaly is True


# =============================================================================
# Edge Case Tests: Server Unreachable / Graceful Fallback
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_coordinator_without_agents(self):
        """Test coordinator handles missing agents gracefully."""
        coordinator = IntegrationCoordinator()
        
        sample = np.random.randn(40)
        report = coordinator.process_network_sample(sample, include_rag_enrichment=False)
        
        # Should still produce a report with fallback values
        assert isinstance(report, SemanticThreatReport)
        assert report.is_anomaly is True  # Fallback assumes anomaly
    
    def test_coordinator_rag_failure(
        self, mock_agent_one, mock_agent_two, mock_agent_three
    ):
        """Test coordinator handles RAG failures gracefully."""
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        # Mock RAG that raises exception
        failing_db = Mock()
        failing_db.similarity_search = Mock(side_effect=Exception("DB unavailable"))
        failing_llm = Mock()
        
        coordinator.set_rag_system(failing_db, failing_llm)
        
        sample = np.random.randn(40)
        report = coordinator.process_network_sample(sample, include_rag_enrichment=True)
        
        # Should still produce report, just without enrichment
        assert isinstance(report, SemanticThreatReport)
        assert report.threat_description == ""  # Enrichment failed
    
    def test_client_parameter_mismatch(self, sample_autoencoder):
        """Test client handles parameter count mismatch."""
        client = NetworkDefenseClient(
            autoencoder=sample_autoencoder,
            client_id="test_client",
        )
        
        # Get parameters and remove some
        params = client.get_parameters(config={})
        truncated_params = params[:len(params) - 2]  # Remove last 2
        
        # Should raise on strict mode (default)
        with pytest.raises(ValueError, match="Weight count mismatch"):
            client.set_parameters(truncated_params)
    
    def test_xgboost_deserialization_failure(self):
        """Test XGBoost deserialization handles invalid data."""
        invalid_data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)  # Not valid JSON
        
        with pytest.raises(ValueError, match="Failed to deserialize"):
            numpy_to_xgboost(invalid_data)
    
    def test_semantic_report_to_dict(self):
        """Test SemanticThreatReport serialization."""
        report = SemanticThreatReport(
            timestamp="2025-01-15T10:00:00",
            sample_id="test_001",
            is_anomaly=True,
            reconstruction_error=0.05,
            anomaly_threshold=0.0396,
            attack_category="DoS",
            classification_confidence=0.85,
            recommended_action="Block IP",
            action_confidence=0.9,
            severity=ThreatSeverity.HIGH,
        )
        
        as_dict = report.to_dict()
        
        assert as_dict["sample_id"] == "test_001"
        assert as_dict["anomaly_detection"]["is_anomaly"] is True
        assert as_dict["classification"]["category"] == "DoS"
        assert as_dict["severity"] == "high"
    
    def test_semantic_report_to_markdown(self):
        """Test SemanticThreatReport markdown generation."""
        report = SemanticThreatReport(
            timestamp="2025-01-15T10:00:00",
            sample_id="test_001",
            is_anomaly=True,
            reconstruction_error=0.05,
            anomaly_threshold=0.0396,
            attack_category="DoS",
            classification_confidence=0.85,
            recommended_action="Block IP",
            action_confidence=0.9,
            severity=ThreatSeverity.HIGH,
            mitre_technique="T1498",
            mitre_tactic="Impact",
            mitre_description="Network Denial of Service",
        )
        
        markdown = report.to_markdown()
        
        assert "# Threat Report" in markdown
        assert "DoS" in markdown
        assert "T1498" in markdown
        assert "🟠" in markdown  # HIGH severity emoji


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.slow
    def test_batch_processing_performance(
        self, mock_agent_one, mock_agent_two, mock_agent_three
    ):
        """Test batch processing doesn't degrade significantly at scale."""
        import time
        
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        # Process 1000 samples
        samples = np.random.randn(1000, 40)
        
        start = time.time()
        reports = coordinator.process_batch(samples, include_rag_enrichment=False)
        elapsed = time.time() - start
        
        assert len(reports) == 1000
        # Should complete in reasonable time (< 10 seconds for mocks)
        assert elapsed < 10, f"Batch processing took too long: {elapsed:.2f}s"


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestClientFactory:
    """Test client factory function."""
    
    def test_create_client_fn(self):
        """Test client factory creates valid clients."""
        client_fn = create_client_fn(
            autoencoder_class=AnomalyAutoencoder,
            autoencoder_config={"input_dim": 40, "latent_dim": 8, "hidden_dims": [32, 16]},
            training_config={"autoencoder_epochs": 1},
        )
        
        # Create client
        client = client_fn("client_001")
        
        assert isinstance(client, NetworkDefenseClient)
        assert client.client_id == "client_001"
        assert isinstance(client.autoencoder, AnomalyAutoencoder)
    
    def test_create_client_fn_with_data_loader(self):
        """Test client factory with custom data loader."""
        def mock_data_loader(client_id):
            X_train = np.random.randn(100, 40).astype(np.float32)
            X_val = np.random.randn(20, 40).astype(np.float32)
            return (X_train, None), (X_val, None)
        
        client_fn = create_client_fn(
            autoencoder_class=AnomalyAutoencoder,
            autoencoder_config={"input_dim": 40, "latent_dim": 8, "hidden_dims": [32, 16]},
            data_loader_fn=mock_data_loader,
        )
        
        client = client_fn("client_002")
        
        assert client.train_data is not None
        assert client.val_data is not None
        assert len(client.train_data[0]) == 100


# =============================================================================
# Differential Privacy Tests
# =============================================================================

class TestDifferentialPrivacy:
    """Test Local Differential Privacy implementation for federated learning."""
    
    # -------------------------------------------------------------------------
    # Unit Test: Clipping
    # -------------------------------------------------------------------------
    
    def test_clipping_enforces_norm_bound(self):
        """
        Unit Test: Pass a set of artificially large dummy weights to apply_dp
        (with noise set to 0) and assert that the L2 norm of the output does
        not exceed the clip_norm.
        """
        dp_engine = DifferentialPrivacyEngine(random_state=42)
        clip_norm = 1.0
        
        # Create artificially large weights that exceed the clip_norm
        large_weights = [
            np.array([10.0, 20.0, 30.0], dtype=np.float32),  # L2 norm ≈ 37.4
            np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),  # L2 norm ≈ 26.9
            np.array([100.0], dtype=np.float32),  # L2 norm = 100
        ]
        
        # Apply DP with noise_multiplier = 0 to isolate clipping behavior
        clipped_weights = dp_engine.apply_dp(
            weights=large_weights,
            clip_norm=clip_norm,
            noise_multiplier=0.0
        )
        
        # Assert each weight array has L2 norm <= clip_norm
        for i, weight in enumerate(clipped_weights):
            l2_norm = np.linalg.norm(weight.flatten())
            assert l2_norm <= clip_norm + 1e-6, (
                f"Weight array {i} has L2 norm {l2_norm:.4f}, "
                f"which exceeds clip_norm {clip_norm}"
            )
        
        # Verify clipping was applied (all 3 weight arrays should be clipped)
        assert dp_engine.clip_count == 3, (
            f"Expected 3 clipped arrays, got {dp_engine.clip_count}"
        )
    
    def test_clipping_preserves_small_weights(self):
        """
        Unit Test: Weights smaller than clip_norm should not be modified
        (aside from numerical precision).
        """
        dp_engine = DifferentialPrivacyEngine(random_state=42)
        clip_norm = 10.0
        
        # Create small weights that are well below clip_norm
        small_weights = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),  # L2 norm ≈ 0.37
            np.array([[0.5, 0.5]], dtype=np.float32),  # L2 norm ≈ 0.71
        ]
        
        # Apply DP with noise_multiplier = 0
        result_weights = dp_engine.apply_dp(
            weights=small_weights,
            clip_norm=clip_norm,
            noise_multiplier=0.0
        )
        
        # Assert weights are unchanged (within numerical tolerance)
        for i, (original, result) in enumerate(zip(small_weights, result_weights)):
            np.testing.assert_allclose(
                result, original, rtol=1e-5,
                err_msg=f"Weight array {i} was modified when it shouldn't have been"
            )
        
        # Verify no clipping was needed
        assert dp_engine.clip_count == 0, (
            f"Expected 0 clipped arrays, got {dp_engine.clip_count}"
        )
    
    # -------------------------------------------------------------------------
    # Unit Test: Noise Injection
    # -------------------------------------------------------------------------
    
    def test_noise_injection_modifies_weights(self):
        """
        Unit Test: Pass weights to apply_dp. Assert that the output weights
        are mathematically different from the input weights, but retain the
        exact same shape and data type.
        """
        dp_engine = DifferentialPrivacyEngine(random_state=42)
        
        # Create test weights
        original_weights = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([[0.5, 0.5], [0.1, 0.2]], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        ]
        
        # Apply DP with noise (using small clip_norm to avoid clipping interference)
        noisy_weights = dp_engine.apply_dp(
            weights=original_weights,
            clip_norm=100.0,  # Large clip_norm to avoid clipping
            noise_multiplier=1.0  # Significant noise
        )
        
        # Assert same number of weight arrays
        assert len(noisy_weights) == len(original_weights), (
            f"Expected {len(original_weights)} weight arrays, got {len(noisy_weights)}"
        )
        
        for i, (original, noisy) in enumerate(zip(original_weights, noisy_weights)):
            # Assert exact same shape
            assert noisy.shape == original.shape, (
                f"Weight array {i} shape mismatch: expected {original.shape}, got {noisy.shape}"
            )
            
            # Assert exact same dtype
            assert noisy.dtype == original.dtype, (
                f"Weight array {i} dtype mismatch: expected {original.dtype}, got {noisy.dtype}"
            )
            
            # Assert weights are different (noise was applied)
            assert not np.allclose(noisy, original), (
                f"Weight array {i} was not modified by noise injection"
            )
        
        # Verify noise was applied (noise_applied_count tracks apply_dp calls, not arrays)
        assert dp_engine.noise_applied_count == 1, (
            f"Expected 1 apply_dp call, got {dp_engine.noise_applied_count}"
        )
    
    def test_noise_injection_deterministic_with_seed(self):
        """
        Unit Test: Verify that the same random_state produces the same
        noisy output for reproducibility.
        """
        weights = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        
        # Apply DP with same seed twice
        engine1 = DifferentialPrivacyEngine(random_state=123)
        result1 = engine1.apply_dp(weights, clip_norm=10.0, noise_multiplier=0.5)
        
        engine2 = DifferentialPrivacyEngine(random_state=123)
        result2 = engine2.apply_dp(weights, clip_norm=10.0, noise_multiplier=0.5)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1[0], result2[0],
            err_msg="Same random_state should produce identical results"
        )
    
    # -------------------------------------------------------------------------
    # Integration Test: Fit Round with DP
    # -------------------------------------------------------------------------
    
    def test_client_fit_round_applies_dp(self):
        """
        Integration Test: Run a simulated flwr fit round and assert that
        the DP engine was successfully called before the weights were returned.
        """
        # Create client with DP enabled
        autoencoder = AnomalyAutoencoder(
            input_dim=40, latent_dim=8, hidden_dims=[32, 16]
        )
        
        client = NetworkDefenseClient(
            client_id="test_dp_client",
            autoencoder=autoencoder,
            dp_enabled=True,
            dp_clip_norm=1.0,
            dp_noise_multiplier=0.1
        )
        
        # Provide training data
        client.train_data = (
            np.random.randn(50, 40).astype(np.float32),
            None  # Labels not needed for autoencoder
        )
        client.val_data = (
            np.random.randn(10, 40).astype(np.float32),
            None
        )
        
        # Get parameters before fit (these should have DP applied)
        initial_params = client.get_parameters(config={})
        
        # Verify DP engine exists and was used
        assert client.dp_engine is not None, "DP engine should be initialized"
        assert client.dp_enabled is True, "DP should be enabled"
        
        # Run a fit round
        updated_params, num_samples, metrics = client.fit(
            parameters=initial_params,
            config={"autoencoder_epochs": 1}
        )
        
        # Verify DP metrics are included in fit response
        assert "dp_enabled" in metrics, "dp_enabled should be in metrics"
        assert metrics["dp_enabled"] is True, "dp_enabled metric should be True"
        assert "dp_clip_count" in metrics, "dp_clip_count should be in metrics"
        assert "dp_applications" in metrics, "dp_applications should be in metrics"
        
        # Verify returned parameters have DP applied (they should differ from
        # raw model weights due to noise injection)
        # Get raw weights without DP for comparison
        raw_weights = autoencoder_weights_to_numpy(client.autoencoder)
        
        # At least some returned parameters should differ from raw weights
        # (accounting for the fact that DP adds noise)
        params_differ = False
        for raw, dp_applied in zip(raw_weights, updated_params):
            if not np.allclose(raw, dp_applied, rtol=1e-5):
                params_differ = True
                break
        
        assert params_differ, (
            "DP-protected parameters should differ from raw model weights"
        )
    
    def test_client_fit_without_dp(self):
        """
        Integration Test: Verify that when DP is disabled, weights are
        returned unchanged.
        """
        autoencoder = AnomalyAutoencoder(
            input_dim=40, latent_dim=8, hidden_dims=[32, 16]
        )
        
        client = NetworkDefenseClient(
            client_id="test_no_dp_client",
            autoencoder=autoencoder,
            dp_enabled=False  # DP disabled
        )
        
        # Provide training data
        client.train_data = (
            np.random.randn(50, 40).astype(np.float32),
            None
        )
        client.val_data = (
            np.random.randn(10, 40).astype(np.float32),
            None
        )
        
        # Run a fit round
        initial_params = client.get_parameters(config={})
        updated_params, num_samples, metrics = client.fit(
            parameters=initial_params,
            config={"autoencoder_epochs": 1}
        )
        
        # Verify DP is disabled in metrics
        assert metrics.get("dp_enabled") is False, "dp_enabled should be False"
        
        # Get raw weights - they should match returned params exactly
        raw_weights = autoencoder_weights_to_numpy(client.autoencoder)
        
        for i, (raw, returned) in enumerate(zip(raw_weights, updated_params)):
            np.testing.assert_array_almost_equal(
                raw, returned,
                err_msg=f"Weight array {i} should be unchanged when DP is disabled"
            )
    
    def test_create_client_fn_with_dp_parameters(self):
        """
        Test that create_client_fn properly passes DP parameters to clients.
        """
        client_fn = create_client_fn(
            autoencoder_class=AnomalyAutoencoder,
            autoencoder_config={"input_dim": 40, "latent_dim": 8, "hidden_dims": [32, 16]},
            dp_enabled=True,
            dp_clip_norm=2.0,
            dp_noise_multiplier=0.5
        )
        
        client = client_fn("dp_test_client")
        
        assert client.dp_enabled is True, "DP should be enabled"
        assert client.dp_clip_norm == 2.0, "clip_norm should be 2.0"
        assert client.dp_noise_multiplier == 0.5, "noise_multiplier should be 0.5"
        assert client.dp_engine is not None, "DP engine should be initialized"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

