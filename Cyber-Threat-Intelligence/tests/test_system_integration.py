"""
System Integration Tests for Privacy-Preserving Threat Intelligence Framework.

This module provides comprehensive integration tests covering:
1. Data dimension flow: DataLoader → Preprocessor → Agent 1/2
2. DP Engine → flwr format verification
3. Agent Two LLM mocking (prevents test hanging)
4. Agent Three RL environment mocking

Run with:
    pytest tests/test_system_integration.py -v --timeout=30
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface that returns immediately without network calls."""
    from agents.interfaces.base import LLMResponse
    
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="This is a mocked LLM response for testing. "
                "Severity: High. Recommended action: Block source IP.",
        model="mock-llm-v1",
        tokens_used=25,
    )
    mock_llm.generate_with_context.return_value = LLMResponse(
        content="Based on retrieved context, this appears to be a DoS attack. "
                "Similar to CVE-2021-44228 patterns.",
        model="mock-llm-v1",
        tokens_used=50,
    )
    return mock_llm


@pytest.fixture
def mock_vector_db_interface():
    """Mock VectorDB interface for RAG retrieval."""
    from agents.interfaces.base import RetrievedContext
    
    mock_db = MagicMock()
    mock_db.similarity_search.return_value = [
        RetrievedContext(
            content="DoS attacks involve overwhelming network resources...",
            metadata={"source": "mitre", "technique": "T1499"},
            similarity_score=0.92,
        ),
        RetrievedContext(
            content="SYN flood attacks exploit the TCP handshake...",
            metadata={"source": "knowledge_base"},
            similarity_score=0.87,
        ),
    ]
    mock_db.count = 100
    return mock_db


@pytest.fixture
def sample_network_features():
    """Generate sample network flow features matching UNSW-NB15 dimensions."""
    # 42 features after preprocessing (40 numerical + encoded categoricals)
    return np.random.randn(100, 42).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample labels for 7-class classification."""
    return np.random.randint(0, 7, size=100)


# =============================================================================
# Test Class: Data Dimension Flow
# =============================================================================

class TestDataDimensionFlow:
    """
    Tests verifying data dimension consistency across the pipeline.
    
    DataLoader → Preprocessor → Agent 1 (Autoencoder) / Agent 2 (XGBoost)
    """
    
    def test_unsw_nb15_feature_count(self):
        """Verify UNSW-NB15 config has expected feature count."""
        from data_pipeline.config import DatasetConfig
        
        config = DatasetConfig()
        
        # Numerical (37) + Categorical (3) + Binary (2) = 42
        total_features = len(config.numerical_features) + \
                        len(config.categorical_features) + \
                        len(config.binary_features)
        
        assert total_features == 42, (
            f"Expected 42 features, got {total_features}"
        )
    
    def test_autoencoder_accepts_preprocessed_data(self, sample_network_features):
        """Test AnomalyAutoencoder accepts preprocessed data dimensions."""
        from agents.models.autoencoder import AnomalyAutoencoder
        
        input_dim = sample_network_features.shape[1]
        
        # Create autoencoder matching data dimensions
        model = AnomalyAutoencoder(
            input_dim=input_dim,
            latent_dim=8,
            hidden_dims=[32, 16],
        )
        
        # Forward pass should work without dimension mismatch
        import torch
        X_tensor = torch.tensor(sample_network_features)
        
        with torch.no_grad():
            output = model(X_tensor)
        
        assert output.shape == X_tensor.shape, (
            f"Output shape {output.shape} should match input {X_tensor.shape}"
        )
    
    def test_xgboost_classifier_accepts_features(
        self, sample_network_features, sample_labels
    ):
        """Test XGBoost classifier trains on correct feature dimensions."""
        import xgboost as xgb
        
        # Create and fit XGBoost with sample data
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )
        
        model.fit(sample_network_features, sample_labels)
        
        # Verify feature dimension matches
        assert model.n_features_in_ == sample_network_features.shape[1]
        
        # Inference should work
        predictions = model.predict(sample_network_features[:10])
        assert len(predictions) == 10
    
    def test_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises appropriate error."""
        from agents.models.autoencoder import AnomalyAutoencoder
        import torch
        
        # Model expects 40 features
        model = AnomalyAutoencoder(input_dim=40, latent_dim=8)
        
        # Data has 42 features - should fail
        X_wrong_dim = torch.randn(10, 42)
        
        with pytest.raises(RuntimeError):
            model(X_wrong_dim)


# =============================================================================
# Test Class: Differential Privacy → Flower Format
# =============================================================================

class TestDPToFlowerFormat:
    """
    Tests verifying DifferentialPrivacyEngine outputs correct format for flwr.
    
    flwr requires List[np.ndarray] (aliased as NDArrays).
    """
    
    def test_dp_returns_list_of_numpy_arrays(self):
        """Test apply_dp returns List[np.ndarray] for flwr compatibility."""
        from federated.differential_privacy import DifferentialPrivacyEngine
        
        dp_engine = DifferentialPrivacyEngine(random_state=42)
        
        # Input weights as List[np.ndarray]
        input_weights = [
            np.random.randn(100, 50).astype(np.float32),
            np.random.randn(50).astype(np.float32),
            np.random.randn(50, 20).astype(np.float32),
        ]
        
        # Apply DP
        output = dp_engine.apply_dp(
            weights=input_weights,
            clip_norm=1.0,
            noise_multiplier=0.1,
        )
        
        # Verify type is list
        assert isinstance(output, list), f"Expected list, got {type(output)}"
        
        # Verify each element is numpy array
        for i, w in enumerate(output):
            assert isinstance(w, np.ndarray), (
                f"Element {i} should be np.ndarray, got {type(w)}"
            )
    
    def test_dp_preserves_weight_structure_for_flwr(self):
        """Test DP preserves shapes/dtypes required by flwr aggregation."""
        from federated.differential_privacy import DifferentialPrivacyEngine
        
        dp_engine = DifferentialPrivacyEngine()
        
        # Simulate autoencoder weights
        original_weights = [
            np.zeros((40, 32), dtype=np.float32),  # encoder layer 1
            np.zeros(32, dtype=np.float32),         # bias 1
            np.zeros((32, 16), dtype=np.float32),  # encoder layer 2
            np.zeros(16, dtype=np.float32),         # bias 2
        ]
        
        private_weights = dp_engine.apply_dp(
            weights=original_weights,
            clip_norm=1.0,
            noise_multiplier=0.1,
        )
        
        # Structure must be preserved for flwr averaging
        assert len(private_weights) == len(original_weights)
        
        for i, (orig, priv) in enumerate(zip(original_weights, private_weights)):
            assert priv.shape == orig.shape, (
                f"Weight {i} shape mismatch: {orig.shape} vs {priv.shape}"
            )
            assert priv.dtype == orig.dtype, (
                f"Weight {i} dtype mismatch: {orig.dtype} vs {priv.dtype}"
            )
    
    def test_federated_client_integration_with_dp(self):
        """Test NetworkDefenseClient properly integrates DP for flwr."""
        from federated.client import NetworkDefenseClient
        from agents.models.autoencoder import AnomalyAutoencoder
        
        autoencoder = AnomalyAutoencoder(input_dim=40, latent_dim=8)
        
        # Client without XGBoost (pure autoencoder weights)
        client = NetworkDefenseClient(
            client_id="test_dp_integration",
            autoencoder=autoencoder,
            xgboost_model=None,  # No XGBoost - pure numpy arrays
            dp_enabled=True,
            dp_clip_norm=1.0,
            dp_noise_multiplier=0.1,
        )
        
        # Get parameters should return flwr-compatible format
        params = client.get_parameters(config={})
        
        # Verify format: should be List[np.ndarray] for pure autoencoder
        assert isinstance(params, list)
        for i, p in enumerate(params):
            assert isinstance(p, np.ndarray), (
                f"Parameter {i} should be np.ndarray, got {type(p)}"
            )


# =============================================================================
# Test Class: Agent Two LLM Mocking
# =============================================================================

class TestAgentTwoLLMMocking:
    """
    Tests verifying Agent Two LLM calls are properly mocked.
    
    Prevents test hanging from real LLM API calls.
    """
    
    def test_agent_two_with_mock_llm_no_hang(
        self, mock_llm_interface, mock_vector_db_interface
    ):
        """
        Test AgentTwo.analyze_threat completes quickly with mocked LLM.
        
        This test should complete in < 1 second with mocking.
        Real LLM calls could take 10+ seconds or hang indefinitely.
        """
        from agents.agent_two import AgentTwo
        from agents.models.xgboost_classifier import ThreatClassifier
        import time
        
        # Create mock classifier
        mock_classifier = MagicMock(spec=ThreatClassifier)
        mock_classifier.is_fitted = True
        mock_classifier.n_classes = 7
        mock_classifier.ATTACK_CATEGORIES = [
            "Normal", "DoS", "Reconnaissance", "Exploits",
            "Brute_Force", "Malware", "Analysis"
        ]
        
        # Mock prediction result
        from agents.models.xgboost_classifier import ClassificationResult
        mock_classifier.predict.return_value = ClassificationResult(
            predicted_category="DoS",
            category_id=1,
            confidence=0.85,
            is_zero_day=True,  # Triggers LLM reasoning
            all_probabilities={"DoS": 0.85, "Normal": 0.1, "Other": 0.05},
        )
        
        # Create agent with mocked dependencies
        agent = AgentTwo(
            classifier=mock_classifier,
            vector_db=mock_vector_db_interface,
            llm=mock_llm_interface,
        )
        
        # Time the analysis
        features = np.random.randn(42).astype(np.float32)
        
        start_time = time.time()
        result = agent.analyze_threat(features)
        elapsed = time.time() - start_time
        
        # Should complete in under 1 second with mocking
        assert elapsed < 1.0, (
            f"analyze_threat took {elapsed:.2f}s - LLM may not be mocked"
        )
        
        # Verify LLM was called (since is_zero_day=True)
        assert mock_llm_interface.generate_with_context.called or \
               mock_llm_interface.generate.called
    
    def test_agent_two_batch_analysis_with_mock(
        self, mock_llm_interface, mock_vector_db_interface
    ):
        """Test batch analysis with mocked LLM doesn't accumulate delay."""
        from agents.agent_two import AgentTwo
        from agents.models.xgboost_classifier import ThreatClassifier, ClassificationResult
        import time
        
        # Setup mock classifier
        mock_classifier = MagicMock(spec=ThreatClassifier)
        mock_classifier.is_fitted = True
        mock_classifier.n_classes = 7
        
        # Return non-zero-day results for batch (avoids LLM per-item)
        mock_classifier.predict_batch.return_value = [
            ClassificationResult(
                predicted_category="DoS",
                category_id=1,
                confidence=0.9,
                is_zero_day=False,  # No LLM
                all_probabilities={},
            )
            for _ in range(50)
        ]
        
        agent = AgentTwo(
            classifier=mock_classifier,
            vector_db=mock_vector_db_interface,
            llm=mock_llm_interface,
        )
        
        features_batch = np.random.randn(50, 42).astype(np.float32)
        
        start_time = time.time()
        results = agent.analyze_threats_batch(features_batch)
        elapsed = time.time() - start_time
        
        # Batch of 50 should complete in < 2 seconds
        assert elapsed < 2.0, f"Batch analysis took {elapsed:.2f}s"
        assert len(results) == 50


# =============================================================================
# Test Class: Agent Three RL Environment Mocking
# =============================================================================

class TestAgentThreeRLMocking:
    """
    Tests verifying RL environment is properly mocked for Agent Three.
    
    Prevents slow training loops in tests.
    """
    
    def test_mock_environment_step(self):
        """Test mocked environment step returns expected format."""
        from agents.environments.network_defense_env import (
            NetworkDefenseEnv, ThreatState, MitigationAction
        )
        
        # Create mock environment
        mock_env = MagicMock(spec=NetworkDefenseEnv)
        
        # Setup reset mock
        mock_observation = np.zeros(14, dtype=np.float32)
        mock_env.reset.return_value = (mock_observation, {})
        
        # Setup step mock
        mock_env.step.return_value = (
            mock_observation,  # next_obs
            1.0,              # reward
            False,            # terminated
            False,            # truncated
            {"action_taken": 2},  # info
        )
        
        # Verify mock works as expected
        obs, info = mock_env.reset()
        assert obs.shape == (14,)
        
        next_obs, reward, term, trunc, step_info = mock_env.step(0)
        assert isinstance(reward, float)
        assert mock_env.step.called
    
    def test_agent_three_with_mocked_env_no_training(self):
        """Test Agent Three inference without running full training."""
        from agents.agent_three import AgentThree
        from agents.environments.network_defense_env import MitigationAction
        from unittest.mock import MagicMock
        
        # Create agent without loading model (uses fallback policy)
        agent = AgentThree()
        
        # Mock a ThreatAnalysisResult from Agent Two (the actual input type)
        mock_analysis_result = MagicMock()
        mock_analysis_result.classification = MagicMock()
        mock_analysis_result.classification.predicted_category = "DoS"
        mock_analysis_result.classification.confidence = 0.85
        mock_analysis_result.classification.is_zero_day = False
        mock_analysis_result.is_zero_day = False
        mock_analysis_result.severity = "high"
        
        # Get action using take_action (the actual method name)
        decision = agent.take_action(mock_analysis_result)
        
        # Decision should be a MitigationDecision namedtuple
        assert hasattr(decision, 'action')
        assert 0 <= decision.action <= 3  # Valid action range
    
    @pytest.mark.slow
    def test_agent_three_env_observation_space(self):
        """Test Agent Three environment observation space is correct."""
        from agents.environments.network_defense_env import NetworkDefenseEnv
        import gymnasium as gym
        
        env = NetworkDefenseEnv()
        
        # Observation space should be 14-dimensional
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (14,)
        
        # Action space should be 4 discrete actions
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4
        
        # Reset should return valid observation
        obs, info = env.reset()
        assert obs.shape == (14,)


# =============================================================================
# Test Class: End-to-End Integration (All Mocked)
# =============================================================================

class TestEndToEndMocked:
    """
    Full pipeline integration test with all external dependencies mocked.
    
    Data → Agent 1 → Agent 2 → Agent 3 → Response
    """
    
    def test_full_pipeline_no_external_calls(
        self, mock_llm_interface, mock_vector_db_interface, sample_network_features
    ):
        """Test complete detection pipeline with all mocks."""
        from agents.models.autoencoder import AnomalyAutoencoder
        from federated.coordinator import IntegrationCoordinator, ThreatSeverity
        
        # Setup Agent 1 (real autoencoder, no external calls)
        autoencoder = AnomalyAutoencoder(
            input_dim=sample_network_features.shape[1],
            latent_dim=8,
        )
        
        # Mock Agent 1
        mock_agent_one = MagicMock()
        mock_agent_one.detect.return_value = True
        mock_agent_one.reconstruction_error = 0.05
        mock_agent_one.threshold = 0.04
        mock_agent_one.model = autoencoder
        
        # Mock Agent 2
        mock_agent_two = MagicMock()
        mock_agent_two.classify.return_value = {
            "category": "DoS",
            "confidence": 0.85,
            "is_zero_day": False,
        }
        
        # Mock Agent 3
        mock_agent_three = MagicMock()
        mock_agent_three.get_action.return_value = 2  # Block IP
        
        # Create coordinator with all mocked agents
        coordinator = IntegrationCoordinator(
            agent_one=mock_agent_one,
            agent_two=mock_agent_two,
            agent_three=mock_agent_three,
        )
        
        # Process sample
        sample = sample_network_features[0]
        report = coordinator.process_network_sample(
            sample,
            sample_id="test_001",
            include_rag_enrichment=False,  # Skip RAG to avoid LLM
        )
        
        # Verify pipeline completed
        assert report is not None
        assert report.sample_id == "test_001"
        
        # All agents should have been called
        assert mock_agent_one.detect.called
        assert mock_agent_two.classify.called
        # Agent 3 only called if threat detected
    
    def test_federated_round_with_dp_fully_mocked(self):
        """Test federated learning round with DP, no real training."""
        from federated.client import NetworkDefenseClient
        from federated.differential_privacy import DifferentialPrivacyEngine
        from agents.models.autoencoder import AnomalyAutoencoder
        
        # Create minimal autoencoder
        autoencoder = AnomalyAutoencoder(input_dim=40, latent_dim=4)
        
        # Create client with DP but WITHOUT XGBoost for pure numpy arrays
        client = NetworkDefenseClient(
            client_id="test_fed_round",
            autoencoder=autoencoder,
            xgboost_model=None,  # No XGBoost
            dp_enabled=True,
            dp_clip_norm=1.0,
            dp_noise_multiplier=0.1,
        )
        
        # Provide minimal training data
        client.train_data = (
            np.random.randn(20, 40).astype(np.float32),
            None,
        )
        client.val_data = (
            np.random.randn(5, 40).astype(np.float32),
            None,
        )
        
        # Get initial parameters
        params = client.get_parameters(config={})
        
        # Simulate fit round
        updated_params, num_samples, metrics = client.fit(
            parameters=params,
            config={"autoencoder_epochs": 1},
        )
        
        # Verify DP was applied
        assert metrics["dp_enabled"] is True
        assert "dp_clip_count" in metrics
        
        # Verify output format for flwr (pure autoencoder = all numpy arrays)
        assert isinstance(updated_params, list)
        for i, p in enumerate(updated_params):
            assert isinstance(p, np.ndarray), (
                f"Parameter {i} should be np.ndarray, got {type(p)}"
            )


# =============================================================================
# Test Class: Import Verification
# =============================================================================

class TestImportChains:
    """Verify all module imports resolve without errors."""
    
    def test_federated_imports(self):
        """Test federated module imports."""
        from federated import (
            NetworkDefenseClient,
            DifferentialPrivacyEngine,
            IntegrationCoordinator,
            create_client_fn,
        )
        
        assert NetworkDefenseClient is not None
        assert DifferentialPrivacyEngine is not None
    
    def test_agent_imports(self):
        """Test agent module imports."""
        from agents.agent_one import AgentOne
        from agents.agent_two import AgentTwo
        from agents.agent_three import AgentThree
        from agents.models.autoencoder import AnomalyAutoencoder
        from agents.models.xgboost_classifier import ThreatClassifier
        
        assert AgentOne is not None
        assert AgentTwo is not None
        assert AgentThree is not None
    
    def test_data_pipeline_imports(self):
        """Test data pipeline imports."""
        from data_pipeline.config import DatasetConfig
        from data_pipeline.preprocessor import Preprocessor
        from data_pipeline.data_loader import DataLoader
        from data_pipeline.unified_taxonomy import UnifiedTaxonomy
        
        assert DatasetConfig is not None
        assert Preprocessor is not None
    
    def test_environment_imports(self):
        """Test RL environment imports."""
        from agents.environments.network_defense_env import (
            NetworkDefenseEnv,
            ThreatState,
            MitigationAction,
        )
        
        assert NetworkDefenseEnv is not None
        assert MitigationAction is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=30"])
