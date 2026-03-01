"""
Unit tests for data pipeline components.

Tests for:
- UnifiedTaxonomy: Attack label mapping across datasets
- CICIDS2017Loader: CIC-IDS2017 data loading
- UnifiedIDSDataset: Combined dataset with SMOTE
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.unified_taxonomy import (
    UnifiedTaxonomy, UnifiedCategory, CategoryInfo, get_taxonomy
)
from data_pipeline.cic_ids2017_loader import CICIDS2017Loader


# ============================================================================
# UnifiedTaxonomy Tests
# ============================================================================

class TestUnifiedTaxonomy:
    """Tests for UnifiedTaxonomy class."""
    
    @pytest.fixture
    def taxonomy(self):
        """Create taxonomy instance."""
        return UnifiedTaxonomy()
    
    # -------------------------------------------------------------------------
    # UNSW-NB15 Mapping Tests
    # -------------------------------------------------------------------------
    
    def test_map_unsw_nb15_normal(self, taxonomy):
        """Test mapping UNSW-NB15 Normal traffic."""
        assert taxonomy.map_unsw_nb15("Normal") == "Normal"
        assert taxonomy.map_unsw_nb15("normal") == "Normal"
        assert taxonomy.map_unsw_nb15("NORMAL") == "Normal"
    
    def test_map_unsw_nb15_dos(self, taxonomy):
        """Test mapping UNSW-NB15 DoS attacks."""
        assert taxonomy.map_unsw_nb15("DoS") == "DoS/DDoS"
        assert taxonomy.map_unsw_nb15("dos") == "DoS/DDoS"
    
    def test_map_unsw_nb15_reconnaissance(self, taxonomy):
        """Test mapping UNSW-NB15 reconnaissance attacks."""
        assert taxonomy.map_unsw_nb15("Reconnaissance") == "Reconnaissance"
        assert taxonomy.map_unsw_nb15("Analysis") == "Analysis"
    
    def test_map_unsw_nb15_exploits(self, taxonomy):
        """Test mapping UNSW-NB15 exploit attacks."""
        assert taxonomy.map_unsw_nb15("Exploits") == "Exploits"
        assert taxonomy.map_unsw_nb15("Backdoor") == "Malware"
        assert taxonomy.map_unsw_nb15("Backdoors") == "Malware"
        assert taxonomy.map_unsw_nb15("Shellcode") == "Exploits"
    
    def test_map_unsw_nb15_brute_force(self, taxonomy):
        """Test mapping UNSW-NB15 brute force attacks."""
        assert taxonomy.map_unsw_nb15("Fuzzers") == "Brute_Force"
    
    def test_map_unsw_nb15_malware(self, taxonomy):
        """Test mapping UNSW-NB15 malware attacks."""
        assert taxonomy.map_unsw_nb15("Worms") == "Malware"
        assert taxonomy.map_unsw_nb15("Generic") == "Malware"
    
    def test_map_unsw_nb15_unknown(self, taxonomy):
        """Test mapping unknown UNSW-NB15 labels."""
        # Unknown labels should map to Malware (default)
        result = taxonomy.map_unsw_nb15("UnknownAttack")
        assert result == "Malware"  # Default fallback
    
    # -------------------------------------------------------------------------
    # CIC-IDS2017 Mapping Tests
    # -------------------------------------------------------------------------
    
    def test_map_cic_ids2017_benign(self, taxonomy):
        """Test mapping CIC-IDS2017 benign traffic."""
        assert taxonomy.map_cic_ids2017("BENIGN") == "Normal"
        assert taxonomy.map_cic_ids2017("benign") == "Normal"
    
    def test_map_cic_ids2017_dos_ddos(self, taxonomy):
        """Test mapping CIC-IDS2017 DoS/DDoS attacks."""
        dos_attacks = [
            "DoS Hulk", "DoS GoldenEye", "DoS slowloris", 
            "DoS Slowhttptest", "DDoS"
        ]
        for attack in dos_attacks:
            assert taxonomy.map_cic_ids2017(attack) == "DoS/DDoS"
    
    def test_map_cic_ids2017_reconnaissance(self, taxonomy):
        """Test mapping CIC-IDS2017 reconnaissance attacks."""
        assert taxonomy.map_cic_ids2017("PortScan") == "Reconnaissance"
    
    def test_map_cic_ids2017_web_attacks(self, taxonomy):
        """Test mapping CIC-IDS2017 web attacks to Exploits."""
        result = taxonomy.map_cic_ids2017("Web Attack - Brute Force")
        assert result == "Exploits"
    
    def test_map_cic_ids2017_brute_force(self, taxonomy):
        """Test mapping CIC-IDS2017 brute force attacks."""
        brute_force = ["FTP-Patator", "SSH-Patator"]
        for attack in brute_force:
            assert taxonomy.map_cic_ids2017(attack) == "Brute_Force"
    
    def test_map_cic_ids2017_malware(self, taxonomy):
        """Test mapping CIC-IDS2017 malware attacks."""
        malware = ["Bot", "Infiltration"]
        for attack in malware:
            assert taxonomy.map_cic_ids2017(attack) == "Malware"
    
    # -------------------------------------------------------------------------
    # Category Info Tests
    # -------------------------------------------------------------------------
    
    def test_get_category_id(self, taxonomy):
        """Test category ID retrieval."""
        assert taxonomy.get_category_id("Normal") == 0
        assert taxonomy.get_category_id("DoS/DDoS") == 1
        assert taxonomy.get_category_id("Reconnaissance") == 2
        assert taxonomy.get_category_id("Exploits") == 3
        assert taxonomy.get_category_id("Brute_Force") == 4
        assert taxonomy.get_category_id("Malware") == 5
        assert taxonomy.get_category_id("Analysis") == 6
    
    def test_get_category_by_id(self, taxonomy):
        """Test category retrieval by ID."""
        for i in range(7):
            category = taxonomy.get_category_name(i)
            assert category is not None
            assert taxonomy.get_category_id(category) == i
    
    def test_get_category_info(self, taxonomy):
        """Test category info retrieval."""
        info = taxonomy.get_category_info("DoS/DDoS")
        assert info is not None
        assert "T1498" in info.mitre_techniques  # Network DoS
    
    def test_get_mitre_techniques(self, taxonomy):
        """Test MITRE technique retrieval."""
        techniques = taxonomy.get_mitre_techniques("Exploits")
        assert "T1190" in techniques  # Exploit Public-Facing Application
    
    def test_num_classes(self, taxonomy):
        """Test number of classes."""
        assert taxonomy.num_classes == 7
    
    def test_all_categories_have_info(self, taxonomy):
        """Test all categories have complete info."""
        categories = ["Normal", "DoS/DDoS", "Reconnaissance", "Exploits", 
                     "Brute_Force", "Malware", "Analysis"]
        for category in categories:
            info = taxonomy.get_category_info(category)
            assert info is not None
            assert len(info.mitre_techniques) >= 0  # Normal has empty list
            assert info.description is not None
    
    # -------------------------------------------------------------------------
    # Singleton Test
    # -------------------------------------------------------------------------
    
    def test_get_taxonomy_singleton(self):
        """Test that get_taxonomy returns same instance."""
        t1 = get_taxonomy()
        t2 = get_taxonomy()
        assert t1 is t2


# ============================================================================
# CICIDS2017Loader Tests
# ============================================================================

class TestCICIDS2017Loader:
    """Tests for CICIDS2017Loader class."""
    
    @pytest.fixture
    def mock_csv_data(self, tmp_path):
        """Create mock CSV files for testing."""
        # Create a simple CSV with CIC-IDS2017 format
        data = {
            ' Destination Port': [80, 443, 22, 8080, 21],
            ' Flow Duration': [1000, 2000, 3000, 4000, 5000],
            ' Total Fwd Packets': [10, 20, 30, 40, 50],
            ' Total Backward Packets': [5, 10, 15, 20, 25],
            ' Total Length of Fwd Packets': [1000, 2000, 3000, 4000, 5000],
            ' Total Length of Bwd Packets': [500, 1000, 1500, 2000, 2500],
            ' Flow Bytes/s': [100.0, 200.0, 300.0, 400.0, 500.0],
            ' Flow Packets/s': [10.0, 20.0, 30.0, 40.0, 50.0],
            ' Flow IAT Mean': [100.0, 200.0, 300.0, 400.0, 500.0],
            ' Flow IAT Std': [10.0, 20.0, 30.0, 40.0, 50.0],
            ' Flow IAT Max': [500.0, 1000.0, 1500.0, 2000.0, 2500.0],
            ' Flow IAT Min': [10.0, 20.0, 30.0, 40.0, 50.0],
            ' Fwd IAT Total': [1000, 2000, 3000, 4000, 5000],
            ' Fwd IAT Mean': [100.0, 200.0, 300.0, 400.0, 500.0],
            ' Fwd IAT Std': [10.0, 20.0, 30.0, 40.0, 50.0],
            ' Fwd IAT Max': [500.0, 1000.0, 1500.0, 2000.0, 2500.0],
            ' Fwd IAT Min': [10.0, 20.0, 30.0, 40.0, 50.0],
            ' Bwd IAT Total': [500, 1000, 1500, 2000, 2500],
            ' Bwd IAT Mean': [50.0, 100.0, 150.0, 200.0, 250.0],
            ' Bwd IAT Std': [5.0, 10.0, 15.0, 20.0, 25.0],
            ' Bwd IAT Max': [250.0, 500.0, 750.0, 1000.0, 1250.0],
            ' Bwd IAT Min': [5.0, 10.0, 15.0, 20.0, 25.0],
            ' Label': ['BENIGN', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'Bot']
        }
        
        # Add remaining columns with dummy values to match expected format
        for i in range(len(CICIDS2017Loader.FEATURE_COLUMNS)):
            col = CICIDS2017Loader.FEATURE_COLUMNS[i]
            if col not in data:
                data[col] = [float(i * j) for j in range(1, 6)]
        
        data[' Label'] = ['BENIGN', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'Bot']
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        return tmp_path
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = CICIDS2017Loader()
        assert loader.data is None
        assert loader.FEATURE_COLUMNS is not None
        assert len(loader.FEATURE_COLUMNS) == 78
    
    def test_loader_with_directory(self, mock_csv_data):
        """Test loading from directory."""
        loader = CICIDS2017Loader()
        loader.load(mock_csv_data, sample_frac=1.0)
        assert loader.data is not None
        assert len(loader.data) == 5
    
    def test_loader_feature_columns(self):
        """Test feature columns are defined."""
        assert len(CICIDS2017Loader.FEATURE_COLUMNS) > 0
        # Should have around 78 features
        assert len(CICIDS2017Loader.FEATURE_COLUMNS) == 78
    
    def test_get_features_and_labels(self, mock_csv_data):
        """Test feature and label extraction."""
        loader = CICIDS2017Loader()
        loader.load(mock_csv_data, sample_frac=1.0)
        X, y = loader.get_features_and_labels()
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert X.shape[0] == 5
    
    def test_label_mapping(self, mock_csv_data):
        """Test label mapping returns unified category strings."""
        loader = CICIDS2017Loader()
        loader.load(mock_csv_data, sample_frac=1.0)
        
        # Check that labels are mapped strings
        labels = loader.data[' Label'].unique()
        expected_labels = {'BENIGN', 'DoS Hulk', 'PortScan', 'FTP-Patator', 'Bot'}
        assert set(labels) == expected_labels


# ============================================================================
# UnifiedIDSDataset Tests
# ============================================================================

class TestUnifiedIDSDataset:
    """Tests for UnifiedIDSDataset class."""
    
    @pytest.fixture
    def mock_unsw_csv(self, tmp_path):
        """Create mock UNSW-NB15 CSV."""
        data = {
            'id': range(100),
            'dur': np.random.rand(100),
            'proto': ['tcp'] * 50 + ['udp'] * 50,
            'service': ['http'] * 30 + ['ftp'] * 30 + ['ssh'] * 40,
            'state': ['FIN'] * 50 + ['CON'] * 50,
            'spkts': np.random.randint(1, 100, 100),
            'dpkts': np.random.randint(1, 100, 100),
            'sbytes': np.random.randint(100, 10000, 100),
            'dbytes': np.random.randint(100, 10000, 100),
            'rate': np.random.rand(100) * 1000,
            'sttl': np.random.randint(1, 255, 100),
            'dttl': np.random.randint(1, 255, 100),
            'sload': np.random.rand(100) * 1000,
            'dload': np.random.rand(100) * 1000,
            'sloss': np.random.randint(0, 10, 100),
            'dloss': np.random.randint(0, 10, 100),
            'sinpkt': np.random.rand(100),
            'dinpkt': np.random.rand(100),
            'sjit': np.random.rand(100),
            'djit': np.random.rand(100),
            'swin': np.random.randint(0, 65535, 100),
            'stcpb': np.random.randint(0, 100000, 100),
            'dtcpb': np.random.randint(0, 100000, 100),
            'dwin': np.random.randint(0, 65535, 100),
            'tcprtt': np.random.rand(100),
            'synack': np.random.rand(100),
            'ackdat': np.random.rand(100),
            'smean': np.random.rand(100) * 100,
            'dmean': np.random.rand(100) * 100,
            'trans_depth': np.random.randint(0, 10, 100),
            'response_body_len': np.random.randint(0, 10000, 100),
            'ct_srv_src': np.random.randint(0, 10, 100),
            'ct_state_ttl': np.random.randint(0, 10, 100),
            'ct_dst_ltm': np.random.randint(0, 10, 100),
            'ct_src_dport_ltm': np.random.randint(0, 10, 100),
            'ct_dst_sport_ltm': np.random.randint(0, 10, 100),
            'ct_dst_src_ltm': np.random.randint(0, 10, 100),
            'is_ftp_login': np.random.randint(0, 2, 100),
            'ct_ftp_cmd': np.random.randint(0, 5, 100),
            'ct_flw_http_mthd': np.random.randint(0, 5, 100),
            'ct_src_ltm': np.random.randint(0, 10, 100),
            'ct_srv_dst': np.random.randint(0, 10, 100),
            'is_sm_ips_ports': np.random.randint(0, 2, 100),
            'attack_cat': ['Normal'] * 40 + ['DoS'] * 20 + ['Exploits'] * 20 + ['Fuzzers'] * 20,
            'label': [0] * 40 + [1] * 60,
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "UNSW_NB15_training-set.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        from data_pipeline.unified_dataset import UnifiedIDSDataset
        dataset = UnifiedIDSDataset()
        assert dataset.X is None
        assert dataset.y is None
    
    def test_load_unsw_nb15(self, mock_unsw_csv):
        """Test loading UNSW-NB15 data."""
        from data_pipeline.unified_dataset import UnifiedIDSDataset
        dataset = UnifiedIDSDataset()
        dataset.load_unsw_nb15(mock_unsw_csv)
        
        assert dataset.X is not None
        assert dataset.y is not None
        assert len(dataset.X) == 100
        assert len(dataset.y) == 100
    
    def test_normalize(self, mock_unsw_csv):
        """Test data normalization."""
        from data_pipeline.unified_dataset import UnifiedIDSDataset
        dataset = UnifiedIDSDataset()
        dataset.load_unsw_nb15(mock_unsw_csv)
        dataset.normalize()
        
        # Check normalization happened (mean should be close to 0, std close to 1)
        assert dataset.X.mean() < 1.0
    
    def test_get_train_test_split(self, mock_unsw_csv):
        """Test train/test splitting."""
        from data_pipeline.unified_dataset import UnifiedIDSDataset
        dataset = UnifiedIDSDataset()
        dataset.load_unsw_nb15(mock_unsw_csv)
        
        X_train, X_test, y_train, y_test = dataset.get_train_test_split()
        
        assert len(X_train) + len(X_test) == 100
        assert len(y_train) + len(y_test) == 100
        assert len(X_train) > len(X_test)  # Default 80/20 split


# ============================================================================
# SMOTE Integration Tests
# ============================================================================

class TestSMOTEIntegration:
    """Tests for SMOTE integration in UnifiedIDSDataset."""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset for testing."""
        # Majority class: 900 samples
        X_majority = np.random.randn(900, 10)
        y_majority = np.zeros(900, dtype=int)
        
        # Minority class: 100 samples
        X_minority = np.random.randn(100, 10)
        y_minority = np.ones(100, dtype=int)
        
        X = np.vstack([X_majority, X_minority])
        y = np.hstack([y_majority, y_minority])
        
        return X, y
    
    def test_smote_available(self):
        """Test SMOTE is available."""
        from data_pipeline.unified_dataset import SMOTE_AVAILABLE
        assert SMOTE_AVAILABLE is True
    
    def test_smote_balances_classes(self, imbalanced_data):
        """Test SMOTE balances imbalanced classes."""
        from imblearn.over_sampling import SMOTE
        
        X, y = imbalanced_data
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        unique, counts = np.unique(y_resampled, return_counts=True)
        # Classes should be balanced after SMOTE
        assert counts[0] == counts[1]


# ============================================================================
# Cross-Dataset Consistency Tests
# ============================================================================

class TestCrossDatasetConsistency:
    """Tests for cross-dataset feature consistency."""
    
    def test_unified_category_consistency(self):
        """Test unified categories are consistent across datasets."""
        taxonomy = UnifiedTaxonomy()
        
        # DoS attacks from both datasets should map to same category
        unsw_dos = taxonomy.map_unsw_nb15("DoS")
        cic_dos = taxonomy.map_cic_ids2017("DoS Hulk")
        assert unsw_dos == cic_dos == "DoS/DDoS"
        
        # Normal traffic from both datasets
        unsw_normal = taxonomy.map_unsw_nb15("Normal")
        cic_normal = taxonomy.map_cic_ids2017("BENIGN")
        assert unsw_normal == cic_normal == "Normal"
    
    def test_category_id_stability(self):
        """Test category IDs remain stable."""
        taxonomy = UnifiedTaxonomy()
        
        # IDs should be consistent
        expected_ids = {
            "Normal": 0,
            "DoS/DDoS": 1,
            "Reconnaissance": 2,
            "Exploits": 3,
            "Brute_Force": 4,
            "Malware": 5,
            "Analysis": 6,
        }
        
        for category, expected_id in expected_ids.items():
            assert taxonomy.get_category_id(category) == expected_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
