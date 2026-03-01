"""
Unified Dataset for Multi-Source IDS Training.

This module combines UNSW-NB15 and CIC-IDS2017 datasets into a unified
training set with consistent attack taxonomy and optional SMOTE balancing.

Features:
- Unified attack categories across datasets
- Feature alignment between datasets
- SMOTE oversampling for minority classes
- Cross-dataset train/test splitting

Usage:
    >>> from data_pipeline.unified_dataset import UnifiedIDSDataset
    >>> dataset = UnifiedIDSDataset()
    >>> dataset.load_unsw_nb15("data/UNSW_NB15_training-set.csv")
    >>> dataset.load_cic_ids2017("data/CIC-IDS2017")
    >>> X_train, X_test, y_train, y_test = dataset.get_train_test_split()
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from .unified_taxonomy import UnifiedTaxonomy, get_taxonomy
from .cic_ids2017_loader import CICIDS2017Loader

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDatasetConfig:
    """Configuration for unified dataset."""
    
    # Normalization method
    normalization: str = "standard"  # "standard", "minmax", or "none"
    
    # Test/validation split ratios
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    
    # SMOTE configuration
    apply_smote: bool = True
    smote_strategy: str = "auto"  # "auto", "minority", "not majority", or dict
    smote_k_neighbors: int = 5
    
    # Minimum samples per class (for SMOTE)
    min_samples_per_class: int = 100
    
    # Maximum samples per class (to prevent memory issues)
    max_samples_per_class: Optional[int] = None
    
    # Random seed
    random_state: int = 42
    
    # Feature selection
    use_common_features: bool = True  # Use only features present in both datasets


# Common features between UNSW-NB15 and CIC-IDS2017
# These are semantically similar features across both datasets
COMMON_FEATURE_MAPPING = {
    # UNSW-NB15 feature -> CIC-IDS2017 feature
    "dur": " Flow Duration",
    "spkts": " Total Fwd Packets",
    "dpkts": " Total Backward Packets",
    "sbytes": "Total Length of Fwd Packets",
    "dbytes": " Total Length of Bwd Packets",
    "rate": "Flow Bytes/s",
    "sload": " Fwd Packet Length Mean",
    "dload": " Bwd Packet Length Mean",
    "sinpkt": " Fwd IAT Mean",
    "dinpkt": " Bwd IAT Mean",
    "sjit": " Fwd IAT Std",
    "djit": " Bwd IAT Std",
    "smean": " Avg Fwd Segment Size",
    "dmean": " Avg Bwd Segment Size",
}


class UnifiedIDSDataset:
    """
    Unified dataset combining UNSW-NB15 and CIC-IDS2017.
    
    This class provides:
    - Unified loading from multiple IDS datasets
    - Consistent attack taxonomy
    - Feature normalization
    - SMOTE oversampling for class balance
    - Train/test/validation splitting
    
    Attributes:
        config: UnifiedDatasetConfig with processing parameters.
        taxonomy: UnifiedTaxonomy for label mapping.
        X: Combined feature matrix (after preprocessing).
        y: Combined labels (unified category IDs).
        source: Array indicating source dataset for each sample.
    
    Example:
        >>> dataset = UnifiedIDSDataset()
        >>> dataset.load_unsw_nb15("data/UNSW_NB15_training-set.csv")
        >>> dataset.load_cic_ids2017("data/CIC-IDS2017")
        >>> dataset.apply_smote()
        >>> X_train, X_test, y_train, y_test = dataset.get_train_test_split()
    """
    
    def __init__(
        self,
        config: Optional[UnifiedDatasetConfig] = None,
        taxonomy: Optional[UnifiedTaxonomy] = None,
    ):
        """
        Initialize unified dataset.
        
        Args:
            config: Dataset configuration.
            taxonomy: Unified taxonomy for labels.
        """
        self.config = config or UnifiedDatasetConfig()
        self.taxonomy = taxonomy or get_taxonomy()
        
        # Data storage
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.source: Optional[np.ndarray] = None  # 0=UNSW, 1=CIC
        
        # Scaler for normalization
        self._scaler = None
        
        # Dataset statistics
        self._stats = {
            "unsw_nb15_samples": 0,
            "cic_ids2017_samples": 0,
            "smote_applied": False,
            "original_distribution": {},
            "balanced_distribution": {},
        }
        
        # Feature info
        self._feature_names: List[str] = []
        self._num_features: int = 0
    
    def load_unsw_nb15(
        self,
        data_path: Union[str, Path],
        sample_frac: Optional[float] = None,
    ) -> "UnifiedIDSDataset":
        """
        Load UNSW-NB15 dataset.
        
        Args:
            data_path: Path to UNSW-NB15 CSV file.
            sample_frac: Fraction of data to sample (None for all).
        
        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading UNSW-NB15 from {data_path}...")
        
        df = pd.read_csv(data_path)
        
        if sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=self.config.random_state)
        
        # Get features (exclude id, attack_cat, label)
        feature_cols = [c for c in df.columns 
                       if c not in ['id', 'attack_cat', 'label']]
        
        # Handle categorical columns
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
        
        X_unsw = df[feature_cols].values.astype(np.float32)
        
        # Handle inf/nan
        X_unsw = np.nan_to_num(X_unsw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get unified labels
        original_labels = df['attack_cat'].fillna('Normal').values
        y_unsw = np.array([
            self.taxonomy.get_category_id(self.taxonomy.map_unsw_nb15(l))
            for l in original_labels
        ])
        
        # Source indicator
        source_unsw = np.zeros(len(X_unsw), dtype=np.int32)
        
        # Combine with existing data
        if self.X is None:
            self.X = X_unsw
            self.y = y_unsw
            self.source = source_unsw
            self._feature_names = feature_cols
            self._num_features = X_unsw.shape[1]
        else:
            # Need to align features - use UNSW dimensions
            if X_unsw.shape[1] != self._num_features:
                logger.warning(f"Feature mismatch: {X_unsw.shape[1]} vs {self._num_features}")
                # Pad or truncate
                if X_unsw.shape[1] < self._num_features:
                    padding = np.zeros((X_unsw.shape[0], self._num_features - X_unsw.shape[1]))
                    X_unsw = np.hstack([X_unsw, padding])
                else:
                    X_unsw = X_unsw[:, :self._num_features]
            
            self.X = np.vstack([self.X, X_unsw])
            self.y = np.concatenate([self.y, y_unsw])
            self.source = np.concatenate([self.source, source_unsw])
        
        self._stats["unsw_nb15_samples"] = len(y_unsw)
        logger.info(f"Loaded {len(y_unsw):,} UNSW-NB15 samples")
        
        return self
    
    def load_cic_ids2017(
        self,
        data_path: Union[str, Path],
        sample_frac: Optional[float] = None,
    ) -> "UnifiedIDSDataset":
        """
        Load CIC-IDS2017 dataset.
        
        Args:
            data_path: Path to CIC-IDS2017 directory.
            sample_frac: Fraction of data to sample (None for all).
        
        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading CIC-IDS2017 from {data_path}...")
        
        loader = CICIDS2017Loader(taxonomy=self.taxonomy)
        loader.load(data_path, sample_frac=sample_frac)
        
        X_cic, y_cic = loader.get_features_and_labels(label_type="unified_id")
        
        # Handle inf/nan
        X_cic = np.nan_to_num(X_cic, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Source indicator
        source_cic = np.ones(len(X_cic), dtype=np.int32)
        
        # Store CIC feature count
        cic_features = X_cic.shape[1]
        
        # Combine with existing data
        if self.X is None:
            self.X = X_cic
            self.y = y_cic
            self.source = source_cic
            self._num_features = cic_features
        else:
            # Align features
            if cic_features != self._num_features:
                logger.info(f"Aligning features: CIC has {cic_features}, target is {self._num_features}")
                if cic_features < self._num_features:
                    padding = np.zeros((X_cic.shape[0], self._num_features - cic_features))
                    X_cic = np.hstack([X_cic, padding])
                else:
                    X_cic = X_cic[:, :self._num_features]
            
            self.X = np.vstack([self.X, X_cic])
            self.y = np.concatenate([self.y, y_cic])
            self.source = np.concatenate([self.source, source_cic])
        
        self._stats["cic_ids2017_samples"] = len(y_cic)
        logger.info(f"Loaded {len(y_cic):,} CIC-IDS2017 samples")
        
        return self
    
    def normalize(self, method: Optional[str] = None) -> "UnifiedIDSDataset":
        """
        Normalize features.
        
        Args:
            method: Normalization method ("standard", "minmax", or "none").
                   Uses config.normalization if None.
        
        Returns:
            Self for method chaining.
        """
        if self.X is None:
            raise ValueError("No data loaded. Call load methods first.")
        
        method = method or self.config.normalization
        
        if method == "none":
            logger.info("Skipping normalization")
            return self
        
        logger.info(f"Applying {method} normalization...")
        
        if method == "standard":
            self._scaler = StandardScaler()
        elif method == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.X = self._scaler.fit_transform(self.X)
        
        return self
    
    def apply_smote(
        self,
        strategy: Optional[str] = None,
        k_neighbors: Optional[int] = None,
    ) -> "UnifiedIDSDataset":
        """
        Apply SMOTE oversampling to balance classes.
        
        Args:
            strategy: SMOTE sampling strategy.
            k_neighbors: Number of neighbors for SMOTE.
        
        Returns:
            Self for method chaining.
        """
        if not SMOTE_AVAILABLE:
            logger.warning("SMOTE not available. Install imbalanced-learn.")
            return self
        
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Call load methods first.")
        
        # Store original distribution
        unique, counts = np.unique(self.y, return_counts=True)
        self._stats["original_distribution"] = dict(zip(
            [self.taxonomy.get_category_name(i) for i in unique],
            counts.tolist()
        ))
        
        strategy = strategy or self.config.smote_strategy
        k_neighbors = k_neighbors or self.config.smote_k_neighbors
        
        logger.info("Applying SMOTE oversampling...")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  k_neighbors: {k_neighbors}")
        
        # Check minimum samples per class
        min_count = min(counts)
        if min_count < k_neighbors + 1:
            logger.warning(f"Minimum class has {min_count} samples, adjusting k_neighbors")
            k_neighbors = max(1, min_count - 1)
        
        try:
            smote = SMOTE(
                sampling_strategy=strategy,
                k_neighbors=k_neighbors,
                random_state=self.config.random_state,
            )
            
            self.X, self.y = smote.fit_resample(self.X, self.y)
            
            # Update source (new samples are synthetic, mark as -1)
            original_len = len(self.source)
            new_len = len(self.y)
            if new_len > original_len:
                synthetic_source = np.full(new_len - original_len, -1, dtype=np.int32)
                self.source = np.concatenate([self.source, synthetic_source])
            
            self._stats["smote_applied"] = True
            
            # Store balanced distribution
            unique, counts = np.unique(self.y, return_counts=True)
            self._stats["balanced_distribution"] = dict(zip(
                [self.taxonomy.get_category_name(i) for i in unique],
                counts.tolist()
            ))
            
            logger.info(f"  Samples after SMOTE: {len(self.y):,}")
            
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            logger.warning("Continuing without SMOTE")
        
        return self
    
    def get_train_test_split(
        self,
        test_ratio: Optional[float] = None,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test split.
        
        Args:
            test_ratio: Test set ratio (uses config if None).
            stratify: Whether to stratify by class.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded.")
        
        test_ratio = test_ratio or self.config.test_ratio
        
        stratify_labels = self.y if stratify else None
        
        return train_test_split(
            self.X, self.y,
            test_size=test_ratio,
            stratify=stratify_labels,
            random_state=self.config.random_state,
        )
    
    def get_train_val_test_split(
        self,
        test_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/validation/test split.
        
        Args:
            test_ratio: Test set ratio (uses config if None).
            val_ratio: Validation set ratio (uses config if None).
            stratify: Whether to stratify by class.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded.")
        
        test_ratio = test_ratio or self.config.test_ratio
        val_ratio = val_ratio or self.config.val_ratio
        
        stratify_labels = self.y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y,
            test_size=test_ratio,
            stratify=stratify_labels,
            random_state=self.config.random_state,
        )
        
        # Second split: train vs val
        val_adjusted = val_ratio / (1 - test_ratio)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adjusted,
            stratify=stratify_temp,
            random_state=self.config.random_state,
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_cross_dataset_split(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cross-dataset split: train on one, test on other.
        
        Returns:
            Tuple of (X_train_unsw, X_test_cic, y_train_unsw, y_test_cic).
        """
        if self.X is None or self.y is None or self.source is None:
            raise ValueError("No data loaded.")
        
        unsw_mask = self.source == 0
        cic_mask = self.source == 1
        
        X_unsw = self.X[unsw_mask]
        y_unsw = self.y[unsw_mask]
        X_cic = self.X[cic_mask]
        y_cic = self.y[cic_mask]
        
        return X_unsw, X_cic, y_unsw, y_cic
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.X is None:
            return {"error": "No data loaded"}
        
        unique, counts = np.unique(self.y, return_counts=True)
        distribution = dict(zip(
            [self.taxonomy.get_category_name(i) for i in unique],
            counts.tolist()
        ))
        
        return {
            "total_samples": len(self.y),
            "num_features": self.X.shape[1],
            "unsw_nb15_samples": self._stats["unsw_nb15_samples"],
            "cic_ids2017_samples": self._stats["cic_ids2017_samples"],
            "smote_applied": self._stats["smote_applied"],
            "class_distribution": distribution,
            "original_distribution": self._stats.get("original_distribution", {}),
            "num_classes": len(unique),
        }
    
    def print_summary(self) -> None:
        """Print dataset summary."""
        stats = self.get_statistics()
        
        print("=" * 65)
        print("UNIFIED IDS DATASET SUMMARY")
        print("=" * 65)
        
        print(f"\nTotal Samples: {stats['total_samples']:,}")
        print(f"Number of Features: {stats['num_features']}")
        print(f"Number of Classes: {stats['num_classes']}")
        
        print(f"\nData Sources:")
        print(f"  UNSW-NB15:    {stats['unsw_nb15_samples']:>10,} samples")
        print(f"  CIC-IDS2017:  {stats['cic_ids2017_samples']:>10,} samples")
        
        if stats['smote_applied']:
            print(f"\nSMOTE Applied: Yes")
            if stats['original_distribution']:
                print("\nOriginal Distribution:")
                for label, count in sorted(stats['original_distribution'].items(),
                                          key=lambda x: -x[1]):
                    print(f"  {label:20} {count:>10,}")
        
        print("\nCurrent Class Distribution:")
        print("-" * 45)
        for label, count in sorted(stats['class_distribution'].items(),
                                  key=lambda x: -x[1]):
            pct = count / stats['total_samples'] * 100
            print(f"  {label:20} {count:>10,} ({pct:5.2f}%)")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Demo usage
    dataset = UnifiedIDSDataset()
    
    # Load both datasets (with sampling for speed)
    if Path("data/UNSW_NB15_training-set.csv").exists():
        dataset.load_unsw_nb15("data/UNSW_NB15_training-set.csv", sample_frac=0.1)
    
    if Path("data/CIC-IDS2017").exists():
        dataset.load_cic_ids2017("data/CIC-IDS2017", sample_frac=0.05)
    
    # Normalize
    dataset.normalize()
    
    # Apply SMOTE
    if SMOTE_AVAILABLE:
        dataset.apply_smote()
    
    # Print summary
    dataset.print_summary()
    
    # Get splits
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
