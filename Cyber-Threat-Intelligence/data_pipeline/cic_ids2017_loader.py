"""
CIC-IDS2017 Dataset Loader.

This module provides data loading functionality for the CIC-IDS2017 
intrusion detection dataset, with automatic label mapping to the
unified attack taxonomy.

The CIC-IDS2017 dataset contains:
- 8 CSV files covering different days/attacks
- 79 network flow features
- ~2.8 million samples total

Usage:
    >>> from data_pipeline.cic_ids2017_loader import CICIDS2017Loader
    >>> loader = CICIDS2017Loader()
    >>> loader.load("data/CIC-IDS2017")
    >>> X, y = loader.get_features_and_labels()
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .unified_taxonomy import UnifiedTaxonomy, get_taxonomy

logger = logging.getLogger(__name__)


@dataclass
class CICIDS2017Config:
    """Configuration for CIC-IDS2017 data loading."""
    
    # Column name for labels
    label_column: str = " Label"
    
    # Columns to drop (non-feature columns)
    drop_columns: List[str] = field(default_factory=lambda: [
        "Flow ID",
        " Source IP",
        " Source Port", 
        " Destination IP",
        " Timestamp",
    ])
    
    # Numeric columns that may have inf values
    inf_columns: List[str] = field(default_factory=lambda: [
        "Flow Bytes/s",
        " Flow Packets/s",
    ])
    
    # Encoding to use when reading files
    encoding: str = "latin-1"
    
    # Whether to balance classes during loading
    balance_classes: bool = False
    
    # Maximum samples per class when balancing
    max_samples_per_class: Optional[int] = None
    
    # Random seed for reproducibility
    random_state: int = 42


class CICIDS2017Loader:
    """
    Data loader for the CIC-IDS2017 intrusion detection dataset.
    
    Handles:
    - Loading multiple CSV files
    - Encoding issues (UTF-8/Latin-1)
    - Infinite value handling
    - Label mapping to unified taxonomy
    - Feature selection and cleaning
    
    Attributes:
        config: CICIDS2017Config with loading parameters.
        taxonomy: UnifiedTaxonomy for label mapping.
        data: Loaded DataFrame (after calling load()).
    
    Example:
        >>> loader = CICIDS2017Loader()
        >>> loader.load("data/CIC-IDS2017")
        >>> print(loader.get_statistics())
        >>> X, y = loader.get_features_and_labels(label_type="unified")
    """
    
    # Feature columns from CIC-IDS2017 (79 features)
    FEATURE_COLUMNS = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets',
        ' Total Backward Packets', 'Total Length of Fwd Packets',
        ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
        ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
        ' Fwd Packet Length Std', 'Bwd Packet Length Max',
        ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
        ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
        ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
        'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
        ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
        ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
        ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
        ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
        ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
        ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
        ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
        ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
        ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
        ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
        ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
        ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
        ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
        ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward',
        'Active Mean', ' Active Std', ' Active Max', ' Active Min',
        'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',
    ]
    
    def __init__(
        self,
        config: Optional[CICIDS2017Config] = None,
        taxonomy: Optional[UnifiedTaxonomy] = None,
    ):
        """
        Initialize CIC-IDS2017 loader.
        
        Args:
            config: Loading configuration. Uses defaults if None.
            taxonomy: Unified taxonomy for label mapping. Uses default if None.
        """
        self.config = config or CICIDS2017Config()
        self.taxonomy = taxonomy or get_taxonomy()
        self.data: Optional[pd.DataFrame] = None
        self._file_stats: Dict[str, int] = {}
    
    def load(
        self,
        data_path: Union[str, Path],
        files: Optional[List[str]] = None,
        sample_frac: Optional[float] = None,
    ) -> "CICIDS2017Loader":
        """
        Load CIC-IDS2017 dataset from directory.
        
        Args:
            data_path: Path to directory containing CIC-IDS2017 CSV files.
            files: Specific files to load. If None, loads all CSV files.
            sample_frac: Fraction of data to sample (for faster loading).
        
        Returns:
            Self for method chaining.
        
        Raises:
            FileNotFoundError: If data_path doesn't exist.
            ValueError: If no CSV files found.
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Find all CSV files
        if files is None:
            csv_files = list(data_path.glob("*.csv"))
        else:
            csv_files = [data_path / f for f in files]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_path}")
        
        logger.info(f"Loading {len(csv_files)} CIC-IDS2017 files...")
        
        # Load and concatenate all files
        dfs = []
        for csv_file in csv_files:
            df = self._load_single_file(csv_file, sample_frac)
            if df is not None and len(df) > 0:
                self._file_stats[csv_file.name] = len(df)
                dfs.append(df)
                logger.info(f"  Loaded {csv_file.name}: {len(df):,} samples")
        
        if not dfs:
            raise ValueError("No data loaded from any file")
        
        # Concatenate all dataframes
        self.data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total samples loaded: {len(self.data):,}")
        
        # Clean the data
        self._clean_data()
        
        return self
    
    def _load_single_file(
        self,
        file_path: Path,
        sample_frac: Optional[float] = None,
    ) -> Optional[pd.DataFrame]:
        """Load a single CSV file with encoding handling."""
        try:
            # Try with specified encoding and C engine first
            df = pd.read_csv(file_path, encoding=self.config.encoding, low_memory=False)
            
            if sample_frac is not None:
                df = df.sample(frac=sample_frac, random_state=self.config.random_state)
            
            return df
            
        except Exception as e:
            # Fallback to Python engine for problematic files
            logger.warning(f"C engine failed for {file_path}, trying Python engine...")
            try:
                df = pd.read_csv(file_path, encoding=self.config.encoding, 
                               engine='python', on_bad_lines='skip')
                
                if sample_frac is not None:
                    df = df.sample(frac=sample_frac, random_state=self.config.random_state)
                
                return df
            except Exception as e2:
                logger.error(f"Error loading {file_path}: {e2}")
                return None
    
    def _clean_data(self) -> None:
        """Clean the loaded data."""
        if self.data is None:
            return
        
        initial_count = len(self.data)
        
        # 1. Handle infinite values
        for col in self.config.inf_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].replace([np.inf, -np.inf], np.nan)
        
        # 2. Replace inf with nan in all numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 3. Fill NaN values
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
        
        # 4. Remove rows with any remaining NaN
        self.data = self.data.dropna()
        
        # 5. Strip whitespace from label column
        if self.config.label_column in self.data.columns:
            self.data[self.config.label_column] = self.data[self.config.label_column].str.strip()
        
        cleaned_count = len(self.data)
        removed = initial_count - cleaned_count
        
        if removed > 0:
            logger.info(f"Cleaned data: removed {removed:,} rows ({removed/initial_count*100:.2f}%)")
    
    def get_features_and_labels(
        self,
        label_type: str = "unified",
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature matrix and labels.
        
        Args:
            label_type: Type of labels to return:
                - "original": Original CIC-IDS2017 labels
                - "unified": Unified taxonomy labels (strings)
                - "unified_id": Unified taxonomy IDs (integers)
                - "binary": Binary (0=Normal, 1=Attack)
            feature_columns: Specific feature columns. Uses all if None.
        
        Returns:
            Tuple of (X, y) where X is feature matrix, y is labels.
        
        Raises:
            ValueError: If data not loaded or invalid label_type.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = [c for c in self.FEATURE_COLUMNS if c in self.data.columns]
        
        X = self.data[feature_columns].values.astype(np.float32)
        
        # Get labels based on type
        original_labels = self.data[self.config.label_column].values
        
        if label_type == "original":
            y = original_labels
        elif label_type == "unified":
            y = np.array([self.taxonomy.map_cic_ids2017(l) for l in original_labels])
        elif label_type == "unified_id":
            unified = [self.taxonomy.map_cic_ids2017(l) for l in original_labels]
            y = np.array([self.taxonomy.get_category_id(l) for l in unified])
        elif label_type == "binary":
            unified = [self.taxonomy.map_cic_ids2017(l) for l in original_labels]
            y = np.array([self.taxonomy.get_binary_label(l) for l in unified])
        else:
            raise ValueError(f"Invalid label_type: {label_type}")
        
        return X, y
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics.
        """
        if self.data is None:
            return {"error": "Data not loaded"}
        
        # Get label distribution (unified)
        original_labels = self.data[self.config.label_column].values
        unified_labels = [self.taxonomy.map_cic_ids2017(l) for l in original_labels]
        
        label_counts = pd.Series(unified_labels).value_counts().to_dict()
        original_counts = pd.Series(original_labels).value_counts().to_dict()
        
        return {
            "total_samples": len(self.data),
            "num_features": len(self.FEATURE_COLUMNS),
            "file_counts": self._file_stats,
            "unified_label_distribution": label_counts,
            "original_label_distribution": original_counts,
            "attack_samples": sum(1 for l in unified_labels if l != "Normal"),
            "normal_samples": sum(1 for l in unified_labels if l == "Normal"),
        }
    
    def get_balanced_sample(
        self,
        samples_per_class: int,
        label_type: str = "unified_id",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a balanced sample of the data.
        
        Args:
            samples_per_class: Number of samples per class.
            label_type: Type of labels (see get_features_and_labels).
        
        Returns:
            Balanced (X, y) tuple.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Get unified labels
        original_labels = self.data[self.config.label_column].values
        unified_labels = np.array([self.taxonomy.map_cic_ids2017(l) for l in original_labels])
        
        # Sample from each class
        balanced_indices = []
        for category in self.taxonomy.category_names:
            class_indices = np.where(unified_labels == category)[0]
            if len(class_indices) > 0:
                n_samples = min(samples_per_class, len(class_indices))
                sampled = np.random.choice(class_indices, n_samples, replace=False)
                balanced_indices.extend(sampled)
        
        # Shuffle
        np.random.shuffle(balanced_indices)
        
        # Get features and labels for sampled indices
        feature_columns = [c for c in self.FEATURE_COLUMNS if c in self.data.columns]
        X = self.data.iloc[balanced_indices][feature_columns].values.astype(np.float32)
        
        # Get appropriate labels
        sampled_original = original_labels[balanced_indices]
        
        if label_type == "unified_id":
            unified = [self.taxonomy.map_cic_ids2017(l) for l in sampled_original]
            y = np.array([self.taxonomy.get_category_id(l) for l in unified])
        elif label_type == "binary":
            unified = [self.taxonomy.map_cic_ids2017(l) for l in sampled_original]
            y = np.array([self.taxonomy.get_binary_label(l) for l in unified])
        else:
            y = np.array([self.taxonomy.map_cic_ids2017(l) for l in sampled_original])
        
        return X, y
    
    def print_summary(self) -> None:
        """Print a summary of the loaded dataset."""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("CIC-IDS2017 DATASET SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Samples: {stats['total_samples']:,}")
        print(f"Number of Features: {stats['num_features']}")
        print(f"Attack Samples: {stats['attack_samples']:,}")
        print(f"Normal Samples: {stats['normal_samples']:,}")
        
        print("\nUnified Label Distribution:")
        print("-" * 40)
        for label, count in sorted(stats['unified_label_distribution'].items(), 
                                   key=lambda x: -x[1]):
            pct = count / stats['total_samples'] * 100
            print(f"  {label:20} {count:>10,} ({pct:5.2f}%)")
        
        print("\nOriginal Label Distribution:")
        print("-" * 40)
        for label, count in sorted(stats['original_label_distribution'].items(),
                                   key=lambda x: -x[1])[:10]:
            pct = count / stats['total_samples'] * 100
            print(f"  {label:25} {count:>10,} ({pct:5.2f}%)")


if __name__ == "__main__":
    # Demo
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    data_path = "data/CIC-IDS2017"
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        sys.exit(1)
    
    # Load with sampling for faster demo
    loader = CICIDS2017Loader()
    loader.load(data_path, sample_frac=0.1)
    loader.print_summary()
    
    # Get features and labels
    X, y = loader.get_features_and_labels(label_type="unified_id")
    print(f"\nFeature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")
