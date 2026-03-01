"""
Data Loader module for UNSW-NB15 dataset.

This module provides the DataLoader class for loading, validating, and
performing initial cleaning of the UNSW-NB15 network intrusion dataset.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from data_pipeline.config import DatasetConfig

# Configure module logger
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial cleaning of UNSW-NB15 tabular data.
    
    This class provides functionality to load CSV files, validate schema,
    perform initial data cleaning, and detect data quality issues.
    
    Attributes:
        config: DatasetConfig instance with dataset parameters.
        data: Loaded DataFrame (None until load() is called).
        is_loaded: Boolean indicating if data has been loaded.
    
    Example:
        >>> config = DatasetConfig()
        >>> loader = DataLoader(config)
        >>> loader.load("path/to/unsw_nb15.csv")
        >>> print(loader.data.shape)
        (100000, 49)
        >>> stats = loader.get_data_statistics()
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        """
        Initializes the DataLoader with configuration.
        
        Args:
            config: DatasetConfig instance. If None, uses default configuration.
        """
        self.config = config if config is not None else DatasetConfig()
        self._data: Optional[pd.DataFrame] = None
        self._original_shape: Optional[Tuple[int, int]] = None
        self._cleaning_report: Dict[str, Any] = {}
        
        logger.info("DataLoader initialized with config: %s", self.config.normalization_method)
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Returns the loaded DataFrame."""
        return self._data
    
    @property
    def is_loaded(self) -> bool:
        """Returns True if data has been successfully loaded."""
        return self._data is not None
    
    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        """Returns the shape of the loaded data, or None if not loaded."""
        return self._data.shape if self._data is not None else None
    
    def load(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        low_memory: bool = False,
    ) -> "DataLoader":
        """
        Loads data from a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            encoding: File encoding (default: utf-8).
            low_memory: If True, uses chunked reading for large files.
        
        Returns:
            Self for method chaining.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or cannot be parsed.
        
        Example:
            >>> loader = DataLoader()
            >>> loader.load("data/UNSW_NB15_training.csv")
            <DataLoader object at ...>
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info("Loading data from: %s", file_path)
        
        try:
            self._data = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=low_memory,
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file: {e}")
        
        self._original_shape = self._data.shape
        logger.info("Loaded data with shape: %s", self._data.shape)
        
        return self
    
    def load_multiple(
        self,
        file_paths: List[Union[str, Path]],
        encoding: str = "utf-8",
    ) -> "DataLoader":
        """
        Loads and concatenates data from multiple CSV files.
        
        Args:
            file_paths: List of paths to CSV files.
            encoding: File encoding (default: utf-8).
        
        Returns:
            Self for method chaining.
        
        Raises:
            FileNotFoundError: If any file does not exist.
            ValueError: If files have incompatible schemas.
        
        Example:
            >>> loader = DataLoader()
            >>> loader.load_multiple([
            ...     "data/UNSW_NB15_1.csv",
            ...     "data/UNSW_NB15_2.csv",
            ... ])
        """
        dataframes: List[pd.DataFrame] = []
        
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            
            logger.info("Loading: %s", path)
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            dataframes.append(df)
        
        # Validate schema consistency
        columns_list = [set(df.columns) for df in dataframes]
        if not all(cols == columns_list[0] for cols in columns_list):
            raise ValueError("Files have incompatible schemas (different columns)")
        
        self._data = pd.concat(dataframes, axis=0, ignore_index=True)
        self._original_shape = self._data.shape
        
        logger.info("Combined data shape: %s", self._data.shape)
        return self
    
    def validate_schema(self, strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validates that the loaded data matches expected UNSW-NB15 schema.
        
        Args:
            strict: If True, requires all expected columns to be present.
        
        Returns:
            Tuple of (is_valid, list_of_issues).
        
        Example:
            >>> loader.load("data.csv")
            >>> is_valid, issues = loader.validate_schema()
            >>> if not is_valid:
            ...     print("Schema issues:", issues)
        """
        if not self.is_loaded:
            return False, ["Data not loaded"]
        
        issues: List[str] = []
        actual_columns = set(self._data.columns)
        
        # Check for expected numerical features
        expected_numerical = set(self.config.numerical_features)
        missing_numerical = expected_numerical - actual_columns
        if missing_numerical:
            issues.append(f"Missing numerical features: {missing_numerical}")
        
        # Check for expected categorical features
        expected_categorical = set(self.config.categorical_features)
        missing_categorical = expected_categorical - actual_columns
        if missing_categorical:
            issues.append(f"Missing categorical features: {missing_categorical}")
        
        # Check for label column
        if self.config.label_column not in actual_columns:
            issues.append(f"Missing label column: {self.config.label_column}")
        
        # Check for attack category column
        if self.config.attack_category_column not in actual_columns:
            issues.append(f"Missing attack category column: {self.config.attack_category_column}")
        
        is_valid = len(issues) == 0 if strict else self.config.label_column in actual_columns
        
        if issues:
            logger.warning("Schema validation issues: %s", issues)
        else:
            logger.info("Schema validation passed")
        
        return is_valid, issues
    
    def clean(
        self,
        drop_duplicates: bool = True,
        drop_id_columns: bool = True,
        standardize_column_names: bool = True,
    ) -> "DataLoader":
        """
        Performs initial data cleaning operations.
        
        Args:
            drop_duplicates: If True, removes duplicate rows.
            drop_id_columns: If True, drops identifier columns.
            standardize_column_names: If True, converts column names to lowercase.
        
        Returns:
            Self for method chaining.
        
        Raises:
            RuntimeError: If data has not been loaded.
        
        Example:
            >>> loader.load("data.csv").clean()
            >>> print(loader.cleaning_report)
        """
        self._ensure_loaded()
        
        self._cleaning_report = {
            "original_shape": self._original_shape,
            "duplicates_removed": 0,
            "columns_dropped": [],
            "columns_renamed": False,
        }
        
        # Standardize column names
        if standardize_column_names:
            self._data.columns = self._data.columns.str.lower().str.strip()
            self._cleaning_report["columns_renamed"] = True
            logger.info("Column names standardized to lowercase")
        
        # Drop duplicate rows
        if drop_duplicates:
            initial_rows = len(self._data)
            self._data = self._data.drop_duplicates()
            duplicates_removed = initial_rows - len(self._data)
            self._cleaning_report["duplicates_removed"] = duplicates_removed
            if duplicates_removed > 0:
                logger.info("Removed %d duplicate rows", duplicates_removed)
        
        # Drop identifier columns
        if drop_id_columns:
            columns_to_drop = [
                col for col in self.config.id_columns
                if col.lower() in self._data.columns
            ]
            if columns_to_drop:
                self._data = self._data.drop(columns=columns_to_drop, errors="ignore")
                self._cleaning_report["columns_dropped"] = columns_to_drop
                logger.info("Dropped identifier columns: %s", columns_to_drop)
        
        self._cleaning_report["final_shape"] = self._data.shape
        logger.info("Cleaning complete. Final shape: %s", self._data.shape)
        
        return self
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Returns the cleaning report from the last clean() operation.
        
        Returns:
            Dictionary containing cleaning statistics.
        """
        return self._cleaning_report.copy()
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Computes and returns comprehensive data statistics.
        
        Returns:
            Dictionary containing data statistics including:
            - shape: (rows, columns)
            - missing_values: Count per column
            - dtypes: Data types per column
            - numerical_stats: Summary statistics for numerical columns
            - categorical_stats: Value counts for categorical columns
            - label_distribution: Distribution of target labels
        
        Example:
            >>> stats = loader.get_data_statistics()
            >>> print(stats["label_distribution"])
            {0: 56000, 1: 119341}
        """
        self._ensure_loaded()
        
        stats: Dict[str, Any] = {
            "shape": self._data.shape,
            "columns": list(self._data.columns),
            "dtypes": self._data.dtypes.astype(str).to_dict(),
            "missing_values": self._data.isnull().sum().to_dict(),
            "total_missing": int(self._data.isnull().sum().sum()),
        }
        
        # Numerical statistics
        numerical_cols = [
            col for col in self.config.numerical_features
            if col in self._data.columns
        ]
        if numerical_cols:
            stats["numerical_stats"] = self._data[numerical_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = [
            col for col in self.config.categorical_features
            if col in self._data.columns
        ]
        if categorical_cols:
            stats["categorical_stats"] = {
                col: self._data[col].value_counts().to_dict()
                for col in categorical_cols
            }
        
        # Label distribution
        label_col = self.config.label_column
        if label_col in self._data.columns:
            stats["label_distribution"] = self._data[label_col].value_counts().to_dict()
        
        # Attack category distribution
        attack_col = self.config.attack_category_column
        if attack_col in self._data.columns:
            stats["attack_category_distribution"] = self._data[attack_col].value_counts().to_dict()
        
        return stats
    
    def detect_data_quality_issues(self) -> Dict[str, Any]:
        """
        Detects potential data quality issues in the loaded data.
        
        Returns:
            Dictionary containing detected issues:
            - missing_value_columns: Columns with missing values
            - high_cardinality_columns: Categorical columns with many unique values
            - constant_columns: Columns with single value
            - highly_correlated_pairs: Pairs of highly correlated features
            - outlier_columns: Columns with potential outliers
        
        Example:
            >>> issues = loader.detect_data_quality_issues()
            >>> if issues["constant_columns"]:
            ...     print("Consider removing:", issues["constant_columns"])
        """
        self._ensure_loaded()
        
        issues: Dict[str, Any] = {
            "missing_value_columns": {},
            "high_cardinality_columns": {},
            "constant_columns": [],
            "potential_outlier_columns": [],
            "infinite_value_columns": [],
        }
        
        # Check for missing values
        missing = self._data.isnull().sum()
        issues["missing_value_columns"] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Check for constant columns
        for col in self._data.columns:
            if self._data[col].nunique() <= 1:
                issues["constant_columns"].append(col)
        
        # Check for high cardinality categorical columns
        for col in self.config.categorical_features:
            if col in self._data.columns:
                nunique = self._data[col].nunique()
                if nunique > 100:  # Threshold for high cardinality
                    issues["high_cardinality_columns"][col] = nunique
        
        # Check for infinite values in numerical columns
        numerical_cols = [
            col for col in self.config.numerical_features
            if col in self._data.columns
        ]
        for col in numerical_cols:
            if self._data[col].dtype in [np.float64, np.float32]:
                if np.isinf(self._data[col]).any():
                    issues["infinite_value_columns"].append(col)
        
        # Check for potential outliers using IQR
        for col in numerical_cols[:10]:  # Check first 10 for performance
            if col in self._data.columns:
                q1 = self._data[col].quantile(0.25)
                q3 = self._data[col].quantile(0.75)
                iqr = q3 - q1
                outlier_count = (
                    (self._data[col] < (q1 - 1.5 * iqr)) | 
                    (self._data[col] > (q3 + 1.5 * iqr))
                ).sum()
                if outlier_count > len(self._data) * 0.05:  # More than 5% outliers
                    issues["potential_outlier_columns"].append(col)
        
        return issues
    
    def get_feature_names(
        self,
        include_numerical: bool = True,
        include_categorical: bool = True,
        include_binary: bool = True,
    ) -> List[str]:
        """
        Returns list of feature column names based on inclusion flags.
        
        Args:
            include_numerical: Include numerical features.
            include_categorical: Include categorical features.
            include_binary: Include binary features.
        
        Returns:
            List of feature column names present in the data.
        """
        self._ensure_loaded()
        
        features: List[str] = []
        actual_columns = set(self._data.columns)
        
        if include_numerical:
            features.extend([
                col for col in self.config.numerical_features
                if col in actual_columns
            ])
        
        if include_categorical:
            features.extend([
                col for col in self.config.categorical_features
                if col in actual_columns
            ])
        
        if include_binary:
            features.extend([
                col for col in self.config.binary_features
                if col in actual_columns
            ])
        
        return features
    
    def get_features_and_labels(
        self,
        label_type: str = "binary",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separates features and labels from the loaded data.
        
        Args:
            label_type: Type of label to return ('binary' or 'multiclass').
        
        Returns:
            Tuple of (features_df, labels_series).
        
        Raises:
            ValueError: If label_type is invalid.
            RuntimeError: If data is not loaded.
        
        Example:
            >>> X, y = loader.get_features_and_labels(label_type="binary")
            >>> print(X.shape, y.shape)
        """
        self._ensure_loaded()
        
        if label_type not in ("binary", "multiclass"):
            raise ValueError(f"label_type must be 'binary' or 'multiclass', got '{label_type}'")
        
        label_col = (
            self.config.label_column if label_type == "binary"
            else self.config.attack_category_column
        )
        
        if label_col not in self._data.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")
        
        feature_cols = self.get_feature_names()
        
        X = self._data[feature_cols].copy()
        y = self._data[label_col].copy()
        
        return X, y
    
    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        stratify_column: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Returns a sample of the loaded data.
        
        Args:
            n: Number of samples to return.
            frac: Fraction of samples to return (0.0 to 1.0).
            stratify_column: Column to use for stratified sampling.
            random_state: Random seed for reproducibility.
        
        Returns:
            Sampled DataFrame.
        
        Example:
            >>> sample = loader.sample(n=1000, stratify_column="label")
        """
        self._ensure_loaded()
        
        random_state = random_state or self.config.random_seed
        
        if stratify_column and stratify_column in self._data.columns:
            # Stratified sampling
            from sklearn.model_selection import train_test_split
            
            if frac:
                n = int(len(self._data) * frac)
            
            _, sample = train_test_split(
                self._data,
                test_size=n,
                stratify=self._data[stratify_column],
                random_state=random_state,
            )
            return sample
        
        return self._data.sample(n=n, frac=frac, random_state=random_state)
    
    def _ensure_loaded(self) -> None:
        """Ensures data is loaded, raises RuntimeError if not."""
        if not self.is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
    
    def __repr__(self) -> str:
        """Returns string representation of the DataLoader."""
        if self.is_loaded:
            return f"DataLoader(loaded=True, shape={self.shape})"
        return "DataLoader(loaded=False)"
    
    def __len__(self) -> int:
        """Returns number of rows in loaded data."""
        if self.is_loaded:
            return len(self._data)
        return 0
