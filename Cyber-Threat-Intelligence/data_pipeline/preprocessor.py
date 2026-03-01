"""
Preprocessor module for UNSW-NB15 dataset.

This module provides the Preprocessor class for feature normalization,
categorical encoding, missing value handling, and data transformation.
"""

import logging
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer

from data_pipeline.config import DatasetConfig

# Configure module logger
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Handles preprocessing of UNSW-NB15 dataset features.
    
    This class provides robust preprocessing capabilities including:
    - Numerical feature normalization (MinMax, Standard, Robust scaling)
    - Categorical feature encoding (Label encoding, One-hot encoding)
    - Missing value imputation
    - Outlier handling
    
    The preprocessor maintains fitted state and can be saved/loaded for
    consistent preprocessing across train and test datasets.
    
    Attributes:
        config: DatasetConfig instance with preprocessing parameters.
        is_fitted: Boolean indicating if preprocessor has been fitted.
        numerical_scaler: Fitted scaler for numerical features.
        categorical_encoders: Dict of fitted encoders for categorical features.
        imputers: Dict of fitted imputers for missing values.
    
    Example:
        >>> from data_pipeline import Preprocessor, DataLoader, DatasetConfig
        >>> config = DatasetConfig(normalization_method="standard")
        >>> preprocessor = Preprocessor(config)
        >>> 
        >>> # Fit on training data
        >>> X_train, y_train = loader_train.get_features_and_labels()
        >>> X_train_processed = preprocessor.fit_transform(X_train)
        >>> 
        >>> # Transform test data using fitted preprocessor
        >>> X_test, y_test = loader_test.get_features_and_labels()
        >>> X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        """
        Initializes the Preprocessor with configuration.
        
        Args:
            config: DatasetConfig instance. If None, uses default configuration.
        """
        self.config = config if config is not None else DatasetConfig()
        
        # Initialize scalers and encoders
        self._numerical_scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None
        self._categorical_encoders: Dict[str, Union[LabelEncoder, OneHotEncoder]] = {}
        self._label_encoder: Optional[LabelEncoder] = None
        self._numerical_imputer: Optional[SimpleImputer] = None
        self._categorical_imputers: Dict[str, SimpleImputer] = {}
        
        # Track fitted state and feature information
        self._is_fitted: bool = False
        self._fitted_numerical_features: List[str] = []
        self._fitted_categorical_features: List[str] = []
        self._feature_statistics: Dict[str, Any] = {}
        self._encoding_type: str = "label"  # 'label' or 'onehot'
        
        logger.info(
            "Preprocessor initialized with normalization=%s, numerical_fill=%s",
            self.config.normalization_method,
            self.config.numerical_fill_strategy,
        )
    
    @property
    def is_fitted(self) -> bool:
        """Returns True if the preprocessor has been fitted."""
        return self._is_fitted
    
    @property
    def numerical_features(self) -> List[str]:
        """Returns list of fitted numerical feature names."""
        return self._fitted_numerical_features.copy()
    
    @property
    def categorical_features(self) -> List[str]:
        """Returns list of fitted categorical feature names."""
        return self._fitted_categorical_features.copy()
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        categorical_encoding: str = "label",
    ) -> "Preprocessor":
        """
        Fits the preprocessor on training data.
        
        Args:
            X: Feature DataFrame to fit on.
            y: Optional label Series for fitting label encoder.
            categorical_encoding: Encoding type ('label' or 'onehot').
        
        Returns:
            Self for method chaining.
        
        Raises:
            ValueError: If categorical_encoding is invalid.
        
        Example:
            >>> preprocessor.fit(X_train, y_train, categorical_encoding="onehot")
        """
        if categorical_encoding not in ("label", "onehot"):
            raise ValueError(f"categorical_encoding must be 'label' or 'onehot', got '{categorical_encoding}'")
        
        self._encoding_type = categorical_encoding
        logger.info("Fitting preprocessor on data with shape: %s", X.shape)
        
        # Identify feature columns in the data
        self._fitted_numerical_features = [
            col for col in self.config.numerical_features
            if col in X.columns
        ]
        self._fitted_categorical_features = [
            col for col in self.config.categorical_features
            if col in X.columns
        ]
        
        # Fit missing value imputers
        self._fit_imputers(X)
        
        # Impute missing values before fitting scalers/encoders
        X_imputed = self._impute_missing_values(X.copy())
        
        # Fit numerical scaler
        self._fit_numerical_scaler(X_imputed)
        
        # Fit categorical encoders
        self._fit_categorical_encoders(X_imputed)
        
        # Fit label encoder if labels provided
        if y is not None:
            self._fit_label_encoder(y)
        
        # Compute and store feature statistics
        self._compute_feature_statistics(X_imputed)
        
        self._is_fitted = True
        logger.info(
            "Preprocessor fitted: %d numerical, %d categorical features",
            len(self._fitted_numerical_features),
            len(self._fitted_categorical_features),
        )
        
        return self
    
    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transforms data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame to transform.
            y: Optional label Series to encode.
        
        Returns:
            Transformed feature array, or tuple of (features, labels) if y provided.
        
        Raises:
            RuntimeError: If preprocessor has not been fitted.
        
        Example:
            >>> X_transformed = preprocessor.transform(X_test)
            >>> # or with labels
            >>> X_transformed, y_encoded = preprocessor.transform(X_test, y_test)
        """
        self._ensure_fitted()
        
        logger.debug("Transforming data with shape: %s", X.shape)
        
        # Create copy to avoid modifying original
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = self._impute_missing_values(X_processed)
        
        # Handle outliers if configured
        if self.config.handle_outliers:
            X_processed = self._handle_outliers(X_processed)
        
        # Transform numerical features
        X_numerical = self._transform_numerical(X_processed)
        
        # Transform categorical features
        X_categorical = self._transform_categorical(X_processed)
        
        # Combine features
        if X_categorical is not None and X_categorical.size > 0:
            X_combined = np.hstack([X_numerical, X_categorical])
        else:
            X_combined = X_numerical
        
        # Transform labels if provided
        if y is not None:
            y_encoded = self._transform_labels(y)
            return X_combined, y_encoded
        
        return X_combined
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        categorical_encoding: str = "label",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fits and transforms data in one step.
        
        Args:
            X: Feature DataFrame to fit and transform.
            y: Optional label Series to encode.
            categorical_encoding: Encoding type ('label' or 'onehot').
        
        Returns:
            Transformed feature array, or tuple of (features, labels) if y provided.
        
        Example:
            >>> X_train_processed, y_train_encoded = preprocessor.fit_transform(
            ...     X_train, y_train, categorical_encoding="onehot"
            ... )
        """
        self.fit(X, y, categorical_encoding)
        return self.transform(X, y)
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Inverse transforms encoded labels back to original values.
        
        Args:
            y_encoded: Encoded label array.
        
        Returns:
            Original label values.
        
        Raises:
            RuntimeError: If label encoder was not fitted.
        """
        if self._label_encoder is None:
            raise RuntimeError("Label encoder not fitted. Fit with labels first.")
        
        return self._label_encoder.inverse_transform(y_encoded)
    
    def get_feature_names(self) -> List[str]:
        """
        Returns the names of all features after transformation.
        
        Returns:
            List of feature names in order they appear in transformed output.
        """
        self._ensure_fitted()
        
        feature_names = self._fitted_numerical_features.copy()
        
        if self._encoding_type == "onehot":
            for col in self._fitted_categorical_features:
                encoder = self._categorical_encoders.get(col)
                if encoder is not None and hasattr(encoder, "categories_"):
                    for category in encoder.categories_[0]:
                        feature_names.append(f"{col}_{category}")
        else:
            feature_names.extend(self._fitted_categorical_features)
        
        return feature_names
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics computed during fitting.
        
        Returns:
            Dictionary containing feature statistics.
        """
        return self._feature_statistics.copy()
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Saves the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor.
        
        Raises:
            RuntimeError: If preprocessor has not been fitted.
        
        Example:
            >>> preprocessor.fit(X_train)
            >>> preprocessor.save("models/preprocessor.pkl")
        """
        self._ensure_fitted()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "numerical_scaler": self._numerical_scaler,
            "categorical_encoders": self._categorical_encoders,
            "label_encoder": self._label_encoder,
            "numerical_imputer": self._numerical_imputer,
            "categorical_imputers": self._categorical_imputers,
            "fitted_numerical_features": self._fitted_numerical_features,
            "fitted_categorical_features": self._fitted_categorical_features,
            "feature_statistics": self._feature_statistics,
            "encoding_type": self._encoding_type,
            "is_fitted": self._is_fitted,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        logger.info("Preprocessor saved to: %s", filepath)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Preprocessor":
        """
        Loads a fitted preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor.
        
        Returns:
            Loaded Preprocessor instance.
        
        Raises:
            FileNotFoundError: If file does not exist.
        
        Example:
            >>> preprocessor = Preprocessor.load("models/preprocessor.pkl")
            >>> X_test_processed = preprocessor.transform(X_test)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        preprocessor = cls(state["config"])
        preprocessor._numerical_scaler = state["numerical_scaler"]
        preprocessor._categorical_encoders = state["categorical_encoders"]
        preprocessor._label_encoder = state["label_encoder"]
        preprocessor._numerical_imputer = state["numerical_imputer"]
        preprocessor._categorical_imputers = state["categorical_imputers"]
        preprocessor._fitted_numerical_features = state["fitted_numerical_features"]
        preprocessor._fitted_categorical_features = state["fitted_categorical_features"]
        preprocessor._feature_statistics = state["feature_statistics"]
        preprocessor._encoding_type = state["encoding_type"]
        preprocessor._is_fitted = state["is_fitted"]
        
        logger.info("Preprocessor loaded from: %s", filepath)
        return preprocessor
    
    # ==================== Private Methods ====================
    
    def _fit_imputers(self, X: pd.DataFrame) -> None:
        """Fits imputers for missing value handling."""
        # Numerical imputer
        numerical_strategy = self.config.numerical_fill_strategy
        if numerical_strategy == "zero":
            self._numerical_imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            self._numerical_imputer = SimpleImputer(strategy=numerical_strategy)
        
        if self._fitted_numerical_features:
            numerical_data = X[self._fitted_numerical_features]
            self._numerical_imputer.fit(numerical_data)
        
        # Categorical imputers
        categorical_strategy = self.config.categorical_fill_strategy
        for col in self._fitted_categorical_features:
            if categorical_strategy == "mode":
                imputer = SimpleImputer(strategy="most_frequent")
            else:  # "unknown"
                imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
            
            imputer.fit(X[[col]].astype(str))
            self._categorical_imputers[col] = imputer
    
    def _impute_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values in the DataFrame."""
        # Impute numerical features
        if self._fitted_numerical_features and self._numerical_imputer is not None:
            numerical_data = X[self._fitted_numerical_features]
            X[self._fitted_numerical_features] = self._numerical_imputer.transform(numerical_data)
        
        # Impute categorical features
        for col in self._fitted_categorical_features:
            if col in self._categorical_imputers:
                X[col] = self._categorical_imputers[col].transform(
                    X[[col]].astype(str)
                ).ravel()
        
        return X
    
    def _fit_numerical_scaler(self, X: pd.DataFrame) -> None:
        """Fits the numerical feature scaler."""
        method = self.config.normalization_method
        
        if method == "minmax":
            self._numerical_scaler = MinMaxScaler()
        elif method == "standard":
            self._numerical_scaler = StandardScaler()
        elif method == "robust":
            self._numerical_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if self._fitted_numerical_features:
            numerical_data = X[self._fitted_numerical_features].values
            self._numerical_scaler.fit(numerical_data)
            logger.debug("Numerical scaler fitted using %s method", method)
    
    def _fit_categorical_encoders(self, X: pd.DataFrame) -> None:
        """Fits encoders for categorical features."""
        for col in self._fitted_categorical_features:
            if self._encoding_type == "onehot":
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                )
                encoder.fit(X[[col]].astype(str))
            else:  # label encoding
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str))
            
            self._categorical_encoders[col] = encoder
        
        logger.debug(
            "Fitted %d categorical encoders using %s encoding",
            len(self._categorical_encoders),
            self._encoding_type,
        )
    
    def _fit_label_encoder(self, y: pd.Series) -> None:
        """Fits the label encoder."""
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(y)
        logger.debug("Label encoder fitted with %d classes", len(self._label_encoder.classes_))
    
    def _transform_numerical(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms numerical features."""
        if not self._fitted_numerical_features:
            return np.array([]).reshape(len(X), 0)
        
        numerical_data = X[self._fitted_numerical_features].values
        
        # Handle infinite values
        numerical_data = np.nan_to_num(
            numerical_data,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=np.finfo(np.float64).min,
        )
        
        return self._numerical_scaler.transform(numerical_data)
    
    def _transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms categorical features."""
        if not self._fitted_categorical_features:
            return np.array([]).reshape(len(X), 0)
        
        encoded_features: List[np.ndarray] = []
        
        for col in self._fitted_categorical_features:
            encoder = self._categorical_encoders[col]
            
            if self._encoding_type == "onehot":
                encoded = encoder.transform(X[[col]].astype(str))
            else:
                # Handle unseen categories for label encoding
                col_values = X[col].astype(str).values
                encoded = np.zeros(len(col_values), dtype=np.int64)
                
                for i, val in enumerate(col_values):
                    if val in encoder.classes_:
                        encoded[i] = encoder.transform([val])[0]
                    else:
                        # Assign -1 for unknown categories
                        encoded[i] = -1
                
                encoded = encoded.reshape(-1, 1)
            
            encoded_features.append(encoded)
        
        return np.hstack(encoded_features) if encoded_features else np.array([]).reshape(len(X), 0)
    
    def _transform_labels(self, y: pd.Series) -> np.ndarray:
        """Transforms labels using fitted encoder."""
        if self._label_encoder is None:
            return y.values
        
        return self._label_encoder.transform(y)
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handles outliers in numerical features using clipping."""
        if not self._fitted_numerical_features:
            return X
        
        for col in self._fitted_numerical_features:
            if col in X.columns and col in self._feature_statistics.get("numerical", {}):
                stats = self._feature_statistics["numerical"][col]
                lower_bound = stats.get("lower_clip", X[col].min())
                upper_bound = stats.get("upper_clip", X[col].max())
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        return X
    
    def _compute_feature_statistics(self, X: pd.DataFrame) -> None:
        """Computes and stores statistics for fitted features."""
        self._feature_statistics = {
            "numerical": {},
            "categorical": {},
        }
        
        # Numerical statistics
        for col in self._fitted_numerical_features:
            if col in X.columns:
                values = X[col].values
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                self._feature_statistics["numerical"][col] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_clip": float(q1 - self.config.outlier_threshold * iqr),
                    "upper_clip": float(q3 + self.config.outlier_threshold * iqr),
                }
        
        # Categorical statistics
        for col in self._fitted_categorical_features:
            if col in X.columns:
                value_counts = X[col].value_counts()
                self._feature_statistics["categorical"][col] = {
                    "n_unique": int(X[col].nunique()),
                    "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "categories": list(X[col].unique()),
                }
    
    def _ensure_fitted(self) -> None:
        """Ensures preprocessor is fitted, raises RuntimeError if not."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
    
    def __repr__(self) -> str:
        """Returns string representation of the Preprocessor."""
        if self._is_fitted:
            return (
                f"Preprocessor(fitted=True, "
                f"numerical={len(self._fitted_numerical_features)}, "
                f"categorical={len(self._fitted_categorical_features)}, "
                f"encoding={self._encoding_type})"
            )
        return "Preprocessor(fitted=False)"
