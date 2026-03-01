"""
Batch Generator module for UNSW-NB15 dataset.

This module provides the BatchGenerator class for efficient iteration
over data in batches during training and evaluation.
"""

import logging
from typing import Optional, Iterator, Tuple, Union, List
import math

import numpy as np
from sklearn.model_selection import train_test_split

from data_pipeline.config import DatasetConfig

# Configure module logger
logger = logging.getLogger(__name__)


class BatchGenerator:
    """
    Generates batches of data for training and evaluation.
    
    This class provides efficient batch iteration with support for:
    - Shuffling with reproducibility
    - Stratified train/test/validation splitting
    - Variable batch sizes
    - Epoch-based iteration
    
    Attributes:
        X: Feature array of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data each epoch.
        random_seed: Random seed for reproducibility.
    
    Example:
        >>> from data_pipeline import BatchGenerator
        >>> 
        >>> # Create generator for training
        >>> train_gen = BatchGenerator(X_train, y_train, batch_size=64, shuffle=True)
        >>> 
        >>> # Iterate over batches
        >>> for X_batch, y_batch in train_gen:
        ...     # Train model on batch
        ...     pass
        >>> 
        >>> # Or iterate for multiple epochs
        >>> for epoch in range(10):
        ...     for X_batch, y_batch in train_gen:
        ...         pass
        ...     train_gen.on_epoch_end()  # Reshuffle for next epoch
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 64,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        """
        Initializes the BatchGenerator.
        
        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Optional label array of shape (n_samples,).
            batch_size: Number of samples per batch (default: 64).
            shuffle: Whether to shuffle data at each epoch (default: True).
            random_seed: Random seed for reproducibility.
            drop_last: If True, drops the last incomplete batch.
        
        Raises:
            ValueError: If X and y have incompatible shapes.
        """
        if y is not None and len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )
        
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_seed = random_seed
        
        self._n_samples = len(X)
        self._indices: np.ndarray = np.arange(self._n_samples)
        self._rng = np.random.RandomState(random_seed)
        self._current_index = 0
        
        if self.shuffle:
            self._shuffle_indices()
        
        logger.debug(
            "BatchGenerator initialized: %d samples, batch_size=%d, shuffle=%s",
            self._n_samples,
            self.batch_size,
            self.shuffle,
        )
    
    @property
    def n_samples(self) -> int:
        """Returns total number of samples."""
        return self._n_samples
    
    @property
    def n_batches(self) -> int:
        """Returns total number of batches per epoch."""
        if self.drop_last:
            return self._n_samples // self.batch_size
        return math.ceil(self._n_samples / self.batch_size)
    
    @property
    def feature_dim(self) -> int:
        """Returns the feature dimension."""
        return self.X.shape[1] if len(self.X.shape) > 1 else 1
    
    def __len__(self) -> int:
        """Returns the number of batches."""
        return self.n_batches
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Iterates over batches of data.
        
        Yields:
            Tuple of (X_batch, y_batch). y_batch is None if no labels provided.
        """
        self._current_index = 0
        
        while self._current_index < self._n_samples:
            batch_end = min(self._current_index + self.batch_size, self._n_samples)
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and (batch_end - self._current_index) < self.batch_size:
                break
            
            batch_indices = self._indices[self._current_index:batch_end]
            
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices] if self.y is not None else None
            
            self._current_index = batch_end
            
            yield X_batch, y_batch
    
    def get_batch(self, batch_index: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns a specific batch by index.
        
        Args:
            batch_index: Index of the batch to retrieve (0-indexed).
        
        Returns:
            Tuple of (X_batch, y_batch).
        
        Raises:
            IndexError: If batch_index is out of range.
        """
        if batch_index < 0 or batch_index >= self.n_batches:
            raise IndexError(
                f"Batch index {batch_index} out of range [0, {self.n_batches})"
            )
        
        start_idx = batch_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self._n_samples)
        
        batch_indices = self._indices[start_idx:end_idx]
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices] if self.y is not None else None
        
        return X_batch, y_batch
    
    def on_epoch_end(self) -> None:
        """
        Called at the end of each epoch. Reshuffles indices if shuffle=True.
        
        Example:
            >>> for epoch in range(10):
            ...     for batch in generator:
            ...         pass
            ...     generator.on_epoch_end()
        """
        if self.shuffle:
            self._shuffle_indices()
        self._current_index = 0
    
    def reset(self) -> None:
        """Resets the generator to the beginning."""
        self._current_index = 0
        if self.shuffle:
            self._shuffle_indices()
    
    def sample_batch(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns a random batch of data.
        
        Returns:
            Tuple of (X_batch, y_batch).
        """
        random_indices = self._rng.choice(
            self._n_samples,
            size=min(self.batch_size, self._n_samples),
            replace=False,
        )
        
        X_batch = self.X[random_indices]
        y_batch = self.y[random_indices] if self.y is not None else None
        
        return X_batch, y_batch
    
    def _shuffle_indices(self) -> None:
        """Shuffles the sample indices."""
        self._rng.shuffle(self._indices)
    
    def __repr__(self) -> str:
        """Returns string representation of the BatchGenerator."""
        return (
            f"BatchGenerator(n_samples={self._n_samples}, "
            f"batch_size={self.batch_size}, "
            f"n_batches={self.n_batches}, "
            f"shuffle={self.shuffle})"
        )


class DataSplitter:
    """
    Handles splitting data into train/validation/test sets.
    
    Supports stratified splitting to maintain class distribution across splits.
    
    Example:
        >>> splitter = DataSplitter(test_ratio=0.2, val_ratio=0.1, random_seed=42)
        >>> splits = splitter.split(X, y, stratify=True)
        >>> X_train, y_train = splits["train"]
        >>> X_val, y_val = splits["validation"]
        >>> X_test, y_test = splits["test"]
    """
    
    def __init__(
        self,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """
        Initializes the DataSplitter.
        
        Args:
            test_ratio: Fraction of data for test set (default: 0.2).
            val_ratio: Fraction of data for validation set (default: 0.1).
            random_seed: Random seed for reproducibility.
        
        Raises:
            ValueError: If ratios are invalid.
        """
        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
        if test_ratio + val_ratio >= 1.0:
            raise ValueError("Sum of test_ratio and val_ratio must be less than 1.0")
        
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        logger.debug(
            "DataSplitter initialized: test=%.2f, val=%.2f",
            test_ratio,
            val_ratio,
        )
    
    @classmethod
    def from_config(cls, config: DatasetConfig) -> "DataSplitter":
        """
        Creates a DataSplitter from a DatasetConfig.
        
        Args:
            config: DatasetConfig instance.
        
        Returns:
            Configured DataSplitter instance.
        """
        return cls(
            test_ratio=config.test_split_ratio,
            val_ratio=config.validation_split_ratio,
            random_seed=config.random_seed,
        )
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        stratify: bool = True,
    ) -> dict:
        """
        Splits data into train/validation/test sets.
        
        Args:
            X: Feature array.
            y: Optional label array for stratified splitting.
            stratify: If True and y is provided, performs stratified split.
        
        Returns:
            Dictionary with keys 'train', 'validation', 'test', each containing
            a tuple of (X_split, y_split).
        
        Example:
            >>> splits = splitter.split(X, y, stratify=True)
            >>> X_train, y_train = splits["train"]
        """
        stratify_labels = y if (stratify and y is not None) else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=stratify_labels,
        )
        
        # Second split: separate validation from training
        if self.val_ratio > 0:
            # Adjust validation ratio for remaining data
            adjusted_val_ratio = self.val_ratio / (1.0 - self.test_ratio)
            
            stratify_temp = y_temp if (stratify and y_temp is not None) else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=adjusted_val_ratio,
                random_state=self.random_seed,
                stratify=stratify_temp,
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        splits = {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
        }
        
        if self.val_ratio > 0:
            splits["validation"] = (X_val, y_val)
        
        logger.info(
            "Data split: train=%d, val=%s, test=%d",
            len(X_train),
            len(X_val) if X_val is not None else 0,
            len(X_test),
        )
        
        return splits
    
    def create_generators(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 64,
        stratify: bool = True,
        shuffle_train: bool = True,
    ) -> dict:
        """
        Splits data and creates BatchGenerators for each split.
        
        Args:
            X: Feature array.
            y: Optional label array.
            batch_size: Batch size for generators.
            stratify: If True, performs stratified split.
            shuffle_train: If True, shuffles training data.
        
        Returns:
            Dictionary with keys 'train', 'validation', 'test', each containing
            a BatchGenerator instance.
        
        Example:
            >>> generators = splitter.create_generators(X, y, batch_size=64)
            >>> train_gen = generators["train"]
            >>> for X_batch, y_batch in train_gen:
            ...     pass
        """
        splits = self.split(X, y, stratify=stratify)
        
        generators = {}
        
        # Training generator (shuffled)
        X_train, y_train = splits["train"]
        generators["train"] = BatchGenerator(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=shuffle_train,
            random_seed=self.random_seed,
        )
        
        # Validation generator (no shuffle)
        if "validation" in splits:
            X_val, y_val = splits["validation"]
            generators["validation"] = BatchGenerator(
                X_val, y_val,
                batch_size=batch_size,
                shuffle=False,
                random_seed=self.random_seed,
            )
        
        # Test generator (no shuffle)
        X_test, y_test = splits["test"]
        generators["test"] = BatchGenerator(
            X_test, y_test,
            batch_size=batch_size,
            shuffle=False,
            random_seed=self.random_seed,
        )
        
        logger.info(
            "Created generators: train=%d batches, val=%s batches, test=%d batches",
            len(generators["train"]),
            len(generators.get("validation", [])) if "validation" in generators else 0,
            len(generators["test"]),
        )
        
        return generators


def create_train_test_generators(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    config: Optional[DatasetConfig] = None,
    stratify: bool = True,
) -> Tuple[BatchGenerator, BatchGenerator, Optional[BatchGenerator]]:
    """
    Convenience function to create train/test/validation generators.
    
    Args:
        X: Feature array.
        y: Optional label array.
        config: DatasetConfig for split ratios and batch size.
        stratify: If True, performs stratified split.
    
    Returns:
        Tuple of (train_generator, test_generator, validation_generator).
        validation_generator is None if val_ratio is 0.
    
    Example:
        >>> train_gen, test_gen, val_gen = create_train_test_generators(
        ...     X, y, config=DatasetConfig(default_batch_size=128)
        ... )
        >>> for X_batch, y_batch in train_gen:
        ...     # Train model
        ...     pass
    """
    config = config or DatasetConfig()
    
    splitter = DataSplitter.from_config(config)
    generators = splitter.create_generators(
        X, y,
        batch_size=config.default_batch_size,
        stratify=stratify,
    )
    
    return (
        generators["train"],
        generators["test"],
        generators.get("validation"),
    )
