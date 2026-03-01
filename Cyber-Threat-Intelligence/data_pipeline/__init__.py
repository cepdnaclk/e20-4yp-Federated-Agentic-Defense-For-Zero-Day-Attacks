"""
Data Pipeline Module for Multi-Dataset Intrusion Detection System.

This module provides a clean, object-oriented data pipeline for loading,
preprocessing, and batching network intrusion datasets including:
- UNSW-NB15
- CIC-IDS2017

Modules:
    data_loader: Handles loading and cleaning tabular data.
    preprocessor: Implements feature normalization, encoding, and missing value handling.
    batch_generator: Provides batch iteration for training and testing.
    config: Contains dataset configuration and feature specifications.
    unified_taxonomy: Unified attack category mapping across datasets.
    cic_ids2017_loader: CIC-IDS2017 dataset loader.
    unified_dataset: Combined multi-dataset loader with SMOTE support.
"""

from data_pipeline.data_loader import DataLoader
from data_pipeline.preprocessor import Preprocessor
from data_pipeline.batch_generator import BatchGenerator
from data_pipeline.config import DatasetConfig
from data_pipeline.unified_taxonomy import UnifiedTaxonomy, get_taxonomy
from data_pipeline.cic_ids2017_loader import CICIDS2017Loader
from data_pipeline.unified_dataset import UnifiedIDSDataset

__all__ = [
    # Original UNSW-NB15 pipeline
    "DataLoader",
    "Preprocessor",
    "BatchGenerator",
    "DatasetConfig",
    # Unified multi-dataset support
    "UnifiedTaxonomy",
    "get_taxonomy",
    "CICIDS2017Loader",
    "UnifiedIDSDataset",
]

__version__ = "2.0.0"
