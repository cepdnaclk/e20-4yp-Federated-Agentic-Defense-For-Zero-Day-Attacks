"""Test SMOTE balancing with unified dataset."""

import sys
sys.path.insert(0, '.')

import numpy as np
from data_pipeline.unified_dataset import UnifiedIDSDataset, UnifiedDatasetConfig

def main():
    # Create config
    config = UnifiedDatasetConfig(
        apply_smote=True,
        random_state=42
    )
    
    # Create dataset
    dataset = UnifiedIDSDataset(config=config)
    
    # Load datasets
    print('Loading UNSW-NB15...')
    dataset.load_unsw_nb15('data/UNSW_NB15_training-set.csv', sample_frac=0.1)
    print(f'  UNSW-NB15 loaded: {dataset._stats["unsw_nb15_samples"]:,} samples')
    
    print('Loading CIC-IDS2017...')
    dataset.load_cic_ids2017('data/CIC-IDS2017', sample_frac=0.01)
    print(f'  CIC-IDS2017 loaded: {dataset._stats["cic_ids2017_samples"]:,} samples')
    
    print()
    print('Before SMOTE:')
    print(f'  Total samples: {len(dataset.X):,}')
    unique, counts = np.unique(dataset.y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f'    Class {u}: {c:,} ({c/len(dataset.y)*100:.2f}%)')
    
    print()
    print('Applying SMOTE...')
    dataset.apply_smote()
    
    print()
    print('After SMOTE:')
    print(f'  Total samples: {len(dataset.X):,}')
    unique, counts = np.unique(dataset.y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f'    Class {u}: {c:,} ({c/len(dataset.y)*100:.2f}%)')
    
    # Get train/test split
    print()
    print('Getting train/test split...')
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()
    print(f'  Train: {X_train.shape}')
    print(f'  Test: {X_test.shape}')

if __name__ == "__main__":
    main()
