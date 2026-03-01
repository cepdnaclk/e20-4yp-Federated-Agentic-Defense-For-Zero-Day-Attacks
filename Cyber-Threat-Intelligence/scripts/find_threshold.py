"""
Script to find optimal threshold with accuracy constraint.
"""
import numpy as np
from pathlib import Path
from data_pipeline import DataLoader, DatasetConfig, Preprocessor
from data_pipeline.batch_generator import DataSplitter
from agents import AgentOne
from agents.models.autoencoder import AnomalyAutoencoder


def main():
    # Load data
    config = DatasetConfig(normalization_method='minmax')
    loader = DataLoader(config)
    loader.load('data/UNSW_NB15_training-set.csv').clean()
    X, y = loader.get_features_and_labels(label_type='binary')

    # Load preprocessor and transform
    preprocessor = Preprocessor.load('models/agent_one/preprocessor.pkl')
    X_processed = preprocessor.transform(X)

    # Split data
    splitter = DataSplitter(test_ratio=0.2, val_ratio=0.1, random_seed=42)
    splits = splitter.split(X_processed, y, stratify=True)
    X_test, y_test = splits['test']

    # Load model and create agent
    model = AnomalyAutoencoder.load('models/agent_one/best_model.pth')
    agent = AgentOne(model, threshold=0.1)

    # Search for threshold with 95% accuracy
    print('=' * 60)
    print('Constrained Threshold Search (Target: 95% Accuracy)')
    print('=' * 60)
    threshold, results = agent.find_constrained_threshold(
        X_test, y_test, 
        min_accuracy=0.95,
        n_thresholds=2000,
        verbose=True
    )

    if not results['found']:
        # Try lower thresholds
        print()
        print('=' * 60)
        print('Trying different accuracy targets...')
        print('=' * 60)
        for target in [0.90, 0.85, 0.80, 0.75]:
            thresh, res = agent.find_constrained_threshold(
                X_test, y_test,
                min_accuracy=target,
                n_thresholds=2000,
                verbose=False
            )
            if res['found']:
                m = res['best_metrics']
                print(f"\n   Target {target*100:.0f}%: ACHIEVABLE")
                print(f"   Best threshold: {thresh:.6f}")
                print(f"   Accuracy: {m['accuracy']:.4f}")
                print(f"   False Negatives: {m['false_negatives']}")
                print(f"   Recall: {m['recall']:.4f}")
            else:
                print(f"   Target {target*100:.0f}%: Not achievable")


if __name__ == "__main__":
    main()
