"""
Federated Learning with Validation Sets Demo
=============================================
This demonstrates how to test clients with validation data to measure
how server updates improve detection on unseen attack types.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated import NetworkDefenseClient
from agents.models.autoencoder import AnomalyAutoencoder


def generate_dos_data(n_samples: int, seed: int = None) -> np.ndarray:
    """Generate DoS attack patterns."""
    if seed:
        np.random.seed(seed)
    data = np.random.randn(n_samples, 40).astype(np.float32) * 0.5
    data[:, 0:10] += np.random.uniform(1.0, 2.0, (n_samples, 10))
    return data


def generate_recon_data(n_samples: int, seed: int = None) -> np.ndarray:
    """Generate Reconnaissance attack patterns."""
    if seed:
        np.random.seed(seed)
    data = np.random.randn(n_samples, 40).astype(np.float32) * 0.5
    data[:, 15:25] += np.random.uniform(0.8, 1.8, (n_samples, 10))
    return data


def generate_normal_data(n_samples: int, seed: int = None) -> np.ndarray:
    """Generate normal traffic patterns."""
    if seed:
        np.random.seed(seed)
    return np.random.randn(n_samples, 40).astype(np.float32) * 0.3


def create_hospital_client():
    """
    Client A: Hospital Network
    - TRAINS on: DoS attacks (what they see locally)
    - VALIDATES on: DoS + Recon + Normal (to measure generalization)
    """
    print("=" * 60)
    print("  CLIENT A - Hospital Network")
    print("  Training: DoS attacks + Normal traffic")
    print("  Validation: DoS + Recon + Normal (measures learning from Client B)")
    print("=" * 60)
    
    # Training data - only DoS (what hospital sees locally)
    train_dos = generate_dos_data(400, seed=42)
    train_normal = generate_normal_data(400, seed=43)
    X_train = np.vstack([train_dos, train_normal])
    y_train = np.array([1] * 400 + [0] * 400)  # 1=attack, 0=normal
    
    # Validation data - includes Recon (attacks client never trained on!)
    val_dos = generate_dos_data(100, seed=100)
    val_recon = generate_recon_data(100, seed=101)  # Never seen in training!
    val_normal = generate_normal_data(100, seed=102)
    X_val = np.vstack([val_dos, val_recon, val_normal])
    y_val = np.array([1] * 100 + [2] * 100 + [0] * 100)  # 1=DoS, 2=Recon, 0=Normal
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"    - DoS attacks: 400")
    print(f"    - Normal: 400")
    print(f"\n  Validation samples: {len(X_val)}")
    print(f"    - DoS attacks: 100 (seen in training)")
    print(f"    - Recon attacks: 100 (NEVER seen - will learn from Client B!)")
    print(f"    - Normal: 100")
    
    model = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    
    client = NetworkDefenseClient(
        autoencoder=model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        client_id='hospital_network'
    )
    
    return client


def create_bank_client():
    """
    Client B: Bank Network
    - TRAINS on: Reconnaissance attacks (what they see locally)
    - VALIDATES on: DoS + Recon + Normal (to measure generalization)
    """
    print("=" * 60)
    print("  CLIENT B - Bank Network")
    print("  Training: Reconnaissance attacks + Normal traffic")
    print("  Validation: DoS + Recon + Normal (measures learning from Client A)")
    print("=" * 60)
    
    # Training data - only Recon (what bank sees locally)
    train_recon = generate_recon_data(400, seed=200)
    train_normal = generate_normal_data(400, seed=201)
    X_train = np.vstack([train_recon, train_normal])
    y_train = np.array([2] * 400 + [0] * 400)  # 2=Recon, 0=normal
    
    # Validation data - includes DoS (attacks client never trained on!)
    val_dos = generate_dos_data(100, seed=300)  # Never seen in training!
    val_recon = generate_recon_data(100, seed=301)
    val_normal = generate_normal_data(100, seed=302)
    X_val = np.vstack([val_dos, val_recon, val_normal])
    y_val = np.array([1] * 100 + [2] * 100 + [0] * 100)  # 1=DoS, 2=Recon, 0=Normal
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"    - Recon attacks: 400")
    print(f"    - Normal: 400")
    print(f"\n  Validation samples: {len(X_val)}")
    print(f"    - DoS attacks: 100 (NEVER seen - will learn from Client A!)")
    print(f"    - Recon attacks: 100 (seen in training)")
    print(f"    - Normal: 100")
    
    model = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    
    client = NetworkDefenseClient(
        autoencoder=model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        client_id='bank_network'
    )
    
    return client


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Start a federated learning client with validation')
    parser.add_argument('--client', type=str, required=True, choices=['hospital', 'bank'],
                        help='Which client to start (hospital or bank)')
    parser.add_argument('--server', type=str, default='localhost:9090',
                        help='Server address (default: localhost:9090)')
    args = parser.parse_args()
    
    import flwr as fl
    
    if args.client == 'hospital':
        client = create_hospital_client()
    else:
        client = create_bank_client()
    
    print(f"\n🔗 Connecting to server at {args.server}...")
    print("   Waiting for federated learning to begin...\n")
    
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )


if __name__ == "__main__":
    main()
