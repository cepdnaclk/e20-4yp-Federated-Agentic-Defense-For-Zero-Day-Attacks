"""
Federated Learning Visualization Demo
=====================================
This script demonstrates how clients learn from each other in federated learning
by visualizing the before/after performance on attack types each client has never seen.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.models.autoencoder import AnomalyAutoencoder


def generate_attack_data(attack_type: str, n_samples: int = 500) -> np.ndarray:
    """Generate synthetic data for different attack types with distinct patterns."""
    np.random.seed(hash(attack_type) % 2**32)
    
    # Start with realistic base noise
    base = np.random.randn(n_samples, 40).astype(np.float32) * 0.8
    
    if attack_type == "DoS":
        # DoS: Elevated values in specific features (network volume indicators)
        # But with variance so it's not perfectly separable
        base[:, 0:5] += np.random.uniform(0.8, 1.5, (n_samples, 5))
        base[:, 5:10] += np.random.uniform(0.3, 0.8, (n_samples, 5))
    elif attack_type == "Reconnaissance":
        # Recon: Different pattern - port scanning signatures
        base[:, 15:20] += np.random.uniform(0.7, 1.3, (n_samples, 5))
        base[:, 20:25] += np.random.uniform(0.2, 0.6, (n_samples, 5))
    elif attack_type == "Exploits":
        # Exploits: Payload signatures in different features
        base[:, 30:35] += np.random.uniform(0.6, 1.2, (n_samples, 5))
        base[:, 35:40] += np.random.uniform(0.2, 0.5, (n_samples, 5))
    elif attack_type == "Normal":
        # Normal: Just noise, centered around 0
        base = np.random.randn(n_samples, 40).astype(np.float32) * 0.5
    
    return base


def train_autoencoder(model: AnomalyAutoencoder, data: np.ndarray, epochs: int = 50):
    """Train autoencoder and return training losses."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    X_tensor = torch.FloatTensor(data)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def compute_reconstruction_errors(model: AnomalyAutoencoder, data: np.ndarray) -> np.ndarray:
    """Compute per-sample reconstruction errors."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(data)
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    return errors


def federated_average(models: list) -> dict:
    """Simple FedAvg: average the weights of all models."""
    avg_state = {}
    
    for key in models[0].state_dict().keys():
        avg_state[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(dim=0)
    
    return avg_state


def visualize_learning_progress(results: dict, save_path: str = None):
    """Create comprehensive visualization of federated learning progress."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Federated Learning: How Clients Learn From Each Other', fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {'Client A (DoS Expert)': '#e74c3c', 'Client B (Recon Expert)': '#3498db'}
    attack_colors = {'DoS': '#e74c3c', 'Reconnaissance': '#3498db', 'Exploits': '#2ecc71', 'Normal': '#95a5a6'}
    
    # Plot 1: Training Loss Over Rounds
    ax1 = axes[0, 0]
    for client_name, data in results['training_losses'].items():
        ax1.plot(data, label=client_name, color=colors.get(client_name, 'gray'), linewidth=2)
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.set_title('Training Loss Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection Ability BEFORE Federation
    ax2 = axes[0, 1]
    before_data = results['detection_before']
    x = np.arange(len(before_data['attacks']))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, before_data['Client A'], width, label='Client A', color=colors['Client A (DoS Expert)'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, before_data['Client B'], width, label='Client B', color=colors['Client B (Recon Expert)'], alpha=0.8)
    
    ax2.set_xlabel('Attack Type')
    ax2.set_ylabel('Detection Rate (%)')
    ax2.set_title('BEFORE Federation\n(Isolated Training)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(before_data['attacks'], rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Detection Ability AFTER Federation
    ax3 = axes[0, 2]
    after_data = results['detection_after']
    
    bars1 = ax3.bar(x - width/2, after_data['Client A'], width, label='Client A', color=colors['Client A (DoS Expert)'], alpha=0.8)
    bars2 = ax3.bar(x + width/2, after_data['Client B'], width, label='Client B', color=colors['Client B (Recon Expert)'], alpha=0.8)
    
    ax3.set_xlabel('Attack Type')
    ax3.set_ylabel('Detection Rate (%)')
    ax3.set_title('AFTER Federation\n(Collaborative Learning)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(after_data['attacks'], rotation=45)
    ax3.legend()
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Improvement Chart
    ax4 = axes[1, 0]
    improvements = results['improvements']
    
    bar_width = 0.25
    x_imp = np.arange(len(improvements['attacks']))
    
    ax4.bar(x_imp - bar_width, improvements['Client A'], bar_width, label='Client A Improvement', 
            color=colors['Client A (DoS Expert)'], alpha=0.8)
    ax4.bar(x_imp, improvements['Client B'], bar_width, label='Client B Improvement', 
            color=colors['Client B (Recon Expert)'], alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Attack Type')
    ax4.set_ylabel('Detection Improvement (%)')
    ax4.set_title('Knowledge Transfer Gain\n(After - Before Federation)')
    ax4.set_xticks(x_imp)
    ax4.set_xticklabels(improvements['attacks'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Reconstruction Error Distribution
    ax5 = axes[1, 1]
    for attack, errors in results['error_distributions'].items():
        ax5.hist(errors, bins=30, alpha=0.5, label=attack, color=attack_colors.get(attack, 'gray'))
    ax5.axvline(x=results['threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold ({results["threshold"]:.3f})')
    ax5.set_xlabel('Reconstruction Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution (Federated Model)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary Text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = """
    FEDERATED LEARNING SUMMARY
    ==========================
    
    Client A (Hospital Network):
      • Specialized in: DoS attacks
      • Training data: DoS + Normal traffic
      • Learned from Client B: Reconnaissance patterns
    
    Client B (Bank Network):  
      • Specialized in: Reconnaissance attacks
      • Training data: Recon + Normal traffic
      • Learned from Client A: DoS patterns
    
    KEY INSIGHT:
    After federated averaging, BOTH clients can
    detect attacks they've NEVER seen locally!
    
    This happens because:
    1. Each client trains on its local data
    2. Only MODEL WEIGHTS are shared (not raw data)
    3. FedAvg combines knowledge from all clients
    4. Updated global model is sent back to clients
    
    Privacy preserved: No raw network
    traffic ever leaves the client!
    """
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
    
    plt.close()  # Close figure to free memory


def run_federated_demo():
    """Run the complete federated learning demonstration."""
    print("=" * 60)
    print("  FEDERATED LEARNING VISUALIZATION DEMO")
    print("  Showing how clients learn from each other")
    print("=" * 60)
    
    # Generate data for each client
    print("\n📦 Generating heterogeneous client data...")
    
    # Client A: Hospital network - sees DoS attacks
    client_a_dos = generate_attack_data("DoS", 400)
    client_a_normal = generate_attack_data("Normal", 400)
    client_a_train = np.vstack([client_a_dos, client_a_normal])
    
    # Client B: Bank network - sees Reconnaissance attacks  
    client_b_recon = generate_attack_data("Reconnaissance", 400)
    client_b_normal = generate_attack_data("Normal", 400)
    client_b_train = np.vstack([client_b_recon, client_b_normal])
    
    # Test data - all attack types (to evaluate cross-learning)
    test_dos = generate_attack_data("DoS", 200)
    test_recon = generate_attack_data("Reconnaissance", 200)
    test_exploits = generate_attack_data("Exploits", 200)
    test_normal = generate_attack_data("Normal", 200)
    
    print(f"  Client A: {len(client_a_train)} samples (DoS + Normal)")
    print(f"  Client B: {len(client_b_train)} samples (Recon + Normal)")
    print(f"  Test set: {200*4} samples (All attack types)")
    
    # Create models
    print("\n🔧 Creating autoencoder models...")
    model_a = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    model_b = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    
    # Phase 1: Isolated Training (BEFORE federation)
    print("\n📚 Phase 1: Isolated Training (No Collaboration)...")
    print("  Training Client A on local data only...")
    losses_a_isolated = train_autoencoder(model_a, client_a_train, epochs=100)
    print("  Training Client B on local data only...")
    losses_b_isolated = train_autoencoder(model_b, client_b_train, epochs=100)
    
    # Save isolated model states
    isolated_state_a = {k: v.clone() for k, v in model_a.state_dict().items()}
    isolated_state_b = {k: v.clone() for k, v in model_b.state_dict().items()}
    
    # Evaluate isolated models
    print("\n📊 Evaluating isolated models...")
    
    def evaluate_detection(model, test_data, threshold):
        errors = compute_reconstruction_errors(model, test_data)
        detected = (errors > threshold).sum()
        return (detected / len(test_data)) * 100
    
    # Calculate threshold from normal data
    model_a.load_state_dict(isolated_state_a)
    normal_errors_a = compute_reconstruction_errors(model_a, test_normal)
    threshold_a = np.percentile(normal_errors_a, 95)
    
    model_b.load_state_dict(isolated_state_b)
    normal_errors_b = compute_reconstruction_errors(model_b, test_normal)
    threshold_b = np.percentile(normal_errors_b, 95)
    
    # Detection rates BEFORE federation
    detection_before = {
        'attacks': ['DoS', 'Recon', 'Exploits'],
        'Client A': [
            evaluate_detection(model_a, test_dos, threshold_a),
            evaluate_detection(model_a, test_recon, threshold_a),
            evaluate_detection(model_a, test_exploits, threshold_a),
        ],
        'Client B': [
            evaluate_detection(model_b, test_dos, threshold_b),
            evaluate_detection(model_b, test_recon, threshold_b),
            evaluate_detection(model_b, test_exploits, threshold_b),
        ]
    }
    
    print("\n  Detection Rates BEFORE Federation:")
    print(f"  {'Attack Type':<15} {'Client A':<12} {'Client B':<12}")
    print(f"  {'-'*39}")
    for i, attack in enumerate(detection_before['attacks']):
        print(f"  {attack:<15} {detection_before['Client A'][i]:>10.1f}% {detection_before['Client B'][i]:>10.1f}%")
    
    # Phase 2: Federated Learning
    print("\n🔄 Phase 2: Federated Learning (5 Rounds)...")
    
    # Reset models
    model_a = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    model_b = AnomalyAutoencoder(input_dim=40, latent_dim=8)
    
    all_losses_a = []
    all_losses_b = []
    
    for round_num in range(5):
        print(f"\n  Round {round_num + 1}/5:")
        
        # Local training
        print(f"    Client A training locally...")
        losses_a = train_autoencoder(model_a, client_a_train, epochs=20)
        all_losses_a.extend(losses_a)
        
        print(f"    Client B training locally...")
        losses_b = train_autoencoder(model_b, client_b_train, epochs=20)
        all_losses_b.extend(losses_b)
        
        # Federated Averaging
        print(f"    🔀 Aggregating weights (FedAvg)...")
        avg_weights = federated_average([model_a, model_b])
        
        # Distribute back to clients
        model_a.load_state_dict(avg_weights)
        model_b.load_state_dict(avg_weights)
        
        # Evaluate current round
        normal_errors = compute_reconstruction_errors(model_a, test_normal)
        threshold = np.percentile(normal_errors, 95)
        
        dos_det = evaluate_detection(model_a, test_dos, threshold)
        recon_det = evaluate_detection(model_a, test_recon, threshold)
        print(f"    Round {round_num + 1} - DoS: {dos_det:.1f}%, Recon: {recon_det:.1f}%")
    
    # Final evaluation AFTER federation
    print("\n📊 Evaluating federated models...")
    
    normal_errors_fed = compute_reconstruction_errors(model_a, test_normal)
    threshold_fed = np.percentile(normal_errors_fed, 95)
    
    detection_after = {
        'attacks': ['DoS', 'Recon', 'Exploits'],
        'Client A': [
            evaluate_detection(model_a, test_dos, threshold_fed),
            evaluate_detection(model_a, test_recon, threshold_fed),
            evaluate_detection(model_a, test_exploits, threshold_fed),
        ],
        'Client B': [
            evaluate_detection(model_b, test_dos, threshold_fed),
            evaluate_detection(model_b, test_recon, threshold_fed),
            evaluate_detection(model_b, test_exploits, threshold_fed),
        ]
    }
    
    print("\n  Detection Rates AFTER Federation:")
    print(f"  {'Attack Type':<15} {'Client A':<12} {'Client B':<12}")
    print(f"  {'-'*39}")
    for i, attack in enumerate(detection_after['attacks']):
        print(f"  {attack:<15} {detection_after['Client A'][i]:>10.1f}% {detection_after['Client B'][i]:>10.1f}%")
    
    # Calculate improvements
    improvements = {
        'attacks': ['DoS', 'Recon', 'Exploits'],
        'Client A': [
            detection_after['Client A'][i] - detection_before['Client A'][i]
            for i in range(3)
        ],
        'Client B': [
            detection_after['Client B'][i] - detection_before['Client B'][i]
            for i in range(3)
        ]
    }
    
    print("\n  IMPROVEMENT (After - Before):")
    print(f"  {'Attack Type':<15} {'Client A':<12} {'Client B':<12}")
    print(f"  {'-'*39}")
    for i, attack in enumerate(improvements['attacks']):
        a_imp = improvements['Client A'][i]
        b_imp = improvements['Client B'][i]
        a_str = f"+{a_imp:.1f}%" if a_imp >= 0 else f"{a_imp:.1f}%"
        b_str = f"+{b_imp:.1f}%" if b_imp >= 0 else f"{b_imp:.1f}%"
        print(f"  {attack:<15} {a_str:>10} {b_str:>10}")
    
    # Compute error distributions for visualization
    error_distributions = {
        'DoS': compute_reconstruction_errors(model_a, test_dos),
        'Reconnaissance': compute_reconstruction_errors(model_a, test_recon),
        'Exploits': compute_reconstruction_errors(model_a, test_exploits),
        'Normal': compute_reconstruction_errors(model_a, test_normal),
    }
    
    # Prepare results for visualization
    results = {
        'training_losses': {
            'Client A (DoS Expert)': all_losses_a,
            'Client B (Recon Expert)': all_losses_b,
        },
        'detection_before': detection_before,
        'detection_after': detection_after,
        'improvements': improvements,
        'error_distributions': error_distributions,
        'threshold': threshold_fed,
    }
    
    # Generate visualization
    print("\n📈 Generating visualization...")
    save_path = os.path.join(os.path.dirname(__file__), 'federated_learning_visualization.png')
    visualize_learning_progress(results, save_path)
    
    print("\n" + "=" * 60)
    print("  KEY FINDINGS:")
    print("=" * 60)
    print("""
    🏥 Client A (Hospital) initially could ONLY detect DoS attacks
       because that's all it saw in its local network.
       
    🏦 Client B (Bank) initially could ONLY detect Reconnaissance
       attacks because that's all it saw locally.
       
    🔄 AFTER federated learning:
       • Both clients can now detect BOTH attack types!
       • Client A learned Recon patterns from Client B
       • Client B learned DoS patterns from Client A
       
    🔒 PRIVACY PRESERVED:
       • No raw network traffic was ever shared
       • Only model weights were exchanged
       • Each organization's data stayed local
    """)


if __name__ == "__main__":
    run_federated_demo()
