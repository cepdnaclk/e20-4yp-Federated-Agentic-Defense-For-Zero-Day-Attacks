"""
Simulate Federation Rounds by sending updates to the FL Server.
This script simulates multiple agents sending model updates and generates
real privacy metrics that can be visualized.
"""

import base64
import io
import json
import random
import time
import numpy as np
import requests

FL_SERVER_URL = "http://localhost:9090"


def encode_weights(weights):
    """Encode numpy weights to base64"""
    encoded = []
    for arr in weights:
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return encoded


def generate_model_weights():
    """Generate simulated autoencoder weights"""
    # Typical autoencoder architecture: encoder + decoder
    weights = [
        np.random.randn(43, 32).astype(np.float32) * 0.1,   # Input to hidden
        np.random.randn(32,).astype(np.float32) * 0.1,      # Bias
        np.random.randn(32, 16).astype(np.float32) * 0.1,   # Hidden to latent
        np.random.randn(16,).astype(np.float32) * 0.1,      # Bias
        np.random.randn(16, 32).astype(np.float32) * 0.1,   # Latent to hidden
        np.random.randn(32,).astype(np.float32) * 0.1,      # Bias
        np.random.randn(32, 43).astype(np.float32) * 0.1,   # Hidden to output
        np.random.randn(43,).astype(np.float32) * 0.1,      # Bias
    ]
    return weights


def generate_signatures(num_sigs, dim=16):
    """Generate anomaly signatures with embeddings"""
    signatures = []
    for _ in range(num_sigs):
        embedding = np.random.randn(dim).astype(np.float32).tolist()
        recon_error = abs(random.gauss(0.8, 0.3))
        signatures.append({
            "embedding": embedding,
            "recon_error": recon_error
        })
    return signatures


def simulate_agent_update(agent_id, sample_count, include_weights=True, round_end=False):
    """Send a simulated agent update to the FL server"""
    payload = {
        "agent_id": agent_id,
        "sample_count": sample_count,
        "anomaly_stats": {
            "total_packets": sample_count,
            "anomalies_detected": random.randint(5, 50),
            "avg_recon_error": random.uniform(0.3, 1.2)
        },
        "round_end": round_end
    }
    
    if include_weights:
        weights = generate_model_weights()
        payload["weights"] = encode_weights(weights)
    
    # Add signatures
    num_sigs = random.randint(5, 25)
    payload["signatures"] = generate_signatures(num_sigs)
    
    try:
        response = requests.post(
            f"{FL_SERVER_URL}/api/submit_update",
            json=payload,
            timeout=10
        )
        return response.json()
    except Exception as e:
        print(f"  Error: {e}")
        return None


def simulate_federation(num_rounds=15, agents=None):
    """Run a simulated federation with multiple agents"""
    if agents is None:
        agents = ["agent_org_a", "agent_org_b", "agent_org_c", "agent_org_d"]
    
    print("=" * 60)
    print("FEDERATED LEARNING SIMULATION")
    print("=" * 60)
    print(f"Simulating {num_rounds} rounds with {len(agents)} agents")
    print(f"FL Server: {FL_SERVER_URL}")
    print()
    
    # Check FL server is running
    try:
        r = requests.get(f"{FL_SERVER_URL}/api/privacy/summary", timeout=5)
        print(f"FL Server Status: OK")
    except Exception as e:
        print(f"Error: FL Server not reachable at {FL_SERVER_URL}")
        print(f"Make sure the FL server is running")
        return
    
    print()
    
    for round_num in range(1, num_rounds + 1):
        print(f"--- Round {round_num}/{num_rounds} ---")
        
        # Select participating agents (not all agents participate every round)
        num_participants = random.randint(2, len(agents))
        participating = random.sample(agents, num_participants)
        
        for i, agent_id in enumerate(participating):
            sample_count = random.randint(80, 200)
            # Last agent in round triggers aggregation
            is_round_end = (i == len(participating) - 1)
            
            result = simulate_agent_update(
                agent_id=agent_id,
                sample_count=sample_count,
                include_weights=True,
                round_end=is_round_end
            )
            
            if result:
                status = result.get("status", "unknown")
                privacy = result.get("privacy_metrics", {})
                
                info_parts = [f"{agent_id}: {sample_count} samples"]
                if privacy:
                    info_parts.append(f"ε={privacy.get('epsilon', 0):.4f}")
                
                print(f"  {' | '.join(info_parts)}")
            else:
                print(f"  {agent_id}: Failed to submit")
        
        # Small delay between rounds
        time.sleep(0.5)
    
    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    # Get final summary
    try:
        r = requests.get(f"{FL_SERVER_URL}/api/privacy/summary", timeout=5)
        summary = r.json()
        print("\nPrivacy Metrics Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error getting summary: {e}")


if __name__ == "__main__":
    import sys
    
    num_rounds = 15
    if len(sys.argv) > 1:
        try:
            num_rounds = int(sys.argv[1])
        except ValueError:
            pass
    
    simulate_federation(num_rounds=num_rounds)
