#!/bin/bash
# ==============================================================================
# Federated Learning Simulation Script
# ==============================================================================
# This script demonstrates how to run federated learning for the Network
# Defense IDS system with multiple clients.
#
# Usage:
#   ./scripts/run_federated_simulation.sh [num_clients] [num_rounds]
#
# Example:
#   ./scripts/run_federated_simulation.sh 3 10
# ==============================================================================

set -e

# Default parameters
NUM_CLIENTS=${1:-3}
NUM_ROUNDS=${2:-10}
SERVER_ADDRESS="localhost:8080"
CHECKPOINT_DIR="./federated_checkpoints"

echo "=============================================="
echo "Network Defense Federated Learning Simulation"
echo "=============================================="
echo "Clients: $NUM_CLIENTS"
echo "Rounds: $NUM_ROUNDS"
echo "Server: $SERVER_ADDRESS"
echo "=============================================="

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Function to start server
start_server() {
    echo "[*] Starting Federated Server..."
    python -m federated.server \
        --address "[::]:8080" \
        --rounds $NUM_ROUNDS \
        --min-clients $NUM_CLIENTS \
        --save-path $CHECKPOINT_DIR \
        --log-level INFO &
    SERVER_PID=$!
    echo "[+] Server started (PID: $SERVER_PID)"
    sleep 3  # Wait for server to initialize
}

# Function to start a client
start_client() {
    CLIENT_ID=$1
    echo "[*] Starting Client $CLIENT_ID..."
    python -c "
import numpy as np
import flwr as fl
from federated.client import NetworkDefenseClient
from agents.models.autoencoder import AnomalyAutoencoder

# Create model
model = AnomalyAutoencoder(input_dim=40, latent_dim=8, hidden_dims=[32, 16])

# Generate synthetic data for demo
X_train = np.random.randn(1000, 40).astype(np.float32)
X_val = np.random.randn(200, 40).astype(np.float32)

# Create client
client = NetworkDefenseClient(
    autoencoder=model,
    train_data=(X_train, None),
    val_data=(X_val, None),
    training_config={'autoencoder_epochs': 3, 'autoencoder_batch_size': 128},
    client_id='client_$CLIENT_ID',
)

# Start client
fl.client.start_numpy_client(server_address='$SERVER_ADDRESS', client=client)
" &
    CLIENT_PIDS[$CLIENT_ID]=$!
    echo "[+] Client $CLIENT_ID started (PID: ${CLIENT_PIDS[$CLIENT_ID]})"
}

# Cleanup function
cleanup() {
    echo "[*] Cleaning up processes..."
    kill $SERVER_PID 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    echo "[+] Cleanup complete"
}

trap cleanup EXIT

# Main execution
declare -A CLIENT_PIDS

# Start server
start_server

# Start clients
for i in $(seq 1 $NUM_CLIENTS); do
    start_client $i
    sleep 1  # Stagger client starts
done

# Wait for server to complete
echo "[*] Waiting for federated training to complete..."
wait $SERVER_PID
echo "[+] Federated training complete!"

# Show results
echo ""
echo "=============================================="
echo "Results"
echo "=============================================="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
ls -la $CHECKPOINT_DIR
