# ==============================================================================
# Federated Learning Simulation Script (PowerShell)
# ==============================================================================
# This script demonstrates how to run federated learning for the Network
# Defense IDS system with multiple clients on Windows.
#
# Usage:
#   .\scripts\run_federated_simulation.ps1 [-NumClients 3] [-NumRounds 10]
#
# Example:
#   .\scripts\run_federated_simulation.ps1 -NumClients 3 -NumRounds 10
# ==============================================================================

param(
    [int]$NumClients = 3,
    [int]$NumRounds = 10,
    [string]$ServerAddress = "localhost:8080",
    [string]$CheckpointDir = ".\federated_checkpoints"
)

Write-Host "=============================================="
Write-Host "Network Defense Federated Learning Simulation"
Write-Host "=============================================="
Write-Host "Clients: $NumClients"
Write-Host "Rounds: $NumRounds"
Write-Host "Server: $ServerAddress"
Write-Host "=============================================="

# Create checkpoint directory
if (-not (Test-Path $CheckpointDir)) {
    New-Item -ItemType Directory -Path $CheckpointDir | Out-Null
    Write-Host "[+] Created checkpoint directory: $CheckpointDir"
}

# Store process objects
$processes = @()

try {
    # Start server
    Write-Host "[*] Starting Federated Server..."
    $serverProcess = Start-Process -FilePath "python" -ArgumentList @(
        "-m", "federated.server",
        "--address", "[::]:8080",
        "--rounds", $NumRounds,
        "--min-clients", $NumClients,
        "--save-path", $CheckpointDir,
        "--log-level", "INFO"
    ) -PassThru -NoNewWindow
    $processes += $serverProcess
    Write-Host "[+] Server started (PID: $($serverProcess.Id))"
    
    # Wait for server to initialize
    Start-Sleep -Seconds 3

    # Start clients
    for ($i = 1; $i -le $NumClients; $i++) {
        Write-Host "[*] Starting Client $i..."
        
        $clientCode = @"
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
    client_id='client_$i',
)

# Start client
fl.client.start_numpy_client(server_address='$ServerAddress', client=client)
"@
        
        $clientProcess = Start-Process -FilePath "python" -ArgumentList "-c", $clientCode -PassThru -NoNewWindow
        $processes += $clientProcess
        Write-Host "[+] Client $i started (PID: $($clientProcess.Id))"
        
        # Stagger client starts
        Start-Sleep -Seconds 1
    }

    # Wait for server to complete
    Write-Host "[*] Waiting for federated training to complete..."
    $serverProcess.WaitForExit()
    Write-Host "[+] Federated training complete!"

    # Show results
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "Results"
    Write-Host "=============================================="
    Write-Host "Checkpoints saved to: $CheckpointDir"
    Get-ChildItem $CheckpointDir | Format-Table Name, Length, LastWriteTime

} finally {
    # Cleanup
    Write-Host "[*] Cleaning up processes..."
    foreach ($proc in $processes) {
        if (-not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "[+] Cleanup complete"
}
