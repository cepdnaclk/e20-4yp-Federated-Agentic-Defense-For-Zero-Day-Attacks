
from agents.Orchestrator.orchestrator import Orchestrator

def start_defense_system():
    print("--- Initializing Federated Agentic Defense ---")
    orchestrator = Orchestrator()

    # Simulated Autoencoder Data
    raw_output = {
        "anomaly_score": 0.94,
        "timestamp": "2024-05-20T10:00:00Z",
        "features": {
            'srcip': '175.45.176.0', 'sport': 39500, 
            'dstip': '149.171.126.15', 'dsport': '80', 
            'proto': 'tcp', 'state': 'FIN', 'dur': 0.177449, 
            'sbytes': 1214, 'dbytes': 268, 'sttl': 254, 'dttl': 252, 
            'service': 'http', 'Spkts': 10, 'Dpkts': 6,
            'tcprtt': 0.05198, 'ct_srv_src': 5
        }
    }

    orchestrator.process_autoencoder_input(raw_output)

if __name__ == "__main__":
    start_defense_system()