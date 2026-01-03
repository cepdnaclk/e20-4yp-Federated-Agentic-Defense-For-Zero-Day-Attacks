import json
# import triage_agent1 as triage_agent_final 
import triage_agent as triage_agent_final

#1.Handler Function Here
def handle_autoencoder_alert(json_data: dict):
    """
    Cleaner function to convert Autoencoder JSON into a Triage-friendly format.
    """
    score = json_data.get("anomaly_score", "Unknown")
    timestamp = json_data.get("timestamp", "Now")
    context = json_data.get("context", {})
    protocol = context.get("protocol", "Unknown")
    direction = context.get("direction", "Unknown")
    
    #Human Readable String
    formatted_alert = (
        f"Alert: Anomaly detected at {timestamp}. "
        f"Drift Score: {score}. "
        f"Traffic Context: Protocol {protocol}, Direction {direction}."
    )
    
    # Send string to the Triage Agent
    print(f"\n[Pre-Process] converted JSON to: '{formatted_alert}'")
    
    # Call the Agent's public API
    triage_agent_final.process_anomaly(formatted_alert)

if __name__ == "__main__":
    # autoencoder logic
    
    # Example raw output from your model
    raw_output = {
        "flow_id": "uuid-1234",
        "timestamp": "2024-05-20T10:00:00Z",
        "anomaly_score": 0.91,
        "feature_vector": [0.12, 1.33], 
        "context": {
            "protocol": "QUIC",
            "direction": "outbound"
        }
    }

    handle_autoencoder_alert(raw_output)