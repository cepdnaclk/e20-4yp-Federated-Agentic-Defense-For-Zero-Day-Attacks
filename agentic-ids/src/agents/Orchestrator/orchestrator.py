import json
import agents.A1_triage_agent.triage_agent as triage_agent_final

class Orchestrator:
    """
    Central controller to manage alert data flows between 
    detection models and triage agents.
    """
    
    def __init__(self):
        self.triage_agent = triage_agent_final

    def process_autoencoder_input(self, json_data: dict):
        """
        Transforms raw Autoencoder JSON into a formatted string 
        and dispatches it to the Triage Agent.
        """
        # Extracting data with defaults
        score = json_data.get("anomaly_score", "Unknown")
        timestamp = json_data.get("timestamp", "Now")
        context = json_data.get("context", {})
        protocol = context.get("protocol", "Unknown")
        direction = context.get("direction", "Unknown")
        
        # Format the alert for LLM/Agent consumption
        formatted_alert = (
            f"Alert: Anomaly detected at {timestamp}. "
            f"Drift Score: {score}. "
            f"Traffic Context: Protocol {protocol}, Direction {direction}."
        )
        
        print(f"\n[Orchestrator] Pre-processed Alert: '{formatted_alert}'")
        
        # Dispatch to Agent
        return self.agent.process_anomaly(formatted_alert)

if __name__ == "__main__":
    # Initialize the Orchestrator
    orchestrator = Orchestrator()

    # Mock raw output from the Autoencoder model
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

    # Execute the pipeline
    orchestrator.process_autoencoder_input(raw_output)