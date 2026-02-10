import json
import time
# Assuming triage_agent_final has a class or method 'process_anomaly'
import agents.A1_triage_agent.triage_agent as triage_agent
import agents.A2_suspicious_agent.suspicious_agent as suspicious_agent
import uuid 
from agents.A3_federation_agent.async_sender import AsyncSignatureSender
import threading
from datetime import datetime
import os


class Orchestrator:
    """
    Central controller to manage alert data flows. 
    Specifically tuned for Zero-Day behavioral detection.
    """

    
    
    def __init__(self):
        # Ensure you are calling the correct class instance here

        # classification, 
        self.A1_agent = triage_agent   

        # known attack mitigation 
        self.A2_agent = suspicious_agent

        # 

    def process_autoencoder_input(self, json_data: dict):
        """
        Transforms raw Autoencoder JSON into behavioral insights
        for the Triage Agent.
        """

        # 1. Extract Metadata
        score = json_data.get("anomaly_score", 0.0)
        timestamp = json_data.get("timestamp", "Unknown Time")
        features = json_data.get("features", {})

        # 2. Extract Basic Network Identifiers
        network_info = {
            "src": f"{features.get('srcip')}:{features.get('sport')}",
            "dst": f"{features.get('dstip')}:{features.get('dsport')}",
            "proto": features.get("proto", "TCP").upper(),
            "service": features.get("service", "General"),
            "state": features.get("state", "Unknown")
        }

        # 3. Behavioral Feature Engineering (Crucial for Zero-Day)
        # Intensity: How frequent is this connection pattern?
        intensity = features.get("ct_srv_src", 0) 
        
        # Efficiency: Packet size consistency (Zero-day exploits often have abnormal ratios)
        spkts = max(features.get("Spkts", 1), 1)
        s_efficiency = features.get("sbytes", 0) / spkts

        # Network TTL Profile: Detects spoofing or unusual routing
        ttl_path = f"Source TTL: {features.get('sttl')} | Dest TTL: {features.get('dttl')}"

        # Latency Profile: High tcprtt can indicate MITM or scanning lag
        latency = features.get("tcprtt", 0)

        # 4. Construct the Behavioral Narrative for the LLM
        # We provide context, not just data.
        formatted_alert = (
            f"--- ZERO-DAY THREAT ALERT ---\n"
            f"Timestamp: {timestamp}\n"
            f"Anomaly Confidence: {score:.2f}\n"
            f"Flow: {network_info['src']} -> {network_info['dst']} ({network_info['proto']})\n"
            f"Service: {network_info['service']} | State: {network_info['state']}\n"
            f"\nBEHAVIORAL INDICATORS:\n"
            f"- Traffic Intensity: {intensity} concurrent connections (High suggests automated attack)\n"
            f"- Payload Profile: {s_efficiency:.2f} bytes/packet (Check for exfiltration/buffer overflow)\n"
            f"- Network Path: {ttl_path}\n"
            f"- TCP Latency: {latency}s\n"
            f"----------------------------"
        )
        
        # print(f"\n[Orchestrator] Dispatched Behavioral Alert:\n{formatted_alert}")
        triage_result = self.A1_agent.process_anomaly(formatted_alert)

        if triage_result["target_pipeline"] == "CorrectiveRAG":
            self.A2_agent.handle_suspicious_alert(triage_result)

        if triage_result["target_pipeline"] == "AdaptiveRAG":
            print("[Orchestrator] AdaptiveRAG pipeline selected - currently not implemented.")
        

        ## Testing
        ## Debug purpose: Send sample signature asynchronously
        fl_server_url = os.getenv("FL_SERVER_URL", "http://localhost:8000")
        sender = AsyncSignatureSender(fl_server_url)
        sample_signature = {
            "signature_id": str(uuid.uuid4()),
            "feature_deviation": {
                "conn_rate": 4.2,
                "dst_entropy": 3.1,
                "avg_pkt_size": -1.7
            },
            "confidence": 0.93,
            "frequency": 5,
            "time_window": "10s",
            "agent_id": "agent_async_01",
            "timestamp": datetime.utcnow().isoformat()
        }

        sender.enqueue(sample_signature)
        time.sleep(10)

        # Dispatch to Agent
        # return self.agent.process_anomaly(formatted_alert)

if __name__ == "__main__":
    orchestrator = Orchestrator()

    # The dataset features you provided earlier
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