import json
import time
# Assuming triage_agent_final has a class or method 'process_anomaly'
import agents.A1_triage_agent.triage_agent as triage_agent
import agents.A2_suspicious_agent.suspicious_agent as suspicious_agent
import agents.A4_action_agent.action_agent as action_agent
import uuid 
# FEDERATION COMMENTED OUT FOR TESTING
# from agents.A3_federation_agent.async_sender import AsyncSignatureSender
import threading
from datetime import datetime
import os

# Import monitoring service
from utils.monitoring_service import log_triage_classification


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

        # FEDERATION COMMENTED OUT FOR TESTING PHASE
        # Federated server configuration from environment
        # self.org_id = os.getenv("ORG_ID", "org-unknown")
        # self.fl_server_url = os.getenv("FL_SERVER_URL", "http://localhost:9090")
        # self.signature_sender = AsyncSignatureSender(self.fl_server_url)
        
        print("[Orchestrator] Initialized - Federation disabled for testing phase")
        print("[Orchestrator] A4 Action Agent loaded for threat response")

    def process_autoencoder_input(self, json_data: dict, flow_id: str = None):
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
        
    def process_autoencoder_input(self, json_data: dict, flow_id: str = None):
        """
        Transforms raw Autoencoder JSON into behavioral insights
        for the Triage Agent.
        """
        
        processing_start = time.time()
        
        if flow_id is None:
            flow_id = json_data.get("flow_id", str(uuid.uuid4()))

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
        
        print(f"[Orchestrator] Processing flow {flow_id} with anomaly score {score:.4f}")
        
        # Process through triage agent
        triage_result = self.A1_agent.process_anomaly(formatted_alert)
        
        # Log triage results with monitoring
        processing_time = (time.time() - processing_start) * 1000  # Convert to ms
        classification = log_triage_classification(flow_id, triage_result, processing_time)

        # Initialize action response tracking
        action_response = None

        # Route to appropriate pipeline
        if triage_result["target_pipeline"] == "CorrectiveRAG":
            print(f"[Orchestrator] Routing {flow_id} to Suspicious Agent for verification")
            suspicious_result = self.A2_agent.handle_suspicious_alert(triage_result)
            
            # Generate action response for verified threats
            if suspicious_result.get("verification_status") == "TRUE_THREAT":
                print(f"[Orchestrator] {flow_id} confirmed as threat - generating action plan")
                action_response = action_agent.process_threat_response(triage_result, json_data, flow_id)
            
            # Log threat action
            from utils.monitoring_service import log_threat_response
            action_type = "investigate" if suspicious_result.get("verification_status") == "TRUE_THREAT" else "monitor"
            log_threat_response(flow_id, action_type, suspicious_result)

        elif triage_result["target_pipeline"] == "AgenticRAG":
            print(f"[Orchestrator] {flow_id} classified as BENIGN - logging for compliance")
            
        elif triage_result["target_pipeline"] == "AdaptiveRAG":
            print(f"[Orchestrator] {flow_id} classified as ZERO-DAY CANDIDATE - generating emergency response")
            
            # Generate immediate action plan for zero-day candidates  
            action_response = action_agent.process_threat_response(triage_result, json_data, flow_id)
            
            # Log as high-priority threat
            from utils.monitoring_service import log_threat_response
            threat_details = {
                "likely_attack_category": "Zero-Day Candidate",
                "confidence": score,
                "verification_status": "REQUIRES_INVESTIGATION", 
                "mitigation_plan": f"Emergency response initiated - Priority {action_response.get('action_plan', {}).get('priority', 5)}",
                "action_response": action_response
            }
            log_threat_response(flow_id, "emergency_response", threat_details)
        

        ## FEDERATION FUNCTIONALITY COMMENTED OUT FOR TESTING
        ## Debug purpose: Send sample signature asynchronously
        # sample_signature = {
        #     "signature_id": str(uuid.uuid4()),
        #     "feature_deviation": {
        #         "conn_rate": 4.2,
        #         "dst_entropy": 3.1,
        #         "avg_pkt_size": -1.7
        #     },
        #     "confidence": 0.93,
        #     "frequency": 5,
        #     "time_window": "10s",
        #     "agent_id": self.org_id,
        #     "timestamp": datetime.utcnow().isoformat()
        # }

        # print(f"[Orchestrator] Org {self.org_id} enqueuing signature to {self.fl_server_url}")
        # self.signature_sender.enqueue(sample_signature)
        # time.sleep(10)

        return {
            "flow_id": flow_id,
            "processing_time_ms": processing_time,
            "triage_result": triage_result,
            "classification": classification,
            "anomaly_score": score,
            "action_response": action_response  # Include action response if generated
        }

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