#!/usr/bin/env python3
"""
Test script for A4 Action Agent - Threat Response System
Demonstrates quick threat analysis and action planning capabilities
"""

import sys
import json
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from agents.A4_action_agent.action_agent import ActionAgent, process_threat_response


def test_action_agent():
    """Test the A4 Action Agent with sample threat data"""
    
    print("=== A4 ACTION AGENT TEST ===")
    
    # Sample triage result (simulating A1 agent output)
    triage_result = {
        "target_pipeline": "AdaptiveRAG",
        "confidence_score": 0.92,
        "behavior_analysis": "Unusual connection pattern detected. High connection frequency with abnormal packet sizes suggests potential data exfiltration attempt.",
        "threat_indicators": ["high_connection_rate", "unusual_packet_size", "external_destination"]
    }
    
    # Sample flow data (simulating autoencoder output)
    flow_data = {
        "anomaly_score": 0.94,
        "timestamp": "2024-05-20T10:00:00Z", 
        "features": {
            "srcip": "192.168.1.100",  # Internal workstation
            "sport": 49152,
            "dstip": "203.0.113.42",   # External IP
            "dsport": "443",
            "proto": "tcp",
            "state": "FIN",
            "service": "https",
            "sbytes": 2048,
            "Spkts": 15,
            "ct_srv_src": 8,
            "tcprtt": 0.12
        }
    }
    
    flow_id = "test_flow_001"
    
    print(f"Processing threat flow: {flow_id}")
    print(f"Anomaly Score: {flow_data['anomaly_score']:.3f}")
    print(f"Source: {flow_data['features']['srcip']} -> Destination: {flow_data['features']['dstip']}")
    print()
    
    # Test the action agent
    try:
        result = process_threat_response(triage_result, flow_data, flow_id)
        
        print("=== THREAT ANALYSIS RESULT ===")
        threat = result["threat_summary"]
        print(f"Threat Type: {threat['threat_type']}")
        print(f"Severity: {threat['severity']}")  
        print(f"Target Asset: {threat['target_asset']}")
        print(f"Attack Vector: {threat['attack_vector']}")
        print(f"Reasoning: {threat['reasoning']}")
        print()
        
        print("=== ACTION PLAN ===")
        actions = result["action_plan"]
        print(f"Priority Level: {actions['priority']}/5")
        
        if actions['immediate_actions']:
            print("\n🚨 IMMEDIATE ACTIONS:")
            for action in actions['immediate_actions']:
                print(f"  • {action}")
        
        if actions['containment_actions']:
            print("\n🔒 CONTAINMENT ACTIONS:")
            for action in actions['containment_actions']:
                print(f"  • {action}")
                
        if actions['monitoring_actions']:
            print("\n📊 MONITORING ACTIONS:")
            for action in actions['monitoring_actions']:
                print(f"  • {action}")
                
        if actions['notification_actions']:
            print("\n📢 NOTIFICATIONS:")
            for action in actions['notification_actions']:
                print(f"  • {action}")
        
        print("\n=== EXECUTION SUMMARY ===")
        exec_result = result["execution_result"]
        print(f"Response ID: {exec_result['response_id']}")
        print(f"Actions Planned: {exec_result['actions_planned']}")
        print(f"Notifications: {exec_result['notifications_sent']}")
        print(f"Status: {exec_result['execution_status']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_network_config():
    """Test network configuration loading"""
    
    print("\n=== NETWORK CONFIG TEST ===")
    
    try:
        agent = ActionAgent()
        config = agent.network_config
        
        print("✅ Network configuration loaded successfully")
        print(f"Organization: {config['organization']['name']}")
        print(f"Network Segments: {len(config['organization']['network_segments'])}")
        print(f"Critical Assets: {len(config['organization']['assets']['critical_servers'])}")
        print(f"Response Levels: {list(config['security_policies']['response_levels'].keys())}")
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")


if __name__ == "__main__":
    test_network_config()
    test_action_agent()
    
    print("\n🎯 A4 Action Agent test completed!")
    print("Check logs/action_responses.jsonl for detailed response records.")