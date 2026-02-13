# test_monitoring.py
"""
Test script to demonstrate the monitoring and logging functionality 
of the Agentic IDS system.

This script sends sample packets to the /detect endpoint and shows
how the monitoring system tracks:
- Packet reception
- Anomaly detection 
- Threat classification
- Actions taken
"""

import requests
import json
import time
import random

# Configuration
SERVER_URL = "http://localhost:5000"
TEST_PACKETS = [
    # Normal web traffic
    {
        "flow_id": "test-normal-001",
        "features": {
            "srcip": "192.168.1.100",
            "sport": 45678, 
            "dstip": "8.8.8.8",
            "dsport": "80",
            "proto": "tcp",
            "state": "FIN",
            "dur": 0.123,
            "sbytes": 512,
            "dbytes": 1024,
            "sttl": 64,
            "dttl": 64,
            "service": "http",
            "Spkts": 3,
            "Dpkts": 3,
            "tcprtt": 0.025,
            "ct_srv_src": 1
        }
    },
    # Suspicious: High anomaly score traffic 
    {
        "flow_id": "test-suspicious-001",
        "features": {
            "srcip": "10.0.0.1",
            "sport": 12345,
            "dstip": "192.168.1.10", 
            "dsport": "22",
            "proto": "tcp",
            "state": "INT",
            "dur": 60.5,
            "sbytes": 15000,
            "dbytes": 200,
            "sttl": 255,
            "dttl": 64,
            "service": "ssh",
            "Spkts": 50,
            "Dpkts": 10,
            "tcprtt": 0.15,
            "ct_srv_src": 25
        }
    },
    # Potential zero-day: Unusual port, high data transfer
    {
        "flow_id": "test-zeroday-001", 
        "features": {
            "srcip": "175.45.176.0",
            "sport": 39500,
            "dstip": "149.171.126.15",
            "dsport": "8888",
            "proto": "tcp",
            "state": "FIN",
            "dur": 180.5,
            "sbytes": 50000,
            "dbytes": 5000,
            "sttl": 254,
            "dttl": 252,
            "service": "unknown",
            "Spkts": 200,
            "Dpkts": 100, 
            "tcprtt": 0.08,
            "ct_srv_src": 1
        }
    },
    # High frequency connection pattern
    {
        "flow_id": "test-scanning-001",
        "features": {
            "srcip": "203.0.113.200",
            "sport": random.randint(30000, 60000),
            "dstip": "192.168.1.50", 
            "dsport": random.choice(["21", "22", "23", "25", "53", "80", "110", "443"]),
            "proto": "tcp",
            "state": "REJ",
            "dur": 0.001,
            "sbytes": 40,
            "dbytes": 0,
            "sttl": 64,
            "dttl": 0,
            "service": "unknown",
            "Spkts": 1,
            "Dpkts": 0,
            "tcprtt": 0.0,
            "ct_srv_src": 0
        }
    }
]

def test_health():
    """Test if the server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Agentic IDS server is healthy")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def send_test_packet(packet_data):
    """Send a test packet to the /detect endpoint"""
    try:
        response = requests.post(
            f"{SERVER_URL}/detect",
            json=packet_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ Packet {packet_data['flow_id']} sent successfully")
            return True
        else:
            print(f"❌ Failed to send packet {packet_data['flow_id']}: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Error sending packet: {e}")
        return False

def get_monitoring_status():
    """Get current monitoring metrics"""
    try:
        response = requests.get(f"{SERVER_URL}/monitoring", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get monitoring status: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"❌ Error getting monitoring status: {e}")
        return None

def print_monitoring_summary():
    """Request monitoring summary from server"""
    try:
        response = requests.get(f"{SERVER_URL}/monitoring/summary", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*50)
            print("SERVER MONITORING SUMMARY")
            print("="*50)
            return result['metrics']
        else:
            print(f"❌ Failed to get monitoring summary: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"❌ Error getting monitoring summary: {e}")
        return None

def main():
    print("🚀 Starting Agentic IDS Monitoring Test")
    print("="*50)
    
    # Test server health
    if not test_health():
        print("Please start the Agentic IDS server first with: python src/main.py")
        return
    
    print("\n📊 Initial monitoring status:")
    initial_metrics = get_monitoring_status()
    if initial_metrics:
        print(f"   Total packets processed: {initial_metrics.get('total_packets', 0)}")
        print(f"   Anomalies detected: {initial_metrics.get('anomalies_detected', 0)}")
        print(f"   Threats detected: {initial_metrics.get('threats_detected', 0)}")
    
    print("\n🎯 Sending test packets...")
    print("-" * 30)
    
    # Send test packets with delays to show processing
    for i, packet in enumerate(TEST_PACKETS):
        print(f"\n📦 Sending packet {i+1}/4: {packet['flow_id']}")
        print(f"   Type: {packet['features']['srcip']} -> {packet['features']['dstip']}:{packet['features']['dsport']}")
        
        if send_test_packet(packet):
            # Wait a bit for processing
            time.sleep(2)
            
            # Show updated metrics
            metrics = get_monitoring_status()
            if metrics:
                print(f"   📈 Total packets: {metrics.get('total_packets', 0)}")
                print(f"   🚨 Anomalies: {metrics.get('anomalies_detected', 0)}")
                print(f"   ⚠️  Threats: {metrics.get('threats_detected', 0)}")
        
        # Small delay between packets
        time.sleep(1)
    
    # Final summary
    print("\n⏭️  Waiting for final processing...")
    time.sleep(3)
    
    final_metrics = print_monitoring_summary()
    
    if final_metrics:
        print(f"\n📋 TEST SUMMARY:")
        print(f"   Packets sent: {len(TEST_PACKETS)}")
        print(f"   Packets processed: {final_metrics.get('total_packets', 0)}")
        print(f"   Anomaly detection rate: {final_metrics.get('anomaly_detection_rate', 0):.1f}%")
        print(f"   Threat detection rate: {final_metrics.get('threat_detection_rate', 0):.1f}%")
        print(f"   Zero-day candidates: {final_metrics.get('zero_day_candidates', 0)}")
        
        log_dir = "logs"  # Default log directory
        print(f"\n📁 Check these log files for detailed analysis:")
        print(f"   📄 {log_dir}/packet_processing.jsonl")
        print(f"   📊 {log_dir}/accuracy_metrics.csv")
        print(f"   🚨 {log_dir}/threat_actions.jsonl")
        print(f"   📈 {log_dir}/system_metrics.jsonl")
    
    print("\n✅ Test completed! The monitoring system is now tracking all IDS activity.")

if __name__ == "__main__":
    main()