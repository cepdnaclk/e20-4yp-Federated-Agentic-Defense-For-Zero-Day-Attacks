# utils/monitoring_service.py
import json
import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
from pathlib import Path
import uuid


class MonitoringService:
    """
    Comprehensive monitoring and logging service for the Agentic IDS system.
    Tracks packet processing, threat detection accuracy, and system performance.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Log files
        self.packet_log_file = self.log_dir / "packet_processing.jsonl"
        self.accuracy_log_file = self.log_dir / "accuracy_metrics.csv"
        self.threat_actions_file = self.log_dir / "threat_actions.jsonl"
        self.system_metrics_file = self.log_dir / "system_metrics.jsonl"
        
        # Initialize CSV headers if file doesn't exist
        self._init_accuracy_csv()
        
        # In-memory metrics for real-time monitoring
        self.session_metrics = {
            "total_packets": 0,
            "anomalies_detected": 0,
            "threats_detected": 0,
            "zero_day_candidates": 0,
            "benign_classifications": 0,
            "suspicious_classifications": 0,
            "false_positives": 0,
            "true_positives": 0,
            "session_start": datetime.utcnow().isoformat()
        }
    
    def _init_accuracy_csv(self):
        """Initialize accuracy CSV file with headers if it doesn't exist"""
        if not self.accuracy_log_file.exists():
            with open(self.accuracy_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'flow_id', 'anomaly_score', 'prediction',
                    'triage_classification', 'triage_reasoning', 'target_pipeline',
                    'final_action', 'threat_type', 'confidence', 'processing_time_ms'
                ])
    
    def log_packet_received(self, flow_id: str, features: Dict[str, Any], 
                          timestamp: Optional[str] = None) -> str:
        """Log incoming packet data"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
            
        packet_log = {
            "event_type": "packet_received",
            "timestamp": timestamp,
            "flow_id": flow_id,
            "src_ip": features.get('srcip'),
            "src_port": features.get('sport'),
            "dst_ip": features.get('dstip'), 
            "dst_port": features.get('dsport'),
            "protocol": features.get('proto'),
            "service": features.get('service'),
            "state": features.get('state'),
            "bytes_sent": features.get('sbytes', 0),
            "bytes_received": features.get('dbytes', 0),
            "packets_sent": features.get('Spkts', 0),
            "packets_received": features.get('Dpkts', 0),
            "duration": features.get('dur', 0),
            "tcp_rtt": features.get('tcprtt', 0)
        }
        
        with self._lock:
            self.session_metrics["total_packets"] += 1
            
        self._append_to_jsonl(self.packet_log_file, packet_log)
        print(f"[MONITOR] Packet received: {flow_id} | {features.get('srcip')}:{features.get('sport')} -> {features.get('dstip')}:{features.get('dsport')}")
        
        return flow_id
    
    def log_inference_result(self, flow_id: str, inference_result: Dict[str, Any]):
        """Log inference service prediction results"""
        anomaly_score = inference_result.get('anomaly_score', 0)
        prediction = inference_result.get('prediction', 0)
        
        inference_log = {
            "event_type": "inference_result",
            "timestamp": datetime.utcnow().isoformat(),
            "flow_id": flow_id,
            "anomaly_score": anomaly_score,
            "prediction": prediction,
            "threshold_exceeded": prediction == 1,
            "reconstruction_error": sum(inference_result.get('reconstruction_error_vector', []))
        }
        
        with self._lock:
            if prediction == 1:
                self.session_metrics["anomalies_detected"] += 1
                
        self._append_to_jsonl(self.system_metrics_file, inference_log)
        
        if prediction == 1:
            print(f"[MONITOR] Anomaly detected: {flow_id} | Score: {anomaly_score:.4f}")
        
    def log_triage_result(self, flow_id: str, triage_result: Dict[str, Any], 
                         processing_time_ms: float = 0):
        """Log A1 triage agent classification results"""
        classification = triage_result.get('semantic_summary', '').split('.')[0]
        pipeline = triage_result.get('target_pipeline', 'Unknown')
        reasoning = triage_result.get('semantic_summary', '')
        
        triage_log = {
            "event_type": "triage_classification",
            "timestamp": datetime.utcnow().isoformat(),
            "flow_id": flow_id,
            "classification": classification,
            "target_pipeline": pipeline,
            "reasoning": reasoning,
            "processing_time_ms": processing_time_ms,
            "feature_vector": triage_result.get('feature_vector', []),
            "temporal_stats": triage_result.get('temporal_stats', {})
        }
        
        # Update session metrics based on classification
        with self._lock:
            if "BENIGN" in classification.upper():
                self.session_metrics["benign_classifications"] += 1
            elif "SUSPICIOUS" in classification.upper():
                self.session_metrics["suspicious_classifications"] += 1
                self.session_metrics["threats_detected"] += 1
            elif "ZERO-DAY" in classification.upper():
                self.session_metrics["zero_day_candidates"] += 1
                self.session_metrics["threats_detected"] += 1
        
        self._append_to_jsonl(self.system_metrics_file, triage_log)
        print(f"[MONITOR] Triage result: {flow_id} | {classification} -> {pipeline}")
        
        return classification
    
    def log_threat_action(self, flow_id: str, action_type: str, threat_details: Dict[str, Any]):
        """Log threat response actions taken by the system"""
        action_log = {
            "event_type": "threat_action",
            "timestamp": datetime.utcnow().isoformat(),
            "flow_id": flow_id,
            "action_type": action_type,  # block, alert, investigate, quarantine, etc.
            "threat_category": threat_details.get('likely_attack_category', 'Unknown'),
            "confidence": threat_details.get('confidence', 0),
            "mitigation_plan": threat_details.get('mitigation_plan', ''),
            "verification_status": threat_details.get('verification_status', ''),
            "kb_context_used": len(threat_details.get('kb_context', '')),
            "action_taken": True
        }
        
        with self._lock:
            if action_type in ['block', 'quarantine', 'investigate']:
                if threat_details.get('verification_status') == 'TRUE_THREAT':
                    self.session_metrics["true_positives"] += 1
                else:
                    self.session_metrics["false_positives"] += 1
        
        self._append_to_jsonl(self.threat_actions_file, action_log)
        print(f"[MONITOR] Threat action: {flow_id} | Action: {action_type} | Category: {threat_details.get('likely_attack_category', 'Unknown')}")
    
    def log_accuracy_metrics(self, flow_id: str, full_pipeline_result: Dict[str, Any]):
        """Log comprehensive accuracy metrics for the end-to-end pipeline"""
        timestamp = datetime.utcnow().isoformat()
        
        # Extract key metrics
        anomaly_score = full_pipeline_result.get('anomaly_score', 0)
        prediction = full_pipeline_result.get('prediction', 0)
        triage_class = full_pipeline_result.get('triage_classification', 'Unknown')
        triage_reasoning = full_pipeline_result.get('triage_reasoning', '')
        target_pipeline = full_pipeline_result.get('target_pipeline', 'Unknown')
        final_action = full_pipeline_result.get('final_action', 'None')
        threat_type = full_pipeline_result.get('threat_type', 'Unknown')
        confidence = full_pipeline_result.get('confidence', 0)
        processing_time = full_pipeline_result.get('processing_time_ms', 0)
        
        # Write to CSV
        with self._lock:
            with open(self.accuracy_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, flow_id, anomaly_score, prediction,
                    triage_class, triage_reasoning, target_pipeline,
                    final_action, threat_type, confidence, processing_time
                ])
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get current session metrics for real-time monitoring"""
        with self._lock:
            metrics = self.session_metrics.copy()
            
        # Calculate derived metrics
        total_packets = metrics["total_packets"]
        if total_packets > 0:
            metrics["anomaly_detection_rate"] = metrics["anomalies_detected"] / total_packets * 100
            metrics["threat_detection_rate"] = metrics["threats_detected"] / total_packets * 100
        else:
            metrics["anomaly_detection_rate"] = 0
            metrics["threat_detection_rate"] = 0
            
        # Calculate accuracy if we have predictions
        total_predictions = metrics["true_positives"] + metrics["false_positives"]
        if total_predictions > 0:
            metrics["precision"] = metrics["true_positives"] / total_predictions * 100
        else:
            metrics["precision"] = 0
            
        return metrics
    
    def print_session_summary(self):
        """Print a summary of the current monitoring session"""
        metrics = self.get_session_metrics()
        
        print("\n" + "="*70)
        print("AGENTIC IDS MONITORING SESSION SUMMARY")
        print("="*70)
        print(f"Session Start: {metrics['session_start']}")
        print(f"Current Time:  {datetime.utcnow().isoformat()}")
        print("\nPACKET PROCESSING:")
        print(f"  Total Packets Processed: {metrics['total_packets']}")
        print(f"  Anomalies Detected:      {metrics['anomalies_detected']} ({metrics['anomaly_detection_rate']:.2f}%)")
        print(f"  Threats Detected:        {metrics['threats_detected']} ({metrics['threat_detection_rate']:.2f}%)")
        
        print("\nTRIAGE CLASSIFICATIONS:")
        print(f"  Benign:                  {metrics['benign_classifications']}")
        print(f"  Suspicious:              {metrics['suspicious_classifications']}")
        print(f"  Zero-Day Candidates:     {metrics['zero_day_candidates']}")
        
        print("\nACCURACY METRICS:")
        print(f"  True Positives:          {metrics['true_positives']}")
        print(f"  False Positives:         {metrics['false_positives']}")
        print(f"  Precision:               {metrics['precision']:.2f}%")
        
        print("\nLOG FILES:")
        print(f"  Packet Log:              {self.packet_log_file}")
        print(f"  Accuracy Metrics:        {self.accuracy_log_file}")
        print(f"  Threat Actions:          {self.threat_actions_file}")
        print(f"  System Metrics:          {self.system_metrics_file}")
        print("="*70)
    
    def _append_to_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """Append data to JSONL file in thread-safe manner"""
        with self._lock:
            with open(file_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
    
    def export_metrics(self, export_path: Optional[str] = None) -> str:
        """Export all metrics to a comprehensive JSON report"""
        if export_path is None:
            export_path = self.log_dir / f"metrics_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        metrics = self.get_session_metrics()
        
        # Read recent logs for context
        recent_packets = self._read_recent_jsonl(self.packet_log_file, 10)
        recent_actions = self._read_recent_jsonl(self.threat_actions_file, 10)
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "session_metrics": metrics,
            "recent_packets": recent_packets,
            "recent_threat_actions": recent_actions,
            "log_files": {
                "packet_log": str(self.packet_log_file),
                "accuracy_log": str(self.accuracy_log_file),
                "threat_actions": str(self.threat_actions_file),
                "system_metrics": str(self.system_metrics_file)
            }
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"[MONITOR] Metrics exported to: {export_path}")
        return str(export_path)
    
    def _read_recent_jsonl(self, file_path: Path, n: int = 10) -> List[Dict]:
        """Read the last n lines from a JSONL file"""
        if not file_path.exists():
            return []
            
        lines = []
        try:
            with open(file_path, 'r') as f:
                all_lines = f.readlines()
                for line in all_lines[-n:]:
                    lines.append(json.loads(line.strip()))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
            
        return lines


# Global monitoring service instance
monitoring_service = None

def get_monitoring_service() -> MonitoringService:
    """Get or create the global monitoring service instance"""
    global monitoring_service
    if monitoring_service is None:
        log_dir = os.getenv("MONITORING_LOG_DIR", "logs")
        monitoring_service = MonitoringService(log_dir)
    return monitoring_service


def log_packet_processing(flow_id: str, features: Dict[str, Any]) -> str:
    """Convenience function to log packet processing"""
    return get_monitoring_service().log_packet_received(flow_id, features)


def log_inference_prediction(flow_id: str, inference_result: Dict[str, Any]):
    """Convenience function to log inference predictions"""
    get_monitoring_service().log_inference_result(flow_id, inference_result)


def log_triage_classification(flow_id: str, triage_result: Dict[str, Any], 
                            processing_time_ms: float = 0) -> str:
    """Convenience function to log triage classifications"""
    return get_monitoring_service().log_triage_result(flow_id, triage_result, processing_time_ms)


def log_threat_response(flow_id: str, action_type: str, threat_details: Dict[str, Any]):
    """Convenience function to log threat response actions"""
    get_monitoring_service().log_threat_action(flow_id, action_type, threat_details)


def print_monitoring_summary():
    """Convenience function to print monitoring summary"""
    get_monitoring_service().print_session_summary()


if __name__ == "__main__":
    # Test the monitoring service
    monitor = MonitoringService()
    
    # Test packet logging
    test_features = {
        'srcip': '192.168.1.100',
        'sport': 12345,
        'dstip': '10.0.0.1', 
        'dsport': 80,
        'proto': 'tcp',
        'service': 'http'
    }
    
    flow_id = monitor.log_packet_received("test-flow-1", test_features)
    
    # Test inference logging
    inference_result = {
        'anomaly_score': 0.85,
        'prediction': 1,
        'reconstruction_error_vector': [0.1, 0.2, 0.3]
    }
    monitor.log_inference_result(flow_id, inference_result)
    
    # Test triage logging
    triage_result = {
        'semantic_summary': 'Classified as SUSPICIOUS. High anomaly score detected.',
        'target_pipeline': 'CorrectiveRAG',
        'feature_vector': [0.1, 0.8, 0.9]
    }
    monitor.log_triage_result(flow_id, triage_result, 150.5)
    
    # Print summary
    monitor.print_session_summary()