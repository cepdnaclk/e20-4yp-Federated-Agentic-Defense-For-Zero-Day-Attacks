# Agentic IDS Monitoring and Accuracy Tracking

This enhanced version of the Agentic IDS includes comprehensive monitoring and logging to track system performance, threat detection accuracy, and packet processing metrics.

## What's New

### 🚨 Federation Disabled for Testing
- Federation functionality has been commented out in the orchestrator
- `.env` file updated to disable federation settings
- Focus is now on local monitoring and accuracy measurement

### 📊 Comprehensive Monitoring System
The new monitoring system tracks:

**Packet Processing:**
- Incoming packet details (source, destination, protocol, etc.)
- Processing times and throughput
- Network flow characteristics

**Threat Detection:**
- Anomaly scores from the autoencoder
- Triage agent classifications (BENIGN/SUSPICIOUS/ZERO-DAY)
- Target pipeline routing decisions
- Final actions taken by the system

**Accuracy Metrics:**
- True positives vs false positives
- Precision calculations
- Detection rates by threat category
- Classification reasoning and confidence

## New Features

### 1. Monitoring Service (`src/utils/monitoring_service.py`)
Centralized logging and metrics collection with:
- **Real-time metrics** tracked in memory
- **Persistent logs** saved to JSON and CSV files  
- **Thread-safe operations** for concurrent processing
- **Export functionality** for detailed reporting

### 2. Enhanced Endpoints
New monitoring endpoints added to the main server:

```
GET /monitoring           - Get current metrics JSON
GET /monitoring/summary   - Print and return full summary
```

### 3. Improved Logging Output
All components now provide detailed, structured logging:

**Inference Service:**
```
[INFERENCE] ANOMALY DETECTED:
            Score: 0.856234 (threshold: 0.5)
            Source: 192.168.1.100:45678
            Dest: 10.0.0.1:22
            Protocol: tcp
            Service: ssh
```

**Triage Agent:**
```
[TRIAGE] Flow Classification:
         Decision: SUSPICIOUS
         Pipeline: CorrectiveRAG  
         Reasoning: High anomaly score with unusual SSH activity
         Context Retrieved: 1247 characters
```

**Suspicious Agent:**
```
[Suspicious Agent] Analysis Results:
                   Status: TRUE_THREAT
                   Attack Category: Brute Force Attack
                   Confidence: 0.87
                   Action Plan: Block source IP and alert SOC
                   KB Context Used: 892 chars
```

**Orchestrator:**
```
[Orchestrator] Processing flow abc-123 with anomaly score 0.8542
[Orchestrator] Routing abc-123 to Suspicious Agent for verification
```

## Log Files Generated

The monitoring system creates these log files in the `logs/` directory:

1. **`packet_processing.jsonl`** - Raw packet data and network flows
2. **`accuracy_metrics.csv`** - Structured accuracy metrics for analysis
3. **`threat_actions.jsonl`** - Actions taken for each threat
4. **`system_metrics.jsonl`** - System performance and processing metrics

## Usage

### Start the System
```bash
cd agentic-ids-local/src
python main.py
```

The server will start with monitoring enabled and show:
```
[INFO] Starting Agentic IDS Local Server on port 5000...
[INFO] Monitoring system initialized - logs will be saved to: logs
[INFO] Federation functionality: DISABLED (commented out for testing)
[INFO] Available endpoints:
       - POST /detect     : Process network packets
       - GET  /health     : Health check
       - GET  /monitoring : Get current metrics
       - GET  /monitoring/summary : Print and get full summary
```

### Send Test Data
Use the provided test script to see monitoring in action:

```bash
python test_monitoring.py
```

This will:
1. Send 4 different types of test packets
2. Show real-time processing and classification  
3. Display accuracy metrics
4. Generate sample log files

### Monitor in Real-Time
Check current system status:
```bash
curl http://localhost:5000/monitoring
```

Get detailed summary:
```bash
curl http://localhost:5000/monitoring/summary
```

## Sample Output

When packets are processed, you'll see detailed tracking:

```
[MONITOR] Packet received: test-001 | 192.168.1.100:45678 -> 8.8.8.8:80
[INFERENCE] Normal traffic: 192.168.1.100 -> 8.8.8.8 | Score: 0.234567
[Orchestrator] Processing flow test-001 with anomaly score 0.2346
[TRIAGE] Flow Classification:
         Decision: BENIGN
         Pipeline: AgenticRAG
         Reasoning: Normal HTTP traffic pattern
         Context Retrieved: 456 characters
[Orchestrator] test-001 classified as BENIGN - logging for compliance
```

For anomalies:
```
[MONITOR] Packet received: test-suspicious-001 | 10.0.0.1:12345 -> 192.168.1.10:22
[INFERENCE] ANOMALY DETECTED:
            Score: 0.934567 (threshold: 0.5)
            Source: 10.0.0.1:12345
            Dest: 192.168.1.10:22
            Protocol: tcp
            Service: ssh
[Orchestrator] Processing flow test-suspicious-001 with anomaly score 0.9346
[TRIAGE] Flow Classification:
         Decision: SUSPICIOUS
         Pipeline: CorrectiveRAG
         Reasoning: High anomaly score with repeated SSH connections
         Context Retrieved: 1203 characters
[Orchestrator] Routing test-suspicious-001 to Suspicious Agent for verification
[Suspicious Agent] Analysis Results:
                   Status: TRUE_THREAT
                   Attack Category: SSH Brute Force
                   Confidence: 0.91
                   Action Plan: Block source IP immediately
                   KB Context Used: 847 chars
[MONITOR] Threat action: test-suspicious-001 | Action: investigate | Category: SSH Brute Force
```

## Metrics Tracked

The system now tracks comprehensive metrics:

- **Packet Processing**: Total packets, processing times, throughput
- **Anomaly Detection**: Detection rates, score distributions, thresholds
- **Threat Classification**: Breakdown by category (BENIGN/SUSPICIOUS/ZERO-DAY)  
- **Action Tracking**: Response actions taken, mitigation plans
- **Accuracy Metrics**: True/false positives, precision, confidence scores

## Configuration

Set monitoring configuration in `.env`:
```bash
MONITORING_LOG_DIR=logs  # Directory for log files
PORT=5000                # Server port
# Federation disabled for testing
```

## Development Notes

- Federation code is commented out but preserved for future re-enabling
- All monitoring operations are thread-safe for concurrent packet processing
- Log files use efficient JSONL format for streaming and analysis
- CSV format used for accuracy metrics to enable easy analysis in Excel/Python
- Memory metrics are maintained for real-time dashboard potential

## Next Steps

1. **Accuracy Validation**: Use the generated logs to validate threat detection accuracy
2. **Performance Tuning**: Monitor processing times to optimize pipeline performance
3. **Baseline Establishment**: Run with known traffic to establish normal baselines
4. **False Positive Analysis**: Use logs to identify and reduce false positive patterns
5. **Federation Re-enabling**: When ready, uncomment federation code and update `.env`

The system is now ready for comprehensive accuracy monitoring and performance analysis!