# A4 Action Agent - Threat Response System

## Overview

The A4 Action Agent is a specialized component that handles threat response actions with concise reasoning and automated action planning. It integrates with the existing orchestrator to provide rapid threat assessment and generate executable response plans.

## Features

### 🎯 Concise Threat Analysis
- **Quick Classification**: Identifies threat types (Scan, Exploit, DDoS, Exfiltration)
- **Severity Assessment**: Assigns Low/Medium/High/Critical severity levels
- **Brief Reasoning**: 2-3 sentence explanations focusing on actionable insights
- **Asset Context**: Considers network topology and asset criticality

### ⚡ Action Planning
- **Immediate Actions**: Tasks to execute within 5 minutes
- **Containment Actions**: Isolation and blocking measures
- **Monitoring Actions**: Enhanced surveillance requirements  
- **Notification Actions**: Alert routing based on severity and policy

### 🔧 Network-Aware Configuration
- **Organization Settings**: Defined network segments and asset inventory
- **Security Policies**: Configurable response levels and escalation rules
- **Asset Criticality**: Critical servers, web servers, DNS, mail infrastructure
- **Automated Firewall Rules**: Template-based blocking and quarantine

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orchestrator  │───▶│  A4 Action      │───▶│  Action Logs    │
│   (Routes)      │    │  Agent          │    │  & Responses    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Triage Result  │    │ Network Config  │    │ SOAR Integration│
│  (A1 Output)    │    │ (Organization)  │    │ (Future)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Files Structure

```
agents/A4_action_agent/
├── __init__.py              # Package initialization
├── action_agent.py          # Main agent implementation  
└── network_config.json      # Organization network settings
```

## Configuration

### Network Config (`network_config.json`)
```json
{
  "organization": {
    "network_segments": {
      "dmz": {"subnets": ["10.0.1.0/24"], "criticality": "Medium"},
      "internal": {"subnets": ["192.168.1.0/24"], "criticality": "High"},
      "critical": {"subnets": ["172.16.1.0/24"], "criticality": "Critical"}
    },
    "assets": {
      "critical_servers": ["172.16.1.10", "172.16.1.11"],
      "web_servers": ["10.0.1.10"],
      "dns_servers": ["192.168.1.1"]
    }
  },
  "security_policies": {
    "response_levels": {
      "low": {"actions": ["monitor", "log"]},
      "medium": {"actions": ["monitor", "log", "alert_admin"]},
      "high": {"actions": ["monitor", "log", "alert_soc", "isolate_host"]},
      "critical": {"actions": ["monitor", "log", "alert_soc", "block_ip", "isolate_host"]}
    }
  }
}
```

## Usage

### Integration with Orchestrator
The A4 Action Agent is automatically invoked by the Orchestrator for:

1. **Verified Threats** (CorrectiveRAG pipeline): When A2 agent confirms TRUE_THREAT
2. **Zero-Day Candidates** (AdaptiveRAG pipeline): Immediate emergency response

### Direct Usage
```python
from agents.A4_action_agent.action_agent import process_threat_response

# Process a threat
result = process_threat_response(triage_result, flow_data, flow_id)

# Access results
threat_summary = result["threat_summary"]
action_plan = result["action_plan"] 
execution_result = result["execution_result"]
```

## Data Models

### ThreatSummary
```python
{
    "threat_type": "Exploit",           # Scan, Exploit, DDoS, Exfiltration, Unknown
    "severity": "High",                 # Low, Medium, High, Critical  
    "target_asset": "Server",           # Server, Workstation, Network, Unknown
    "attack_vector": "Network",         # Network, Web, Email, Internal, Unknown
    "reasoning": "Brief explanation"    # 2-3 sentences max
}
```

### ActionPlan  
```python
{
    "immediate_actions": ["Enable monitoring for source IP"],
    "monitoring_actions": ["Check target for compromise indicators"],
    "containment_actions": ["Firewall rule: deny ip from 1.2.3.4"],
    "notification_actions": ["SOC notification: #incident-response"],
    "priority": 4                       # 1-5 priority level (5=urgent)
}
```

## Logging

### Action Response Logs
- **File**: `logs/action_responses.jsonl`
- **Format**: One JSON record per line
- **Contents**: Complete threat assessment, action plan, and execution details

### Log Fields
```json
{
    "timestamp": "2024-02-13T10:30:00Z",
    "flow_id": "flow_12345",
    "threat_assessment": {...},
    "action_plan": {...},
    "source_data": {"src_ip": "1.2.3.4", "dst_ip": "5.6.7.8"},
    "execution_status": "PLANNED"
}
```

## Testing

Run the test script:
```bash
python test_action_agent.py
```

### Test Coverage
- ✅ Network configuration loading
- ✅ Threat analysis with sample data
- ✅ Action plan generation  
- ✅ Response logging
- ✅ Priority assignment
- ✅ Asset criticality assessment

## Integration Points

### With Orchestrator
- Receives triage results from A1 agent
- Processes flow data from autoencoder
- Returns structured action response

### With Monitoring Service  
- Logs action responses to threat_actions.jsonl
- Integrates with existing monitoring infrastructure

### Future SOAR Integration
- Action execution automation
- Ticket creation and tracking
- Workflow orchestration

## Response Examples

### High Severity Threat
```
🚨 IMMEDIATE ACTIONS:
  • Enable enhanced monitoring for 192.168.1.100
  • Log incident to security database

🔒 CONTAINMENT ACTIONS:  
  • Firewall rule: deny ip from 192.168.1.100 to any

📊 MONITORING ACTIONS:
  • Monitor all traffic from 192.168.1.100 for 24 hours
  • Check 203.0.113.42 for compromise indicators

📢 NOTIFICATIONS:
  • SOC notification: #incident-response
```

### Zero-Day Response
```
Priority: 5/5 (URGENT)
Threat: Unknown (Critical)
Target: Critical Server
Emergency response initiated - automated containment active
```

## Security Considerations

- **Principle of Least Privilege**: Actions only logged, not executed automatically
- **Network Segmentation**: Respects organization network boundaries  
- **Escalation Paths**: Clear notification hierarchies
- **Audit Trail**: Complete action logging for compliance

## Future Enhancements

- [ ] SOAR platform integration (ServiceNow, Phantom, etc.)
- [ ] Machine learning for action effectiveness scoring
- [ ] Automated action execution with approval workflows
- [ ] Integration with network security tools (firewalls, SIEM)
- [ ] Custom action templates per organization
- [ ] Real-time threat intelligence integration