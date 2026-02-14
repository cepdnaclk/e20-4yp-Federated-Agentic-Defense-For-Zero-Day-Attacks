import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: Please set your GROQ_API_KEY environment variable.")
    sys.exit(1)


class ThreatSummary(BaseModel):
    """Concise threat analysis output"""
    threat_type: str = Field(description="Brief threat category: Scan, Exploit, DDoS, Exfiltration, Unknown")
    severity: str = Field(description="Low, Medium, High, Critical")
    target_asset: str = Field(description="What's being targeted: Server, Workstation, Network, Unknown")
    attack_vector: str = Field(description="How: Network, Web, Email, Internal, Unknown")
    reasoning: str = Field(description="2-3 sentence explanation of why this is a threat", max_length=200)


class ActionPlan(BaseModel):
    """Executable action plan"""
    immediate_actions: List[str] = Field(description="Actions to take now (next 5 minutes)")
    monitoring_actions: List[str] = Field(description="Enhanced monitoring to implement")
    containment_actions: List[str] = Field(description="Isolation/blocking actions if needed") 
    notification_actions: List[str] = Field(description="Who to notify and how")
    priority: int = Field(description="1-5 priority level (5=urgent)")


class ActionAgent:
    """
    A4 Action Agent: Rapid threat assessment and response orchestration.
    Provides concise analysis and generates actionable response plans.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load network configuration
        if config_path is None:
            config_path = Path(__file__).parent / "network_config.json"
        
        with open(config_path, 'r') as f:
            self.network_config = json.load(f)
        
        # Initialize LLM for threat reasoning
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=100
        )
        
        # Initialize logs
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.action_log = self.log_dir / "action_responses.jsonl"
        
        print("[A4_ActionAgent] Initialized with network config")
    
    def analyze_threat(self, triage_result: Dict[str, Any], flow_data: Dict[str, Any]) -> ThreatSummary:
        """
        Quick threat analysis - identify what kind of threat this is and severity.
        Keep reasoning short and actionable.
        """
        
        # Extract key indicators
        anomaly_score = flow_data.get("anomaly_score", 0.0)
        network_info = self._extract_network_context(flow_data.get("features", {}))
        triage_classification = triage_result.get("target_pipeline", "Unknown")
        
        # Build context for LLM
        threat_context = f"""
THREAT ANALYSIS REQUEST:
Anomaly Score: {anomaly_score:.3f}
Classification: {triage_classification}  
Network Flow: {network_info['src']} → {network_info['dst']} ({network_info['proto']})
Service: {network_info['service']} | State: {network_info['state']}
Asset Context: {self._get_asset_context(network_info['dst_ip'])}

BEHAVIORAL SIGNS:
{triage_result.get('behavior_analysis', 'No behavioral analysis available')}

Provide CONCISE threat assessment. Focus on actionable insights, not technical details.
"""

        system_prompt = """You are a cybersecurity analyst providing rapid threat assessment.

Your job: Analyze the threat and provide SHORT, ACTIONABLE insights.

Rules:
- Reasoning must be 2-3 sentences max
- Focus on WHAT the threat is and WHY it matters
- Consider asset criticality from network context
- No technical jargon - clear, direct language
- If unclear, say "Unknown" rather than guess"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=threat_context)
        ]
        
        try:
            response = self.llm.invoke(messages)
            # Parse and structure the response
            return self._parse_threat_analysis(response.content, anomaly_score, network_info)
            
        except Exception as e:
            print(f"[A4_ActionAgent] Error in threat analysis: {e}")
            return ThreatSummary(
                threat_type="Unknown",
                severity="Medium",
                target_asset="Unknown",
                attack_vector="Network", 
                reasoning=f"Analysis failed: {str(e)[:100]}"
            )
    
    def generate_actions(self, threat_summary: ThreatSummary, network_info: Dict[str, Any]) -> ActionPlan:
        """
        Generate specific, executable actions based on threat assessment and network config.
        """
        
        # Get response level from config
        severity_map = {"Low": "low", "Medium": "medium", "High": "high", "Critical": "critical"}
        response_level = severity_map.get(threat_summary.severity, "medium")
        policy_actions = self.network_config["security_policies"]["response_levels"][response_level]["actions"]
        
        # Build action plan based on threat type and network config
        immediate_actions = []
        monitoring_actions = []
        containment_actions = []
        notification_actions = []
        
        # Immediate actions based on severity
        if "monitor" in policy_actions:
            immediate_actions.append(f"Enable enhanced monitoring for {network_info['src_ip']}")
            
        if "log" in policy_actions:
            immediate_actions.append("Log incident to security database")
            
        if "alert_admin" in policy_actions:
            notification_actions.append(f"Email alert to {self.network_config['contact_info']['network_admin']['email']}")
            
        if "alert_soc" in policy_actions:
            notification_actions.append(f"SOC notification: {self.network_config['contact_info']['soc_team']['slack']}")
            
        # Containment actions for higher severity
        if "block_ip" in policy_actions:
            firewall_rule = self.network_config["security_policies"]["firewall_rules"]["block_template"]
            containment_actions.append(f"Firewall rule: {firewall_rule.format(src_ip=network_info['src_ip'])}")
            
        if "isolate_host" in policy_actions:
            containment_actions.append(f"Quarantine host {network_info['dst_ip']} to isolated VLAN")
            
        # Monitoring actions based on threat type
        if threat_summary.threat_type in ["Scan", "Exploit"]:
            monitoring_actions.append(f"Monitor all traffic from {network_info['src_ip']} for 24 hours")
            monitoring_actions.append(f"Check {network_info['dst_ip']} for compromise indicators")
            
        if threat_summary.threat_type == "DDoS":
            monitoring_actions.append("Activate DDoS protection mechanisms")
            monitoring_actions.append("Monitor bandwidth utilization")
            
        # Priority based on severity and asset criticality
        priority_map = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}
        priority = priority_map.get(threat_summary.severity, 3)
        
        # Boost priority for critical assets
        if self._is_critical_asset(network_info['dst_ip']):
            priority = min(5, priority + 1)
            
        return ActionPlan(
            immediate_actions=immediate_actions,
            monitoring_actions=monitoring_actions,
            containment_actions=containment_actions,
            notification_actions=notification_actions,
            priority=priority
        )
    
    def execute_response(self, flow_id: str, threat_summary: ThreatSummary, action_plan: ActionPlan, 
                        flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log the response plan and return execution summary.
        Future: Could integrate with SOAR platforms for automated execution.
        """
        
        response_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "flow_id": flow_id,
            "threat_assessment": threat_summary.dict(),
            "action_plan": action_plan.dict(),
            "source_data": {
                "src_ip": flow_data.get("features", {}).get("srcip"),
                "dst_ip": flow_data.get("features", {}).get("dstip"),
                "anomaly_score": flow_data.get("anomaly_score")
            },
            "execution_status": "PLANNED"  # Future: track actual execution
        }
        
        # Log to file
        self._log_action_response(response_record)
        
        # Print immediate actions for visibility
        print(f"[A4_ActionAgent] Response Plan for {flow_id}")
        print(f"Threat: {threat_summary.threat_type} ({threat_summary.severity})")
        print(f"Priority: {action_plan.priority}/5")
        
        if action_plan.immediate_actions:
            print("Immediate Actions:")
            for action in action_plan.immediate_actions:
                print(f"  - {action}")
        
        return {
            "response_id": response_record["timestamp"],
            "actions_planned": len(action_plan.immediate_actions + action_plan.containment_actions),
            "notifications_sent": len(action_plan.notification_actions),
            "priority_level": action_plan.priority,
            "execution_status": "LOGGED"
        }
    
    def _extract_network_context(self, features: Dict[str, Any]) -> Dict[str, str]:
        """Extract and format network information"""
        return {
            "src": f"{features.get('srcip', 'unknown')}:{features.get('sport', 'unknown')}",
            "dst": f"{features.get('dstip', 'unknown')}:{features.get('dsport', 'unknown')}",
            "src_ip": features.get('srcip', 'unknown'),
            "dst_ip": features.get('dstip', 'unknown'),
            "proto": features.get("proto", "TCP").upper(),
            "service": features.get("service", "unknown"),
            "state": features.get("state", "unknown")
        }
    
    def _get_asset_context(self, dst_ip: str) -> str:
        """Determine asset criticality based on network config"""
        assets = self.network_config["organization"]["assets"]
        
        if dst_ip in assets.get("critical_servers", []):
            return "CRITICAL SERVER"
        elif dst_ip in assets.get("web_servers", []):
            return "WEB SERVER"
        elif dst_ip in assets.get("dns_servers", []):
            return "DNS SERVER" 
        elif dst_ip in assets.get("mail_servers", []):
            return "MAIL SERVER"
        
        # Check network segments
        for segment, config in self.network_config["organization"]["network_segments"].items():
            # Simplified IP range check (would need proper subnet checking in production)
            if any(dst_ip.startswith(subnet.split('/')[0][:7]) for subnet in config["subnets"]):
                return f"{config['description']} ({config['criticality']})"
                
        return "Unknown Asset"
    
    def _is_critical_asset(self, ip: str) -> bool:
        """Check if IP is a critical asset"""
        critical_assets = self.network_config["organization"]["assets"]["critical_servers"]
        return ip in critical_assets
    
    def _parse_threat_analysis(self, llm_response: str, anomaly_score: float, network_info: Dict[str, str]) -> ThreatSummary:
        """Parse LLM response into structured threat summary"""
        # Simple parsing - in production could use structured output
        response_lower = llm_response.lower()
        
        # Determine threat type
        threat_type = "Unknown"
        if any(word in response_lower for word in ["scan", "probe", "recon"]):
            threat_type = "Scan"
        elif any(word in response_lower for word in ["exploit", "attack", "vulnerability"]):
            threat_type = "Exploit"
        elif any(word in response_lower for word in ["dos", "ddos", "flood"]):
            threat_type = "DDoS"
        elif any(word in response_lower for word in ["exfiltrat", "data", "steal"]):
            threat_type = "Exfiltration"
        
        # Determine severity based on anomaly score and content
        if anomaly_score > 0.8 or "critical" in response_lower:
            severity = "Critical"
        elif anomaly_score > 0.6 or "high" in response_lower:
            severity = "High"
        elif anomaly_score > 0.3 or "medium" in response_lower:
            severity = "Medium"
        else:
            severity = "Low"
        
        # Determine target asset type
        target_asset = "Unknown"
        if "server" in response_lower:
            target_asset = "Server"
        elif "workstation" in response_lower:
            target_asset = "Workstation"
        elif "network" in response_lower:
            target_asset = "Network"
        
        return ThreatSummary(
            threat_type=threat_type,
            severity=severity,
            target_asset=target_asset,
            attack_vector="Network",
            reasoning=llm_response[:200]  # Truncate to max length
        )
    
    def _log_action_response(self, response_record: Dict[str, Any]):
        """Log action response to JSONL file"""
        try:
            with open(self.action_log, 'a') as f:
                f.write(json.dumps(response_record) + '\n')
        except Exception as e:
            print(f"[A4_ActionAgent] Failed to log response: {e}")


# Convenience function for direct usage
def process_threat_response(triage_result: Dict[str, Any], flow_data: Dict[str, Any], flow_id: str) -> Dict[str, Any]:
    """
    Main interface function for threat response processing
    """
    agent = ActionAgent()
    
    # Analyze threat
    threat_summary = agent.analyze_threat(triage_result, flow_data)
    
    # Generate action plan
    network_info = agent._extract_network_context(flow_data.get("features", {}))
    action_plan = agent.generate_actions(threat_summary, network_info)
    
    # Execute response (log and plan)
    execution_result = agent.execute_response(flow_id, threat_summary, action_plan, flow_data)
    
    return {
        "threat_summary": threat_summary.dict(),
        "action_plan": action_plan.dict(),
        "execution_result": execution_result
    }