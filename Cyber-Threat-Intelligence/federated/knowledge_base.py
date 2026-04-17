"""
Threat Knowledge Base Populator for RAG Pipeline.

This module provides utilities to seed the FAISS vector database with
threat intelligence from MITRE ATT&CK and CVE/NVD sources.

The knowledge base enables the RAG system to:
- Retrieve relevant attack descriptions for detected threats
- Map network anomalies to known vulnerability patterns
- Provide context for zero-day threat analysis

Usage:
    >>> from federated.knowledge_base import ThreatKnowledgeBase
    >>> kb = ThreatKnowledgeBase(vector_db=faiss_db)
    >>> kb.load_mitre_attack()
    >>> kb.load_cve_data("data/nvd_cves.json")
    >>> kb.get_stats()
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MITRE ATT&CK Data (Embedded for reliability)
# =============================================================================

MITRE_ATTACK_TECHNIQUES = [
    # Reconnaissance
    {
        "technique_id": "T1595",
        "name": "Active Scanning",
        "tactic": "Reconnaissance",
        "description": "Adversaries may execute active reconnaissance scans to gather information that can be used during targeting. Active scans involve probing victim infrastructure via network traffic to identify potential entry points, open services, and vulnerabilities.",
        "mitigations": ["Monitor for suspicious network traffic patterns", "Implement network intrusion detection", "Use honeypots to detect scanning activity"],
        "detection": "Monitor network traffic for unusual scanning patterns, port sweeps, or service enumeration attempts.",
        "platforms": ["Network"],
        "severity": "medium",
    },
    {
        "technique_id": "T1592",
        "name": "Gather Victim Host Information",
        "tactic": "Reconnaissance",
        "description": "Adversaries may gather information about the victim's hosts that can be used during targeting. Information about hosts may include administrative data, hardware details, and configuration information.",
        "mitigations": ["Limit exposure of system information", "Monitor for information gathering attempts"],
        "detection": "Monitor for attempts to gather system information through various protocols.",
        "platforms": ["Network", "Windows", "Linux"],
        "severity": "low",
    },
    # Initial Access
    {
        "technique_id": "T1190",
        "name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "description": "Adversaries may attempt to exploit vulnerabilities in internet-facing systems. Web applications, databases, standard services (SSH, SMB), and network device administration interfaces are common targets.",
        "mitigations": ["Web Application Firewall", "Regular patching", "Input validation", "Network segmentation"],
        "detection": "Monitor application logs for exploitation attempts, unusual requests, or error patterns.",
        "platforms": ["Windows", "Linux", "Network", "Containers"],
        "severity": "critical",
        "related_cves": ["CVE-2021-44228", "CVE-2021-26855", "CVE-2019-19781"],
    },
    {
        "technique_id": "T1133",
        "name": "External Remote Services",
        "tactic": "Initial Access",
        "description": "Adversaries may leverage external-facing remote services to initially access and/or persist within a network. Remote services such as VPNs, Citrix, and other access mechanisms allow users to connect to internal resources.",
        "mitigations": ["Multi-factor authentication", "Limit remote access services", "Monitor remote access logs"],
        "detection": "Monitor authentication logs for brute force attempts or unusual access patterns.",
        "platforms": ["Windows", "Linux"],
        "severity": "high",
    },
    # Execution
    {
        "technique_id": "T1059",
        "name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "description": "Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries. These interfaces include PowerShell, Windows Command Shell, Unix Shell, Python, JavaScript, and others.",
        "mitigations": ["Restrict script execution", "Application whitelisting", "Monitor command-line activity"],
        "detection": "Monitor process creation and command-line arguments for suspicious patterns.",
        "platforms": ["Windows", "Linux", "macOS"],
        "severity": "high",
    },
    {
        "technique_id": "T1203",
        "name": "Exploitation for Client Execution",
        "tactic": "Execution",
        "description": "Adversaries may exploit software vulnerabilities in client applications to execute code. Vulnerabilities exist in browser-based content, office documents, and other common applications.",
        "mitigations": ["Application isolation", "Exploit protection", "User training"],
        "detection": "Monitor for unusual process creation patterns from client applications.",
        "platforms": ["Windows", "Linux", "macOS"],
        "severity": "critical",
    },
    # Persistence
    {
        "technique_id": "T1098",
        "name": "Account Manipulation",
        "tactic": "Persistence",
        "description": "Adversaries may manipulate accounts to maintain access to victim systems. Account manipulation may consist of modifying permissions, credentials, or other attributes.",
        "mitigations": ["Multi-factor authentication", "Privileged account management", "Audit account changes"],
        "detection": "Monitor for changes to account attributes, permissions, or credential updates.",
        "platforms": ["Windows", "Linux", "Azure AD", "Office 365"],
        "severity": "high",
    },
    # Privilege Escalation
    {
        "technique_id": "T1055",
        "name": "Process Injection",
        "tactic": "Privilege Escalation",
        "description": "Adversaries may inject code into processes to evade process-based defenses and elevate privileges. Process injection runs code in the address space of a separate live process.",
        "mitigations": ["Endpoint detection and response", "Behavior blocking", "Process integrity monitoring"],
        "detection": "Monitor for unexpected memory allocations, API calls associated with injection techniques.",
        "platforms": ["Windows", "Linux", "macOS"],
        "severity": "critical",
        "sub_techniques": ["DLL Injection", "Thread Execution Hijacking", "Ptrace System Calls"],
    },
    {
        "technique_id": "T1068",
        "name": "Exploitation for Privilege Escalation",
        "tactic": "Privilege Escalation",
        "description": "Adversaries may exploit software vulnerabilities to gain elevated privileges. Exploitation of vulnerabilities can enable adversaries to run programs with higher privileges.",
        "mitigations": ["Regular patching", "Exploit protection", "Least privilege principle"],
        "detection": "Monitor for exploitation attempts and unusual privilege elevation.",
        "platforms": ["Windows", "Linux", "macOS"],
        "severity": "critical",
    },
    # Defense Evasion
    {
        "technique_id": "T1036",
        "name": "Masquerading",
        "tactic": "Defense Evasion",
        "description": "Adversaries may attempt to manipulate features of their artifacts to make them appear legitimate. Masquerading occurs when the name or location of an object is manipulated to evade defenses.",
        "mitigations": ["Code signing enforcement", "File integrity monitoring", "Execution prevention"],
        "detection": "Monitor for files with names similar to legitimate system files in unusual locations.",
        "platforms": ["Windows", "Linux", "macOS", "Containers"],
        "severity": "medium",
    },
    # Credential Access
    {
        "technique_id": "T1110",
        "name": "Brute Force",
        "tactic": "Credential Access",
        "description": "Adversaries may use brute force techniques to gain access to accounts when passwords are unknown. Techniques include password guessing, password spraying, and credential stuffing.",
        "mitigations": ["Account lockout policies", "Multi-factor authentication", "Rate limiting"],
        "detection": "Monitor authentication logs for multiple failed attempts or password spray patterns.",
        "platforms": ["Windows", "Linux", "Azure AD", "Office 365", "SaaS"],
        "severity": "high",
    },
    # Discovery
    {
        "technique_id": "T1046",
        "name": "Network Service Discovery",
        "tactic": "Discovery",
        "description": "Adversaries may attempt to get a listing of services running on remote hosts. Methods include port scanning, service identification, and banner grabbing.",
        "mitigations": ["Network segmentation", "Disable unnecessary services", "Network intrusion detection"],
        "detection": "Monitor for port scanning activity and unusual network traffic patterns.",
        "platforms": ["Windows", "Linux", "macOS", "Network"],
        "severity": "medium",
    },
    # Lateral Movement
    {
        "technique_id": "T1080",
        "name": "Taint Shared Content",
        "tactic": "Lateral Movement",
        "description": "Adversaries may taint shared network content by adding programs, scripts, or exploit code to otherwise valid files. Infected shared content may be used to move laterally.",
        "mitigations": ["Restrict write access to shared resources", "Application whitelisting", "Endpoint protection"],
        "detection": "Monitor shared drives for newly introduced executable content or script files.",
        "platforms": ["Windows", "Linux", "macOS"],
        "severity": "high",
    },
    {
        "technique_id": "T1021",
        "name": "Remote Services",
        "tactic": "Lateral Movement",
        "description": "Adversaries may use Valid Accounts to log into a service specifically designed to accept remote connections, such as SSH, RDP, or VNC.",
        "mitigations": ["Disable or restrict remote services", "Multi-factor authentication", "Network segmentation"],
        "detection": "Monitor authentication logs and network traffic for remote service connections.",
        "platforms": ["Windows", "Linux", "macOS"],
        "severity": "high",
    },
    # Collection
    {
        "technique_id": "T1040",
        "name": "Network Sniffing",
        "tactic": "Collection",
        "description": "Adversaries may sniff network traffic to capture information about an environment. Network sniffing refers to using packet capture tools to intercept network traffic.",
        "mitigations": ["Encrypt sensitive traffic", "Network segmentation", "Use secure protocols"],
        "detection": "Monitor for promiscuous mode on network interfaces or known packet capture tools.",
        "platforms": ["Windows", "Linux", "macOS", "Network"],
        "severity": "medium",
    },
    # Command and Control  
    {
        "technique_id": "T1071",
        "name": "Application Layer Protocol",
        "tactic": "Command and Control",
        "description": "Adversaries may communicate using application layer protocols to avoid detection. Commands to the remote system, and often the results of those commands, will be embedded within the protocol traffic.",
        "mitigations": ["Network intrusion detection", "SSL/TLS inspection", "Web proxy filtering"],
        "detection": "Monitor for anomalous traffic patterns within standard protocols.",
        "platforms": ["Windows", "Linux", "macOS", "Network"],
        "severity": "high",
    },
    # Exfiltration
    {
        "technique_id": "T1048",
        "name": "Exfiltration Over Alternative Protocol",
        "tactic": "Exfiltration",
        "description": "Adversaries may steal data by exfiltrating it over a different protocol than that of the existing command and control channel. Examples include DNS tunneling, ICMP tunneling.",
        "mitigations": ["Network segmentation", "Data loss prevention", "Network monitoring"],
        "detection": "Monitor for unusual traffic patterns in protocols not typically used for data transfer.",
        "platforms": ["Windows", "Linux", "macOS", "Network"],
        "severity": "critical",
    },
    # Impact
    {
        "technique_id": "T1498",
        "name": "Network Denial of Service",
        "tactic": "Impact",
        "description": "Adversaries may perform Network Denial of Service (DoS) attacks to degrade or block the availability of targeted resources. Network DoS includes flooding traffic to a target.",
        "mitigations": ["DDoS protection services", "Rate limiting", "Network monitoring", "Traffic filtering"],
        "detection": "Monitor for sudden spikes in network traffic, SYN floods, or amplification attacks.",
        "platforms": ["Network"],
        "severity": "critical",
        "sub_techniques": ["Direct Network Flood", "Reflection Amplification"],
    },
    {
        "technique_id": "T1499",
        "name": "Endpoint Denial of Service",
        "tactic": "Impact",
        "description": "Adversaries may perform Endpoint Denial of Service (DoS) attacks to degrade or block the availability of services to users. This includes overwhelming web applications or services.",
        "mitigations": ["Web Application Firewall", "Rate limiting", "Resource quotas"],
        "detection": "Monitor for resource exhaustion patterns, application crashes, or service unavailability.",
        "platforms": ["Windows", "Linux", "macOS", "Containers"],
        "severity": "high",
    },
    {
        "technique_id": "T1486",
        "name": "Data Encrypted for Impact",
        "tactic": "Impact",
        "description": "Adversaries may encrypt data on target systems or on large numbers of systems in a network to interrupt availability. Ransomware uses encryption to hold data hostage.",
        "mitigations": ["Data backup", "User training", "Endpoint protection", "Network segmentation"],
        "detection": "Monitor for mass file encryption, known ransomware indicators, or ransom notes.",
        "platforms": ["Windows", "Linux", "macOS"],
        "severity": "critical",
    },
]


# =============================================================================
# CVE Data (Sample IDS-relevant CVEs)
# =============================================================================

SAMPLE_CVE_DATA = [
    {
        "cve_id": "CVE-2021-44228",
        "name": "Log4Shell",
        "description": "Apache Log4j2 <=2.14.1 JNDI features do not protect against attacker controlled LDAP and other JNDI related endpoints. An attacker who can control log messages or log message parameters can execute arbitrary code loaded from LDAP servers.",
        "cvss_score": 10.0,
        "attack_vector": "Network",
        "related_techniques": ["T1190", "T1059"],
        "affected_products": ["Apache Log4j"],
        "patches": ["Upgrade to Log4j 2.17.0 or later"],
        "ioc_patterns": ["${jndi:ldap://", "${jndi:rmi://"],
    },
    {
        "cve_id": "CVE-2021-26855",
        "name": "ProxyLogon",
        "description": "Microsoft Exchange Server Remote Code Execution Vulnerability. This vulnerability allows an attacker to bypass authentication and impersonate as the admin.",
        "cvss_score": 9.8,
        "attack_vector": "Network",
        "related_techniques": ["T1190", "T1133"],
        "affected_products": ["Microsoft Exchange Server 2013", "2016", "2019"],
        "patches": ["Apply Microsoft security updates"],
        "ioc_patterns": ["X-AnonResource-Backend", "X-BEResource"],
    },
    {
        "cve_id": "CVE-2019-19781",
        "name": "Citrix ADC Path Traversal",
        "description": "Citrix Application Delivery Controller (ADC) and Gateway allows directory traversal, leading to remote code execution without credentials.",
        "cvss_score": 9.8,
        "attack_vector": "Network",
        "related_techniques": ["T1190", "T1059"],
        "affected_products": ["Citrix ADC", "Citrix Gateway"],
        "patches": ["Apply Citrix firmware updates"],
        "ioc_patterns": ["/vpn/../vpns/", "NSC_USER"],
    },
    {
        "cve_id": "CVE-2020-1472",
        "name": "Zerologon",
        "description": "An elevation of privilege vulnerability exists when an attacker establishes a vulnerable Netlogon secure channel connection to a domain controller.",
        "cvss_score": 10.0,
        "attack_vector": "Network",
        "related_techniques": ["T1068", "T1098"],
        "affected_products": ["Windows Server 2008-2019"],
        "patches": ["Apply Microsoft security updates", "Enable secure RPC"],
        "ioc_patterns": ["Netlogon authentication", "Empty password attempts"],
    },
    {
        "cve_id": "CVE-2017-0144",
        "name": "EternalBlue",
        "description": "The SMBv1 server in Microsoft Windows allows remote attackers to execute arbitrary code via crafted packets, aka 'Windows SMB Remote Code Execution Vulnerability'.",
        "cvss_score": 9.8,
        "attack_vector": "Network",
        "related_techniques": ["T1190", "T1080"],
        "affected_products": ["Windows Vista/7/8/10", "Windows Server 2008-2016"],
        "patches": ["MS17-010", "Disable SMBv1"],
        "ioc_patterns": ["SMBv1 negotiation", "DoublePulsar backdoor"],
    },
    {
        "cve_id": "CVE-2018-13379",
        "name": "FortiGate SSL VPN Path Traversal",
        "description": "A path traversal vulnerability in the FortiOS SSL VPN web portal may allow an unauthenticated attacker to download FortiOS system files.",
        "cvss_score": 9.8,
        "attack_vector": "Network",
        "related_techniques": ["T1133", "T1190"],
        "affected_products": ["FortiOS 5.6.3-6.0.4"],
        "patches": ["Upgrade FortiOS", "Enable MFA"],
        "ioc_patterns": ["/remote/fgt_lang?", "sslvpn_websession"],
    },
    {
        "cve_id": "CVE-2022-22965",
        "name": "Spring4Shell",
        "description": "A Spring MVC or Spring WebFlux application running on JDK 9+ may be vulnerable to remote code execution via data binding.",
        "cvss_score": 9.8,
        "attack_vector": "Network",
        "related_techniques": ["T1190", "T1059"],
        "affected_products": ["Spring Framework 5.3.0-5.3.17", "5.2.0-5.2.19"],
        "patches": ["Upgrade Spring Framework", "Disable data binding"],
        "ioc_patterns": ["class.module.classLoader", "tomcatwar.jsp"],
    },
    {
        "cve_id": "CVE-2023-34362",
        "name": "MOVEit Transfer SQL Injection",
        "description": "SQL injection vulnerability in Progress MOVEit Transfer web application allows unauthenticated attackers to gain access to the database.",
        "cvss_score": 9.8,
        "attack_vector": "Network",
        "related_techniques": ["T1190", "T1059"],
        "affected_products": ["MOVEit Transfer"],
        "patches": ["Apply Progress security patches"],
        "ioc_patterns": ["human2.aspx", "GUESTACCESS"],
    },
]


# Network Attack Pattern Descriptions (for IDS context)
NETWORK_ATTACK_PATTERNS = [
    {
        "attack_type": "DoS",
        "description": "Denial of Service attacks aim to overwhelm target systems with traffic or requests, making services unavailable to legitimate users. Common patterns include SYN floods, UDP floods, and application-layer attacks.",
        "indicators": ["High packet rates", "Single source to single destination", "Abnormal protocol distributions", "Resource exhaustion"],
        "mitre_techniques": ["T1498", "T1499"],
        "recommended_actions": ["Enable rate limiting", "Deploy DDoS protection", "Block offending IPs", "Scale infrastructure"],
    },
    {
        "attack_type": "Reconnaissance",
        "description": "Reconnaissance attacks involve scanning and probing networks to identify vulnerabilities, open ports, running services, and potential entry points. These often precede more sophisticated attacks.",
        "indicators": ["Port scans", "Service enumeration", "Banner grabbing", "Sequential connection attempts"],
        "mitre_techniques": ["T1595", "T1046", "T1592"],
        "recommended_actions": ["Monitor for scan patterns", "Deploy honeypots", "Block known scanner IPs", "Alert security team"],
    },
    {
        "attack_type": "Exploits",
        "description": "Exploit attacks leverage known or zero-day vulnerabilities in software to gain unauthorized access, execute code, or escalate privileges. These target specific weaknesses in applications or operating systems.",
        "indicators": ["Unusual payload patterns", "Known exploit signatures", "Anomalous requests to vulnerable endpoints", "Shellcode patterns"],
        "mitre_techniques": ["T1190", "T1203", "T1068"],
        "recommended_actions": ["Block source IP", "Patch vulnerable systems", "Isolate affected hosts", "Conduct forensic analysis"],
    },
    {
        "attack_type": "Backdoor",
        "description": "Backdoor attacks establish persistent unauthorized access channels, allowing attackers to return to compromised systems. These often use command-and-control infrastructure for remote management.",
        "indicators": ["Unusual outbound connections", "Periodic beaconing", "Encrypted C2 traffic", "Non-standard port usage"],
        "mitre_techniques": ["T1059", "T1071", "T1098"],
        "recommended_actions": ["Isolate affected systems", "Block C2 communications", "Full malware analysis", "Credential reset"],
    },
    {
        "attack_type": "Worms",
        "description": "Worm attacks involve self-propagating malware that spreads across networks without user interaction. They exploit vulnerabilities to move laterally and can cause rapid, widespread infection.",
        "indicators": ["Lateral movement patterns", "Multiple systems contacting same external IP", "Exploitation attempts across subnet", "Abnormal SMB/RPC traffic"],
        "mitre_techniques": ["T1080", "T1021", "T1210"],
        "recommended_actions": ["Isolate affected subnet", "Disable vulnerable services", "Emergency patching", "Network-wide scan"],
    },
    {
        "attack_type": "Fuzzers",
        "description": "Fuzzing attacks send malformed or unexpected input to applications to discover vulnerabilities. While sometimes used for legitimate security testing, malicious fuzzing probes for exploitable weaknesses.",
        "indicators": ["Malformed packets", "Boundary condition testing", "Repeated requests with variations", "Application errors"],
        "mitre_techniques": ["T1499", "T1190"],
        "recommended_actions": ["Monitor for crash patterns", "Enable input validation", "Update WAF rules", "Log anomalous requests"],
    },
    {
        "attack_type": "Shellcode",
        "description": "Shellcode attacks inject and execute malicious code directly in memory, often as part of buffer overflow or code injection exploits. This provides attackers with direct system control.",
        "indicators": ["NOP sleds", "Position-independent code patterns", "Syscall sequences", "Encoded payloads"],
        "mitre_techniques": ["T1055", "T1203"],
        "recommended_actions": ["Enable DEP/ASLR", "Deploy endpoint detection", "Memory monitoring", "Immediate isolation"],
    },
    {
        "attack_type": "Analysis",
        "description": "Analysis attacks involve passive or active network traffic monitoring to gather intelligence. Attackers capture and analyze traffic to find credentials, map infrastructure, or identify vulnerabilities.",
        "indicators": ["Promiscuous mode interfaces", "ARP spoofing", "Traffic mirroring", "Unusual packet capture"],
        "mitre_techniques": ["T1040", "T1557"],
        "recommended_actions": ["Encrypt sensitive traffic", "Enable ARP protection", "Segment networks", "Monitor for sniffing tools"],
    },
    {
        "attack_type": "Generic",
        "description": "Generic attacks represent unclassified malicious activity that doesn't fit specific categories. These may include novel attack techniques, combined attacks, or activity requiring further analysis.",
        "indicators": ["Anomalous traffic patterns", "Unknown signatures", "Statistical outliers", "Behavioral anomalies"],
        "mitre_techniques": ["T1595"],
        "recommended_actions": ["Capture traffic for analysis", "Alert security team", "Enhanced monitoring", "Threat hunting investigation"],
    },
]


@dataclass
class KnowledgeBaseStats:
    """Statistics about the loaded knowledge base."""
    mitre_techniques: int = 0
    cve_entries: int = 0
    attack_patterns: int = 0
    total_documents: int = 0
    last_updated: str = ""
    

class ThreatKnowledgeBase:
    """
    Threat Knowledge Base for RAG Pipeline.
    
    This class manages a FAISS vector database populated with threat
    intelligence from MITRE ATT&CK, CVE databases, and network attack
    pattern descriptions.
    
    The knowledge base supports:
    - Semantic search for relevant threat context
    - Attack pattern matching for detected anomalies
    - CVE correlation for exploit identification
    - Zero-day analysis through similar attack retrieval
    
    Attributes:
        vector_db: FAISS vector database interface.
        embedding_fn: Function to generate text embeddings.
        stats: Knowledge base statistics.
    
    Example:
        >>> from agents.interfaces.vector_db import FAISSVectorDB
        >>> from sentence_transformers import SentenceTransformer
        >>> 
        >>> embeddings = SentenceTransformer("all-MiniLM-L6-v2")
        >>> vector_db = FAISSVectorDB(embeddings.encode)
        >>> 
        >>> kb = ThreatKnowledgeBase(vector_db)
        >>> kb.load_all()
        >>> results = kb.search("SQL injection vulnerability")
    """
    
    def __init__(
        self,
        vector_db: Any,
        embedding_fn: Optional[Any] = None,
    ):
        """
        Initialize the Threat Knowledge Base.
        
        Args:
            vector_db: VectorDBInterface implementation.
            embedding_fn: Optional embedding function override.
        """
        self.vector_db = vector_db
        self.embedding_fn = embedding_fn
        self.stats = KnowledgeBaseStats()
        self._initialized = False
        
        logger.info("ThreatKnowledgeBase initialized")
    
    def load_all(self) -> "ThreatKnowledgeBase":
        """
        Load all knowledge sources into the vector database.
        
        Returns:
            Self for method chaining.
        """
        logger.info("Loading all threat knowledge sources...")
        
        self.load_mitre_attack()
        self.load_cve_data()
        self.load_network_attack_patterns()
        
        self.stats.last_updated = datetime.now().isoformat()
        self._initialized = True
        
        logger.info(
            f"Knowledge base loaded: {self.stats.total_documents} documents "
            f"({self.stats.mitre_techniques} MITRE, {self.stats.cve_entries} CVE, "
            f"{self.stats.attack_patterns} patterns)"
        )
        
        return self
    
    def load_mitre_attack(
        self,
        techniques: Optional[List[Dict]] = None,
    ) -> int:
        """
        Load MITRE ATT&CK techniques into the knowledge base.
        
        Args:
            techniques: Optional list of technique dicts. Uses embedded
                       data if not provided.
        
        Returns:
            Number of techniques loaded.
        """
        techniques = techniques or MITRE_ATTACK_TECHNIQUES
        
        texts = []
        metadatas = []
        ids = []
        
        for tech in techniques:
            # Build comprehensive document text
            doc_text = self._format_mitre_document(tech)
            texts.append(doc_text)
            
            metadatas.append({
                "source": "mitre_attack",
                "technique_id": tech["technique_id"],
                "tactic": tech["tactic"],
                "severity": tech.get("severity", "medium"),
                "name": tech["name"],
            })
            
            ids.append(f"mitre_{tech['technique_id']}")
        
        # Add to vector database
        self.vector_db.add_documents(texts, metadatas=metadatas, ids=ids)
        
        self.stats.mitre_techniques = len(techniques)
        self.stats.total_documents += len(techniques)
        
        logger.info(f"Loaded {len(techniques)} MITRE ATT&CK techniques")
        return len(techniques)
    
    def _format_mitre_document(self, tech: Dict) -> str:
        """Format MITRE technique as searchable document."""
        mitigations = ", ".join(tech.get("mitigations", []))
        sub_techniques = ", ".join(tech.get("sub_techniques", []))
        related_cves = ", ".join(tech.get("related_cves", []))
        
        doc = f"""MITRE ATT&CK Technique: {tech['technique_id']} - {tech['name']}

Tactic: {tech['tactic']}
Severity: {tech.get('severity', 'medium')}

Description:
{tech['description']}

Detection Methods:
{tech.get('detection', 'Monitor for suspicious activity related to this technique.')}

Recommended Mitigations:
{mitigations}

Platforms: {', '.join(tech.get('platforms', ['All']))}
"""
        
        if sub_techniques:
            doc += f"\nSub-techniques: {sub_techniques}"
        
        if related_cves:
            doc += f"\nRelated CVEs: {related_cves}"
        
        return doc
    
    def load_cve_data(
        self,
        cve_list: Optional[List[Dict]] = None,
        json_path: Optional[str] = None,
    ) -> int:
        """
        Load CVE data into the knowledge base.
        
        Args:
            cve_list: Optional list of CVE dicts.
            json_path: Optional path to CVE JSON file.
        
        Returns:
            Number of CVEs loaded.
        """
        # Load from file if provided
        if json_path and Path(json_path).exists():
            with open(json_path, 'r') as f:
                cve_list = json.load(f)
        
        # Use embedded data if nothing provided
        cve_list = cve_list or SAMPLE_CVE_DATA
        
        texts = []
        metadatas = []
        ids = []
        
        for cve in cve_list:
            doc_text = self._format_cve_document(cve)
            texts.append(doc_text)
            
            metadatas.append({
                "source": "cve_nvd",
                "cve_id": cve["cve_id"],
                "cvss_score": cve.get("cvss_score", 0.0),
                "attack_vector": cve.get("attack_vector", "Unknown"),
                "name": cve.get("name", cve["cve_id"]),
            })
            
            ids.append(f"cve_{cve['cve_id']}")
        
        self.vector_db.add_documents(texts, metadatas=metadatas, ids=ids)
        
        self.stats.cve_entries = len(cve_list)
        self.stats.total_documents += len(cve_list)
        
        logger.info(f"Loaded {len(cve_list)} CVE entries")
        return len(cve_list)
    
    def _format_cve_document(self, cve: Dict) -> str:
        """Format CVE as searchable document."""
        techniques = ", ".join(cve.get("related_techniques", []))
        products = ", ".join(cve.get("affected_products", []))
        patches = ", ".join(cve.get("patches", []))
        iocs = ", ".join(cve.get("ioc_patterns", []))
        
        doc = f"""CVE: {cve['cve_id']} - {cve.get('name', 'Unnamed')}

CVSS Score: {cve.get('cvss_score', 'N/A')}
Attack Vector: {cve.get('attack_vector', 'Unknown')}

Description:
{cve['description']}

Affected Products:
{products}

Related MITRE Techniques: {techniques}

Patches/Mitigations:
{patches}

Indicators of Compromise (IoC):
{iocs if iocs else 'None documented'}
"""
        return doc
    
    def load_network_attack_patterns(
        self,
        patterns: Optional[List[Dict]] = None,
    ) -> int:
        """
        Load network attack pattern descriptions.
        
        Args:
            patterns: Optional list of attack pattern dicts.
        
        Returns:
            Number of patterns loaded.
        """
        patterns = patterns or NETWORK_ATTACK_PATTERNS
        
        texts = []
        metadatas = []
        ids = []
        
        for pattern in patterns:
            doc_text = self._format_attack_pattern_document(pattern)
            texts.append(doc_text)
            
            metadatas.append({
                "source": "network_patterns",
                "attack_type": pattern["attack_type"],
                "mitre_techniques": ",".join(pattern.get("mitre_techniques", [])),
            })
            
            ids.append(f"pattern_{pattern['attack_type'].lower()}")
        
        self.vector_db.add_documents(texts, metadatas=metadatas, ids=ids)
        
        self.stats.attack_patterns = len(patterns)
        self.stats.total_documents += len(patterns)
        
        logger.info(f"Loaded {len(patterns)} network attack patterns")
        return len(patterns)
    
    def _format_attack_pattern_document(self, pattern: Dict) -> str:
        """Format network attack pattern as searchable document."""
        indicators = "\n- ".join(pattern.get("indicators", []))
        techniques = ", ".join(pattern.get("mitre_techniques", []))
        actions = "\n- ".join(pattern.get("recommended_actions", []))
        
        doc = f"""Network Attack Pattern: {pattern['attack_type']}

Description:
{pattern['description']}

Indicators:
- {indicators}

Related MITRE Techniques: {techniques}

Recommended Response Actions:
- {actions}
"""
        return doc
    
    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant threat intelligence.
        
        Args:
            query: Search query string.
            k: Number of results to return.
            source_filter: Optional filter by source (mitre_attack, cve_nvd, network_patterns).
        
        Returns:
            List of search results with content and metadata.
        """
        filter_dict = None
        if source_filter:
            filter_dict = {"source": source_filter}
        
        results = self.vector_db.similarity_search(query, k=k, filter=filter_dict)
        
        return [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.similarity_score,
                "doc_id": r.document_id,
            }
            for r in results
        ]
    
    def search_by_attack_type(
        self,
        attack_type: str,
        include_cves: bool = True,
        include_mitre: bool = True,
        k: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        Search for all intelligence related to an attack type.
        
        Args:
            attack_type: Attack category (e.g., "DoS", "Exploits").
            include_cves: Whether to search CVE database.
            include_mitre: Whether to search MITRE techniques.
            k: Results per source.
        
        Returns:
            Dict with 'patterns', 'mitre', and 'cves' keys.
        """
        results = {"patterns": [], "mitre": [], "cves": []}
        
        # Search attack patterns
        pattern_results = self.search(
            attack_type, k=k, source_filter="network_patterns"
        )
        results["patterns"] = pattern_results
        
        if include_mitre:
            mitre_results = self.search(
                attack_type, k=k, source_filter="mitre_attack"
            )
            results["mitre"] = mitre_results
        
        if include_cves:
            cve_results = self.search(
                attack_type, k=k, source_filter="cve_nvd"
            )
            results["cves"] = cve_results
        
        return results
    
    def get_context_for_threat(
        self,
        attack_category: str,
        mitre_technique: str = "",
        is_zero_day: bool = False,
        max_context_length: int = 3000,
    ) -> Tuple[str, List[str]]:
        """
        Get formatted context for LLM threat analysis.
        
        This method builds a comprehensive context string for the RAG
        pipeline by combining relevant MITRE, CVE, and pattern data.
        
        Args:
            attack_category: Detected attack type.
            mitre_technique: MITRE technique ID if known.
            is_zero_day: Whether this is a potential zero-day.
            max_context_length: Maximum context string length.
        
        Returns:
            Tuple of (context_string, list_of_cve_ids).
        """
        context_parts = []
        cve_ids = []
        
        # Build search query
        query_parts = [attack_category]
        if mitre_technique:
            query_parts.append(mitre_technique)
        if is_zero_day:
            query_parts.append("unknown vulnerability zero-day")
        
        query = " ".join(query_parts)
        
        # Get attack pattern context
        patterns = self.search(query, k=2, source_filter="network_patterns")
        for p in patterns:
            if len("\n".join(context_parts)) < max_context_length:
                context_parts.append(f"[Attack Pattern]\n{p['content'][:500]}")
        
        # Get MITRE context
        mitre_results = self.search(query, k=2, source_filter="mitre_attack")
        for m in mitre_results:
            if len("\n".join(context_parts)) < max_context_length:
                context_parts.append(f"[MITRE ATT&CK]\n{m['content'][:500]}")
        
        # Get CVE context
        cve_results = self.search(query, k=3, source_filter="cve_nvd")
        for c in cve_results:
            if len("\n".join(context_parts)) < max_context_length:
                context_parts.append(f"[CVE Reference]\n{c['content'][:400]}")
                if "cve_id" in c.get("metadata", {}):
                    cve_ids.append(c["metadata"]["cve_id"])
        
        context_string = "\n\n---\n\n".join(context_parts)
        
        return context_string, cve_ids
    
    def add_custom_intelligence(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Add custom threat intelligence to the knowledge base.
        
        Args:
            text: Intelligence document text.
            metadata: Document metadata.
            doc_id: Optional document ID.
        
        Returns:
            Assigned document ID.
        """
        metadata["source"] = metadata.get("source", "custom")
        doc_id = doc_id or f"custom_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.vector_db.add_documents([text], metadatas=[metadata], ids=[doc_id])
        self.stats.total_documents += 1
        
        logger.info(f"Added custom intelligence: {doc_id}")
        return doc_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "mitre_techniques": self.stats.mitre_techniques,
            "cve_entries": self.stats.cve_entries,
            "attack_patterns": self.stats.attack_patterns,
            "total_documents": self.stats.total_documents,
            "last_updated": self.stats.last_updated,
            "initialized": self._initialized,
        }
    
    def persist(self) -> None:
        """Persist the knowledge base to disk."""
        if hasattr(self.vector_db, 'persist'):
            self.vector_db.persist()
            logger.info("Knowledge base persisted to disk")


def create_knowledge_base(
    persist_directory: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> ThreatKnowledgeBase:
    """
    Factory function to create a fully initialized knowledge base.
    
    Args:
        persist_directory: Optional directory for persistence.
        embedding_model: Sentence transformer model name.
    
    Returns:
        Initialized ThreatKnowledgeBase.
    
    Example:
        >>> kb = create_knowledge_base("./knowledge_db")
        >>> results = kb.search("SQL injection")
    """
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from agents.interfaces.vector_db import FAISSVectorDB
    except ImportError as e:
        raise ImportError(
            f"Required dependencies not installed: {e}. "
            "Install with: pip install sentence-transformers langchain-community faiss-cpu"
        )
    
    # Create embedding function
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create vector database
    vector_db = FAISSVectorDB(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    
    # Create and initialize knowledge base
    kb = ThreatKnowledgeBase(vector_db=vector_db)
    kb.load_all()
    
    if persist_directory:
        kb.persist()
    
    return kb
