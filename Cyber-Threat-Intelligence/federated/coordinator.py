"""
Integration Coordinator - Semantic Translation Bridge.

This module bridges the federated detection system with the RAG
(Retrieval Augmented Generation) pipeline. It translates raw
detection outputs into semantically enriched threat reports by
querying MITRE ATT&CK, CVE databases, and threat intelligence.

Key Concepts:
    - SemanticThreatReport: Enriched threat report with MITRE/CVE context
    - IntegrationCoordinator: Orchestrates detection → analysis → report flow
    - Zero-day handling: Unknown threats routed to RAG for reasoning

Classes:
    SemanticThreatReport: Dataclass for enriched threat information.
    IntegrationCoordinator: Main coordination bridge.

Example:
    >>> from federated.coordinator import IntegrationCoordinator
    >>> coordinator = IntegrationCoordinator(agent_one, agent_two, agent_three)
    >>> coordinator.set_rag_system(vector_db, llm)
    >>> 
    >>> report = coordinator.process_network_sample(sample)
    >>> print(report.mitre_technique)
    >>> print(report.recommended_action)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SemanticThreatReport:
    """
    Enriched threat report with semantic context.
    
    This dataclass contains the complete analysis of a detected threat,
    including raw detection outputs, MITRE ATT&CK mappings, CVE references,
    and recommended mitigation actions.
    
    Attributes:
        timestamp: When the report was generated.
        sample_id: Unique identifier for the analyzed sample.
        
        # Agent One: Anomaly Detection
        is_anomaly: Whether sample was flagged as anomalous.
        reconstruction_error: Autoencoder reconstruction error.
        anomaly_threshold: Threshold used for detection.
        
        # Agent Two: Threat Classification
        attack_category: Predicted attack type (e.g., "DoS", "Exploits").
        classification_confidence: Model confidence in prediction.
        is_zero_day: Whether this appears to be an unknown threat.
        
        # Agent Three: Mitigation Decision
        recommended_action: Suggested response (Block IP, Isolate Subnet, etc.).
        action_confidence: Confidence in the recommendation.
        
        # RAG Enrichment
        mitre_technique: MITRE ATT&CK technique ID (e.g., T1040).
        mitre_tactic: MITRE ATT&CK tactic (e.g., Collection).
        mitre_description: Description of the technique.
        cve_references: Related CVE identifiers.
        threat_description: Natural language threat description.
        
        # Metadata
        severity: Overall threat severity.
        metadata: Additional context and debug info.
    
    Example:
        >>> report = SemanticThreatReport(
        ...     timestamp="2025-01-15T10:30:00",
        ...     sample_id="sample_001",
        ...     is_anomaly=True,
        ...     reconstruction_error=0.0845,
        ...     anomaly_threshold=0.0334,
        ...     attack_category="Reconnaissance",
        ...     classification_confidence=0.92,
        ...     is_zero_day=False,
        ...     recommended_action="Block IP",
        ...     action_confidence=0.87,
        ...     mitre_technique="T1595",
        ...     mitre_tactic="Reconnaissance",
        ...     mitre_description="Active Scanning",
        ...     severity=ThreatSeverity.HIGH,
        ... )
    """
    # Identifiers
    timestamp: str
    sample_id: str
    
    # Agent One: Anomaly Detection
    is_anomaly: bool
    reconstruction_error: float
    anomaly_threshold: float
    
    # Agent Two: Threat Classification
    attack_category: str
    classification_confidence: float
    is_zero_day: bool = False
    
    # Agent Three: Mitigation Decision
    recommended_action: str = ""
    action_confidence: float = 0.0
    
    # RAG Enrichment
    mitre_technique: str = ""
    mitre_tactic: str = ""
    mitre_description: str = ""
    cve_references: List[str] = field(default_factory=list)
    threat_description: str = ""
    
    # Severity and metadata
    severity: ThreatSeverity = ThreatSeverity.INFO
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "sample_id": self.sample_id,
            "anomaly_detection": {
                "is_anomaly": self.is_anomaly,
                "reconstruction_error": self.reconstruction_error,
                "threshold": self.anomaly_threshold,
            },
            "classification": {
                "category": self.attack_category,
                "confidence": self.classification_confidence,
                "is_zero_day": self.is_zero_day,
            },
            "mitigation": {
                "action": self.recommended_action,
                "confidence": self.action_confidence,
            },
            "intelligence": {
                "mitre_technique": self.mitre_technique,
                "mitre_tactic": self.mitre_tactic,
                "mitre_description": self.mitre_description,
                "cve_references": self.cve_references,
                "threat_description": self.threat_description,
            },
            "severity": self.severity.value,
            "metadata": self.metadata,
        }
    
    def to_markdown(self) -> str:
        """Generate markdown-formatted report."""
        severity_emoji = {
            ThreatSeverity.CRITICAL: "🔴",
            ThreatSeverity.HIGH: "🟠",
            ThreatSeverity.MEDIUM: "🟡",
            ThreatSeverity.LOW: "🟢",
            ThreatSeverity.INFO: "🔵",
        }
        
        return f"""
# Threat Report: {self.sample_id}

**Generated**: {self.timestamp}
**Severity**: {severity_emoji.get(self.severity, "⚪")} {self.severity.value.upper()}

## Detection Summary

| Agent | Result | Confidence |
|-------|--------|------------|
| Anomaly (Agent 1) | {'⚠️ ANOMALY' if self.is_anomaly else '✅ Normal'} | Error: {self.reconstruction_error:.4f} |
| Classification (Agent 2) | {self.attack_category} | {self.classification_confidence:.2%} |
| Mitigation (Agent 3) | {self.recommended_action} | {self.action_confidence:.2%} |

## Threat Intelligence

- **MITRE Technique**: {self.mitre_technique or 'N/A'} - {self.mitre_description or 'N/A'}
- **MITRE Tactic**: {self.mitre_tactic or 'N/A'}
- **CVE References**: {', '.join(self.cve_references) if self.cve_references else 'None identified'}
- **Zero-Day**: {'⚠️ Possible unknown threat' if self.is_zero_day else 'Known pattern'}

## Analysis

{self.threat_description or 'No additional analysis available.'}

---
*Report generated by Network Defense IDS*
"""


# MITRE ATT&CK mapping for common attack categories
ATTACK_TO_MITRE_MAP: Dict[str, Dict[str, str]] = {
    "Normal": {
        "technique": "",
        "tactic": "",
        "description": "Benign network traffic",
    },
    "Fuzzers": {
        "technique": "T1499",
        "tactic": "Impact",
        "description": "Endpoint Denial of Service through malformed input",
    },
    "Analysis": {
        "technique": "T1040",
        "tactic": "Collection",
        "description": "Network Sniffing and traffic analysis",
    },
    "Backdoors": {
        "technique": "T1059",
        "tactic": "Execution",
        "description": "Command and Scripting Interpreter abuse",
    },
    "DoS": {
        "technique": "T1498",
        "tactic": "Impact",
        "description": "Network Denial of Service attack",
    },
    "Exploits": {
        "technique": "T1190",
        "tactic": "Initial Access",
        "description": "Exploit Public-Facing Application",
    },
    "Generic": {
        "technique": "T1595",
        "tactic": "Reconnaissance",
        "description": "Active Scanning and probing",
    },
    "Reconnaissance": {
        "technique": "T1595",
        "tactic": "Reconnaissance",
        "description": "Active network reconnaissance",
    },
    "Shellcode": {
        "technique": "T1055",
        "tactic": "Privilege Escalation",
        "description": "Process Injection through shellcode",
    },
    "Worms": {
        "technique": "T1080",
        "tactic": "Lateral Movement",
        "description": "Taint Shared Content for propagation",
    },
}


class IntegrationCoordinator:
    """
    Semantic Translation Bridge between detection and intelligence.
    
    This coordinator:
    1. Receives detection signals from Agents One/Two/Three
    2. Queries RAG system for threat intelligence enrichment
    3. Maps threats to MITRE ATT&CK framework
    4. Produces SemanticThreatReports for downstream consumption
    
    The coordinator acts as a bridge between the ML detection pipeline
    and human-readable threat intelligence, enabling:
    - Automated SOAR integration
    - Analyst dashboards
    - Compliance reporting
    - Threat hunting workflows
    
    Attributes:
        agent_one: Anomaly detection agent (Autoencoder).
        agent_two: Classification agent (XGBoost + RAG).
        agent_three: Mitigation agent (PPO RL).
        vector_db: Vector database for similarity search.
        llm: Language model for threat analysis.
        zero_day_threshold: Confidence below which to flag zero-day.
    
    Example:
        >>> coordinator = IntegrationCoordinator(
        ...     agent_one=agent_one,
        ...     agent_two=agent_two,
        ...     agent_three=agent_three,
        ... )
        >>> coordinator.set_rag_system(faiss_db, groq_llm)
        >>> 
        >>> # Process a single sample
        >>> report = coordinator.process_network_sample(sample, sample_id="test_001")
        >>> 
        >>> # Batch processing
        >>> reports = coordinator.process_batch(samples)
    """
    
    def __init__(
        self,
        agent_one=None,
        agent_two=None,
        agent_three=None,
        zero_day_threshold: float = 0.4,
    ):
        """
        Initialize the Integration Coordinator.
        
        Args:
            agent_one: Anomaly detection agent (optional).
            agent_two: Classification agent (optional).
            agent_three: Mitigation agent (optional).
            zero_day_threshold: Confidence threshold for zero-day flag.
        """
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.agent_three = agent_three
        self.zero_day_threshold = zero_day_threshold
        
        # RAG components (set via set_rag_system)
        self.vector_db = None
        self.llm = None
        
        # Stats tracking
        self._samples_processed = 0
        self._anomalies_detected = 0
        self._zero_days_flagged = 0
        
        # Federated learning state
        self._federated_round = 0
        self._pending_samples_for_reanalysis: List[Dict[str, Any]] = []
        self._on_reanalysis_complete: Optional[callable] = None
        
        # Knowledge base (set via set_knowledge_base)
        self._knowledge_base = None
        
        logger.info(
            f"IntegrationCoordinator initialized: "
            f"zero_day_threshold={zero_day_threshold}"
        )
    
    def set_rag_system(self, vector_db, llm) -> None:
        """
        Configure the RAG system for threat enrichment.
        
        Args:
            vector_db: VectorDBInterface implementation (e.g., FAISSVectorDB).
            llm: LLMInterface implementation (e.g., GroqLLM, OpenAILLM).
        """
        self.vector_db = vector_db
        self.llm = llm
        logger.info("RAG system configured")
    
    def process_network_sample(
        self,
        sample: np.ndarray,
        sample_id: Optional[str] = None,
        include_rag_enrichment: bool = True,
    ) -> SemanticThreatReport:
        """
        Process a single network sample through the full pipeline.
        
        This method:
        1. Runs anomaly detection (Agent One)
        2. Classifies the threat (Agent Two)
        3. Determines mitigation action (Agent Three)
        4. Enriches with RAG context if enabled
        5. Builds and returns a SemanticThreatReport
        
        Args:
            sample: Network feature vector (1D numpy array).
            sample_id: Optional identifier for the sample.
            include_rag_enrichment: Whether to query RAG for context.
        
        Returns:
            SemanticThreatReport with full analysis.
        
        Raises:
            ValueError: If required agents are not configured.
        """
        timestamp = datetime.now().isoformat()
        sample_id = sample_id or f"sample_{self._samples_processed:06d}"
        
        # Ensure sample is 2D for model input
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Step 1: Anomaly Detection (Agent One)
        is_anomaly, reconstruction_error, threshold = self._run_anomaly_detection(sample)
        
        if is_anomaly:
            self._anomalies_detected += 1
        
        # Step 2: Threat Classification (Agent Two)
        attack_category, confidence, is_zero_day = self._run_classification(sample)
        
        if is_zero_day:
            self._zero_days_flagged += 1
        
        # Step 3: Mitigation Decision (Agent Three)
        recommended_action, action_confidence = self._run_mitigation(sample, attack_category)
        
        # Step 4: MITRE Mapping
        mitre_info = self._get_mitre_mapping(attack_category)
        
        # Step 5: RAG Enrichment (if enabled and available)
        threat_description = ""
        cve_references = []
        
        if include_rag_enrichment and is_anomaly:
            threat_description, cve_references = self._run_rag_enrichment(
                attack_category, mitre_info, is_zero_day
            )
        
        # Determine severity
        severity = self._calculate_severity(
            is_anomaly, confidence, is_zero_day, attack_category
        )
        
        # Build report
        report = SemanticThreatReport(
            timestamp=timestamp,
            sample_id=sample_id,
            is_anomaly=is_anomaly,
            reconstruction_error=reconstruction_error,
            anomaly_threshold=threshold,
            attack_category=attack_category,
            classification_confidence=confidence,
            is_zero_day=is_zero_day,
            recommended_action=recommended_action,
            action_confidence=action_confidence,
            mitre_technique=mitre_info.get("technique", ""),
            mitre_tactic=mitre_info.get("tactic", ""),
            mitre_description=mitre_info.get("description", ""),
            cve_references=cve_references,
            threat_description=threat_description,
            severity=severity,
            metadata={
                "pipeline_version": "1.0",
                "rag_enabled": include_rag_enrichment,
            },
        )
        
        self._samples_processed += 1
        logger.debug(f"Processed sample {sample_id}: {attack_category} ({severity.value})")
        
        return report
    
    def _run_anomaly_detection(
        self, sample: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Run anomaly detection using Agent One.
        
        Returns:
            Tuple of (is_anomaly, reconstruction_error, threshold).
        """
        if self.agent_one is None:
            # Fallback: treat all as potential anomalies
            logger.warning("Agent One not configured, assuming anomaly")
            return True, 0.05, 0.0334
        
        try:
            is_anomaly = self.agent_one.detect(sample)
            
            # Get reconstruction error if available
            if hasattr(self.agent_one, 'reconstruction_error'):
                reconstruction_error = float(self.agent_one.reconstruction_error)
            else:
                reconstruction_error = 0.05 if is_anomaly else 0.01
            
            threshold = getattr(self.agent_one, 'threshold', 0.0396)
            
            return bool(is_anomaly), reconstruction_error, threshold
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return True, 0.1, 0.0334  # Conservative fallback
    
    def _run_classification(
        self, sample: np.ndarray
    ) -> Tuple[str, float, bool]:
        """
        Run threat classification using Agent Two.
        
        Returns:
            Tuple of (attack_category, confidence, is_zero_day).
        """
        if self.agent_two is None:
            logger.warning("Agent Two not configured, returning Unknown")
            return "Unknown", 0.0, True
        
        try:
            result = self.agent_two.classify(sample)
            
            # Handle different result formats
            if isinstance(result, dict):
                category = result.get("category", "Unknown")
                confidence = result.get("confidence", 0.5)
            elif isinstance(result, tuple):
                category, confidence = result[:2]
            else:
                category = str(result)
                confidence = 0.5
            
            # Check for zero-day (low confidence or unknown category)
            is_zero_day = (
                confidence < self.zero_day_threshold
                or category in ["Unknown", "Zero-Day"]
            )
            
            return str(category), float(confidence), is_zero_day
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "Unknown", 0.0, True
    
    def _run_mitigation(
        self, sample: np.ndarray, attack_category: str
    ) -> Tuple[str, float]:
        """
        Run mitigation decision using Agent Three.
        
        Returns:
            Tuple of (action_name, confidence).
        """
        if self.agent_three is None:
            # Default mitigation based on category
            default_actions = {
                "DoS": "Rate Limit",
                "Exploits": "Block IP",
                "Backdoors": "Isolate Subnet",
                "Worms": "Quarantine",
            }
            action = default_actions.get(attack_category, "Monitor")
            return action, 0.5
        
        try:
            # Get mitigation action from Agent Three
            action_idx = self.agent_three.get_action(sample)
            
            # Map action index to name
            action_names = [
                "Monitor",
                "Rate Limit",
                "Block IP",
                "Isolate Subnet",
                "Quarantine",
            ]
            
            if isinstance(action_idx, int):
                action = action_names[action_idx % len(action_names)]
                confidence = 0.8  # RL confidence approximation
            else:
                action = str(action_idx)
                confidence = 0.7
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Mitigation decision error: {e}")
            return "Monitor", 0.3
    
    def _get_mitre_mapping(self, attack_category: str) -> Dict[str, str]:
        """
        Map attack category to MITRE ATT&CK technique.
        
        Args:
            attack_category: Detected attack type.
        
        Returns:
            Dict with technique, tactic, and description.
        """
        mapping = ATTACK_TO_MITRE_MAP.get(
            attack_category,
            {
                "technique": "T1595",
                "tactic": "Reconnaissance",
                "description": "Unknown attack pattern",
            }
        )
        return mapping
    
    def _run_rag_enrichment(
        self,
        attack_category: str,
        mitre_info: Dict[str, str],
        is_zero_day: bool,
    ) -> Tuple[str, List[str]]:
        """
        Query RAG system for threat intelligence enrichment.
        
        Args:
            attack_category: Detected attack type.
            mitre_info: MITRE mapping information.
            is_zero_day: Whether this is a potential zero-day.
        
        Returns:
            Tuple of (threat_description, cve_references).
        """
        if self.llm is None:
            logger.debug("LLM not configured, skipping enrichment")
            return "", []
        
        try:
            technique = mitre_info.get("technique", "")
            
            # Use knowledge base if available, otherwise fall back to vector_db
            if self._knowledge_base is not None:
                context_text, cve_ids_from_kb = self._knowledge_base.get_context_for_threat(
                    attack_category=attack_category,
                    mitre_technique=technique,
                    is_zero_day=is_zero_day,
                )
            elif self.vector_db is not None:
                # Fallback to direct vector search
                query = f"{attack_category} attack {technique} {mitre_info.get('description', '')}"
                contexts = self.vector_db.similarity_search(query, k=5)
                context_text = "\n\n".join([ctx.content for ctx in contexts])
                cve_ids_from_kb = []
            else:
                logger.debug("No knowledge source configured, skipping enrichment")
                return "", []
            
            # Build LLM prompt with rich context
            if is_zero_day:
                llm_prompt = f"""You are a cybersecurity expert analyzing a potential zero-day threat detected by a federated intrusion detection system.

## Detection Summary
- **Attack Category**: {attack_category}
- **MITRE Technique**: {technique}
- **Classification Confidence**: Low (potential zero-day)

## Threat Intelligence Context
{context_text}

## Analysis Required
Based on the detection and threat intelligence context above, provide:

1. **Threat Assessment** (2-3 sentences): What type of attack is this most likely? How does it compare to known patterns?

2. **Severity Level**: Critical/High/Medium/Low and brief justification

3. **Related Vulnerabilities**: Any CVEs that match this pattern (reference specific CVE IDs if found in context)

4. **Recommended Actions**:
   - Immediate response actions
   - Investigation steps
   - Long-term mitigations

5. **Confidence Notes**: What makes this a potential zero-day vs. known attack variant?

Provide a structured, actionable analysis:"""
            else:
                llm_prompt = f"""You are a cybersecurity expert providing a threat analysis summary for a security operations team.

## Detection Details
- **Attack Category**: {attack_category}
- **MITRE Technique**: {technique} - {mitre_info.get('description', '')}
- **Tactic**: {mitre_info.get('tactic', 'Unknown')}

## Threat Intelligence Context
{context_text}

## Required Output
Provide a concise threat summary (3-5 sentences) including:
- What the attack is attempting to accomplish
- Key indicators to monitor
- Recommended immediate action

Keep the response focused and actionable for SOC analysts."""
            
            # Query LLM
            response = self.llm.generate(llm_prompt)
            threat_description = response.content
            
            # Extract CVE references from response and combine with KB CVEs
            cve_refs = self._extract_cve_references(threat_description)
            all_cves = list(set(cve_refs + cve_ids_from_kb if 'cve_ids_from_kb' in dir() else cve_refs))
            
            return threat_description, all_cves
            
        except Exception as e:
            logger.error(f"RAG enrichment error: {e}")
            return "", []
    
    def _extract_cve_references(self, text: str) -> List[str]:
        """Extract CVE IDs from text."""
        import re
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        matches = re.findall(cve_pattern, text, re.IGNORECASE)
        return list(set(matches))
    
    def _calculate_severity(
        self,
        is_anomaly: bool,
        confidence: float,
        is_zero_day: bool,
        attack_category: str,
    ) -> ThreatSeverity:
        """
        Calculate overall threat severity.
        
        Args:
            is_anomaly: Whether anomaly was detected.
            confidence: Classification confidence.
            is_zero_day: Whether possibly a zero-day.
            attack_category: Detected attack category.
        
        Returns:
            ThreatSeverity enum value.
        """
        if not is_anomaly:
            return ThreatSeverity.INFO
        
        # High-severity categories
        critical_categories = {"Backdoors", "Exploits", "Shellcode"}
        high_categories = {"DoS", "Worms", "Reconnaissance"}
        
        if is_zero_day:
            return ThreatSeverity.CRITICAL
        
        if attack_category in critical_categories and confidence > 0.7:
            return ThreatSeverity.CRITICAL
        
        if attack_category in high_categories:
            return ThreatSeverity.HIGH
        
        if confidence > 0.8:
            return ThreatSeverity.MEDIUM
        
        return ThreatSeverity.LOW
    
    def process_batch(
        self,
        samples: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        include_rag_enrichment: bool = False,  # Disabled for performance
    ) -> List[SemanticThreatReport]:
        """
        Process a batch of network samples.
        
        Args:
            samples: Array of network samples (N x features).
            sample_ids: Optional list of sample identifiers.
            include_rag_enrichment: Whether to run RAG (slower).
        
        Returns:
            List of SemanticThreatReports.
        """
        reports = []
        n_samples = len(samples)
        
        if sample_ids is None:
            sample_ids = [f"batch_{i:06d}" for i in range(n_samples)]
        
        for i, (sample, sid) in enumerate(zip(samples, sample_ids)):
            report = self.process_network_sample(
                sample, sample_id=sid, include_rag_enrichment=include_rag_enrichment
            )
            reports.append(report)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")
        
        logger.info(f"Batch processing complete: {n_samples} samples")
        return reports
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict with samples_processed, anomalies_detected, zero_days_flagged.
        """
        return {
            "samples_processed": self._samples_processed,
            "anomalies_detected": self._anomalies_detected,
            "zero_days_flagged": self._zero_days_flagged,
            "anomaly_rate": (
                self._anomalies_detected / self._samples_processed
                if self._samples_processed > 0 else 0.0
            ),
        }
    
    def on_federated_update(
        self, round_number: int, aggregated_params: Any
    ) -> None:
        """
        Callback for federated learning updates.
        
        This method is called after each federated aggregation round
        to update local models with new global weights.
        
        Args:
            round_number: Current federation round.
            aggregated_params: New aggregated model parameters.
        """
        logger.info(f"Federated update received: round {round_number}")
        
        # Update Agent One (Autoencoder) if available
        if self.agent_one is not None and hasattr(self.agent_one, 'model'):
            try:
                from .utils import numpy_to_autoencoder_weights
                ae_weight_count = len(list(self.agent_one.model.state_dict().keys()))
                ae_weights = aggregated_params[:ae_weight_count]
                numpy_to_autoencoder_weights(self.agent_one.model, ae_weights)
                logger.info("Agent One weights updated")
            except Exception as e:
                logger.error(f"Failed to update Agent One: {e}")
        
        # Update Agent Two (XGBoost) if using neural classifier variant
        if self.agent_two is not None and hasattr(self.agent_two, 'update_model'):
            try:
                self.agent_two.update_model(aggregated_params)
                logger.info("Agent Two model updated")
            except Exception as e:
                logger.error(f"Failed to update Agent Two: {e}")
        
        # Trigger RAG re-analysis if configured
        if self._pending_samples_for_reanalysis:
            self._trigger_rag_reanalysis(round_number)
        
        # Reset stats for new model version
        self._samples_processed = 0
        self._anomalies_detected = 0
        self._zero_days_flagged = 0
        self._federated_round = round_number
    
    def _trigger_rag_reanalysis(self, round_number: int) -> None:
        """
        Re-analyze pending samples with updated model weights.
        
        This is called after federated updates to generate new
        explanations using the globally improved model.
        """
        if not self._pending_samples_for_reanalysis:
            return
        
        logger.info(
            f"Re-analyzing {len(self._pending_samples_for_reanalysis)} "
            f"samples after federated round {round_number}"
        )
        
        updated_reports = []
        for sample_info in self._pending_samples_for_reanalysis:
            try:
                report = self.process_network_sample(
                    sample_info["sample"],
                    sample_id=f"{sample_info['sample_id']}_r{round_number}",
                    include_rag_enrichment=True,
                )
                updated_reports.append(report)
            except Exception as e:
                logger.error(f"Re-analysis failed for {sample_info['sample_id']}: {e}")
        
        # Notify callbacks if configured
        if self._on_reanalysis_complete:
            self._on_reanalysis_complete(updated_reports, round_number)
        
        # Clear pending samples
        self._pending_samples_for_reanalysis.clear()
    
    def queue_for_reanalysis(
        self,
        sample: np.ndarray,
        sample_id: str,
    ) -> None:
        """
        Queue a sample for re-analysis after the next federated update.
        
        Use this for zero-day detections that may benefit from globally
        updated model weights.
        
        Args:
            sample: Network flow features.
            sample_id: Unique identifier for the sample.
        """
        self._pending_samples_for_reanalysis.append({
            "sample": sample.copy(),
            "sample_id": sample_id,
            "queued_at": datetime.now().isoformat(),
        })
        logger.debug(f"Queued sample {sample_id} for post-FL reanalysis")
    
    def set_reanalysis_callback(
        self,
        callback: callable,
    ) -> None:
        """
        Set callback for when re-analysis completes after federated update.
        
        Args:
            callback: Function(reports: List[SemanticThreatReport], round: int)
        """
        self._on_reanalysis_complete = callback
    
    def set_knowledge_base(self, knowledge_base: Any) -> None:
        """
        Configure the threat knowledge base for enhanced RAG.
        
        Args:
            knowledge_base: ThreatKnowledgeBase instance.
        """
        self._knowledge_base = knowledge_base
        logger.info("Knowledge base configured for coordinator")
    
    def get_enhanced_context(
        self,
        attack_category: str,
        mitre_technique: str,
        is_zero_day: bool,
    ) -> Tuple[str, List[str]]:
        """
        Get enhanced context from knowledge base for LLM analysis.
        
        Args:
            attack_category: Detected attack type.
            mitre_technique: MITRE technique ID.
            is_zero_day: Whether this is a potential zero-day.
        
        Returns:
            Tuple of (context_string, cve_references).
        """
        if self._knowledge_base is None:
            return "", []
        
        return self._knowledge_base.get_context_for_threat(
            attack_category=attack_category,
            mitre_technique=mitre_technique,
            is_zero_day=is_zero_day,
        )
    
    def analyze_with_federated_context(
        self,
        sample: np.ndarray,
        sample_id: Optional[str] = None,
    ) -> SemanticThreatReport:
        """
        Full analysis pipeline with federated context awareness.
        
        This method:
        1. Runs detection with current (federated) model weights
        2. For zero-days, queries knowledge base for similar patterns
        3. Generates explanation grounded in MITRE/CVE context
        4. Optionally queues for re-analysis after next FL round
        
        Args:
            sample: Network flow features.
            sample_id: Optional sample identifier.
        
        Returns:
            SemanticThreatReport with full analysis.
        """
        report = self.process_network_sample(
            sample,
            sample_id=sample_id,
            include_rag_enrichment=True,
        )
        
        # For zero-days, queue for re-analysis after next FL round
        if report.is_zero_day:
            self.queue_for_reanalysis(sample, report.sample_id)
            report.metadata["queued_for_reanalysis"] = True
            report.metadata["current_fl_round"] = getattr(self, '_federated_round', 0)
        
        return report
