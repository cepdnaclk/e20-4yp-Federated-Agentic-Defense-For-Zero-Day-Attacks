"""
Explanation Evaluation Utilities

This module provides metrics and evaluation tools for assessing the quality of
RAG-generated threat explanations. These metrics are essential for validating
the core novelty: that federated model updates improve explanation quality.

Evaluation Dimensions:
1. Factual Accuracy: Do explanations correctly cite CVEs/MITRE techniques?
2. Relevance: Does the explanation match the detected threat type?
3. Completeness: Does it cover severity, indicators, and actions?
4. Consistency: Do explanations remain consistent across similar threats?
5. Improvement Over Rounds: Does quality improve with federated learning?
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

@dataclass
class ExplanationMetrics:
    """Metrics for a single explanation."""
    sample_id: str
    federated_round: int
    
    # Factual accuracy
    cve_accuracy: float = 0.0  # % of cited CVEs that are valid
    mitre_accuracy: float = 0.0  # % of MITRE techniques that match attack
    
    # Relevance
    category_match: bool = False  # Does explanation match detected category?
    technique_relevance: float = 0.0  # Semantic relevance score [0-1]
    
    # Completeness
    has_severity: bool = False
    has_indicators: bool = False
    has_actions: bool = False
    completeness_score: float = 0.0  # Combined completeness [0-1]
    
    # Quality
    confidence: float = 0.0
    explanation_length: int = 0
    specificity_score: float = 0.0  # How specific vs generic
    
    # Timing
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "federated_round": self.federated_round,
            "cve_accuracy": self.cve_accuracy,
            "mitre_accuracy": self.mitre_accuracy,
            "category_match": self.category_match,
            "technique_relevance": self.technique_relevance,
            "completeness": {
                "has_severity": self.has_severity,
                "has_indicators": self.has_indicators,
                "has_actions": self.has_actions,
                "score": self.completeness_score,
            },
            "quality": {
                "confidence": self.confidence,
                "length": self.explanation_length,
                "specificity": self.specificity_score,
            },
            "latency_ms": self.latency_ms,
        }
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score [0-1]."""
        weights = {
            "cve_accuracy": 0.15,
            "mitre_accuracy": 0.20,
            "category_match": 0.15,
            "completeness": 0.25,
            "specificity": 0.15,
            "confidence": 0.10,
        }
        
        score = (
            weights["cve_accuracy"] * self.cve_accuracy +
            weights["mitre_accuracy"] * self.mitre_accuracy +
            weights["category_match"] * (1.0 if self.category_match else 0.0) +
            weights["completeness"] * self.completeness_score +
            weights["specificity"] * self.specificity_score +
            weights["confidence"] * self.confidence
        )
        
        return min(1.0, max(0.0, score))


@dataclass
class RoundMetrics:
    """Aggregated metrics for a federated round."""
    round_number: int
    num_samples: int
    
    avg_cve_accuracy: float = 0.0
    avg_mitre_accuracy: float = 0.0
    category_match_rate: float = 0.0
    avg_completeness: float = 0.0
    avg_specificity: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    
    overall_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_number,
            "samples": self.num_samples,
            "cve_accuracy": self.avg_cve_accuracy,
            "mitre_accuracy": self.avg_mitre_accuracy,
            "category_match_rate": self.category_match_rate,
            "completeness": self.avg_completeness,
            "specificity": self.avg_specificity,
            "confidence": self.avg_confidence,
            "latency_ms": self.avg_latency_ms,
            "overall_quality": self.overall_quality,
        }


@dataclass
class ImprovementReport:
    """Report on explanation quality improvement over federated rounds."""
    start_round: int
    end_round: int
    rounds_analyzed: int
    
    quality_trend: List[float] = field(default_factory=list)
    quality_improvement: float = 0.0  # % improvement
    quality_slope: float = 0.0  # Trend slope
    
    cve_accuracy_trend: List[float] = field(default_factory=list)
    mitre_accuracy_trend: List[float] = field(default_factory=list)
    completeness_trend: List[float] = field(default_factory=list)
    
    significant_improvements: List[Dict[str, Any]] = field(default_factory=list)
    regressions: List[Dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# Evaluation Functions
# ==============================================================================

class ExplanationEvaluator:
    """
    Evaluator for RAG-generated threat explanations.
    
    This class provides comprehensive evaluation of explanation quality,
    validating that federated learning improves explanations over time.
    """
    
    def __init__(
        self,
        valid_cves: Optional[Set[str]] = None,
        mitre_techniques: Optional[Dict[str, Dict[str, Any]]] = None,
        attack_categories: Optional[List[str]] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            valid_cves: Set of valid CVE IDs for accuracy checking
            mitre_techniques: Dict of MITRE technique IDs to metadata
            attack_categories: List of valid attack category names
        """
        self.valid_cves = valid_cves or set()
        self.mitre_techniques = mitre_techniques or {}
        self.attack_categories = attack_categories or [
            "Normal", "DoS", "Reconnaissance", "Exploits", 
            "Fuzzers", "Generic", "Analysis", "Backdoor", 
            "Shellcode", "Worms"
        ]
        
        # Track metrics over rounds
        self._round_metrics: Dict[int, List[ExplanationMetrics]] = defaultdict(list)
        self._all_metrics: List[ExplanationMetrics] = []
        
        # Patterns for parsing explanations
        self._severity_patterns = [
            r'\b(critical|high|medium|low|severe|severe)\b',
            r'severity[:\s]+(critical|high|medium|low)',
            r'\*\*severity\*\*[:\s]*(critical|high|medium|low)',
        ]
        
        self._indicator_patterns = [
            r'\b(indicator|ioc|sign|symptom|characteristic)s?\b',
            r'look for|monitor for|watch for',
            r'traffic pattern|packet signature|connection attempt',
        ]
        
        self._action_patterns = [
            r'\b(recommend|action|response|mitigation|remediation)s?\b',
            r'should (block|isolate|investigate|monitor)',
            r'immediately|urgently|as soon as',
        ]
        
        logger.info("ExplanationEvaluator initialized")
    
    def load_knowledge_base_references(self, knowledge_base: Any) -> None:
        """Load CVEs and MITRE techniques from a knowledge base."""
        if hasattr(knowledge_base, 'cve_ids'):
            self.valid_cves = set(knowledge_base.cve_ids)
        
        if hasattr(knowledge_base, 'mitre_techniques'):
            self.mitre_techniques = knowledge_base.mitre_techniques
    
    def evaluate_explanation(
        self,
        explanation: str,
        detected_category: str,
        cited_cves: List[str],
        cited_mitre: List[str],
        confidence: float,
        federated_round: int,
        latency_ms: float,
        sample_id: str = "",
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> ExplanationMetrics:
        """
        Evaluate a single explanation.
        
        Args:
            explanation: The generated explanation text
            detected_category: The attack category detected
            cited_cves: CVE IDs cited in the explanation
            cited_mitre: MITRE technique IDs cited
            confidence: Model confidence for the detection
            federated_round: Which FL round this is from
            latency_ms: Time to generate explanation
            sample_id: Unique sample identifier
            ground_truth: Optional ground truth for accuracy
            
        Returns:
            ExplanationMetrics with evaluation results
        """
        metrics = ExplanationMetrics(
            sample_id=sample_id or f"sample_{len(self._all_metrics)}",
            federated_round=federated_round,
            confidence=confidence,
            latency_ms=latency_ms,
            explanation_length=len(explanation),
        )
        
        # CVE Accuracy
        if cited_cves:
            valid_count = sum(1 for cve in cited_cves if cve in self.valid_cves)
            metrics.cve_accuracy = valid_count / len(cited_cves)
        else:
            metrics.cve_accuracy = 1.0  # No claims = no errors
        
        # MITRE Accuracy
        if cited_mitre:
            valid_count = sum(1 for tech in cited_mitre if tech in self.mitre_techniques)
            metrics.mitre_accuracy = valid_count / len(cited_mitre)
        else:
            metrics.mitre_accuracy = 1.0
        
        # Category match
        metrics.category_match = self._check_category_match(
            explanation, detected_category
        )
        
        # Technique relevance
        metrics.technique_relevance = self._calculate_technique_relevance(
            cited_mitre, detected_category
        )
        
        # Completeness checks
        explanation_lower = explanation.lower()
        
        metrics.has_severity = any(
            re.search(p, explanation_lower, re.IGNORECASE)
            for p in self._severity_patterns
        )
        
        metrics.has_indicators = any(
            re.search(p, explanation_lower, re.IGNORECASE)
            for p in self._indicator_patterns
        )
        
        metrics.has_actions = any(
            re.search(p, explanation_lower, re.IGNORECASE)
            for p in self._action_patterns
        )
        
        completeness_factors = [
            metrics.has_severity,
            metrics.has_indicators,
            metrics.has_actions,
            len(explanation) > 100,  # Minimum substantive length
            len(cited_cves) > 0 or len(cited_mitre) > 0,  # Has references
        ]
        metrics.completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Specificity score
        metrics.specificity_score = self._calculate_specificity(
            explanation, detected_category, cited_cves, cited_mitre
        )
        
        # Compare with ground truth if provided
        if ground_truth:
            metrics = self._adjust_for_ground_truth(metrics, ground_truth)
        
        # Track metrics
        self._round_metrics[federated_round].append(metrics)
        self._all_metrics.append(metrics)
        
        return metrics
    
    def _check_category_match(
        self,
        explanation: str,
        detected_category: str,
    ) -> bool:
        """Check if explanation mentions the detected category appropriately."""
        category_lower = detected_category.lower()
        explanation_lower = explanation.lower()
        
        # Direct mention
        if category_lower in explanation_lower:
            return True
        
        # Category synonyms
        synonyms = {
            "dos": ["denial of service", "ddos", "flooding", "resource exhaustion"],
            "reconnaissance": ["scan", "probe", "enumeration", "discovery", "recon"],
            "exploits": ["exploit", "vulnerability", "attack vector", "payload"],
            "fuzzers": ["fuzz", "malformed", "invalid input", "boundary"],
            "backdoor": ["backdoor", "trojan", "remote access", "persistence"],
            "shellcode": ["shellcode", "code execution", "payload", "buffer overflow"],
            "worms": ["worm", "self-propagating", "spreading", "lateral movement"],
            "analysis": ["analysis", "protocol", "inspection", "fingerprint"],
        }
        
        related_terms = synonyms.get(category_lower, [])
        return any(term in explanation_lower for term in related_terms)
    
    def _calculate_technique_relevance(
        self,
        cited_mitre: List[str],
        detected_category: str,
    ) -> float:
        """Calculate how relevant cited MITRE techniques are to detected category."""
        if not cited_mitre:
            return 0.5  # Neutral if no techniques cited
        
        # Map categories to expected MITRE tactics
        category_tactics = {
            "DoS": ["impact"],
            "Reconnaissance": ["reconnaissance", "discovery"],
            "Exploits": ["initial-access", "execution", "privilege-escalation"],
            "Fuzzers": ["initial-access", "execution"],
            "Backdoor": ["persistence", "command-and-control"],
            "Shellcode": ["execution"],
            "Worms": ["lateral-movement", "execution"],
            "Analysis": ["reconnaissance", "collection"],
        }
        
        expected_tactics = category_tactics.get(detected_category, [])
        if not expected_tactics:
            return 0.5
        
        relevant_count = 0
        for tech_id in cited_mitre:
            tech_info = self.mitre_techniques.get(tech_id, {})
            tech_tactic = tech_info.get("tactic", "").lower().replace(" ", "-")
            if tech_tactic in expected_tactics:
                relevant_count += 1
        
        return relevant_count / len(cited_mitre)
    
    def _calculate_specificity(
        self,
        explanation: str,
        detected_category: str,
        cited_cves: List[str],
        cited_mitre: List[str],
    ) -> float:
        """
        Calculate how specific (vs generic) the explanation is.
        
        A specific explanation:
        - Mentions specific CVEs or techniques
        - Includes specific numbers, IPs, or indicators
        - References particular tools or methods
        - Avoids generic phrases like "may be malicious"
        """
        specificity_indicators = [
            # Specific references
            len(cited_cves) > 0,
            len(cited_mitre) > 0,
            
            # Specific patterns
            bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', explanation)),  # IP
            bool(re.search(r'port \d+', explanation, re.IGNORECASE)),  # Port
            bool(re.search(r'\d+ (?:bytes|packets|connections)', explanation)),  # Numbers
            
            # Tool/technique mentions
            bool(re.search(r'nmap|metasploit|cobalt|mimikatz|powershell', explanation, re.IGNORECASE)),
            
            # Specific category mention
            detected_category.lower() in explanation.lower(),
        ]
        
        generic_indicators = [
            # Vague phrases reduce specificity
            "may be" in explanation.lower(),
            "possibly" in explanation.lower(),
            "could be" in explanation.lower(),
            "unknown" in explanation.lower(),
            "generic" in explanation.lower(),
        ]
        
        specific_score = sum(specificity_indicators) / len(specificity_indicators)
        generic_penalty = sum(generic_indicators) * 0.1
        
        return max(0.0, min(1.0, specific_score - generic_penalty))
    
    def _adjust_for_ground_truth(
        self,
        metrics: ExplanationMetrics,
        ground_truth: Dict[str, Any],
    ) -> ExplanationMetrics:
        """Adjust metrics based on ground truth when available."""
        # If we have true category
        if "category" in ground_truth:
            explanation_mentions_truth = ground_truth["category"].lower() in metrics.sample_id.lower()
            # Could adjust metrics.category_match based on true label
        
        # If we have true CVEs
        if "cves" in ground_truth:
            true_cves = set(ground_truth["cves"])
            # Could calculate precision/recall for CVE citations
        
        return metrics
    
    def aggregate_round_metrics(self, round_number: int) -> Optional[RoundMetrics]:
        """Aggregate all metrics for a federated round."""
        round_data = self._round_metrics.get(round_number, [])
        
        if not round_data:
            return None
        
        n = len(round_data)
        
        round_metrics = RoundMetrics(
            round_number=round_number,
            num_samples=n,
            avg_cve_accuracy=sum(m.cve_accuracy for m in round_data) / n,
            avg_mitre_accuracy=sum(m.mitre_accuracy for m in round_data) / n,
            category_match_rate=sum(1 for m in round_data if m.category_match) / n,
            avg_completeness=sum(m.completeness_score for m in round_data) / n,
            avg_specificity=sum(m.specificity_score for m in round_data) / n,
            avg_confidence=sum(m.confidence for m in round_data) / n,
            avg_latency_ms=sum(m.latency_ms for m in round_data) / n,
        )
        
        # Calculate overall quality
        round_metrics.overall_quality = sum(m.overall_score for m in round_data) / n
        
        return round_metrics
    
    def generate_improvement_report(
        self,
        start_round: Optional[int] = None,
        end_round: Optional[int] = None,
    ) -> ImprovementReport:
        """
        Generate a report on explanation quality improvement over rounds.
        
        This is the key metric for validating that federated learning
        improves threat explanation quality.
        """
        rounds = sorted(self._round_metrics.keys())
        
        if not rounds:
            return ImprovementReport(
                start_round=0,
                end_round=0,
                rounds_analyzed=0,
            )
        
        start = start_round if start_round is not None else rounds[0]
        end = end_round if end_round is not None else rounds[-1]
        
        relevant_rounds = [r for r in rounds if start <= r <= end]
        
        report = ImprovementReport(
            start_round=start,
            end_round=end,
            rounds_analyzed=len(relevant_rounds),
        )
        
        if not relevant_rounds:
            return report
        
        # Collect metrics per round
        for round_num in relevant_rounds:
            round_metrics = self.aggregate_round_metrics(round_num)
            if round_metrics:
                report.quality_trend.append(round_metrics.overall_quality)
                report.cve_accuracy_trend.append(round_metrics.avg_cve_accuracy)
                report.mitre_accuracy_trend.append(round_metrics.avg_mitre_accuracy)
                report.completeness_trend.append(round_metrics.avg_completeness)
        
        if len(report.quality_trend) >= 2:
            # Calculate improvement
            first_quality = report.quality_trend[0]
            last_quality = report.quality_trend[-1]
            
            if first_quality > 0:
                report.quality_improvement = ((last_quality - first_quality) / first_quality) * 100
            
            # Calculate trend slope using linear regression
            x = np.arange(len(report.quality_trend))
            y = np.array(report.quality_trend)
            
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
                report.quality_slope = slope
            
            # Find significant improvements and regressions
            for i in range(1, len(report.quality_trend)):
                change = report.quality_trend[i] - report.quality_trend[i-1]
                
                if change > 0.05:  # 5% improvement threshold
                    report.significant_improvements.append({
                        "from_round": relevant_rounds[i-1],
                        "to_round": relevant_rounds[i],
                        "improvement": change,
                    })
                elif change < -0.05:  # 5% regression threshold
                    report.regressions.append({
                        "from_round": relevant_rounds[i-1],
                        "to_round": relevant_rounds[i],
                        "regression": abs(change),
                    })
        
        return report
    
    def get_best_round(self) -> Optional[Tuple[int, float]]:
        """Get the round with best overall explanation quality."""
        best_round = None
        best_quality = -1
        
        for round_num in self._round_metrics:
            metrics = self.aggregate_round_metrics(round_num)
            if metrics and metrics.overall_quality > best_quality:
                best_quality = metrics.overall_quality
                best_round = round_num
        
        return (best_round, best_quality) if best_round else None
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external analysis."""
        return {
            "total_samples": len(self._all_metrics),
            "rounds_tracked": len(self._round_metrics),
            "per_round": {
                round_num: self.aggregate_round_metrics(round_num).to_dict()
                for round_num in self._round_metrics
            },
            "improvement": self.generate_improvement_report().__dict__,
        }


# ==============================================================================
# Zero-Day Explanation Evaluator
# ==============================================================================

class ZeroDayExplanationEvaluator:
    """
    Specialized evaluator for zero-day threat explanations.
    
    Zero-day explanations require different evaluation criteria:
    - Cannot verify against known CVEs
    - Should emphasize anomaly explanation
    - Should provide comparison to nearest known patterns
    - Should express appropriate uncertainty
    """
    
    def __init__(self, known_patterns: Optional[Dict[str, Dict[str, Any]]] = None):
        self.known_patterns = known_patterns or {}
        self._evaluations: List[Dict[str, Any]] = []
    
    def evaluate_zero_day_explanation(
        self,
        explanation: str,
        detection_confidence: float,
        anomaly_score: float,
        nearest_category: str,
        nearest_pattern_similarity: float,
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate a zero-day threat explanation.
        
        Returns:
            Dict with evaluation metrics specific to zero-day explanations
        """
        result = {
            "sample_id": sample_id,
            "detection_confidence": detection_confidence,
            "anomaly_score": anomaly_score,
            "nearest_category": nearest_category,
            "nearest_similarity": nearest_pattern_similarity,
        }
        
        explanation_lower = explanation.lower()
        
        # Check for appropriate uncertainty expression
        uncertainty_phrases = [
            "potential", "possible", "suspected", "may be",
            "resembles", "similar to", "appears to be",
            "further investigation", "requires analysis",
        ]
        result["expresses_uncertainty"] = any(
            phrase in explanation_lower for phrase in uncertainty_phrases
        )
        
        # Check for comparison to known patterns
        result["compares_to_known"] = (
            nearest_category.lower() in explanation_lower or
            "similar" in explanation_lower or
            "resembles" in explanation_lower
        )
        
        # Check for investigation guidance
        investigation_phrases = [
            "investigate", "analyze", "examine", "review",
            "correlate", "monitor", "track", "collect",
        ]
        result["provides_investigation_steps"] = any(
            phrase in explanation_lower for phrase in investigation_phrases
        )
        
        # Check for anomaly explanation
        anomaly_phrases = [
            "anomal", "unusual", "abnormal", "deviation",
            "unexpected", "outlier", "novel", "new pattern",
        ]
        result["explains_anomaly"] = any(
            phrase in explanation_lower for phrase in anomaly_phrases
        )
        
        # Calculate zero-day explanation score
        score_factors = [
            result["expresses_uncertainty"],
            result["compares_to_known"],
            result["provides_investigation_steps"],
            result["explains_anomaly"],
            detection_confidence > 0.5,
            len(explanation) > 150,
        ]
        result["zero_day_score"] = sum(score_factors) / len(score_factors)
        
        self._evaluations.append(result)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics for zero-day evaluations."""
        if not self._evaluations:
            return {"total_evaluated": 0}
        
        n = len(self._evaluations)
        return {
            "total_evaluated": n,
            "avg_zero_day_score": sum(e["zero_day_score"] for e in self._evaluations) / n,
            "uncertainty_rate": sum(1 for e in self._evaluations if e["expresses_uncertainty"]) / n,
            "comparison_rate": sum(1 for e in self._evaluations if e["compares_to_known"]) / n,
            "investigation_rate": sum(1 for e in self._evaluations if e["provides_investigation_steps"]) / n,
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_evaluator_from_knowledge_base(knowledge_base: Any) -> ExplanationEvaluator:
    """Create an evaluator pre-configured with knowledge base data."""
    evaluator = ExplanationEvaluator()
    
    if knowledge_base:
        # Load CVEs
        if hasattr(knowledge_base, 'cve_data'):
            evaluator.valid_cves = set(knowledge_base.cve_data.keys())
        
        # Load MITRE techniques
        if hasattr(knowledge_base, 'mitre_techniques'):
            evaluator.mitre_techniques = knowledge_base.mitre_techniques
    
    return evaluator


def create_evaluation_callback(
    evaluator: ExplanationEvaluator,
) -> Callable[[Dict[str, Any]], ExplanationMetrics]:
    """
    Create a callback function for integration with FederatedRAGBridge.
    
    Usage:
        evaluator = ExplanationEvaluator()
        callback = create_evaluation_callback(evaluator)
        bridge.set_explanation_callback(callback)
    """
    def evaluate_callback(result_dict: Dict[str, Any]) -> ExplanationMetrics:
        return evaluator.evaluate_explanation(
            explanation=result_dict.get("explanation", ""),
            detected_category=result_dict.get("category", "Unknown"),
            cited_cves=result_dict.get("cve_references", []),
            cited_mitre=result_dict.get("mitre_techniques", []),
            confidence=result_dict.get("confidence", 0.0),
            federated_round=result_dict.get("federated_round", 0),
            latency_ms=result_dict.get("processing_time_ms", 0.0),
            sample_id=result_dict.get("sample_id", ""),
        )
    
    return evaluate_callback
