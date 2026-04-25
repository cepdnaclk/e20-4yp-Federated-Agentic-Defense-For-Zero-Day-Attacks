"""Agent Three: RAG/LLM Action Recommendation Agent.

This agent is responsible for generating *actionable response steps* using a
knowledge base (RAG) and an LLM.

Per April 2026 refactor:
- Agent 2 is XGBoost-only classification (`agents/agent_two.py`).
- This Agent 3 focuses on action recommendations + explanation using the KB.
- The previous RL mitigation agent is preserved as `agents/agent_three_rl.py`.

The key output is a structured list of recommended actions that a SOC/IR team
can take, grounded in retrieved threat intelligence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.interfaces.base import LLMInterface, VectorDBInterface, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ActionRecommendation:
    """Structured action recommendation result from the LLM agent."""

    recommended_actions: List[str] = field(default_factory=list)
    primary_action: str = ""
    confidence: float = 0.0
    threat_summary: str = ""
    cve_references: List[str] = field(default_factory=list)
    model: Optional[str] = None
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_action": self.primary_action,
            "recommended_actions": self.recommended_actions,
            "confidence": self.confidence,
            "threat_summary": self.threat_summary,
            "cve_references": self.cve_references,
            "model": self.model,
        }


class AgentThree:
    """RAG/LLM-based action recommender (agent #3)."""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        vector_db: Optional[VectorDBInterface] = None,
        knowledge_base: Optional[Any] = None,
        context_k: int = 5,
    ):
        self._llm = llm
        self._vector_db = vector_db
        self._knowledge_base = knowledge_base
        self._context_k = int(context_k)

        logger.info(
            "AgentThree (RAG) initialized: llm=%s, vector_db=%s, knowledge_base=%s",
            getattr(llm, "model_name", None) if llm else "None",
            type(vector_db).__name__ if vector_db else "None",
            type(knowledge_base).__name__ if knowledge_base else "None",
        )

    def set_rag_system(self, vector_db: Optional[VectorDBInterface], llm: Optional[LLMInterface]) -> None:
        self._vector_db = vector_db
        self._llm = llm

    def set_knowledge_base(self, knowledge_base: Any) -> None:
        self._knowledge_base = knowledge_base

    def recommend_actions(
        self,
        attack_category: str,
        mitre_info: Optional[Dict[str, str]] = None,
        is_zero_day: bool = False,
        severity: str = "info",
        classification_confidence: float = 0.0,
        is_anomaly: bool = True,
    ) -> ActionRecommendation:
        """Generate a set of recommended actions using KB + LLM."""

        mitre_info = mitre_info or {}
        technique = mitre_info.get("technique", "")
        tactic = mitre_info.get("tactic", "")
        technique_desc = mitre_info.get("description", "")

        # If no LLM is configured, fall back to deterministic actions.
        if self._llm is None:
            actions = self._fallback_actions(attack_category=attack_category, severity=severity, is_zero_day=is_zero_day)
            return ActionRecommendation(
                recommended_actions=actions,
                primary_action=actions[0] if actions else "Monitor",
                confidence=0.4,
                threat_summary="LLM not configured; using rule-based recommendations.",
                cve_references=[],
                model=None,
                raw_response=None,
            )

        context_text, cves_from_kb = self._get_context(
            attack_category=attack_category,
            mitre_technique=technique,
            is_zero_day=is_zero_day,
        )

        prompt = self._build_prompt(
            attack_category=attack_category,
            technique=technique,
            tactic=tactic,
            technique_desc=technique_desc,
            is_zero_day=is_zero_day,
            severity=severity,
            classification_confidence=classification_confidence,
            is_anomaly=is_anomaly,
            context_text=context_text,
        )

        response: LLMResponse = self._llm.generate(prompt, temperature=0.2, max_tokens=1024)
        raw = response.content or ""

        actions = self._extract_actions(raw)
        if not actions:
            actions = self._fallback_actions(attack_category=attack_category, severity=severity, is_zero_day=is_zero_day)

        cves = sorted(set(self._extract_cves(raw) + cves_from_kb))
        threat_summary = raw.strip()

        return ActionRecommendation(
            recommended_actions=actions,
            primary_action=actions[0] if actions else "Monitor",
            confidence=0.7,
            threat_summary=threat_summary,
            cve_references=cves,
            model=getattr(self._llm, "model_name", None),
            raw_response=raw,
        )

    def _get_context(
        self,
        attack_category: str,
        mitre_technique: str,
        is_zero_day: bool,
    ) -> Tuple[str, List[str]]:
        # Prefer the ThreatKnowledgeBase interface if supplied.
        if self._knowledge_base is not None and hasattr(self._knowledge_base, "get_context_for_threat"):
            try:
                return self._knowledge_base.get_context_for_threat(
                    attack_category=attack_category,
                    mitre_technique=mitre_technique,
                    is_zero_day=is_zero_day,
                )
            except Exception as e:
                logger.warning("Knowledge base context fetch failed: %s", e)

        if self._vector_db is None:
            return "", []

        try:
            query = f"{attack_category} {mitre_technique} zero-day" if is_zero_day else f"{attack_category} {mitre_technique}"
            contexts = self._vector_db.similarity_search(query, k=self._context_k)
            context_text = "\n\n---\n\n".join([c.content for c in contexts])
            return context_text, []
        except Exception as e:
            logger.warning("Vector DB context fetch failed: %s", e)
            return "", []

    def _build_prompt(
        self,
        attack_category: str,
        technique: str,
        tactic: str,
        technique_desc: str,
        is_zero_day: bool,
        severity: str,
        classification_confidence: float,
        is_anomaly: bool,
        context_text: str,
    ) -> str:
        zero_day_line = "Yes" if is_zero_day else "No"
        anomaly_line = "Yes" if is_anomaly else "No"

        return f"""You are a cybersecurity incident responder. Produce an actionable response plan.

## Detection Summary
- Attack Category: {attack_category}
- Classification Confidence: {classification_confidence:.2f}
- Potential Zero-Day: {zero_day_line}
- Anomaly Detected: {anomaly_line}
- Severity (system): {severity}

## MITRE Context
- Technique: {technique}
- Tactic: {tactic}
- Description: {technique_desc}

## Retrieved Intelligence (Knowledge Base)
{context_text}

## Output Format (IMPORTANT)
Return:
1) A short threat summary (2-4 sentences)
2) A section titled 'Recommended Actions' with 3-7 bullet points, ordered by priority.
3) If any CVEs are relevant, list them as CVE-YYYY-NNNN.
"""

    def _extract_actions(self, text: str) -> List[str]:
        # Try to locate the "Recommended Actions" section.
        lowered = text.lower()
        idx = lowered.find("recommended actions")
        if idx != -1:
            section = text[idx:]
        else:
            section = text

        lines = [ln.strip() for ln in section.splitlines()]
        actions: List[str] = []
        for ln in lines:
            if not ln:
                continue

            # Stop if we hit another major heading after capturing some actions.
            if actions and (ln.startswith("##") or ln.lower().startswith("severity") or ln.lower().startswith("indicators")):
                break

            bullet = None
            if ln.startswith("-") or ln.startswith("*"):
                bullet = ln[1:].strip()
            else:
                m = re.match(r"^\d+\.?\s+(.*)$", ln)
                if m:
                    bullet = m.group(1).strip()

            if bullet:
                # Avoid capturing empty or ultra-long paragraphs as "actions".
                if 3 <= len(bullet) <= 200:
                    actions.append(bullet)

        # De-dup while preserving order
        seen = set()
        uniq: List[str] = []
        for a in actions:
            key = a.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(a)
        return uniq

    def _extract_cves(self, text: str) -> List[str]:
        return sorted(set(re.findall(r"CVE-\d{4}-\d{4,7}", text, flags=re.IGNORECASE)))

    def _fallback_actions(self, attack_category: str, severity: str, is_zero_day: bool) -> List[str]:
        base = {
            "DoS": [
                "Rate-limit affected services",
                "Block high-volume sources at edge controls",
                "Enable/verify DDoS protections and alerts",
            ],
            "Exploits": [
                "Isolate affected systems",
                "Patch vulnerable services and validate versions",
                "Hunt for post-exploitation activity in logs",
            ],
            "Reconnaissance": [
                "Enable enhanced logging on probed services",
                "Block or tarpitting suspicious scanners",
                "Review exposed services and tighten firewall rules",
            ],
            "Normal": [
                "No action required; continue monitoring",
            ],
        }.get(attack_category, ["Investigate the traffic and collect logs", "Monitor for recurrence"])

        if is_zero_day:
            base = [
                "Preserve evidence (pcaps, logs) for deeper analysis",
                "Escalate to incident response and threat hunting",
            ] + base

        if severity in {"critical", "high"}:
            base = ["Act immediately; treat as high-priority incident"] + base

        return base
