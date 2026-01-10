import csv
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: Please set your GROQ_API_KEY environment variable.")
    sys.exit(1)


class SuspiciousAgentState:
    """
    State for the Suspicious Traffic Agent.
    """
    def __init__(self):
        self.messages: List[str] = []
        self.triage_data: Dict[str, Any] = {}
        self.verification_status: str = ""
        self.action_plan: str = ""
        self.kb_context: str = ""
        self.likely_attack_category: str = ""
        self.mitigation_plan: Dict[str, Any] = {}

# Structured Output for the verification decision
class VerificationDecision(BaseModel):
    is_threat: bool = Field(description="True if this is a real threat, False if benign/false positive.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation for the decision.")
    recommended_action: str = Field(description="Action to take: 'Block IP', 'Quarantine Host', 'Ignore', etc.")


class MitigationPlan(BaseModel):
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Severity based on confidence, trend, recurrence, and KB evidence."
    )
    likely_attack_category: str = Field(
        description="Best-effort category guess (e.g., Reconnaissance, Exploits, DoS, Generic, Normal)."
    )
    immediate_actions: List[str] = Field(description="Actions to take right now (first 15 minutes).")
    containment_actions: List[str] = Field(description="Containment steps to stop spread/impact.")
    monitoring_actions: List[str] = Field(description="What to watch/alert on next.")
    iocs_to_collect: List[str] = Field(description="IOCs to capture (IPs, ports, flow features, logs).")
    notes: str = Field(description="Short rationale and any assumptions.")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# KNOWLEDGE BASE (Vector DB)


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
KB_FILE = Path(__file__).resolve().parent / "UNSW_NB15_training-set.csv"

# UNSW_NB15_training-set.csv is large (~175k rows). Embedding every row is expensive.
# Keep a representative subset so retrieval is fast and repeatable.
MAX_DOCS_TOTAL = 3000
MAX_DOCS_PER_ATTACK_CAT = 250

suspicious_vector_store: Optional[FAISS] = None
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_mock_kb_data() -> List[Document]:
    return [
        Document(page_content="UNSW_NB15: attack_cat=Normal label=0 proto=tcp service=- state=FIN spkts=6 dpkts=4 sbytes=258 dbytes=172 rate=74.08749"),
        Document(page_content="UNSW_NB15: attack_cat=Exploits label=1 proto=tcp service=http state=FIN spkts=12 dpkts=10 sbytes=900 dbytes=1200 rate=25.1"),
        Document(page_content="UNSW_NB15: attack_cat=Reconnaissance label=1 proto=tcp service=- state=CON spkts=20 dpkts=0 sbytes=1200 dbytes=0 rate=300.0"),
    ]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _row_to_unsw_text(row: Dict[str, Any]) -> str:
    """Convert an UNSW_NB15 row (many numeric fields) into compact text for embedding."""

    attack_cat = (row.get("attack_cat") or "").strip() or "Unknown"
    label = (row.get("label") or "").strip()

    # Keep a small subset of high-signal fields
    keys = [
        "proto",
        "service",
        "state",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sttl",
        "dttl",
        "sload",
        "dload",
        "tcprtt",
        "synack",
        "ackdat",
        "trans_depth",
        "response_body_len",
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "ct_flw_http_mthd",
        "is_sm_ips_ports",
    ]

    parts: List[str] = [f"UNSW_NB15: attack_cat={attack_cat}"]
    if label != "":
        parts.append(f"label={label}")

    for key in keys:
        val = row.get(key)
        if val is None:
            continue
        sval = str(val).strip()
        if sval == "" or sval == "-":
            continue
        parts.append(f"{key}={sval}")

    return " ".join(parts)


def initialize_suspicious_kb():
    """Creates a local in-memory Knowledge Base (FAISS) for Suspicious verification."""
    global suspicious_vector_store
    if suspicious_vector_store is not None:
        return

    print("[INIT] Building Suspicious Knowledge Base (FAISS)...")

    docs: List[Document] = []
    if KB_FILE.exists():
        try:
            per_cat_counts: Dict[str, int] = {}
            with open(KB_FILE, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    attack_cat = (row.get("attack_cat") or "Unknown").strip() or "Unknown"
                    label = _to_int(row.get("label"), default=0)

                    # Prefer malicious examples (label=1), but keep some normal too.
                    # Cap by category to get balanced coverage.
                    current = per_cat_counts.get(attack_cat, 0)
                    if current >= MAX_DOCS_PER_ATTACK_CAT:
                        continue

                    # For "Normal" only keep a smaller subset.
                    if attack_cat.lower() == "normal" and current >= max(25, MAX_DOCS_PER_ATTACK_CAT // 8):
                        continue

                    text = _row_to_unsw_text(row)
                    if not text:
                        continue

                    docs.append(Document(page_content=text))
                    per_cat_counts[attack_cat] = current + 1

                    if len(docs) >= MAX_DOCS_TOTAL:
                        break

            print(f"Loaded {len(docs)} sampled records from {KB_FILE.name}")
        except Exception:
            docs = get_mock_kb_data()
    else:
        print("KB CSV not found. Using mock KB data.")
        docs = get_mock_kb_data()

    suspicious_vector_store = FAISS.from_documents(docs, embedding_model)
    print("[INIT] Suspicious Knowledge Base Ready.")


def retrieve_kb_context(query: str, k: int = 4) -> str:
    initialize_suspicious_kb()
    if suspicious_vector_store is None:
        return ""
    docs = suspicious_vector_store.similarity_search(query, k=k)
    return "\n".join([f"- {d.page_content}" for d in docs])


def infer_attack_category_from_kb(kb_context: str) -> str:
    """Infer likely attack category from retrieved UNSW samples."""
    counts: Dict[str, int] = {}
    for line in kb_context.splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines look like: "- UNSW_NB15: attack_cat=Exploits label=1 ..."
        lower = line.lower()
        if "attack_cat=" not in lower:
            continue
        # Extract the token after attack_cat=
        try:
            start = lower.index("attack_cat=") + len("attack_cat=")
            rest = line[start:]
            token = rest.split()[0]
            cat = token.strip().strip(",;")
        except Exception:
            continue
        if not cat:
            continue
        # Normalize common variants
        cat_norm = cat.capitalize() if cat.islower() else cat
        counts[cat_norm] = counts.get(cat_norm, 0) + 1

    if not counts:
        return "Unknown"

    # Prefer non-Normal if present
    sorted_cats = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    for cat, _ in sorted_cats:
        if cat.lower() != "normal":
            return cat
    return sorted_cats[0][0]


def mitigation_playbook(likely_attack_category: str, trend: str, recent_count: int) -> List[str]:
    """Deterministic baseline mitigation steps by category."""
    cat = (likely_attack_category or "").lower()
    base = [
        "Preserve evidence: snapshot logs/flows for the alert window.",
        "Identify source/destination IPs and affected asset(s).",
        "Increase monitoring/alerting for recurrence over next 60 minutes.",
    ]

    if trend == "increasing" or recent_count >= 10:
        base.append("Escalate to SOC Tier 2 (active campaign suspected).")

    if "recon" in cat:
        return base + [
            "Rate-limit or block scanning source IPs at the edge.",
            "Harden exposed services; verify firewall rules and open ports.",
            "Enable/verify IDS signatures for port scans and service probing.",
        ]
    if "dos" in cat:
        return base + [
            "Apply rate-limiting / DDoS protections; engage upstream mitigation if needed.",
            "Block or tarpitting suspicious sources; enforce SYN cookies / connection limits.",
            "Validate service health; scale or failover critical endpoints.",
        ]
    if "exploit" in cat:
        return base + [
            "Quarantine impacted host(s) if exploitation suspected.",
            "Patch vulnerable services or temporarily disable the exposed surface.",
            "Collect process tree, auth logs, and web/app logs for post-exploitation evidence.",
        ]
    if "backdoor" in cat or "shellcode" in cat or "worm" in cat:
        return base + [
            "Isolate host immediately from the network (containment priority).",
            "Run EDR scan; capture memory dump if feasible.",
            "Reset credentials and rotate keys if compromise is possible.",
        ]

    # Generic / unknown
    return base + [
        "Block high-confidence suspicious indicators (IPs/ports) if available.",
        "Run targeted endpoint scan on affected assets.",
    ]


def build_mitigation_plan(semantic_summary: str, kb_context: str, trend: str, recent_count: int) -> Dict[str, Any]:
    """Combine KB evidence + agent knowledge to produce a mitigation plan."""

    likely_cat = infer_attack_category_from_kb(kb_context)
    baseline_steps = mitigation_playbook(likely_cat, trend, recent_count)

    # If KB is missing or LLM unavailable, return deterministic plan.
    # (We still prefer deterministic steps for reliability.)
    try:
        structured_llm = llm.with_structured_output(MitigationPlan)
        prompt = f"""
        You are a SOC response assistant. Produce a concise mitigation plan.

        ALERT SUMMARY:
        {semantic_summary}

        TEMPORAL SIGNALS:
        - trend: {trend}
        - recent_anomalies: {recent_count}

        UNSW VECTOR KB CONTEXT (similar historical flows):
        {kb_context if kb_context else "(no KB context)"}

        BASELINE PLAYBOOK STEPS (agent knowledge):
        {chr(10).join([f"- {s}" for s in baseline_steps])}

        Constraints:
        - Be actionable and SOC-friendly.
        - Prefer containment first, then monitoring.
        - Do not invent specific IPs unless provided.
        """

        plan: MitigationPlan = structured_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Generate the mitigation plan."),
        ])
        return plan.model_dump()
    except Exception:
        return {
            "severity": "High" if (trend == "increasing" or recent_count >= 10) else "Medium",
            "likely_attack_category": likely_cat,
            "immediate_actions": baseline_steps[:4],
            "containment_actions": baseline_steps[4:],
            "monitoring_actions": [
                "Alert on repeated similar flows and spikes in rate/sbytes/dbytes.",
                "Correlate with auth failures and new outbound destinations.",
            ],
            "iocs_to_collect": [
                "src_ip, dst_ip, src_port, dst_port (if available)",
                "proto, service, state", 
                "time window for pcap/log retention",
            ],
            "notes": "Fallback deterministic plan (LLM structured output unavailable).",
        }


def _heuristic_override(semantic_summary: str, kb_context: str, trend: str, recent_count: int) -> Optional[VerificationDecision]:
    """Lightweight deterministic boosts for very clear known-signature matches."""
    text = (semantic_summary + "\n" + kb_context).lower()

    if "sqlmap" in text:
        return VerificationDecision(
            is_threat=True,
            confidence=0.95,
            reasoning="Detected strong known signature match: sqlmap user agent / automated SQLi tooling.",
            recommended_action="Block source IP; add WAF rule; inspect web server logs.",
        )

    if "port 4444" in text or "metasploit" in text:
        return VerificationDecision(
            is_threat=True,
            confidence=0.93,
            reasoning="Detected strong known compromise indicator: Metasploit default port 4444.",
            recommended_action="Quarantine affected host; block C2; capture forensics.",
        )

    if "port 445" in text and ("external" in text or "internet" in text):
        return VerificationDecision(
            is_threat=True,
            confidence=0.9,
            reasoning="Detected policy violation / high-risk vector: SMB (445) exposure from external sources.",
            recommended_action="Block source IP; confirm firewall rules; scan host for lateral movement.",
        )

    if "203.0.113.55" in text or "apt29" in text:
        return VerificationDecision(
            is_threat=True,
            confidence=0.92,
            reasoning="Matched known threat intel (APT-associated C2 indicator).",
            recommended_action="Block indicator; isolate host; perform incident response triage.",
        )

    if "guest wifi" in text and trend == "stable" and recent_count < 5:
        return VerificationDecision(
            is_threat=False,
            confidence=0.75,
            reasoning="Guest network activity can be noisy; stable trend and low recurrence suggest benign.",
            recommended_action="Ignore; keep monitoring; review if recurrence increases.",
        )

    # UNSW KB evidence: if retrieval returns malicious-labelled samples, treat as verified.
    # This helps when the summary includes known categories (e.g., DoS, Exploits, Reconnaissance)
    # and the vector DB pulls similar attack rows.
    malicious_hits = 0
    normal_hits = 0
    for line in kb_context.splitlines():
        l = line.lower()
        if "label=1" in l and "attack_cat=normal" not in l:
            malicious_hits += 1
        if "label=0" in l or "attack_cat=normal" in l:
            normal_hits += 1

    if malicious_hits > 0 and malicious_hits >= normal_hits:
        return VerificationDecision(
            is_threat=True,
            confidence=0.9,
            reasoning=f"Vector KB retrieved {malicious_hits} similar UNSW malicious samples (label=1).",
            recommended_action="Escalate; block suspicious indicators; collect packet capture; validate affected host.",
        )

    return None

def analyze_suspicious_activity(state: SuspiciousAgentState) -> SuspiciousAgentState:
    """
    Analyzes the suspicious alert using the context provided by the Triage Agent.
    It acts as a 'Corrective RAG' step to verify if the suspicion is valid.
    """
    triage_data = state.triage_data
    semantic_summary = triage_data.get("semantic_summary", "No summary provided.")
    temporal_stats = triage_data.get("temporal_stats", {})
    
    # Check temporal trends - a key factor for verification
    trend = temporal_stats.get("trend", "stable")
    recent_count = temporal_stats.get("recent_anomalies", 0)
    
    print(f"   [Suspicious Agent] Analyzing: {semantic_summary}")
    print(f"   [Suspicious Agent] Context: Trend is {trend}, Count: {recent_count}")

    kb_context = retrieve_kb_context(semantic_summary, k=4)
    state.kb_context = kb_context
    state.likely_attack_category = infer_attack_category_from_kb(kb_context) if kb_context else "Unknown"
    if kb_context:
        print("[Suspicious Agent] Retrieved KB context")

    system_prompt = f"""
    You are a Cyber Defense Verification Agent specializing in 'Suspicious' traffic analysis.
    
    Your goal is to distinguish between False Positives (e.g., admin activity, misconfiguration) and True Threats (e.g., scans, policy violations).
    
    INPUT CONTEXT:
    - Alert Summary: {semantic_summary}
    - Temporal Trend: {trend} (Increasing trend suggests active attack).
    - Recent Anomalies Count: {recent_count} (High count suggests persistence).

    INTERNAL KB (known assets/policies/threat intel):
    {kb_context if kb_context else "(no KB context available)"}
    
    VERIFICATION LOGIC:
    1. If the trend is 'increasing' AND recent anomalies > 10, it is likely a TRUE THREAT (e.g., Brute Force).
    2. If the trend is 'stable' AND recent anomalies < 5, consider if it could be a FALSE POSITIVE (e.g., periodic maintenance).
    3. Look for keywords like 'known attack signature', 'policy violation' in the summary.
    
    Provide a verification decision.
    """
    
    structured_llm = llm.with_structured_output(VerificationDecision)
    decision = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Verify this alert.")
    ])

    override = _heuristic_override(semantic_summary, kb_context, trend, recent_count)
    if override is not None:
        decision = override
    
    status = "Verified Threat" if decision.is_threat else "False Positive"
    
    log_msg = (
        f"VERIFICATION RESULT: {status}\n"
        f"CONFIDENCE: {decision.confidence}\n"
        f"REASONING: {decision.reasoning}\n"
        f"ACTION: {decision.recommended_action}"
    )
    
    state.messages.append(log_msg)
    state.verification_status = status
    state.action_plan = decision.recommended_action

    if status == "Verified Threat":
        state.mitigation_plan = build_mitigation_plan(
            semantic_summary=semantic_summary,
            kb_context=kb_context,
            trend=trend,
            recent_count=recent_count,
        )
        state.messages.append(
            "MITIGATION PLAN: " + json.dumps(state.mitigation_plan, indent=2)
        )
    
    return state

def execute_response(state: SuspiciousAgentState) -> SuspiciousAgentState:
    """
    Simulates taking action based on the verification.
    """
    status = state.verification_status
    action = state.action_plan
    
    if status == "Verified Threat":
        extra = ""
        if state.likely_attack_category:
            extra = f" Likely UNSW category: {state.likely_attack_category}."
        response_msg = f"[RESPONSE] Initiating mitigation: {action}.{extra} Alert escalated to SOC Tier 2."
    else:
        response_msg = f"[RESPONSE] Alert dismissed as False Positive. Logged for audit."
    
    state.messages.append(response_msg)
    return state

# 4. WORKFLOW EXECUTION (without langgraph)

def run_suspicious_agent_workflow(triage_output_json: dict) -> SuspiciousAgentState:
    """
    Runs the suspicious agent workflow as a linear pipeline.
    """
    state = SuspiciousAgentState()
    state.triage_data = triage_output_json
    
    # Step 1: Analyze threat
    state = analyze_suspicious_activity(state)
    
    # Step 2: Execute response
    state = execute_response(state)
    
    return state

# 5. EXTERNAL API (Hook for Triage Agent)

def handle_suspicious_alert(triage_output_json: dict) -> Dict[str, Any]:
    """
    Entry point for the Suspicious Agent.
    Receives the JSON output from the Triage Agent.
    """
    print("\n---SUSPICIOUS TRAFFIC AGENT ACTIVATED---")
    
    final_state = run_suspicious_agent_workflow(triage_output_json)
    
    for message in final_state.messages:
        print(f"{message}")
    
    print("--------------------------------------------------\n")
    
    return {
        "messages": final_state.messages,
        "verification_status": final_state.verification_status,
        "action_plan": final_state.action_plan,
        "triage_data": final_state.triage_data,
        "kb_context": final_state.kb_context,
        "likely_attack_category": final_state.likely_attack_category,
        "mitigation_plan": final_state.mitigation_plan,
    }

# 6. TEST EXECUTION

# if __name__ == "__main__":
#     # Simulate input from Triage Agent (Scenario 1: True Threat)
#     mock_triage_output_threat = {
#         "semantic_summary": "Classified as Suspicious. Pattern matches known SQL Injection attempt on Web Server.",
#         "feature_vector": [0.1, 0.5, 0.9],
#         "temporal_stats": {
#             "recent_anomalies": 25,
#             "trend": "increasing"
#         },
#         "target_pipeline": "CorrectiveRAG"
#     }

#     # Simulate input from Triage Agent (Scenario 2: False Positive)
#     mock_triage_output_benign = {
#         "semantic_summary": "Classified as Suspicious. High traffic volume on Guest WiFi.",
#         "feature_vector": [0.1, 0.1, 0.1],
#         "temporal_stats": {
#             "recent_anomalies": 2,
#             "trend": "stable"
#         },
#         "target_pipeline": "CorrectiveRAG"
#     }
    
#     print("Test 1: Handling Probable Threat...")
#     handle_suspicious_alert(mock_triage_output_threat)
    
#     print("\nTest 2: Handling Probable False Positive...")
#     handle_suspicious_alert(mock_triage_output_benign)