import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field

# LangChain Imports (LLM + Embeddings + Vector Store)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


"""Plain (raw) Python version of the triage pipeline.

This keeps the same:
- Groq LLM model: llama-3.3-70b-versatile (via ChatGroq)
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (via HuggingFaceEmbeddings)
- Retrieval: FAISS vector store

But removes LangGraph state machines and runs the flow sequentially.
"""


# CONFIGURATION & SETUP
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: Please set your GROQ_API_KEY environment variable.")
    sys.exit(1)

# Structured Output for reliability (Triage Decision)
class TriageDecision(BaseModel):
    category: Literal["BENIGN", "SUSPICIOUS", "ZERO-DAY"] = Field(
        description="The classification of the security event."
    )
    routing: Literal["AgenticRAG", "CorrectiveRAG", "AdaptiveRAG"] = Field(
        description="The pipeline to route this event to."
    )
    reasoning: str = Field(description="Why this classification was chosen.")

LLM_MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# KNOWLEDGE BASE (RAG SETUP)

# Use Local Embeddings (privacy preserving; embeddings computed locally)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
triage_vector_store: Optional[FAISS] = None
KB_FILE = Path(__file__).resolve().parent / "./KB/soc_knowledge.csv"

def get_mock_data():
    """Fallback data if no CSV is found."""
    return [
        Document(page_content="Asset: Finance-DB (10.0.5.50). Criticality: HIGH. Allowed Ports: 443, 22 only. Behavior: No outbound internet access."),
        Document(page_content="Asset: Guest-WiFi-Gateway (192.168.1.1). Criticality: LOW. Behavior: High traffic volume expected."),
        Document(page_content="Pattern: Sequential port knocking (ports 7000, 8000, 9000) is a known sign of C2 initiation."),
        Document(page_content="Pattern: Encrypted large payloads on non-standard ports (e.g., 8888) targeting Databases are indicative of Data Exfiltration."),
        Document(page_content="Policy: Failed logins < 5 on Low Criticality assets are BENIGN."),
        Document(page_content="Policy: ANY unexpected outbound connection from High Criticality assets is a ZERO-DAY CANDIDATE until proven otherwise."),
        Document(page_content="Policy: Known signature matches should be treated as SUSPICIOUS and verified.")
    ]

def initialize_triage_kb():
    """Creates a local in-memory Knowledge Base for the Triage Agent."""
    global triage_vector_store
    if triage_vector_store is not None:
        return
    
    print("[INIT] Building Triage Knowledge Base (FAISS)...")
    
    # Load from CSV or fallback to Mock Data
    docs: List[Document] = []
    if KB_FILE.exists():
        try:
            with open(KB_FILE, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    full_text = f"{row.get('Type', 'Info')}: {row.get('Content', '')}"
                    docs.append(Document(page_content=full_text))
            print(f"Loaded {len(docs)} records from {KB_FILE.name}")
        except Exception:
            docs = get_mock_data()
    else:
        print("CSV not found. Using Mock Data.")
        docs = get_mock_data()

    triage_vector_store = FAISS.from_documents(docs, embedding_model)
    print("[INIT] Triage Knowledge Base Ready.")

# AGENT NODES (plain Python functions)

# Initialize Groq LLM
llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)


def retrieve_context(raw_log: str, k: int = 3) -> str:
    """Step 1: Look up asset/policy context in the Knowledge Base."""
    initialize_triage_kb()
    assert triage_vector_store is not None

    docs = triage_vector_store.similarity_search(raw_log, k=k)
    return "\n".join([f"- {d.page_content}" for d in docs])


def triage_analysis(raw_log: str, retrieved_context: str) -> TriageDecision:
    """Step 2: LLM classifies the log based on Context + Rules."""
    
    system_prompt = f"""
    You are the SOC Triage Agent for a Zero-Day Detection System.
    
    YOUR GOAL: Classify the incoming security log and route it.
    
    INTERNAL KNOWLEDGE CONTEXT:
    {retrieved_context}
    
    CLASSIFICATION RULES:
    1. BENIGN -> Route to 'AgenticRAG'.
       (Routine events, low-risk asset noise, known safe patterns).
       
    2. SUSPICIOUS -> Route to 'CorrectiveRAG'.
       (Known attack signatures, policy violations, failed logins on high-value assets).
       
    3. ZERO-DAY CANDIDATE -> Route to 'AdaptiveRAG'.
       (Anomalous behavior on Critical Assets, encrypted payloads on weird ports, behavior contradicting asset profile).
       
    Analyze the log below. Be conservative: if an anomaly matches a Critical Asset's "Never Do This" rule, it is likely a Zero-Day Candidate.
    """
    
    # Enforce Structured Output
    structured_llm = llm.with_structured_output(TriageDecision)
    decision: TriageDecision = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=raw_log)
    ])

    return decision

# # DOWNSTREAM PIPELINES (Integration of your Concepts)

# def call_agentic_rag(raw_log: str) -> str:
#     """Handles BENIGN traffic."""
#     return "[AgenticRAG] Event classified as Benign. Logged to compliance archive."


# def call_corrective_rag(raw_log: str, retrieved_context: str) -> str:
#     """Handles SUSPICIOUS traffic (self-check/grade step)."""
#     grader_prompt = (
#         f"Given context: {retrieved_context}, is this log: {raw_log} definitely malicious? "
#         "Reply YES or NO."
#     )
#     grade = llm.invoke([HumanMessage(content=grader_prompt)])
#     return f"[CorrectiveRAG] Suspicious Activity. Verification result: {grade.content}"


# def call_adaptive_rag(raw_log: str) -> str:
#     """Handles ZERO-DAY CANDIDATES (feature extraction step)."""
#     analysis = llm.invoke([
#         SystemMessage(content="Extract unique features (IP, Port, Payload) from this Zero-Day candidate."),
#         HumanMessage(content=raw_log),
#     ])
#     return (
#         f"[AdaptiveRAG] Zero-Day Detected. Features Extracted: {analysis.content}. "
#         "Updating FL Model."
#     )


# def _run_pipeline(target_pipeline: str, raw_log: str, retrieved_context: str) -> str:
#     if target_pipeline == "AgenticRAG":
#         return call_agentic_rag(raw_log)
#     if target_pipeline == "CorrectiveRAG":
#         return call_corrective_rag(raw_log, retrieved_context)
#     if target_pipeline == "AdaptiveRAG":
#         return call_adaptive_rag(raw_log)
#     return "[Error] Unknown routing target."

def process_anomaly(alert_text: str, raw_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Unified HOOK:
    - Streams agent reasoning (for visibility)
    - Returns structured JSON (for downstream systems)
    """
    # print(f"\n[System] Autoencoder sent: {alert_text}")
    # print("Processing Pipeline...\n")

    # Step 1: retrieve
    # print("Retrieving Context...")
    retrieved_context = retrieve_context(alert_text, k=3)

    # Step 2: triage
    # print("Triaging Event...")
    decision = triage_analysis(alert_text, retrieved_context)


    #  Print log 
    print(f"[TRIAGE] Decision: {decision.category.upper()} | Routed: {decision.routing} | Reason: {decision.reasoning}")
    # print(f"TRIAGE RESULT: {decision.category.upper()}")
    # print(f"ROUTING TO: {decision.routing}")
    # print(f"REASONING: {decision.reasoning}")
    # print("---------------------------")

    # # Step 3: route
    # pipeline_message = _run_pipeline(decision.routing, alert_text, retrieved_context)
    # print(pipeline_message)

    # Step 4: build structured output
    semantic_summary = f"Classified as {decision.category}. {decision.reasoning}"

    feature_vector = raw_data.get("feature_vector", []) if raw_data else []
    score = raw_data.get("anomaly_score", 0.5) if raw_data else 0.5

    trend = "increasing" if score > 0.8 else "stable"
    recent_count = random.randint(15, 30) if score > 0.8 else random.randint(0, 5)

    output_json: Dict[str, Any] = {
        "semantic_summary": semantic_summary,
        "feature_vector": feature_vector,
        "temporal_stats": {
            "recent_anomalies": recent_count,
            "trend": trend
        },
        "target_pipeline": decision.routing,
    }

    # print("Final Structured Output Ready\n")
    # print(json.dumps(output_json, indent=4))

    return output_json



# 7. EXECUTION SIMULATION

# if __name__ == "__main__":
#     print("SYSTEM ONLINE: Zero-Day Agentic Detection\n")
    
#     # Scenario: High Criticality Asset (Finance-DB) doing something it NEVER does.
#     test_alert = (
#         "Alert: Server-Alpha (192.168.1.10) transferred 1.4GB of encrypted data "
#         "to unknown external IP 203.0.113.200 within 3 minutes over port 8888."
#     )
    
#     process_anomaly(test_alert)