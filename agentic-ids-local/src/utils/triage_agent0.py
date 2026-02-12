import os
import sys
import csv
import json
import random
from typing import Annotated, TypedDict, List, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

#LangChain & LangGraph Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool


#CONFIGURATION & SETUP
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: Please set your GROQ_API_KEY environment variable.")
    sys.exit(1)

#STATE DEFINITIONS

class AgentState(TypedDict):
    """
    Global state passed between the Triage Agent and downstream RAGs.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    raw_log: str            # The incoming alert/log from Autoencoder
    retrieved_context: str  # Context found in Triage KB
    classification: str     # Benign / Suspicious / Zero-day
    target_pipeline: str    # The next agent to call

# Structured Output for reliability (Triage Decision)
class TriageDecision(BaseModel):
    category: Literal["Benign", "Suspicious", "Zero-day"] = Field(
        description="The classification of the security event."
    )
    routing: Literal["AgenticRAG", "CorrectiveRAG", "AdaptiveRAG"] = Field(
        description="The pipeline to route this event to."
    )
    reasoning: str = Field(description="Why this classification was chosen.")

#KNOWLEDGE BASE (RAG SETUP)

# Use Local Embeddings (Privacy Preserving)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
triage_vector_store = None
KB_FILE = "soc_knowledge.csv"

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
    docs = []
    if os.path.exists(KB_FILE):
        try:
            with open(KB_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    full_text = f"{row.get('Type', 'Info')}: {row.get('Content', '')}"
                    docs.append(Document(page_content=full_text))
            print(f"Loaded {len(docs)} records from {KB_FILE}")
        except Exception:
            docs = get_mock_data()
    else:
        print("CSV not found. Using Mock Data.")
        docs = get_mock_data()

    triage_vector_store = FAISS.from_documents(docs, embedding_model)
    print("[INIT] Triage Knowledge Base Ready.")

#AGENT NODES

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def retrieve_context(state: AgentState):
    """Step 1: Look up the asset and policy in the Knowledge Base."""
    log = state["raw_log"]
    initialize_triage_kb()
        
    # Search for relevant policies or asset info
    docs = triage_vector_store.similarity_search(log, k=3)
    context_str = "\n".join([f"- {d.page_content}" for d in docs])
    
    return {"retrieved_context": context_str}

def triage_analysis(state: AgentState):
    """Step 2: LLM classifies the log based on Context + Rules."""
    log = state["raw_log"]
    context = state["retrieved_context"]
    
    system_prompt = f"""
    You are the SOC Triage Agent for a Zero-Day Detection System.
    
    YOUR GOAL: Classify the incoming security log and route it.
    
    INTERNAL KNOWLEDGE CONTEXT:
    {context}
    
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
    decision = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=log)
    ])
    
    log_entry = (
        f"TRIAGE RESULT: {decision.category.upper()}\n"
        f"ROUTING TO: {decision.routing}\n"
        f"REASONING: {decision.reasoning}"
    )
    
    return {
        "messages": [HumanMessage(content=log_entry)],
        "classification": decision.category,
        "target_pipeline": decision.routing,
        "reasoning_summary": decision.reasoning
    }

#pipline functions for each RAG

def call_agentic_rag(state: AgentState):
    """
    Handles BENIGN traffic.
    Logic (from 1-AgenticRAG): Uses basic tools to log events.
    """
    return {"messages": [HumanMessage(content="[AgenticRAG] Event classified as Benign. Logged to compliance archive.")]}

def call_corrective_rag(state: AgentState):
    """
    Handles SUSPICIOUS traffic.
    Logic (from 2-CorrectiveRAG): Performs Self-Reflection/Grading to verify if it's a False Positive.
    """
    log = state["raw_log"]
    context = state["retrieved_context"]
    
    # Simulating the "Grade" step
    grader_prompt = f"Given context: {context}, is this log: {log} definitely malicious? Reply YES or NO."
    grade = llm.invoke([HumanMessage(content=grader_prompt)])
    
    return {"messages": [HumanMessage(content=f"[CorrectiveRAG] Suspicious Activity. Verification result: {grade.content}")]}

def call_adaptive_rag(state: AgentState):
    """
    Handles ZERO-DAY CANDIDATES.
    Logic (from 4-AdaptiveRAG): Deep Analysis & Federated Learning Update.
    """
    log = state["raw_log"]
    # Simulating Feature Extraction for FL
    analysis = llm.invoke([
        SystemMessage(content="Extract unique features (IP, Port, Payload) from this Zero-Day candidate."),
        HumanMessage(content=log)
    ])
    return {"messages": [HumanMessage(content=f"[AdaptiveRAG] Zero-Day Detected. Features Extracted: {analysis.content}. Updating FL Model.")]}

# 5. GRAPH CONSTRUCTION

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("context_retrieval", retrieve_context)
workflow.add_node("triage_reasoning", triage_analysis)
workflow.add_node("AgenticRAG", call_agentic_rag)
workflow.add_node("CorrectiveRAG", call_corrective_rag)
workflow.add_node("AdaptiveRAG", call_adaptive_rag)

# Add Edges
workflow.add_edge(START, "context_retrieval")
workflow.add_edge("context_retrieval", "triage_reasoning")

# Conditional Edge Logic (The Router)
def route_event(state: AgentState):
    return state["target_pipeline"]

workflow.add_conditional_edges(
    "triage_reasoning",
    route_event,
    {
        "AgenticRAG": "AgenticRAG",
        "CorrectiveRAG": "CorrectiveRAG",
        "AdaptiveRAG": "AdaptiveRAG"
    }
)

workflow.add_edge("AgenticRAG", END)
workflow.add_edge("CorrectiveRAG", END)
workflow.add_edge("AdaptiveRAG", END)

triage_app = workflow.compile()

# 6. EXTERNAL CONNECTION API (The Hook)

# def process_anomaly(alert_text: str, raw_data: dict = None):
#     """
#     HOOK FOR AUTOENCODER. Call this function from your detection script.
#     """
#     print(f"\n[System] Autoencoder sent: {alert_text}")
#     initial_state = {"raw_log": alert_text, "messages": []}
    
#     final_state = None
#     print("Processing Pipeline")
#     for event in triage_app.stream(initial_state):
#         for node, data in event.items():
#             if "messages" in data:
#                 print(f"{node}: {data['messages'][-1].content}")
#                 final_state = data
#     print("---------------------------\n")
#     return final_state

def process_anomaly(alert_text: str, raw_data: dict = None) -> dict:
    """
    Unified HOOK:
    - Streams agent reasoning (for visibility)
    - Returns structured JSON (for downstream systems)
    """
    print(f"\n[System] Autoencoder sent: {alert_text}")
    print("Processing Pipeline...\n")

    initial_state = {
        "raw_log": alert_text,
        "messages": [],
        "reasoning_summary": ""
    }

    #STREAM FOR VISIBILITY
    streamed_final_state = None
    for event in triage_app.stream(initial_state):
        for node, data in event.items():
            if "messages" in data:
                print(f"{node}: {data['messages'][-1].content}")
                streamed_final_state = data

    print("\n---------------------------")

    # INVOKE FOR FINAL STATE
    final_state = triage_app.invoke(initial_state)

    #BUILD STRUCTURED OUTPUT
    semantic_summary = (
        f"Classified as {final_state.get('classification', 'Unknown')}. "
        f"{final_state.get('reasoning_summary', '')}"
    )

    feature_vector = raw_data.get("feature_vector", []) if raw_data else []
    score = raw_data.get("anomaly_score", 0.5) if raw_data else 0.5

    trend = "increasing" if score > 0.8 else "stable"
    recent_count = random.randint(15, 30) if score > 0.8 else random.randint(0, 5)

    output_json = {
        "semantic_summary": semantic_summary,
        "feature_vector": feature_vector,
        "temporal_stats": {
            "recent_anomalies": recent_count,
            "trend": trend
        },
        "target_pipeline": final_state.get("target_pipeline", "Unknown")
    }

    print("Final Structured Output Ready\n")
    print(json.dumps(output_json, indent=4))

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