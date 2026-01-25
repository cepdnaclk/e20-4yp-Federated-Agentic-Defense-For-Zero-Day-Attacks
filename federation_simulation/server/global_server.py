"""
Global Server (Aggregator) for Federated Learning Simulation
Receives attack intelligence updates from distributed agents and maintains
a Global Knowledge Base of known attack signatures.

Supports persistent storage of the knowledge base to JSON files.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import uvicorn
import json

# Initialize FastAPI app
app = FastAPI(
    title="Federated Learning Global Server",
    description="Aggregator server for Zero-Day attack intelligence sharing",
    version="1.0.0"
)

# ============================================================================
# Configuration
# ============================================================================

# Directory for storing knowledge base
KB_STORAGE_DIR = Path(__file__).parent.parent / "knowledge_base"
GLOBAL_KB_FILE = KB_STORAGE_DIR / "global_knowledge_base.json"

# ============================================================================
# Data Models
# ============================================================================

class UpdatePacket(BaseModel):
    """Schema for incoming agent updates"""
    agent_id: str
    attack_signature: List[float]
    mitigation_policy: str
    is_zero_day: bool
    attack_category: Optional[str] = None
    timestamp: Optional[str] = None


class GlobalKnowledgeEntry(BaseModel):
    """Schema for entries in the Global Knowledge Base"""
    agent_id: str
    attack_signature: List[float]
    mitigation_policy: str
    attack_category: Optional[str]
    received_at: str
    is_zero_day: bool


# ============================================================================
# Global Knowledge Base (In-Memory Storage with Persistence)
# ============================================================================

class GlobalKnowledgeBase:
    """In-memory storage for aggregated attack intelligence with file persistence"""
    
    def __init__(self, storage_path: Path = GLOBAL_KB_FILE):
        self.storage_path = storage_path
        self.updates: List[GlobalKnowledgeEntry] = []
        self.zero_day_count: int = 0
        self.total_updates: int = 0
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing knowledge base if available
        self._load_from_file()
    
    def _load_from_file(self):
        """Load knowledge base from JSON file"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.updates = [
                    GlobalKnowledgeEntry(**entry) for entry in data.get("updates", [])
                ]
                self.zero_day_count = data.get("zero_day_count", 0)
                self.total_updates = data.get("total_updates", 0)
                
                print(f"📂 Loaded {len(self.updates)} entries from {self.storage_path}")
            except Exception as e:
                print(f"⚠️  Error loading knowledge base: {e}. Starting fresh.")
                self.updates = []
                self.zero_day_count = 0
                self.total_updates = 0
    
    def _save_to_file(self):
        """Save knowledge base to JSON file"""
        try:
            data = {
                "updates": [entry.model_dump() for entry in self.updates],
                "zero_day_count": self.zero_day_count,
                "total_updates": self.total_updates,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Error saving knowledge base: {e}")
    
    def add_update(self, packet: UpdatePacket) -> GlobalKnowledgeEntry:
        """Add a new update to the knowledge base"""
        entry = GlobalKnowledgeEntry(
            agent_id=packet.agent_id,
            attack_signature=packet.attack_signature,
            mitigation_policy=packet.mitigation_policy,
            attack_category=packet.attack_category,
            received_at=datetime.now().isoformat(),
            is_zero_day=packet.is_zero_day
        )
        self.updates.append(entry)
        self.total_updates += 1
        
        if packet.is_zero_day:
            self.zero_day_count += 1
        
        # Auto-save after each update (can be optimized for batch saves)
        self._save_to_file()
        
        return entry
    
    def get_all_signatures(self) -> List[List[float]]:
        """Return all known attack signatures"""
        return [entry.attack_signature for entry in self.updates]
    
    def get_statistics(self) -> dict:
        """Return statistics about the knowledge base"""
        return {
            "total_updates": self.total_updates,
            "zero_day_count": self.zero_day_count,
            "unique_agents": len(set(e.agent_id for e in self.updates)),
            "attack_categories": list(set(e.attack_category for e in self.updates if e.attack_category)),
            "storage_path": str(self.storage_path)
        }
    
    def clear(self):
        """Clear the knowledge base and remove the file"""
        self.updates = []
        self.zero_day_count = 0
        self.total_updates = 0
        if self.storage_path.exists():
            self.storage_path.unlink()
        print("🗑️  Knowledge base cleared")


# Initialize Global Knowledge Base
knowledge_base = GlobalKnowledgeBase()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Federated Learning Global Server",
        "message": "Global aggregator is ready to receive updates"
    }


@app.post("/upload_update")
async def upload_update(packet: UpdatePacket):
    """
    Receive and store intelligence updates from local agents.
    
    Expected JSON payload:
    {
        "agent_id": str,
        "attack_signature": list[float],
        "mitigation_policy": str,
        "is_zero_day": bool,
        "attack_category": str (optional)
    }
    """
    try:
        # Add update to knowledge base
        entry = knowledge_base.add_update(packet)
        
        # Log Zero-Day discoveries prominently
        if packet.is_zero_day:
            print(f"\n{'='*60}")
            print(f"Global Server: Received new Zero-Day intel from {packet.agent_id}")
            print(f"   Attack Category: {packet.attack_category or 'Unknown'}")
            print(f"   Policy: {packet.mitigation_policy[:50]}...")
            print(f"   Total Zero-Days in KB: {knowledge_base.zero_day_count}")
            print(f"{'='*60}\n")
        else:
            print(f"Global Server: Received update from {packet.agent_id} (Category: {packet.attack_category})")
        
        return {
            "status": "success",
            "message": f"Update received from {packet.agent_id}",
            "zero_day_registered": packet.is_zero_day,
            "total_updates_in_kb": knowledge_base.total_updates
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process update: {str(e)}")


@app.get("/get_global_model")
async def get_global_model():
    """
    Return the aggregated global model containing all known attack signatures.
    Local agents can use this to update their local knowledge.
    """
    signatures = knowledge_base.get_all_signatures()
    stats = knowledge_base.get_statistics()
    
    print(f"Global Server: Serving global model ({len(signatures)} signatures)")
    
    return {
        "status": "success",
        "attack_signatures": signatures,
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/statistics")
async def get_statistics():
    """Return detailed statistics about the Global Knowledge Base"""
    return {
        "status": "success",
        "statistics": knowledge_base.get_statistics(),
        "recent_updates": [
            {
                "agent_id": e.agent_id,
                "attack_category": e.attack_category,
                "is_zero_day": e.is_zero_day,
                "received_at": e.received_at
            }
            for e in knowledge_base.updates[-10:]  # Last 10 updates
        ]
    }


@app.delete("/reset")
async def reset_knowledge_base():
    """Reset the knowledge base and delete the persisted file"""
    global knowledge_base
    knowledge_base.clear()
    knowledge_base = GlobalKnowledgeBase()
    print("🔄 Global Server: Knowledge base has been reset")
    return {"status": "success", "message": "Knowledge base reset and file deleted"}


@app.post("/save")
async def save_knowledge_base():
    """Manually trigger saving the knowledge base to file"""
    knowledge_base._save_to_file()
    return {
        "status": "success", 
        "message": f"Knowledge base saved to {knowledge_base.storage_path}",
        "entries": len(knowledge_base.updates)
    }


# ============================================================================
# Server Runner
# ============================================================================

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    print(f"\n{'='*60}")
    print("Starting Federated Learning Global Server")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Endpoints:")
    print(f"     - POST /upload_update  : Receive agent updates")
    print(f"     - GET  /get_global_model: Retrieve aggregated model")
    print(f"     - GET  /statistics     : View KB statistics")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
