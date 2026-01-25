"""
Local Agent Node for Federated Learning Simulation
Processes local network traffic, detects Zero-Day attacks, and shares intelligence
with the global aggregator server.

Supports optional RAG (Retrieval-Augmented Generation) integration for
LLM-powered mitigation policy generation.

Includes persistent local knowledge base storage.
"""

import requests
import time
import hashlib
import json
from typing import List, Set, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

from .dataset_loader import UNSW_NB15_Loader, create_loader

# Optional RAG integration
if TYPE_CHECKING:
    from .rag_integration import RAGPolicyGenerator

# ============================================================================
# Configuration
# ============================================================================

# Directory for storing local knowledge bases
LOCAL_KB_STORAGE_DIR = Path(__file__).parent.parent / "knowledge_base" / "agents"


class LocalAgent:
    """
    Local Federated Learning Agent
    
    Responsibilities:
    - Process local network traffic data
    - Detect potential Zero-Day attacks (unseen attack categories)
    - Generate mitigation policies
    - Share attack intelligence with global server
    - Learn from global model updates
    """
    
    # Mitigation policy templates based on attack category
    MITIGATION_POLICIES = {
        'Fuzzers': "Block anomalous packet sequences; Enable input validation; Rate limit suspicious sources",
        'Analysis': "Monitor reconnaissance patterns; Implement honeypots; Enable deep packet inspection",
        'Backdoor': "Isolate affected systems; Revoke compromised credentials; Audit network connections",
        'DoS': "Enable rate limiting; Activate DDoS mitigation; Scale infrastructure resources",
        'Exploits': "Patch vulnerable systems; Enable IPS signatures; Segment network zones",
        'Generic': "Apply general security policies; Increase logging verbosity; Alert security team",
        'Reconnaissance': "Mask network topology; Enable port scan detection; Limit information disclosure",
        'Shellcode': "Enable DEP/ASLR; Block executable payloads; Sandbox suspicious processes",
        'Worms': "Quarantine infected hosts; Block lateral movement; Update antivirus signatures",
        'default': "Apply baseline security controls; Monitor for anomalies; Escalate to security team"
    }
    
    def __init__(
        self,
        agent_id: str,
        server_url: str = "http://localhost:8000",
        data_path: Optional[str] = None,
        use_rag: bool = False,
        rag_provider: str = "mock",
        rag_api_key: Optional[str] = None,
        persist_knowledge: bool = True
    ):
        """
        Initialize the Local Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            server_url: URL of the global aggregator server
            data_path: Path to the dataset directory
            use_rag: If True, use RAG/LLM for policy generation
            rag_provider: LLM provider ("mock", "openai", "anthropic", "ollama")
            rag_api_key: API key for cloud LLM providers
            persist_knowledge: If True, save/load local knowledge base to file
        """
        self.agent_id = agent_id
        self.server_url = server_url.rstrip('/')
        self.use_rag = use_rag
        self.persist_knowledge = persist_knowledge
        
        # Setup local knowledge base storage
        self.kb_storage_path = LOCAL_KB_STORAGE_DIR / f"{agent_id}_knowledge_base.json"
        LOCAL_KB_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Local Knowledge Base
        self.known_attacks: Set[str] = set()  # Attack categories we've seen
        self.signature_hashes: Set[str] = set()  # Hash of signatures we've uploaded
        self.learned_signatures: List[Dict[str, Any]] = []  # Detailed signature records
        
        # Load existing knowledge if persistence is enabled
        if persist_knowledge:
            self._load_local_knowledge()
        
        # Statistics
        self.stats = {
            "packets_processed": 0,
            "normal_traffic": 0,
            "attacks_detected": 0,
            "zero_days_discovered": 0,
            "updates_sent": 0,
            "connection_errors": 0
        }
        
        # Initialize RAG if enabled
        self.rag_generator: Optional['RAGPolicyGenerator'] = None
        if use_rag:
            try:
                from .rag_integration import create_rag_generator
                self.rag_generator = create_rag_generator(
                    provider=rag_provider,
                    api_key=rag_api_key
                )
                print(f"   🧠 RAG-enhanced policy generation enabled ({rag_provider})")
            except Exception as e:
                print(f"   ⚠️  RAG initialization failed: {e}. Using rule-based policies.")
                self.use_rag = False
        
        # Dataset loader
        if data_path:
            csv_path = Path(data_path) / "UNSW_NB15_testing-set.csv"
            self.loader = UNSW_NB15_Loader(str(csv_path))
        else:
            self.loader = create_loader(use_testing=True)
        
        print(f"\n🤖 Agent '{self.agent_id}' initialized")
        print(f"   Server URL: {self.server_url}")
        if persist_knowledge:
            print(f"   📁 Knowledge Base: {self.kb_storage_path}")
            print(f"   📚 Known Attacks: {self.known_attacks if self.known_attacks else 'None (starting fresh)'}")
    
    # ========================================================================
    # Local Knowledge Base Persistence
    # ========================================================================
    
    def _load_local_knowledge(self):
        """Load local knowledge base from JSON file"""
        if self.kb_storage_path.exists():
            try:
                with open(self.kb_storage_path, 'r') as f:
                    data = json.load(f)
                
                self.known_attacks = set(data.get("known_attacks", []))
                self.signature_hashes = set(data.get("signature_hashes", []))
                self.learned_signatures = data.get("learned_signatures", [])
                
                print(f"   📂 Loaded local KB: {len(self.known_attacks)} attack types, {len(self.learned_signatures)} signatures")
            except Exception as e:
                print(f"   ⚠️  Error loading local KB: {e}. Starting fresh.")
    
    def _save_local_knowledge(self):
        """Save local knowledge base to JSON file"""
        if not self.persist_knowledge:
            return
            
        try:
            data = {
                "agent_id": self.agent_id,
                "known_attacks": list(self.known_attacks),
                "signature_hashes": list(self.signature_hashes),
                "learned_signatures": self.learned_signatures,
                "stats": self.stats,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.kb_storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"   ⚠️  Error saving local KB: {e}")
    
    def _add_to_local_knowledge(
        self, 
        attack_category: str, 
        features: List[float], 
        policy: str,
        is_zero_day: bool
    ):
        """Add a learned attack to the local knowledge base"""
        self.known_attacks.add(attack_category)
        
        # Store detailed signature record
        signature_record = {
            "attack_category": attack_category,
            "signature_hash": self._generate_signature_hash(features),
            "feature_sample": features[:10],  # Store first 10 features as sample
            "mitigation_policy": policy[:200],  # Truncate policy for storage
            "is_zero_day": is_zero_day,
            "learned_at": datetime.now().isoformat()
        }
        self.learned_signatures.append(signature_record)
        
        # Auto-save
        self._save_local_knowledge()
    
    def clear_local_knowledge(self):
        """Clear the local knowledge base"""
        self.known_attacks = set()
        self.signature_hashes = set()
        self.learned_signatures = []
        if self.kb_storage_path.exists():
            self.kb_storage_path.unlink()
        print(f"   🗑️  {self.agent_id}: Local knowledge base cleared")
    
    def get_local_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of the local knowledge base"""
        return {
            "agent_id": self.agent_id,
            "known_attack_types": list(self.known_attacks),
            "total_signatures": len(self.learned_signatures),
            "unique_hashes": len(self.signature_hashes),
            "storage_path": str(self.kb_storage_path)
        }
    
    def _generate_signature_hash(self, features: List[float]) -> str:
        """Generate a hash of the feature vector for deduplication"""
        feature_str = ','.join(f"{f:.4f}" for f in features[:10])  # Use first 10 features
        return hashlib.md5(feature_str.encode()).hexdigest()[:16]
    
    def _generate_mitigation_policy(self, attack_category: str, features: List[float]) -> str:
        """
        Generate a mitigation policy based on attack category.
        
        If RAG is enabled, uses LLM-powered generation with retrieved context.
        Otherwise, falls back to rule-based template lookup.
        """
        # Use RAG if available
        if self.use_rag and self.rag_generator:
            try:
                return self.rag_generator.generate_policy(
                    attack_category=attack_category,
                    features=features,
                    agent_id=self.agent_id
                )
            except Exception as e:
                print(f"   ⚠️  RAG generation failed: {e}. Using fallback.")
        
        # Fallback: Rule-based policy generation
        base_policy = self.MITIGATION_POLICIES.get(
            attack_category, 
            self.MITIGATION_POLICIES['default']
        )
        
        # Add contextual information
        policy = f"[{attack_category}] {base_policy}"
        policy += f" | Signature hash: {self._generate_signature_hash(features)}"
        policy += f" | Detected by: {self.agent_id} at {datetime.now().isoformat()}"
        
        return policy
    
    def _send_update_to_server(
        self,
        features: List[float],
        attack_category: str,
        is_zero_day: bool
    ) -> bool:
        """
        Send attack intelligence update to the global server.
        
        Args:
            features: Attack signature (feature vector)
            attack_category: Type of attack
            is_zero_day: Whether this is a newly discovered attack
            
        Returns:
            True if update was sent successfully
        """
        update_packet = {
            "agent_id": self.agent_id,
            "attack_signature": features,
            "mitigation_policy": self._generate_mitigation_policy(attack_category, features),
            "is_zero_day": is_zero_day,
            "attack_category": attack_category,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/upload_update",
                json=update_packet,
                timeout=10
            )
            
            if response.status_code == 200:
                self.stats["updates_sent"] += 1
                return True
            else:
                print(f"⚠️  {self.agent_id}: Server returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.stats["connection_errors"] += 1
            if self.stats["connection_errors"] <= 3:
                print(f"⚠️  {self.agent_id}: Cannot connect to server. Retrying...")
            return False
            
        except requests.exceptions.Timeout:
            print(f"⚠️  {self.agent_id}: Request timed out")
            return False
            
        except Exception as e:
            print(f"❌ {self.agent_id}: Error sending update: {e}")
            return False
    
    def _fetch_global_model(self) -> Optional[Dict[str, Any]]:
        """Fetch the latest global model from the server"""
        try:
            response = requests.get(
                f"{self.server_url}/get_global_model",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️  {self.agent_id}: Could not fetch global model: {e}")
        return None
    
    def _is_zero_day(self, attack_category: str) -> bool:
        """Check if this attack category is a Zero-Day (never seen before)"""
        return attack_category not in self.known_attacks
    
    def process_packet(self, features: List[float], label: str) -> Dict[str, Any]:
        """
        Process a single network packet.
        
        Args:
            features: Feature vector representing the packet
            label: Traffic label ('Normal' or attack category)
            
        Returns:
            Dictionary with processing results
        """
        self.stats["packets_processed"] += 1
        
        result = {
            "action": "none",
            "is_attack": False,
            "is_zero_day": False,
            "uploaded": False
        }
        
        # Handle normal traffic
        if label.lower() == 'normal' or label == '':
            self.stats["normal_traffic"] += 1
            return result
        
        # This is an attack
        result["is_attack"] = True
        self.stats["attacks_detected"] += 1
        
        # Generate policy for this attack
        policy = self._generate_mitigation_policy(label, features)
        
        # Check if this is a Zero-Day
        if self._is_zero_day(label):
            result["is_zero_day"] = True
            self.stats["zero_days_discovered"] += 1
            
            print(f"\n🚨 {self.agent_id}: ZERO-DAY DETECTED!")
            print(f"   Attack Category: {label}")
            print(f"   Feature Vector Size: {len(features)}")
            
            # Generate and send update packet
            success = self._send_update_to_server(features, label, is_zero_day=True)
            result["uploaded"] = success
            
            if success:
                # Learn: Add to local known attacks and persist
                self._add_to_local_knowledge(label, features, policy, is_zero_day=True)
                print(f"   ✅ Intelligence shared with global server")
                print(f"   📚 Added '{label}' to local knowledge base")
            else:
                print(f"   ⚠️  Failed to share with global server")
        else:
            # Known attack - still might share if signature is unique
            sig_hash = self._generate_signature_hash(features)
            if sig_hash not in self.signature_hashes:
                # New variant of known attack
                self._send_update_to_server(features, label, is_zero_day=False)
                self.signature_hashes.add(sig_hash)
                
                # Also save the new variant to local KB
                self._add_to_local_knowledge(label, features, policy, is_zero_day=False)
        
        return result
    
    def run_simulation(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        delay: float = 0.01
    ) -> Dict[str, Any]:
        """
        Run the federated learning simulation on a range of dataset rows.
        
        Args:
            start_idx: Starting row index in the dataset
            end_idx: Ending row index (exclusive)
            delay: Delay between packets (simulates real-time traffic)
            
        Returns:
            Statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"🚀 {self.agent_id}: Starting Federated Learning Simulation")
        print(f"   Processing rows {start_idx} to {end_idx or 'end'}")
        print(f"{'='*60}\n")
        
        # Load dataset if not already loaded
        if self.loader.dataframe is None:
            if not self.loader.load_dataset():
                print(f"❌ {self.agent_id}: Failed to load dataset")
                return self.stats
        
        # Wait for server to be ready (with retries)
        self._wait_for_server()
        
        # Process packets
        start_time = time.time()
        
        for features, label in self.loader.yield_packet(start_idx, end_idx):
            self.process_packet(features, label)
            
            # Progress indicator every 50 packets
            if self.stats["packets_processed"] % 50 == 0:
                print(f"   {self.agent_id}: Processed {self.stats['packets_processed']} packets...")
            
            # Simulate real-time traffic
            if delay > 0:
                time.sleep(delay)
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 {self.agent_id}: Simulation Complete")
        print(f"{'='*60}")
        print(f"   Duration: {elapsed_time:.2f} seconds")
        print(f"   Packets Processed: {self.stats['packets_processed']}")
        print(f"   Normal Traffic: {self.stats['normal_traffic']}")
        print(f"   Attacks Detected: {self.stats['attacks_detected']}")
        print(f"   Zero-Days Discovered: {self.stats['zero_days_discovered']}")
        print(f"   Updates Sent: {self.stats['updates_sent']}")
        print(f"   Known Attack Categories: {self.known_attacks}")
        print(f"{'='*60}\n")
        
        return self.stats
    
    def _wait_for_server(self, max_retries: int = 10, retry_delay: float = 1.0):
        """Wait for the global server to become available"""
        print(f"   {self.agent_id}: Checking server connectivity...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.server_url}/", timeout=5)
                if response.status_code == 200:
                    print(f"   {self.agent_id}: ✅ Connected to global server")
                    return True
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print(f"   {self.agent_id}: Waiting for server... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
            except Exception as e:
                print(f"   {self.agent_id}: Connection error: {e}")
        
        print(f"   {self.agent_id}: ⚠️  Server not available, will retry during simulation")
        return False


# ============================================================================
# Standalone Agent Runner
# ============================================================================

def run_agent(
    agent_id: str,
    start_idx: int,
    end_idx: int,
    server_url: str = "http://localhost:8000",
    data_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run an agent simulation.
    Can be used with multiprocessing.
    
    Args:
        agent_id: Unique identifier for the agent
        start_idx: Starting row index
        end_idx: Ending row index
        server_url: Global server URL
        data_path: Path to dataset directory
        
    Returns:
        Agent statistics
    """
    agent = LocalAgent(
        agent_id=agent_id,
        server_url=server_url,
        data_path=data_path
    )
    return agent.run_simulation(start_idx, end_idx)


if __name__ == "__main__":
    # Test the agent standalone
    agent = LocalAgent("Test_Agent")
    stats = agent.run_simulation(start_idx=0, end_idx=50)
    print(f"Final stats: {stats}")
