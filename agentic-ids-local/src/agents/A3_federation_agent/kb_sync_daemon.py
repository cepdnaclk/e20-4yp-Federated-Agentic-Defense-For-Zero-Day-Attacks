# agents/A3_federation_agent/kb_sync_daemon.py
"""
Background KB Synchronization Daemon

Periodically fetches new signatures from the FL server and updates local RAG context.
This enables real-time sharing of zero-day signatures across all federated agents.

Single-agent sharing (N_min=1): When any agent detects a new attack, all other agents
receive the signature through this sync mechanism.
"""

import os
import time
import threading
from typing import Optional, Callable, List, Dict
from datetime import datetime, timezone
from pathlib import Path

from agents.A3_federation_agent.fl_client import FLClient
from agents.A3_federation_agent.kb_sync import fetch_server_signatures, merge_with_local
from agents.knowledge_base import LocalKnowledgeBase

# Configuration
FL_SERVER_URL = os.getenv("FL_SERVER_URL", "http://localhost:5000")
SYNC_INTERVAL_SECONDS = int(os.getenv("KB_SYNC_INTERVAL", "30"))  # Sync every 30 seconds
LOCAL_KB_PATH = Path(__file__).resolve().parent.parent / "local_knowledge_base.json"
AGENT_ID = os.getenv("AGENT_ID", "agent_local")


class KBSyncDaemon:
    """
    Background daemon that periodically syncs federated signatures to local KB
    and triggers RAG context updates for the LLM agents.
    """

    def __init__(
        self,
        fl_server_url: str = None,
        sync_interval: int = None,
        kb_path: str = None,
        agent_id: str = None,
        on_new_signatures: Optional[Callable[[List[Dict]], None]] = None
    ):
        """
        Args:
            fl_server_url: URL of the FL server
            sync_interval: Seconds between sync attempts
            kb_path: Path to local knowledge base JSON file
            agent_id: This agent's identifier
            on_new_signatures: Callback when new signatures are received
        """
        self.fl_server_url = fl_server_url or FL_SERVER_URL
        self.sync_interval = sync_interval or SYNC_INTERVAL_SECONDS
        self.kb_path = str(kb_path or LOCAL_KB_PATH)
        self.agent_id = agent_id or AGENT_ID
        self.on_new_signatures = on_new_signatures or self._default_callback
        
        self.fl_client = FLClient(server_url=self.fl_server_url, timeout=10)
        self.last_sync_version = 0
        self.running = False
        self.sync_thread: Optional[threading.Thread] = None
        self.sync_errors: List[str] = []
        self.sync_count = 0
        self.last_sync_time: Optional[datetime] = None
        
        # Load current KB version
        self._load_current_version()
        
        print(f"[KBSyncDaemon] Initialized - server: {self.fl_server_url}, interval: {self.sync_interval}s")

    def _load_current_version(self):
        """Load current signature version from local KB."""
        try:
            kb = LocalKnowledgeBase.load(self.kb_path)
            self.last_sync_version = kb.version.signature_version
            print(f"[KBSyncDaemon] Current KB version: {self.last_sync_version}")
        except Exception as e:
            print(f"[KBSyncDaemon] Warning: Could not load KB version: {e}")
            self.last_sync_version = 0

    def _default_callback(self, new_signatures: List[Dict]) -> None:
        """Default callback - just log new signatures."""
        print(f"[KBSyncDaemon] Received {len(new_signatures)} new signatures")
        for sig in new_signatures:
            print(f"  - {sig.get('id', 'unknown')}: {sig.get('attack_description', 'No description')[:50]}...")

    def _sync_once(self) -> Dict:
        """
        Perform one synchronization cycle.
        
        Returns:
            Dict with sync results including new signatures count
        """
        result = {
            "success": False,
            "new_signatures": [],
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Fetch signatures from server
            server_response = fetch_server_signatures(
                self.fl_server_url,
                since_version=self.last_sync_version
            )
            
            if server_response is None:
                result["error"] = "No response from server"
                return result
            
            signatures = server_response.get("signatures", [])
            new_version = server_response.get("version", self.last_sync_version)
            
            # Filter out our own signatures
            external_signatures = [
                sig for sig in signatures
                if sig.get("source_agent") != self.agent_id
            ]
            
            if external_signatures:
                # Merge into local KB
                changes = merge_with_local(self.kb_path, {
                    "signature_version": new_version,
                    "signatures": external_signatures
                })
                
                result["new_signatures"] = external_signatures
                result["changes"] = changes
                
                # Trigger callback for RAG update
                self.on_new_signatures(external_signatures)
            
            # Update version tracker
            self.last_sync_version = new_version
            result["success"] = True
            result["version"] = new_version
            
        except Exception as e:
            result["error"] = str(e)
            self.sync_errors.append(f"{datetime.now().isoformat()}: {e}")
            # Keep only last 10 errors
            self.sync_errors = self.sync_errors[-10:]
        
        return result

    def _sync_loop(self):
        """Main sync loop running in background thread."""
        print(f"[KBSyncDaemon] Starting sync loop (interval: {self.sync_interval}s)")
        
        while self.running:
            try:
                result = self._sync_once()
                self.sync_count += 1
                self.last_sync_time = datetime.now(timezone.utc)
                
                if result.get("new_signatures"):
                    print(f"[KBSyncDaemon] Sync #{self.sync_count}: {len(result['new_signatures'])} new signatures")
                elif result.get("error"):
                    print(f"[KBSyncDaemon] Sync #{self.sync_count}: Error - {result['error']}")
                # else: no new signatures, stay quiet
                
            except Exception as e:
                print(f"[KBSyncDaemon] Unexpected error in sync loop: {e}")
            
            # Wait for next sync interval
            for _ in range(self.sync_interval):
                if not self.running:
                    break
                time.sleep(1)
        
        print("[KBSyncDaemon] Sync loop stopped")

    def start(self):
        """Start the background sync daemon."""
        if self.running:
            print("[KBSyncDaemon] Already running")
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        print("[KBSyncDaemon] Started")

    def stop(self):
        """Stop the background sync daemon."""
        if not self.running:
            print("[KBSyncDaemon] Not running")
            return
        
        print("[KBSyncDaemon] Stopping...")
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        print("[KBSyncDaemon] Stopped")

    def sync_now(self) -> Dict:
        """Trigger an immediate sync (can be called from any thread)."""
        return self._sync_once()

    def get_status(self) -> Dict:
        """Get current daemon status."""
        return {
            "running": self.running,
            "sync_count": self.sync_count,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "current_version": self.last_sync_version,
            "recent_errors": self.sync_errors[-5:],
            "fl_server": self.fl_server_url,
            "sync_interval": self.sync_interval
        }


# Singleton instance and global callbacks
_daemon_instance: Optional[KBSyncDaemon] = None
_rag_update_callbacks: List[Callable[[List[Dict]], None]] = []


def register_rag_update_callback(callback: Callable[[List[Dict]], None]):
    """
    Register a callback to be invoked when new signatures are received.
    Use this to update RAG vector stores in LLM agents.
    
    Args:
        callback: Function(signatures: List[Dict]) -> None
    """
    global _rag_update_callbacks
    _rag_update_callbacks.append(callback)
    print(f"[KBSyncDaemon] Registered RAG update callback ({len(_rag_update_callbacks)} total)")


def _broadcast_to_callbacks(new_signatures: List[Dict]):
    """Broadcast new signatures to all registered callbacks."""
    for callback in _rag_update_callbacks:
        try:
            callback(new_signatures)
        except Exception as e:
            print(f"[KBSyncDaemon] Callback error: {e}")


def get_kb_sync_daemon() -> KBSyncDaemon:
    """Get or create singleton KB sync daemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = KBSyncDaemon(on_new_signatures=_broadcast_to_callbacks)
    return _daemon_instance


def start_kb_sync_daemon():
    """Start the global KB sync daemon."""
    daemon = get_kb_sync_daemon()
    daemon.start()
    return daemon


def stop_kb_sync_daemon():
    """Stop the global KB sync daemon."""
    global _daemon_instance
    if _daemon_instance:
        _daemon_instance.stop()
