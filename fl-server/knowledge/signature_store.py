import json
import os
from typing import Dict, List, Optional

import numpy as np

from .versioning import VersionManager


class SignatureStore:
    """
    Persistent store for anomaly signatures with lifecycle:
    Candidate → Verified → Global → Deprecated

    Each signature record:
    - id: unique string
    - embedding: list[float] (latent centroid)
    - state: str (Candidate/Verified/Global/Deprecated)
    - confidence: float
    - metadata: dict (e.g., feature ranges, labels, explanation)
    - agents: list[str] where observed
    - version_added: int (signature_version)
    - version_updated: int (signature_version)
    """

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.vm = VersionManager()
        self._signatures: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            self._signatures = {s["id"]: s for s in doc.get("signatures", [])}
            # restore versions
            mv = doc.get("model_version", 0)
            sv = doc.get("signature_version", 0)
            self.vm._versions.model_version = mv
            self.vm._versions.signature_version = sv
        else:
            self._persist()

    def _persist(self):
        doc = {
            "model_version": self.vm.versions.model_version,
            "signature_version": self.vm.versions.signature_version,
            "signatures": list(self._signatures.values())
        }
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

    def get_known_embeddings(self, states: Optional[List[str]] = None) -> np.ndarray:
        states = states or ["Verified", "Global"]
        embs = [np.array(s["embedding"], dtype=np.float32) for s in self._signatures.values() if s["state"] in states]
        if not embs:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack(embs)

    def add_candidate(self, centroid: List[float], confidence: float, agents: List[str], metadata: Optional[Dict] = None) -> Dict:
        sid = f"sig_{len(self._signatures)+1}"
        rec = {
            "id": sid,
            "embedding": centroid,
            "state": "Candidate",
            "confidence": float(confidence),
            "metadata": metadata or {},
            "agents": agents,
            "version_added": self.vm.bump_signatures().signature_version,
            "version_updated": self.vm.versions.signature_version
        }
        self._signatures[sid] = rec
        self._persist()
        return rec

    def verify_signature(self, sid: str, extra_metadata: Optional[Dict] = None) -> Optional[Dict]:
        rec = self._signatures.get(sid)
        if not rec:
            return None
        rec["state"] = "Verified"
        rec["metadata"].update(extra_metadata or {})
        rec["version_updated"] = self.vm.bump_signatures().signature_version
        self._persist()
        return rec

    def promote_to_global(self, sid: str) -> Optional[Dict]:
        rec = self._signatures.get(sid)
        if not rec:
            return None
        rec["state"] = "Global"
        rec["version_updated"] = self.vm.bump_signatures().signature_version
        self._persist()
        return rec

    def deprecate_signature(self, sid: str, reason: str) -> Optional[Dict]:
        rec = self._signatures.get(sid)
        if not rec:
            return None
        rec["state"] = "Deprecated"
        rec["metadata"]["deprecation_reason"] = reason
        rec["version_updated"] = self.vm.bump_signatures().signature_version
        self._persist()
        return rec

    def get_updates_since(self, version: int) -> Dict:
        updates = [s for s in self._signatures.values() if s["version_updated"] > version]
        return {
            "signature_version": self.vm.versions.signature_version,
            "updates": updates
        }

    def all(self) -> Dict:
        return {
            "signature_version": self.vm.versions.signature_version,
            "signatures": list(self._signatures.values())
        }
