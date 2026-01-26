import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class KBVersion:
    signature_version: int = 0


@dataclass
class LocalKnowledgeBase:
    normal_embeddings: List[List[float]] = field(default_factory=list)
    anomaly_signatures: Dict[str, Dict] = field(default_factory=dict)
    zero_day_candidates: Dict[str, Dict] = field(default_factory=dict)
    version: KBVersion = field(default_factory=KBVersion)

    def to_dict(self) -> Dict:
        return {
            "normal_embeddings": self.normal_embeddings,
            "anomaly_signatures": self.anomaly_signatures,
            "zero_day_candidates": self.zero_day_candidates,
            "signature_version": self.version.signature_version
        }

    @staticmethod
    def from_dict(doc: Dict) -> "LocalKnowledgeBase":
        kb = LocalKnowledgeBase()
        kb.normal_embeddings = doc.get("normal_embeddings", [])
        kb.anomaly_signatures = doc.get("anomaly_signatures", {})
        kb.zero_day_candidates = doc.get("zero_day_candidates", {})
        kb.version.signature_version = int(doc.get("signature_version", 0))
        return kb

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "LocalKnowledgeBase":
        if not os.path.exists(path):
            return LocalKnowledgeBase()
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return LocalKnowledgeBase.from_dict(doc)

    def merge_updates(self, updates: Dict) -> Dict:
        """
        Merge signatures updates from server safely.

        Strategy:
        - Accept Global/Verified signatures directly.
        - Add Candidates with caution: mark locally and require local corroboration.
        - Handle deprecations by marking disabled.
        - Resolve conflicts by preferring latest version and higher confidence.
        """
        changed = {"added": [], "updated": [], "deprecated": []}
        server_version = int(updates.get("signature_version", self.version.signature_version))
        for sig in updates.get("updates", updates.get("signatures", [])):
            sid = sig.get("id")
            state = sig.get("state")
            confidence = float(sig.get("confidence", 0.0))
            if sid in self.anomaly_signatures:
                # Conflict resolution: prefer higher version and higher confidence
                local = self.anomaly_signatures[sid]
                if server_version >= local.get("version", 0) or confidence >= float(local.get("confidence", 0.0)):
                    self.anomaly_signatures[sid] = {**local, **sig, "version": server_version}
                    changed["updated"].append(sid)
            else:
                if state in ("Verified", "Global", "Candidate"):
                    rec = {**sig, "version": server_version}
                    self.anomaly_signatures[sid] = rec
                    changed["added"].append(sid)
            if state == "Deprecated":
                # Mark deprecated locally
                if sid in self.anomaly_signatures:
                    self.anomaly_signatures[sid]["deprecated"] = True
                    changed["deprecated"].append(sid)
        # Update local version
        self.version.signature_version = max(self.version.signature_version, server_version)
        return changed

    def verify_candidate_locally(self, sid: str, embeddings: List[List[float]], recon_errors: List[float]) -> Optional[Dict]:
        """
        Locally corroborate a candidate by checking:
        - embeddings close to candidate centroid
        - reconstruction errors above local threshold
        """
        rec = self.anomaly_signatures.get(sid)
        if not rec or rec.get("state") != "Candidate":
            return None
        centroid = np.array(rec.get("embedding", []), dtype=np.float32)
        local_embs = np.array(embeddings, dtype=np.float32)
        dists = np.linalg.norm(local_embs - centroid, axis=1)
        close = np.mean(dists <= float(rec.get("metadata", {}).get("eps", 0.8)))
        high_err = np.mean(np.array(recon_errors, dtype=np.float32) >= float(rec.get("metadata", {}).get("t_recon", 0.9)))
        corroborated = (close > 0.5) and (high_err > 0.5)
        if corroborated:
            rec["state"] = "Verified"
            rec["confidence"] = max(float(rec.get("confidence", 0.0)), 0.8)
            self.anomaly_signatures[sid] = rec
            return rec
        return None
