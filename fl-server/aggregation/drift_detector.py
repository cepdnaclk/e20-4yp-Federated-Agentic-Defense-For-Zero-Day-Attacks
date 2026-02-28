from typing import List, Dict, Optional, Tuple

import numpy as np


class DriftDetector:
    """
    Detects cross-agent drift and zero-day candidates using latent-space anomaly clusters.

    Approach:
    - Agents submit anomaly embeddings (latent vectors) with per-sample reconstruction errors.
    - We form online clusters using centroid-based assignment with L2 threshold (eps).
    - A cluster is a zero-day candidate if:
      * mean reconstruction error >= T_recon
      * low similarity to known signatures (cosine < T_sim)
      * observed by >= N_min distinct agents (default N_min=1 for single-agent sharing)
    
    MODIFIED: Default N_min=1 enables single-agent sharing - any agent detecting
    a new attack pattern can immediately share it with the federation.
    """

    def __init__(self, eps: float = 0.8, t_recon: float = 0.9, t_sim: float = 0.6, n_min_agents: int = 1):
        self.eps = eps
        self.t_recon = t_recon
        self.t_sim = t_sim
        self.n_min_agents = n_min_agents
        # Buffer of submissions: list of (agent_id, embeddings [N,D], recon_errors [N])
        self._buffer: List[Tuple[str, np.ndarray, np.ndarray]] = []

    def submit_signatures(self, agent_id: str, embeddings: np.ndarray, recon_errors: np.ndarray) -> None:
        if embeddings.size == 0:
            return
        assert embeddings.ndim == 2, "Embeddings must be 2D (N,D)"
        assert recon_errors.ndim == 1 and recon_errors.shape[0] == embeddings.shape[0]
        self._buffer.append((agent_id, embeddings.astype(np.float32), recon_errors.astype(np.float32)))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _cluster_embeddings(self) -> List[Dict]:
        """
        Online centroid-based clustering with L2 threshold eps.
        Returns clusters: [{centroid, members, agent_ids, mean_recon}]
        """
        clusters: List[Dict] = []
        for agent_id, E, R in self._buffer:
            for e, r in zip(E, R):
                assigned = False
                for c in clusters:
                    dist = np.linalg.norm(e - c["centroid"])  # L2 distance
                    if dist <= self.eps:
                        c["members"].append(e)
                        c["agent_ids"].add(agent_id)
                        c["recon"].append(r)
                        # update centroid incrementally
                        c["centroid"] = np.mean(np.stack(c["members"]), axis=0)
                        assigned = True
                        break
                if not assigned:
                    clusters.append({
                        "centroid": e.copy(),
                        "members": [e.copy()],
                        "agent_ids": {agent_id},
                        "recon": [r]
                    })
        # finalize cluster stats
        for c in clusters:
            c["mean_recon"] = float(np.mean(np.array(c["recon"], dtype=np.float32)))
        return clusters

    def detect_zero_day_candidates(self, known_signature_embeddings: Optional[np.ndarray] = None) -> List[Dict]:
        clusters = self._cluster_embeddings()
        candidates: List[Dict] = []
        for c in clusters:
            if c["mean_recon"] < self.t_recon:
                continue
            # compute max similarity to known signatures
            max_sim = 0.0
            if known_signature_embeddings is not None and known_signature_embeddings.size > 0:
                for ks in known_signature_embeddings:
                    max_sim = max(max_sim, self._cosine_similarity(c["centroid"], ks))
            if max_sim >= self.t_sim:
                # too similar to known signatures; not a zero-day candidate
                continue
            if len(c["agent_ids"]) >= self.n_min_agents:
                candidates.append({
                    "embedding": c["centroid"].tolist(),
                    "mean_recon": c["mean_recon"],
                    "agents": list(c["agent_ids"]),
                    "member_count": len(c["members"]),
                    "confidence": float(min(1.0, c["mean_recon"]))
                })
        # clear buffer after detection window
        self._buffer = []
        return candidates
