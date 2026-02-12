from typing import Dict, List, Optional

import numpy as np


class ZeroDayClassifier:
    """
    Classifies candidate clusters as zero-day or not.

    Criteria:
    - High reconstruction error (>= T_recon)
    - Low similarity to known signatures (cosine < T_sim)
    - Observed across >= N_min agents

    Outputs an explanation for auditability.
    """

    def __init__(self, t_recon: float = 0.9, t_sim: float = 0.6, n_min_agents: int = 2):
        self.t_recon = t_recon
        self.t_sim = t_sim
        self.n_min_agents = n_min_agents

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def classify(self, candidate: Dict, known_signature_embeddings: Optional[np.ndarray] = None) -> Dict:
        emb = np.array(candidate["embedding"], dtype=np.float32)
        mean_recon = float(candidate["mean_recon"])
        agent_count = len(candidate.get("agents", []))
        max_sim = 0.0
        if known_signature_embeddings is not None and known_signature_embeddings.size > 0:
            for ks in known_signature_embeddings:
                max_sim = max(max_sim, self._cosine_similarity(emb, ks))

        is_zero_day = (mean_recon >= self.t_recon) and (max_sim < self.t_sim) and (agent_count >= self.n_min_agents)
        explanation = {
            "t_recon": self.t_recon,
            "t_sim": self.t_sim,
            "n_min_agents": self.n_min_agents,
            "observed_agent_count": agent_count,
            "mean_recon": mean_recon,
            "max_similarity_to_known": max_sim,
            "reason": "High reconstruction error, low similarity to known signatures, cross-agent recurrence"
        }

        result = dict(candidate)
        result["is_zero_day"] = bool(is_zero_day)
        result["explanation"] = explanation
        # confidence scaled by recon and low similarity
        result["confidence"] = float(min(1.0, mean_recon) * (1.0 - max_sim))
        return result
