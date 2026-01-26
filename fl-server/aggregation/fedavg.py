import base64
import io
from typing import List, Dict, Optional

import numpy as np


class FedAvgAggregator:
    """
    FedAvg aggregator for autoencoder weights.
    - Accepts per-agent weight updates (list of arrays, same shapes).
    - Aggregates via element-wise mean.
    - Optionally weights by sample_count.
    - Does NOT handle raw gradients; only weights or weight diffs.
    """

    def __init__(self):
        self._updates: List[Dict] = []

    @staticmethod
    def _decode_weights(encoded_weights: List[str]) -> List[np.ndarray]:
        arrays = []
        for w in encoded_weights:
            buf = io.BytesIO(base64.b64decode(w))
            arrays.append(np.load(buf, allow_pickle=False))
        return arrays

    @staticmethod
    def encode_weights(weights: List[np.ndarray]) -> List[str]:
        encoded = []
        for arr in weights:
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return encoded

    def add_update(self, agent_id: str, encoded_weights: List[str], sample_count: Optional[int] = None):
        weights = self._decode_weights(encoded_weights)
        self._updates.append({
            "agent_id": agent_id,
            "weights": weights,
            "sample_count": sample_count if sample_count is not None else 1
        })

    def aggregate(self) -> Optional[List[np.ndarray]]:
        if not self._updates:
            return None

        # Ensure consistent shapes across updates
        num_layers = len(self._updates[0]["weights"])
        for upd in self._updates:
            assert len(upd["weights"]) == num_layers, "Mismatched number of layers in updates"

        # Weighted average by sample_count
        totals = [np.zeros_like(self._updates[0]["weights"][i]) for i in range(num_layers)]
        total_weight = 0.0
        for upd in self._updates:
            sc = float(upd["sample_count"]) if upd["sample_count"] else 1.0
            for i in range(num_layers):
                totals[i] += upd["weights"][i] * sc
            total_weight += sc

        aggregated = [totals[i] / max(total_weight, 1e-8) for i in range(num_layers)]
        # Reset for next round
        self._updates = []
        return aggregated
