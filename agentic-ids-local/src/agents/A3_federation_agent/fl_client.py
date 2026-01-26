import base64
import io
import requests
from typing import List, Dict, Optional


class FLClient:
    """
    Federated Learning client for agent-side communication with FL server.
    Backward compatibility:
    - Existing send_signatures() and fetch_global_patterns() are retained but routed to new endpoints.
    """

    def __init__(self, server_url: str, timeout: int = 5):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    # Backward-compatible: route to /api/submit_update
    def send_signatures(self, signatures: List[Dict]) -> bool:
        try:
            response = requests.post(
                f"{self.server_url}/api/submit_update",
                json={
                    "agent_id": signatures[0].get("agent_id", "unknown"),
                    "signatures": [
                        {"embedding": s.get("embedding", []), "recon_error": s.get("recon_error", 0.0)}
                        for s in signatures
                    ]
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"[FLClient] Send failed: {e}")
            return False

    # Backward-compatible: route to /api/broadcast/signatures
    def fetch_global_patterns(self):
        try:
            response = requests.get(
                f"{self.server_url}/api/broadcast/signatures",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[FLClient] Fetch failed: {e}")
            return None

    # New: register agent
    def register(self, agent_id: str, capabilities: Optional[Dict] = None) -> Optional[Dict]:
        try:
            response = requests.post(
                f"{self.server_url}/api/register",
                json={"agent_id": agent_id, "capabilities": capabilities or {}},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[FLClient] Register failed: {e}")
            return None

    # New: submit full update (weights + anomaly stats + signatures)
    def submit_update(self, agent_id: str, weights: Optional[List], sample_count: int,
                      anomaly_stats: Dict, signatures: List[Dict], round_end: bool = False) -> Optional[Dict]:
        payload = {
            "agent_id": agent_id,
            "sample_count": int(sample_count),
            "anomaly_stats": anomaly_stats,
            "signatures": [
                {"embedding": s.get("embedding", []), "recon_error": float(s.get("recon_error", 0.0))}
                for s in signatures
            ],
            "round_end": bool(round_end)
        }
        if weights is not None:
            # weights should be list[np.ndarray]; encode to base64 npy
            encoded = []
            for arr in weights:
                buf = io.BytesIO()
                import numpy as np
                np.save(buf, arr, allow_pickle=False)
                encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            payload["weights"] = encoded
        try:
            response = requests.post(
                f"{self.server_url}/api/submit_update",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[FLClient] submit_update failed: {e}")
            return None

    # New: fetch global model
    def fetch_global_model(self) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.server_url}/api/broadcast/model",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[FLClient] Fetch global model failed: {e}")
            return None

    # New: fetch signatures with optional version
    def fetch_signatures(self, since_version: Optional[int] = None) -> Optional[Dict]:
        try:
            params = {}
            if since_version is not None:
                params["since"] = int(since_version)
            response = requests.get(
                f"{self.server_url}/api/broadcast/signatures",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[FLClient] Fetch signatures failed: {e}")
            return None
