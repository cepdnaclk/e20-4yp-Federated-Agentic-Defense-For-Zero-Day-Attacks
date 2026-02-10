# federation/fl_client.py

import requests
from typing import List, Dict


class FLClient:
    def __init__(self, server_url: str, timeout: int = 5):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def send_signatures(self, signatures: List[Dict]) -> bool:
        """Send a batch of signatures to the FL server, one at a time"""
        try:
            for signature in signatures:
                response = requests.post(
                    f"{self.server_url}/upload_update",
                    json=signature,
                    timeout=self.timeout
                )
                response.raise_for_status()
            return True

        except requests.exceptions.RequestException as e:
            print(f"[FLClient] Send failed: {e}")
            return False

    def fetch_global_patterns(self):
        try:
            response = requests.get(
                f"{self.server_url}/get_global_model",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"[FLClient] Fetch failed: {e}")
            return None
