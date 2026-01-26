import requests
from typing import Optional

from ..knowledge_base import LocalKnowledgeBase


def fetch_server_signatures(server_url: str, since_version: Optional[int] = None, timeout: int = 5) -> Optional[dict]:
    try:
        url = f"{server_url.rstrip('/')}/api/broadcast/signatures"
        params = {}
        if since_version is not None:
            params["since"] = int(since_version)
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[kb_sync] Failed to fetch signatures: {e}")
        return None


def merge_with_local(kb_path: str, server_updates: dict) -> dict:
    kb = LocalKnowledgeBase.load(kb_path)
    changes = kb.merge_updates(server_updates)
    kb.save(kb_path)
    return changes
