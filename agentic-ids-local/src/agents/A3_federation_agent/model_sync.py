import requests
from typing import List, Optional

import numpy as np


def fetch_global_model(server_url: str, timeout: int = 5) -> Optional[List[np.ndarray]]:
    try:
        r = requests.get(f"{server_url.rstrip('/')}/api/broadcast/model", timeout=timeout)
        r.raise_for_status()
        doc = r.json()
        weights_enc = doc.get("weights", [])
        import base64, io
        arrays = []
        for w in weights_enc:
            buf = io.BytesIO(base64.b64decode(w))
            arrays.append(np.load(buf, allow_pickle=False))
        return arrays
    except requests.RequestException as e:
        print(f"[model_sync] Failed to fetch model: {e}")
        return None


def apply_to_keras(model, weights: List[np.ndarray]) -> bool:
    """
    Apply aggregated weights to a Keras model.
    Assumes order/shape match the server aggregation schema.
    """
    try:
        import tensorflow as tf  # local agents already use Keras/TensorFlow
        keras_weights = []
        for arr in weights:
            keras_weights.append(tf.convert_to_tensor(arr))
        model.set_weights([np.array(w) for w in weights])
        return True
    except Exception as e:
        print(f"[model_sync] Failed to apply weights: {e}")
        return False
