import numpy as np

def cosine_similarity(sig1: dict, sig2: dict) -> float:
    keys = set(sig1.keys()).union(sig2.keys())
    v1 = np.array([sig1.get(k, 0.0) for k in keys])
    v2 = np.array([sig2.get(k, 0.0) for k in keys])
    return float(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    )
