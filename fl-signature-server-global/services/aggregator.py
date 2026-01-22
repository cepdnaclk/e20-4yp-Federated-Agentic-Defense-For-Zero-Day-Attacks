import hashlib
from collections import defaultdict
from typing import List

from models.signature import AnomalySignature
from models.global_pattern import GlobalPattern
from utils.similarity import cosine_similarity

SIMILARITY_THRESHOLD = 0.85


def aggregate_signatures(signatures: List[AnomalySignature]) -> List[GlobalPattern]:
    clusters = []

    for sig in signatures:
        assigned = False

        for cluster in clusters:
            similarity = cosine_similarity(
                sig.feature_deviation,
                cluster["centroid"]
            )

            if similarity >= SIMILARITY_THRESHOLD:
                cluster["members"].append(sig)
                assigned = True
                break

        if not assigned:
            clusters.append({
                "members": [sig],
                "centroid": sig.feature_deviation.copy()
            })

    global_patterns = []

    for cluster in clusters:
        centroid = defaultdict(float)
        agent_ids = set()
        total_freq = 0
        conf_sum = 0.0

        for sig in cluster["members"]:
            for k, v in sig.feature_deviation.items():
                centroid[k] += v
            total_freq += sig.frequency
            conf_sum += sig.confidence
            agent_ids.add(sig.agent_id)

        for k in centroid:
            centroid[k] /= len(cluster["members"])

        pattern_id = hashlib.sha256(str(dict(centroid)).encode()).hexdigest()[:16]

        global_patterns.append(
            GlobalPattern(
                pattern_id=pattern_id,
                centroid=dict(centroid),
                agent_count=len(agent_ids),
                total_frequency=total_freq,
                confidence=conf_sum / len(cluster["members"])
            )
        )

    return global_patterns
