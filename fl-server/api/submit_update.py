import base64
import io
from typing import List

import numpy as np
from flask import Blueprint, request, jsonify, current_app
import json

bp_submit = Blueprint("submit_update", __name__, url_prefix="/api")


def _decode_weights(encoded_weights: List[str]) -> List[np.ndarray]:
    arrays = []
    for w in encoded_weights:
        buf = io.BytesIO(base64.b64decode(w))
        arrays.append(np.load(buf, allow_pickle=False))
    return arrays


@bp_submit.route("/submit_update", methods=["POST"])
def submit_update():
    payload = request.get_json(force=True, silent=True) or {}

    agent_id = payload.get("agent_id")
    if not agent_id:
        return jsonify({"error": "agent_id required"}), 400

    weights = payload.get("weights")  # list of base64 npy buffers
    sample_count = int(payload.get("sample_count", 1))
    anomaly_stats = payload.get("anomaly_stats", {})
    signatures = payload.get("signatures", [])  # [{embedding: [...], recon_error: float}]
    round_end = bool(payload.get("round_end", False))

    # Add to FedAvg aggregator
    agg = current_app.config["AGGREGATOR"]
    if weights:
        agg.add_update(agent_id, weights, sample_count)

    # Submit signatures to drift detector
    dd = current_app.config["DRIFT_DETECTOR"]
    if signatures:
        embeddings = np.array([s.get("embedding", []) for s in signatures], dtype=np.float32)
        recons = np.array([float(s.get("recon_error", 0.0)) for s in signatures], dtype=np.float32)
        dd.submit_signatures(agent_id, embeddings, recons)

    result = {"status": "accepted"}

    # Aggregate model if round ends or enough updates are present
    round_size = int(current_app.config.get("ROUND_SIZE", 2))
    updates_count = len(agg._updates)
    aggregated_weights = None
    if round_end or updates_count >= round_size:
        aggregated_weights = agg.aggregate()
        if aggregated_weights:
            vm = current_app.config["VERSION_MANAGER"]
            vm.bump_model()
            current_app.config["GLOBAL_WEIGHTS"] = aggregated_weights
            result["model_version"] = vm.versions.model_version
            # Optional MQTT broadcast of model
            mqtt = current_app.config.get("MQTT")
            if mqtt:
                try:
                    encoded = agg.encode_weights(aggregated_weights)
                    payload = {"model_version": vm.versions.model_version, "weights": encoded}
                    mqtt["client"].publish(mqtt["topic_model"], json.dumps(payload))
                except Exception as e:
                    print(f"[FL Server] MQTT model publish failed: {e}")

    # Detect zero-day candidates
    ss = current_app.config["SIGNATURE_STORE"]
    known_embs = ss.get_known_embeddings()
    candidates = current_app.config["DRIFT_DETECTOR"].detect_zero_day_candidates(known_embs)
    zdc = current_app.config["ZERO_DAY_CLASSIFIER"]
    promoted = []
    for cand in candidates:
        classified = zdc.classify(cand, known_embs)
        if classified.get("is_zero_day"):
            rec = ss.add_candidate(classified["embedding"], classified["confidence"], cand.get("agents", []), metadata={
                "mean_recon": classified["mean_recon"],
                "explanation": classified["explanation"]
            })
            promoted.append(rec)
    if promoted:
        result["signature_version"] = ss.vm.versions.signature_version
        result["new_candidates"] = promoted
        # Optional MQTT broadcast of signature updates
        mqtt = current_app.config.get("MQTT")
        if mqtt:
            try:
                payload = {"signature_version": ss.vm.versions.signature_version, "updates": promoted}
                mqtt["client"].publish(mqtt["topic_sigs"], json.dumps(payload))
            except Exception as e:
                print(f"[FL Server] MQTT signatures publish failed: {e}")

    return jsonify(result), 200
