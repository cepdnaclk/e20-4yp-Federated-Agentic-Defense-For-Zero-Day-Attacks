from flask import Flask, request, jsonify
from datetime import datetime

from models.signature import AnomalySignature
from services.aggregator import aggregate_signatures
from store.memory import SIGNATURE_STORE, GLOBAL_PATTERNS

app = Flask(__name__)


@app.route("/submit", methods=["POST"])
def submit_signatures():
    payload = request.get_json(force=True)
    signatures = []

    for item in payload:
        item["timestamp"] = datetime.fromisoformat(item["timestamp"])
        signatures.append(AnomalySignature(**item))

    SIGNATURE_STORE.extend(signatures)

    return jsonify({
        "status": "received",
        "count": len(signatures)
    })


@app.route("/aggregate", methods=["POST"])
def aggregate():
    global GLOBAL_PATTERNS
    GLOBAL_PATTERNS = aggregate_signatures(SIGNATURE_STORE)

    return jsonify({
        "patterns_generated": len(GLOBAL_PATTERNS)
    })


@app.route("/patterns", methods=["GET"])
def get_patterns():
    return jsonify([p.dict() for p in GLOBAL_PATTERNS])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
