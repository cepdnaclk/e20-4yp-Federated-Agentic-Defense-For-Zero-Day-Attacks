from flask import Flask, request, jsonify
import time
import json
from service_container import inference_service
import agents.Orchestrator.orchestrator as orchestrator
import os
# FORCE TensorFlow to use the legacy Keras (tf-keras package)
os.environ["TF_USE_LEGACY_KERAS"] = "1"


app = Flask(__name__)
myOrchestrator = orchestrator.Orchestrator()

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True)
    # print(data)

    if not data:
        print("[WARN] Empty or invalid JSON received")
        return jsonify({"status": "ignored"}), 200

    flow_id = data.get("flow_id")
    features = data.get("features", {})


    try:  
        result = inference_service.predict(features)
        myOrchestrator.process_autoencoder_input(result)
        # print(result)
        #
        # Send this result to orchestrator for full pipeline processing


        print(f"[INFO] Flow ID: {flow_id} | Prediction: {result['prediction']} | Score: {result['anomaly_score']:.6f}")
    
    except Exception as e:
        print(f"[ERROR] Inference failed for Flow ID:s {flow_id} | Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    # Fire-and-forget response
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
