from flask import Flask, request, jsonify
import time
import json
from service_container import inference_service

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/detect", methods=["POST"])
def detect():
    recv_time = time.time()

    data = request.get_json(silent=True)
    # print(data)

    if not data:
        print("[WARN] Empty or invalid JSON received")
        return jsonify({"status": "ignored"}), 200

    flow_id = data.get("flow_id")
    features = data.get("features", {})

    # print("\n================= FLOW RECEIVED =================")
    # print(f"Time     : {recv_time}")
    # print(f"Flow ID  : {flow_id}")
    # print(f"Features : {json.dumps(features, indent=2)}")
    # print("================================================\n")

    try:

        
        result = inference_service.predict(features)
        print(result)
        #
        # Send this result to orchestrator for full pipeline processing


        print(f"[INFO] Flow ID: {flow_id} | Prediction: {result['prediction']} | Score: {result['score']:.6f}")
        # return jsonify({
        #     "flow_id": flow_id,
        #     "prediction": result["prediction"],
        #     "score": result["score"],
        #     "recv_time": recv_time
        # }), 200
    except Exception as e:
        print(f"[ERROR] Inference failed for Flow ID: {flow_id} | Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    # Fire-and-forget response
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
