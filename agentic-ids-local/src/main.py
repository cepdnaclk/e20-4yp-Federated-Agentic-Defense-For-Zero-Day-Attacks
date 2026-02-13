from flask import Flask, request, jsonify
import time
import json
import os
import dotenv
import warnings
import uuid

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load environment variables from .env file
dotenv.load_dotenv()

PORT = int(os.getenv("PORT", 5000))

# FORCE TensorFlow to use the legacy Keras (tf-keras package) 
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from service_container import inference_service
import agents.Orchestrator.orchestrator as orchestrator

# Import monitoring service
from utils.monitoring_service import (
    get_monitoring_service, 
    log_packet_processing, 
    log_inference_prediction,
    print_monitoring_summary
)


app = Flask(__name__)
myOrchestrator = orchestrator.Orchestrator()

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/monitoring", methods=["GET"])
def monitoring_status():
    """Endpoint to get current monitoring metrics"""
    monitor = get_monitoring_service()
    metrics = monitor.get_session_metrics()
    return jsonify(metrics), 200

@app.route("/monitoring/summary", methods=["GET"])
def monitoring_summary():
    """Endpoint to trigger monitoring summary print and return metrics"""
    monitor = get_monitoring_service()
    monitor.print_session_summary()
    metrics = monitor.get_session_metrics()
    return jsonify({"message": "Summary printed to console", "metrics": metrics}), 200

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True)
    
    # Return 200 immediately when data comes in
    from threading import Thread
    
    def process_data():
        processing_start = time.time()
        
        if not data:
            print("[WARN] Empty or invalid JSON received")
            return

        flow_id = data.get("flow_id") or str(uuid.uuid4())
        features = data.get("features", {})
        
        # Log incoming packet
        log_packet_processing(flow_id, features)

        try:  
            # Get inference prediction
            result = inference_service.predict(features)
            result['flow_id'] = flow_id  # Ensure flow_id consistency
            
            # Log inference result
            log_inference_prediction(flow_id, result)
            
            # Process through orchestrator 
            if result['prediction'] == 1:
                myOrchestrator.process_autoencoder_input(result, flow_id)
            
            processing_time = (time.time() - processing_start) * 1000  # Convert to ms
            print(f"[INFO] Flow ID: {flow_id} | Prediction: {result['prediction']} | Score: {result['anomaly_score']:.6f} | Processing: {processing_time:.2f}ms")
        
        except Exception as e:
            print(f"[ERROR] Inference failed for Flow ID: {flow_id} | Error: {str(e)}")

    # Start processing in background thread
    Thread(target=process_data, daemon=True).start()
    
    # Return 200 immediately
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    # Print startup message
    print(f"[INFO] Starting Agentic IDS Local Server on port {PORT}...")
    print(f"[INFO] Monitoring system initialized - logs will be saved to: {get_monitoring_service().log_dir}")
    print(f"[INFO] Federation functionality: DISABLED (commented out for testing)")
    print(f"[INFO] Available endpoints:")
    print(f"       - POST /detect     : Process network packets")
    print(f"       - GET  /health     : Health check")  
    print(f"       - GET  /monitoring : Get current metrics")
    print(f"       - GET  /monitoring/summary : Print and get full summary")
    app.run(host="0.0.0.0", port=PORT, debug=True)
