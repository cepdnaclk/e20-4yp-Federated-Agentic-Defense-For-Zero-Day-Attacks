import os
from flask import Flask
import numpy as np

from aggregation.fedavg import FedAvgAggregator
from aggregation.drift_detector import DriftDetector
from aggregation.zero_day_classifier import ZeroDayClassifier
from knowledge.signature_store import SignatureStore
from knowledge.versioning import VersionManager

from api.register import bp_register
from api.submit_update import bp_submit
from api.broadcast import bp_broadcast


def create_app():
    app = Flask(__name__)

    # Core components
    app.config["AGGREGATOR"] = FedAvgAggregator()
    app.config["DRIFT_DETECTOR"] = DriftDetector(
        eps=float(os.environ.get("FL_EPS", 0.8)),
        t_recon=float(os.environ.get("FL_T_RECON", 0.9)),
        t_sim=float(os.environ.get("FL_T_SIM", 0.6)),
        n_min_agents=int(os.environ.get("FL_N_MIN_AGENTS", 2))
    )
    app.config["ZERO_DAY_CLASSIFIER"] = ZeroDayClassifier(
        t_recon=float(os.environ.get("FL_T_RECON", 0.9)),
        t_sim=float(os.environ.get("FL_T_SIM", 0.6)),
        n_min_agents=int(os.environ.get("FL_N_MIN_AGENTS", 2))
    )
    app.config["SIGNATURE_STORE"] = SignatureStore(storage_path=os.environ.get("FL_SIGNATURE_PATH", "./fl-server/knowledge/signatures.json"))
    app.config["VERSION_MANAGER"] = app.config["SIGNATURE_STORE"].vm
    # Provide a dummy global model so broadcast/model works immediately
    app.config["GLOBAL_WEIGHTS"] = [np.zeros((1,), dtype=np.float32), np.ones((1,), dtype=np.float32)]

    app.config["ROUND_SIZE"] = int(os.environ.get("FL_ROUND_SIZE", 2))

    # Optional MQTT config
    mqtt_host = os.environ.get("FL_MQTT_HOST")
    mqtt_port = int(os.environ.get("FL_MQTT_PORT", 1883))
    mqtt_topic_model = os.environ.get("FL_MQTT_TOPIC_MODEL", "fl/global/model")
    mqtt_topic_sigs = os.environ.get("FL_MQTT_TOPIC_SIGS", "fl/global/signatures")
    app.config["MQTT"] = None
    if mqtt_host:
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client()
            client.connect(mqtt_host, mqtt_port, 60)
            client.loop_start()
            app.config["MQTT"] = {
                "client": client,
                "topic_model": mqtt_topic_model,
                "topic_sigs": mqtt_topic_sigs
            }
        except Exception as e:
            print(f"[FL Server] MQTT init failed: {e}")

    # Register blueprints
    app.register_blueprint(bp_register)
    app.register_blueprint(bp_submit)
    app.register_blueprint(bp_broadcast)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9090))
    print(f"[FL Server] Starting on port {port}...")
    app = create_app()
    app.run(host="0.0.0.0", port=port)
