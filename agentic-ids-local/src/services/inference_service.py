# services/inference_service.py
import numpy as np
import uuid
from datetime import datetime
import pandas as pd

class InferenceService:
    def __init__(self, autoencoder, feature_service, threshold):
        self.autoencoder = autoencoder
        self.feature_service = feature_service
        self.threshold = threshold

    def predict(self, features: dict, context: dict = None):
        df = pd.DataFrame([features])

        # Preprocess
        X_scaled = self.feature_service.preprocess(df)

        # Autoencoder reconstruction
        recon = self.autoencoder.predict(X_scaled, verbose=0)

        # Per-feature reconstruction error
        recon_error_vector = np.square(X_scaled - recon)[0]
        anomaly_score = float(np.mean(recon_error_vector))

        prediction = int(anomaly_score > self.threshold)

        if prediction == 1:
            print(f"[INFERENCE] ANOMALY DETECTED:")
            print(f"            Score: {anomaly_score:.6f} (threshold: {self.threshold})")
            print(f"            Source: {features.get('srcip', 'Unknown')}:{features.get('sport', 'Unknown')}")
            print(f"            Dest: {features.get('dstip', 'Unknown')}:{features.get('dsport', 'Unknown')}")
            print(f"            Protocol: {features.get('proto', 'Unknown')}")
            print(f"            Service: {features.get('service', 'Unknown')}")
        else:
            print(f"[INFERENCE] Normal traffic: {features.get('srcip', 'Unknown')} -> {features.get('dstip', 'Unknown')} | Score: {anomaly_score:.6f}")

        return {
            "flow_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "anomaly_score": anomaly_score,
            "feature_vector": X_scaled[0].tolist(),
            "reconstruction_error_vector": recon_error_vector.tolist(),
            "features": features
        }
