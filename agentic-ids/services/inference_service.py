# services/inference_service.py
import numpy as np

class InferenceService:
    def __init__(self, autoencoder, feature_service, threshold):
        self.autoencoder = autoencoder
        self.feature_service = feature_service
        self.threshold = threshold

    def predict(self, features: dict):
        import pandas as pd
        df = pd.DataFrame([features])

        # Preprocess
        X_scaled = self.feature_service.preprocess(df)

        # Autoencoder reconstruction
        recon = self.autoencoder.predict(X_scaled, verbose=0)
        recon_error = np.mean(np.square(X_scaled - recon), axis=1)

        # Threshold
        prediction = (recon_error > self.threshold).astype(int)

        return {
            "prediction": int(prediction[0]),
            "score": float(recon_error[0]),
            "details": {
                "reconstruction_error": recon_error[0],
                "features": features
            }
        }
