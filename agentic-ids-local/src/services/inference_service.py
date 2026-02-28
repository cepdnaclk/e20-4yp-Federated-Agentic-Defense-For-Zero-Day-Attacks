# services/inference_service.py
import numpy as np
import uuid
from datetime import datetime
import pandas as pd
from keras import Model

class InferenceService:
    def __init__(self, autoencoder, feature_service, threshold):
        self.autoencoder = autoencoder
        self.feature_service = feature_service
        self.threshold = threshold
        # Build encoder model to extract latent embeddings (bottleneck layer)
        self.encoder_model = self._build_encoder_model()

    def _build_encoder_model(self):
        """
        Extract encoder portion from autoencoder to get latent embeddings.
        Assumes symmetric autoencoder architecture where bottleneck is at middle layer.
        """
        layers = self.autoencoder.layers
        num_layers = len(layers)
        # Find bottleneck (middle layer - encoder ends at layer with minimum units)
        encoder_end_idx = num_layers // 2
        
        # Get the bottleneck layer output
        bottleneck_layer = layers[encoder_end_idx]
        encoder_model = Model(
            inputs=self.autoencoder.input,
            outputs=bottleneck_layer.output,
            name="encoder"
        )
        print(f"[INFERENCE] Built encoder model - bottleneck dim: {bottleneck_layer.output.shape[-1]}")
        return encoder_model

    def predict(self, features: dict, context: dict = None):
        df = pd.DataFrame([features])

        # Preprocess
        X_scaled = self.feature_service.preprocess(df)

        # Autoencoder reconstruction
        recon = self.autoencoder.predict(X_scaled, verbose=0)

        # Extract latent embedding (z) from bottleneck layer
        latent_embedding = self.encoder_model.predict(X_scaled, verbose=0)[0]

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
            "latent_embedding": latent_embedding.tolist(),  # NEW: For federation signature
            "feature_vector": X_scaled[0].tolist(),
            "reconstruction_error_vector": recon_error_vector.tolist(),
            "features": features
        }
