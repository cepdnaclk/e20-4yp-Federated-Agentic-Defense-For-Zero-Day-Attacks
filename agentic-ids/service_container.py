from services.feature_service import FeatureService
from services.inference_service import InferenceService
import joblib
import json
import tensorflow as tf 

# Load artifacts
with open(".config/model_columns.json") as f:
    model_columns = json.load(f)["model_columns"]

with open(".config/column_rename_map.json") as f:
    rename_map = json.load(f)

#  Loading artifacts
autoencoder = tf.keras.models.load_model(f"models/artifacts/M002/autoencoder.keras")
scaler = joblib.load(f"models/artifacts/M002/scaler.pkl")
encoder = joblib.load(f"models/artifacts/M002/encoder.pkl")

with open(f"models/artifacts/M002/metadata.json") as f:
    metadata = json.load(f)

numerical_cols = metadata["selected_numerical_columns"]
categorical_cols = metadata["categorical_columns"]
onehot_cols = metadata["onehot_columns"]
threshold = metadata["best_threshold"]



# Initialize services
feature_service = FeatureService(rename_map, model_columns,
                                 categorical_cols, onehot_cols, numerical_cols, scaler, encoder=encoder)

inference_service = InferenceService(autoencoder, feature_service, threshold)
