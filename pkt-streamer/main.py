import pandas as pd
import requests
import time
import json
import numpy as np
from dotenv import load_dotenv
import os

# Load variables from .env into environment
load_dotenv(dotenv_path="config.env")

# Configuration fetched from environment variables
CSV_PATH = os.getenv("CSV_PATH")
FEATURES_METADATA = os.getenv("FEATURES_METADATA")
API_URL = os.getenv("API_URL")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", 0.05))
STREAM_DELAY = float(os.getenv("STREAM_DELAY", 1))
DROP_COLUMNS = os.getenv("DROP_COLUMNS", "attack_cat,Label").split(",")



def load_feature_names():
    with open(FEATURES_METADATA, "r") as f:
        metadata_json = json.load(f)

    # handle both possible structures safely
    if isinstance(metadata_json, dict):
        metadata = metadata_json.get("features")
    else:
        metadata = metadata_json

    if not isinstance(metadata, list):
        raise ValueError("Invalid features_metadata.json format")

    metadata_sorted = sorted(metadata, key=lambda x: x["index"])

    print("[SUCCESS] Metadata processed successfully.")
    return [f["name"] for f in metadata_sorted]


def stream_data():
    feature_names = load_feature_names()

    # CSV has NO headers
    df = pd.read_csv(CSV_PATH, header=None)

    # sanity check
    assert len(df.columns) == len(feature_names), \
        "Column count mismatch between CSV and metadata"

    # assign feature names
    df.columns = feature_names

    DROP_COLUMNS = {"attack_cat", "Label"}
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    print(f"[INFO] Starting data stream to {API_URL}...")

    session = requests.Session()

    for i, row in df.iterrows():
        # convert numpy → native Python types
        features = {
            k: (None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v)
            for k, v in row.to_dict().items()
        }

        payload = {
            "flow_id": int(i),
            "features": features
        }

        try:
            response = session.post(API_URL, json=payload, timeout=API_TIMEOUT)
            if response.status_code == 200:
                print(f"[SUCCESS] Flow ID {i} processed successfully")
            else:
                print(f"[WARNING] Request to {API_URL} failed for flow_id {i} - HTTP {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Connection failed to {API_URL} for flow_id {i} - Service not running: {e}")
        except requests.exceptions.Timeout as e:
            print(f"[ERROR] Request timeout to {API_URL} for flow_id {i} - Increase API_TIMEOUT: {e}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request to {API_URL} failed for flow_id {i} - {type(e).__name__}: {e}")
            pass  # fire-and-forget

        time.sleep(STREAM_DELAY)


if __name__ == "__main__":
    print("[INFO] Initializing packet streamer...")
    stream_data()
