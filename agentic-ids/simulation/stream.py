import pandas as pd
import requests
import time
import json
import numpy as np

CSV_PATH = "../data/UNSW_NB15/UNSW-NB15_1.csv"
FEATURES_METADATA = "../data/UNSW_NB15/dataset_features.json"
API_URL = "http://localhost:5000/detect"


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

    print(f"Streaming started")

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
            session.post(API_URL, json=payload, timeout=0.05)
        except requests.exceptions.RequestException:
            pass  # fire-and-forget

        time.sleep(0.3)


if __name__ == "__main__":
    stream_data()
