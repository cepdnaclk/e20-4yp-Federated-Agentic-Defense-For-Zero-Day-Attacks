import time, json, joblib
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from datetime import datetime
import json
import csv
import os


# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


CSVPATH = "../../data/UNSW_NB15/UNSW_NB15_testing-set.csv"
MODEL_VERSION="M002"

#  Loading artifacts
autoencoder = tf.keras.models.load_model(f"../artifacts/{MODEL_VERSION}/autoencoder.keras")
scaler = joblib.load(f"../artifacts/{MODEL_VERSION}/scaler.pkl")
encoder = joblib.load(f"../artifacts/{MODEL_VERSION}/encoder.pkl")

with open(f"../artifacts/{MODEL_VERSION}/metadata.json") as f:
    metadata = json.load(f)

selected_numerical_cols = metadata["selected_numerical_columns"]
categorical_cols = metadata["categorical_columns"]
onehot_cols = metadata["onehot_columns"]
threshold = metadata["best_threshold"]


# Logging
def init_csv_logger(csv_file):
    with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "flow_id",
                "true_label",
                "predicted_label",
                "result",        # TP / TN / FP / FN
                "score"
            ])
        

# preprocess function
def preprocess_flow(flow_df):
    df = flow_df.copy()
    
    y_true = df['attack_cat'].to_string(index=False)


    # 1. Handle categorical features with one-hot
    X_cat = pd.get_dummies(df[categorical_cols], drop_first=False)  # keep all for consistency

    # 2. Reindex to match training columns
    X_cat = X_cat.reindex(columns=[c for c in onehot_cols if c.startswith(tuple(categorical_cols))], fill_value=0)

    # 3. Numeric features
    X_num = df[selected_numerical_cols]

    # 4. Combine
    X_final = pd.concat([X_num, X_cat], axis=1)


    # 5. Ensure order matches training
    X_final = X_final.reindex(columns=selected_numerical_cols + onehot_cols, fill_value=0)

    # 6. Scale numeric + onehot if scaler was fitted on full dataset
    X_scaled = scaler.transform(X_final)


    return X_scaled, y_true

# Detection Function
def detect(flow_df):
    X, y_true = preprocess_flow(flow_df)

    # Autoencoder reconstruction error
    recon = autoencoder.predict(X, verbose=0)
    recon_error = np.mean(np.square(X - recon), axis=1)

    # Ensemble score
    final_score = recon_error

    prediction = (final_score > threshold).astype(int)

    return prediction, final_score, y_true


# Simulate real time streaming
def stream_flows(csv_path, delay = 0.5, log_file="detection_logs.csv", verbose=0):
    df=pd.read_csv(csv_path)
    init_csv_logger(log_file)

    for i in range(len(df)):
        flow = df.iloc[i:i+1]
        pred, score, y_true = detect(flow)

        y_pred_val = "Attack" if pred[0] == 1 else "Normal"
        timestamp = datetime.now().strftime("%H:%M:%S")

        y_true_val = y_true


        # Determine detection outcome
        if y_true_val != "Normal" and y_pred_val == "Attack":
            result = "TP"
        elif y_true_val != "Normal" and y_pred_val == "Normal":
            result = "FN"
        elif y_true_val == "Normal" and y_pred_val == "Attack":
            result = "FP"
        else:
            result = "TN"


        if (verbose == 1):
            # --- Detection logic ---
            if y_true_val != "Normal" and y_pred_val == "Attack":
                # True Positive
                print(
                    f"{GREEN}[TP][{timestamp}] Flow {i} -> ATTACK detected correctly "
                    f"| Score={score[0]:.4f}{RESET}"
                )

            elif y_true_val != "Normal" and y_pred_val == "Normal":
                # False Negative (missed attack)
                print(
                    f"{RED}[FN][{timestamp}] Flow {i} -> MISSED ATTACK "
                    f"| True={y_true_val} | Score={score[0]:.4f}{RESET}"
                )

            elif y_true_val == "Normal" and y_pred_val == "Attack":
                # False Positive
                print(
                    f"{YELLOW}[FP][{timestamp}] Flow {i} -> FALSE ALARM "
                    f"| Score={score[0]:.4f}{RESET}"
                )

            else:
                # True Negative (optional logging)
                print(
                    f"{BLUE}[TN][{timestamp}] Flow {i} -> Normal traffic "
                    f"| Score={score[0]:.4f}{RESET}"
                )
            time.sleep(delay)
        
        # Append to CSV
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                i,
                y_true_val,
                y_pred_val,
                result,
                float(score[0])
            ])
        

if __name__== "__main__":
    stream_flows(CSVPATH, delay = 0.3,log_file="detection_logs.csv", verbose=1)