# Real-Time Network Flow Detection with CSV Logging

This simulates **real-time network traffic streaming** and performs **attack detection** using a trained ML/DL model. 
Each network flow is processed sequentially, classified as **Normal** or **Attack**, and logged into a **CSV file** with detailed evaluation results.

The system is designed for:
- Intrusion Detection System (IDS) experiments
- Model evaluation (TP / FP / FN / TN)
- Real-time demo and presentation use

---

## Features

- Simulated real-time flow streaming from CSV
- Model-based attack detection
- Colored console logs for:
  - True Positives (TP)
  - False Negatives (FN – missed attacks)
  - False Positives (FP – false alarms)
  - True Negatives (TN)
- CSV logging (new file per run)
- Easy offline analysis using Pandas / Excel

---
## CSV Log Format

Each run creates a **fresh CSV file** with the following columns:

| Column | Description |
|------|------------|
| timestamp | Detection time |
| flow_id | Flow index |
| true_label | Ground truth label |
| predicted_label | Model prediction |
| result | TP / FP / FN / TN |
| score | Model confidence / anomaly score |

## How It Works

1. Network flows are read one-by-one from a CSV file
2. Each flow is passed to `detect(flow)`
3. Prediction and ground truth are compared
4. Detection result is logged:
   - Console (colored)
   - CSV (persistent)

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the stream   

```bash
python infer.py
```



Contribution:

* Author: Rodger Jay
* Date: 12/20/2025
* Research purposes only