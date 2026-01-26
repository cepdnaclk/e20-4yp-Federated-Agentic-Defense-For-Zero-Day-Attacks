# Federated Architecture Design

This document specifies the federated learning (FL) extension for the agentic intrusion detection system (IDS). It ensures: minimal changes to local agents, privacy (no raw packets leave agents), and federated coordination via Flask-based REST with optional MQTT.

## Design Overview

- Local detection remains autonomous: agents run autoencoder-based anomaly detection on live packet streams.
- Federated coordination is out-of-band: agents periodically participate in FL rounds by submitting model updates, anomaly statistics, and latent-space signatures.
- Knowledge dissemination is controlled: server aggregates model weights (FedAvg baseline), detects cross-agent novelty (zero-day), and broadcasts global model + curated signatures.

## Components

- Local Agent
  - Packet Streamer: captures live traffic, extracts features.
  - Autoencoder: computes reconstruction error; embeds samples in latent space.
  - Local Signature Builder: clusters local anomalies; produces human-interpretable signatures.
  - Local KB: stores normal embeddings, anomaly signatures, zero-day candidates, confidence scores; versioned.
  - FL Client: registers with server, submits updates, synchronizes model and KB.

- FL Server (Flask)
  - Registration API: tracks agents and capabilities.
  - Aggregation: FedAvg for model updates; secure baseline (no raw data).
  - Drift & Zero-day: analyzes cross-agent latent anomalies and reconstruction errors.
  - Knowledge Store: manages signature lifecycle; versioned updates.
  - Broadcast: disseminates global model, signatures, deprecations via REST and optional MQTT.

## Participation in FL Rounds

1. Local agents collect anomalies during a window W.
2. Agents compute model update (weights or gradients) and package:
   - weights_diff: list of numpy arrays serialized to base64
   - anomaly_stats: mean/std of reconstruction errors; counts
   - signature_embeddings: representative latent vectors for local anomaly clusters
   - signature_metadata: labels, confidence, feature ranges
3. Agents send POST /api/submit_update.
4. Server aggregates using FedAvg after ≥ M updates or on explicit round_end flag.
5. Server runs drift and zero-day detection across submitted signatures.
6. Server updates global knowledge base with versioning and signature lifecycle.
7. Server broadcasts updated model and signatures.

## Global Aggregation

- FedAvg: element-wise average of submitted weight tensors.
- Client-weighting: optionally weight contributions by sample_count.
- Secure baseline: no raw packets; only model updates and high-level statistics/signatures.

## Knowledge Redistribution

- Broadcast provides:
  - global_model: weights and version
  - signatures: new, verified, deprecated; with confidence and rationale
- Agents merge via `kb_sync`: validate versions, resolve conflicts, ensure rollback safety.

## Separation of Concerns

- Local Detection: packet streaming, autoencoder inference, local KB maintenance.
- Federated Coordination: register, submit updates, participate in rounds.
- Knowledge Dissemination: receive global model/signatures, merge safely.

## Flow Diagram (Text-based)

[Packet Stream]
      ↓
[Local Autoencoder]
      ↓
[Anomaly Score]
      ↓
[Local Signature Builder]
      ↓
[Local KB]
      ↓
[FL Update Sender]
      ↓
[FL Server Aggregation]
      ↓
[Zero-Day Detector]
      ↓
[Global Knowledge Base]
      ↓
[Broadcast Updates]
      ↓
[Local Agent KB Update]

### Inputs / Outputs / Processing

- Packet Stream → features → Autoencoder → reconstruction error (score)
- Signature Builder → latent clusters → signatures (embeddings + metadata)
- FL Update Sender → weights_diff + anomaly_stats + signature_embeddings
- Server Aggregation → global_model_weights
- Zero-Day Detector → cross-agent novelty clusters → zero-day candidates
- Global KB → versioned signatures (Candidate → Verified → Global → Deprecated)
- Broadcast → model + signatures + deprecations
- Local KB Update → merge, validate, rollback if needed

## Novelty & Research Value

- Cross-agent latent-space novelty detection: true zero-day detection by identifying clusters that are simultaneously high-error, low-similarity to existing signatures, and repeatedly occurring across ≥ N agents.
- Agentic alignment: agents autonomously curate local knowledge; federation coordinates discovery and dissemination.
- Privacy: no raw packets or per-flow features leave agents; only model updates and compact latent fingerprints.
- Robust KB lifecycle: versioned signatures with confidence, conflict resolution and rollback reduce false positives and poisoning risk.

## Thresholds and False-positive Avoidance

- Reconstruction error threshold T_recon: set based on training distribution percentiles (e.g., > 95th percentile) per agent.
- Similarity threshold T_sim: cosine similarity below s_min (e.g., < 0.6) vs existing signatures.
- Cross-agent frequency N_min: a candidate must recur across ≥ N agents within window W.
- Confidence scoring combines error magnitude, cluster density, and cross-agent recurrence.

