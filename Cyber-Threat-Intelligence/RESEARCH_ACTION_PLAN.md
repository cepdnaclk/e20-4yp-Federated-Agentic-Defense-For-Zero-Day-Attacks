# RESEARCH ACTION PLAN: Privacy-Preserving Multi-Agent IDS Framework

> **Document Status:** Critical Peer Review & Optimization Roadmap  
> **Evaluation Date:** 2026-03-07  
> **Dataset:** UNSW-NB15 (5,000 test samples, 42 features)  
> **Verdict:** ❌ **NOT PUBLICATION-READY** — Major revisions required

---

## Executive Summary

This framework exhibits **catastrophic failure** in multi-class threat classification and **unacceptably high false positive rates** in anomaly detection. The current metrics would result in immediate desk rejection at any top-tier venue (NDSS, CCS, USENIX Security, IEEE S&P). The RL mitigation agent shows promise but cannot compensate for upstream failures.

| Component | Current | SOTA Baseline | Gap | Severity |
|-----------|---------|---------------|-----|----------|
| Agent 1 AUC-ROC | 0.752 | ≥0.95 | -20.8% | 🔴 Critical |
| Agent 1 FPR | 23.6% | ≤5% | +18.6pp | 🔴 Critical |
| Agent 1 Latency | 2.77ms | ≤1.0ms | +177% | 🟡 High |
| Agent 2 F1-macro | 0.086 | ≥0.85 | -89.9% | 🔴 **FATAL** |
| Agent 2 Attack Recall | 0.6% | ≥80% | -79.4pp | 🔴 **FATAL** |
| Agent 3 Convergence | None | Round 10-15 | ∞ | 🟡 High |
| FL Final Accuracy | 88.6% | ≥92% | -3.4% | 🟢 Medium |

---

## Part I: Harsh Metric Analysis

### 1.1 Agent One (Autoencoder Anomaly Detection) — POOR

**Raw Metrics:**
```
AUC-ROC:     0.7524 (target: ≥0.95)
TPR:         0.608  (missing 39.2% of attacks)
FPR:         0.236  (23.6% false alarms)
Threshold:   1.189
Latency:     2.77ms ± 1.06ms (P99: 7.45ms)
Throughput:  361 samples/sec
```

**Critical Issues:**

1. **False Positive Rate is catastrophically high.** At 23.6%, a network processing 100,000 flows/hour would generate **23,600 false alerts/hour** — rendering the system operationally useless. Security Operation Centers typically tolerate ≤1-2% FPR for first-pass filters.

2. **AUC-ROC of 0.752 indicates severe class overlap.** The reconstruction error distributions show:
   - Benign: μ=1.031, σ=0.231
   - Malicious: μ=1.284, σ=0.293
   
   The means differ by only **0.253** with overlapping standard deviations. The autoencoder has failed to learn a discriminative latent representation. This is characteristic of:
   - Insufficient training epochs
   - Latent dimension too large (model memorizing instead of compressing)
   - Missing bottleneck regularization (β-VAE, sparse coding)

3. **Latency is 2.77× above target.** Real-time IDS requires sub-millisecond inference for 10Gbps+ network links. Current P99 latency of 7.45ms creates unacceptable packet processing delays.

**Verdict:** The autoencoder is functioning as a **near-random classifier** with excessive computational overhead. It would be rejected as a standalone contribution.

---

### 1.2 Agent Two (XGBoost Classification) — CATASTROPHIC FAILURE

**Raw Metrics:**
```
Accuracy:        69.04% (misleading due to class imbalance)
F1-macro:        0.0865 (near-random: 0.10 for 10-class)
F1-micro:        0.6904
Precision-macro: 0.0992
Recall-macro:    0.1009
```

**Per-Class Breakdown:**

| Class | Precision | Recall | F1 | Support | Assessment |
|-------|-----------|--------|-----|---------|------------|
| Normal | 70.3% | **98.5%** | 0.820 | 3,500 | Over-predicted |
| Fuzzers | 0.0% | 0.0% | 0.000 | 191 | ❌ Complete miss |
| Analysis | 0.0% | 0.0% | 0.000 | 171 | ❌ Complete miss |
| Backdoor | 0.0% | 0.0% | 0.000 | 161 | ❌ Complete miss |
| DoS | 0.0% | 0.0% | 0.000 | 155 | ❌ Complete miss |
| Exploits | 0.0% | 0.0% | 0.000 | 157 | ❌ Complete miss |
| Generic | 7.7% | 0.6% | 0.011 | 164 | ❌ Near-miss |
| Reconnaissance | 6.7% | 0.6% | 0.011 | 162 | ❌ Near-miss |
| Shellcode | 8.3% | 0.6% | 0.011 | 164 | ❌ Near-miss |
| Worms | 6.3% | 0.6% | 0.010 | 175 | ❌ Near-miss |

**Diagnosis: Complete Class Collapse**

The XGBoost classifier has collapsed to predicting "Normal" for **98.5% of samples**, regardless of true class. The confusion matrix confirms this:
- Row 0 (Normal true): 3,448 correct, 52 misclassified
- Rows 1-9 (Attack true): **1,460 samples classified as Normal out of 1,500 attacks**

This is **not a model** — it is a constant predictor. Root causes:

1. **Severe class imbalance not addressed.** The 70:30 Normal:Attack ratio with 10 classes means minority classes have ~1.5-3.5% representation each. Without SMOTE, class weights, or focal loss, the model converges to majority-class prediction.

2. **Training data likely corrupted or insufficient.** With only 500 training samples for DP evaluation (visible in logs), the model cannot learn attack signatures.

3. **Hyperparameters suboptimal for imbalanced data:**
   - `n_estimators=50` is too low (need 200-500)
   - `max_depth=6` may be insufficient for complex attack patterns
   - `scale_pos_weight` not configured
   - No `sample_weight` based on inverse class frequency

**Verdict:** This component alone **invalidates the entire framework**. No reviewer would accept a "threat classifier" with 0% recall on 6/9 attack categories.

---

### 1.3 Differential Privacy Impact Analysis — SUSPICIOUS

**Raw Metrics:**
```
σ=0.0: 69.92% accuracy (baseline)
σ=0.1: 69.92% (Δ=0.00%)
σ=0.5: 69.96% (Δ=+0.04%)
σ=1.0: 69.90% (Δ=-0.02%)
σ=2.0: 69.92% (Δ=0.00%)
σ=5.0: 69.94% (Δ=+0.02%)
```

**Critical Analysis:**

This data is **physically implausible**. Adding Gaussian noise with σ=5.0 to training gradients should degrade accuracy by 10-30% according to established DP-SGD literature (Abadi et al., 2016). The near-zero degradation indicates:

1. **DP is not being applied correctly.** The noise is likely added to features rather than gradients, which does not provide meaningful privacy guarantees.

2. **The model is already at random-chance performance.** A model predicting the majority class cannot degrade further — it has hit the floor.

3. **Evaluation methodology flaw.** If the same test set is used after each noisy training run without proper holdout separation, results will be biased.

**Privacy Budget Concern:** No ε (epsilon) value is reported. For (ε, δ)-DP to be meaningful:
- ε ≤ 1.0 for strong privacy
- ε ≤ 10.0 for moderate privacy
- ε > 10.0 offers negligible privacy guarantees

Without explicit ε calculation via moments accountant or Rényi DP, **no privacy claims can be made**.

**Verdict:** The "DP impact" results are not credible and would be challenged during peer review.

---

### 1.4 Agent Three (RL Mitigation Policy) — ACCEPTABLE WITH CAVEATS

**Raw Metrics:**
```
Mean Reward:       0.552
Convergence Round: None (not detected)
```

**Policy Matrix (action frequency %):**

| Severity | Do Nothing | Alert Admin | Block IP | Isolate Subnet |
|----------|------------|-------------|----------|----------------|
| Low | **71.1%** | 20.5% | 6.8% | 1.5% |
| Medium | 28.3% | **42.6%** | 22.9% | 6.2% |
| High | 11.5% | 17.3% | **52.3%** | 18.9% |
| Critical | 2.5% | 10.2% | 24.2% | **63.1%** |

**Positive Findings:**

1. **Policy gradient is correct.** The agent learned that:
   - Low severity → passive response (Do Nothing)
   - Critical severity → aggressive response (Isolate Subnet)
   
   This is the expected cost-sensitive behavior.

2. **No policy collapse.** Unlike Agent 2, the RL agent distributes actions across the action space appropriately.

3. **Mean reward is positive.** Indicates the policy is better than random.

**Concerns:**

1. **Convergence not achieved.** The FL accuracy oscillates between 85-91% without stabilizing, suggesting:
   - Learning rate too high
   - Insufficient exploration (entropy coefficient too low)
   - Reward signal too noisy

2. **Evaluation is simulated, not real.** The policy matrix was generated without a trained PPO model (`RL policy not loaded, using simulated policy`). Real policy behavior may differ.

3. **"Do Nothing" at Critical severity (2.5%)** is a security concern. Even 2.5% passive responses to critical threats could be exploited.

**Verdict:** Promising component, but needs re-evaluation with the actual trained policy.

---

### 1.5 Federated Learning Convergence — UNSTABLE

**Raw Metrics:**
```
Initial Accuracy:  68.5% (Round 1)
Peak Accuracy:     90.8% (Round 17)
Final Accuracy:    88.6% (Round 20)
Variance (R10-20): ±3.2%
Convergence:       Not detected
```

**Convergence Curve Analysis:**

The accuracy exhibits high-frequency oscillation in later rounds:
- Round 11: 88.4% → Round 12: 88.2% → Round 13: 85.3% → Round 14: **90.8%**

This "sawtooth" pattern indicates:
1. **Client data heterogeneity (non-IID distribution).** Different clients have different attack distributions, causing the global model to oscillate.
2. **Learning rate not decayed.** Constant learning rate prevents fine-tuning convergence.
3. **Aggregation strategy suboptimal.** FedAvg struggles with heterogeneous data; FedProx or SCAFFOLD may be needed.

**Verdict:** FL is functional but not optimized. Would require additional experiments to claim "convergent federated learning."

---

### 1.6 Semantic Bridge (Agentic RAG) — NOT EVALUATED

The evaluation did not measure:
- LLM inference latency for Llama 3
- Vector database retrieval accuracy (FAISS hit rate)
- Context window utilization
- End-to-end RAG pipeline throughput

**Verdict:** Cannot assess. Must be added to evaluation suite.

---

## Part II: Critical Bottlenecks (Publication Blockers)

### 🚨 Blocker #1: Agent 2 XGBoost Classification Collapse

**Impact:** Entire framework is invalidated. A "threat intelligence system" that cannot identify threats is useless.

**Evidence:**
- F1-macro = 0.086 (random = 0.10)
- 0% recall on 6/9 attack categories
- 98.5% of predictions are "Normal"

**Required Fix:** Complete retraining with class imbalance handling.

---

### 🚨 Blocker #2: Agent 1 False Positive Rate

**Impact:** Operational infeasibility. 23.6% FPR makes the system unusable in production.

**Evidence:**
- FPR = 0.236 (target ≤0.05)
- Reconstruction error distributions overlap significantly
- AUC = 0.752 (target ≥0.95)

**Required Fix:** Architectural changes to autoencoder (β-VAE, contrastive learning, or replace with isolation forest).

---

### 🚨 Blocker #3: Missing Privacy Budget (ε) Calculation

**Impact:** No valid privacy claims can be made without formal DP accounting.

**Evidence:**
- Only σ values reported, no ε
- Accuracy invariant to σ (suspicious)
- No moments accountant or Rényi DP composition

**Required Fix:** Implement proper DP-SGD with privacy budget tracking.

---

## Part III: Hyperparameter Tuning Roadmap

### 3.1 Autoencoder (Agent 1)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `latent_dim` | 8 | **4** | Force tighter bottleneck for better compression |
| `hidden_dims` | [32, 16] | **[64, 32, 16]** | Deeper encoder for complex patterns |
| `dropout_rate` | 0.2 | **0.3** | Stronger regularization |
| `learning_rate` | 0.001 | **0.0005** | Slower convergence, better minima |
| `epochs` | Unknown | **200+** | Ensure full convergence |
| `batch_size` | 64 | **128** | Smoother gradient estimates |
| **NEW: β (VAE)** | N/A | **2.0-4.0** | KL divergence weighting for disentanglement |

**Threshold Selection:**
- Current: Youden's J (maximizes TPR - FPR)
- Recommended: **Fix FPR at 5%, solve for threshold** — operational constraint should drive threshold, not statistical optimum.

### 3.2 XGBoost Classifier (Agent 2)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `n_estimators` | 50 | **300** | More trees for complex boundaries |
| `max_depth` | 6 | **10** | Deeper trees for multi-class |
| `learning_rate` | 0.1 | **0.05** | Slower learning, more trees |
| `min_child_weight` | 3 | **1** | Allow smaller leaves for minority classes |
| `scale_pos_weight` | None | **Computed per-class** | Handle imbalance |
| `subsample` | 0.8 | **0.7** | More regularization |
| `colsample_bytree` | 0.8 | **0.6** | Feature bagging |
| **NEW: `sample_weight`** | None | **Inverse class frequency** | Critical for imbalance |

**Class Weight Calculation:**
```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
```

**Alternative: SMOTE oversampling:**
```python
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

### 3.3 Differential Privacy (FL/DP)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `noise_multiplier (σ)` | 0.1-5.0 | **0.5-1.0** | Balance utility/privacy |
| `max_grad_norm (C)` | 1.0 | **0.5** | Tighter clipping for stability |
| `delta (δ)` | Not set | **1e-5** | Standard for |S|≈10^5 |
| `epochs` | Unknown | **Track via accountant** | Stop when ε exceeds budget |
| **NEW: Target ε** | N/A | **ε ≤ 8.0** | Reasonable utility-privacy trade-off |

**Recommended Implementation:**
Replace feature-level noise with proper DP-SGD using `opacus` or `tensorflow-privacy`:
```python
from opacus import PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=0.8,
    max_grad_norm=0.5,
)
```

### 3.4 PPO (Agent 3)

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `learning_rate` | 3e-4 | **1e-4** | Slower, more stable |
| `n_steps` | 2048 | **4096** | More samples per update |
| `batch_size` | 64 | **256** | Larger batches reduce variance |
| `n_epochs` | 10 | **20** | More passes over buffer |
| `gamma` | 0.99 | **0.95** | Shorter horizon for IDS |
| `gae_lambda` | 0.95 | **0.9** | Less variance in advantage |
| `ent_coef` | 0.01 | **0.05** | More exploration initially |
| `clip_range` | 0.2 | **0.1** | Tighter clipping for stability |

**Reward Shaping (Current likely sub-optimal):**
```python
# Proposed reward structure
REWARD_MATRIX = {
    # (severity, action): reward
    ('Low', 'Do Nothing'): +1.0,
    ('Low', 'Isolate Subnet'): -2.0,  # Over-reaction penalty
    ('Critical', 'Do Nothing'): -5.0,  # Security failure
    ('Critical', 'Isolate Subnet'): +2.0,
    # ... intermediate values for other combinations
}
```

---

## Part IV: Architectural Tweaks

### 4.1 Agent 1 → Agent 2 Data Handoff

**Current Issue:** Agent 2 receives raw features, ignoring Agent 1's anomaly score.

**Proposed Change:**
```python
# In pipeline handoff
agent_two_input = np.concatenate([
    original_features,           # (42,)
    [agent_one_reconstruction_error],  # (1,) - anomaly score
    [agent_one_is_anomaly],      # (1,) - binary flag
], axis=-1)  # Final: (44,)
```

**Rationale:** Agent 2 can use Agent 1's confidence as an additional feature, improving attack detection for samples near the decision boundary.

### 4.2 Replace Autoencoder with Variational Autoencoder (VAE)

**Rationale:** VAE with β>1 (β-VAE) enforces a more structured latent space, improving separation between normal and anomalous distributions.

**File:** `agents/models/autoencoder.py`

**Change:** Add KL divergence loss term:
```python
def loss_function(self, x, x_recon, mu, logvar, beta=2.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

### 4.3 Add Focal Loss for XGBoost Alternative

If XGBoost continues to struggle, replace with a focal loss neural classifier:

**Focal Loss Formula:**
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $\alpha_t$ = class weight
- $\gamma$ = focusing parameter (2.0 recommended)
- $p_t$ = predicted probability for true class

### 4.4 Latency Optimization

**Current Bottleneck:** PyTorch autoencoder inference on CPU.

**Fixes:**
1. **Enable TorchScript JIT compilation:**
   ```python
   model = torch.jit.script(autoencoder)
   ```

2. **Batch inference:** Process flows in batches of 32-64 instead of single samples.

3. **ONNX export for production:**
   ```python
   torch.onnx.export(model, dummy_input, "autoencoder.onnx")
   # Use ONNX Runtime for inference
   ```

4. **Quantization (INT8):**
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

---

## Part V: Next Experiment Command

**Priority:** Fix Agent 2 classification collapse (Blocker #1).

After adding SMOTE and class weights to the training script:

```bash
# Step 1: Install imbalanced-learn if not present
pip install imbalanced-learn

# Step 2: Run targeted retraining with class balancing
cd Cyber-Threat-Intelligence

python -c "
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from agents.models.xgboost_classifier import ThreatClassifier
from data_pipeline.data_loader import DataLoader
import joblib

# Load data
loader = DataLoader()
X, y = loader.load_unswnb15()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE
print(f'Before SMOTE: {np.bincount(y_train)}')
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f'After SMOTE: {np.bincount(y_resampled)}')

# Train with tuned hyperparameters
clf = ThreatClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.7,
    colsample_bytree=0.6,
    zero_day_threshold=0.4,
)
clf.fit(X_resampled.astype(np.float32), y_resampled)

# Evaluate
from sklearn.metrics import classification_report
y_pred = [clf.predict(X_test[i:i+1].astype(np.float32)).category_id for i in range(len(X_test))]
print(classification_report(y_test, y_pred, digits=4))

# Save
clf.save('models/agent_two_balanced')
print('Saved to models/agent_two_balanced')
"
```

**Expected Outcome:**
- F1-macro should increase from 0.086 to ≥0.60
- Per-class recall should be ≥50% for all attack categories
- Overall accuracy may decrease slightly (from 69% to 65-70%) — this is acceptable as it reflects better minority class handling

---

## Part VI: Publication Readiness Checklist

| Requirement | Status | Blocking? |
|-------------|--------|-----------|
| Agent 1 AUC ≥ 0.95 | ❌ 0.752 | Yes |
| Agent 1 FPR ≤ 5% | ❌ 23.6% | Yes |
| Agent 1 Latency ≤ 1ms | ❌ 2.77ms | No (can explain) |
| Agent 2 F1-macro ≥ 0.85 | ❌ 0.086 | **YES (FATAL)** |
| Agent 2 Per-class recall ≥ 50% | ❌ 0-0.6% | **YES (FATAL)** |
| DP ε budget reported | ❌ Missing | Yes |
| DP utility-privacy curve valid | ❌ Suspicious | Yes |
| Agent 3 policy converged | ❌ Not detected | No (can show learning) |
| FL convergence demonstrated | ⚠️ Oscillating | No (within variance) |
| RAG latency benchmarked | ❌ Missing | Yes |
| End-to-end throughput | ⚠️ 361/sec | No (acceptable for research) |

**Current Score:** 1/11 requirements met  
**Target for Submission:** 9/11 minimum

---

## Appendix: Reviewer Anticipated Questions

1. **"Why does your classifier have 0% recall on 6 attack categories?"**
   - Unacceptable answer: "Class imbalance"
   - Required answer: "We addressed this via SMOTE/focal loss, achieving X% recall"

2. **"What is your privacy budget (ε)?"**
   - Required answer: "ε = X with δ = 1e-5 over Y training epochs"

3. **"How does your system compare to DeepLog / DAGMM / USAD?"**
   - Must include quantitative comparison table

4. **"What is the latency breakdown per agent?"**
   - Must provide: Agent 1 (X ms) + Agent 2 (Y ms) + Agent 3 (Z ms) = Total

5. **"How do you handle concept drift in production?"**
   - Must address: Online learning / periodic retraining strategy

---

*Document generated by automated peer review analysis. All metrics sourced from `results/evaluation_metrics.json`.*
