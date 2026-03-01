# Agent One: Autoencoder-based Anomaly Detection

This agent uses an autoencoder neural network to detect anomalies in network traffic.

## How It Works

### The Autoencoder Architecture

```
Input (40 features) → Encoder → Latent Space (8 dims) → Decoder → Reconstruction (40 features)
```

**Architecture Details:**
```
Encoder:
  Linear(40 → 32) → BatchNorm → LeakyReLU → Dropout
  Linear(32 → 16) → BatchNorm → LeakyReLU → Dropout
  Linear(16 → 8)   [Bottleneck]

Decoder (Mirror of Encoder):
  Linear(8 → 16)  → BatchNorm → LeakyReLU → Dropout
  Linear(16 → 32) → BatchNorm → LeakyReLU → Dropout
  Linear(32 → 40)  [Output]
```

### Why Autoencoders for Anomaly Detection?

1. **Unsupervised Learning**: The model learns patterns from normal traffic without needing labeled attack data.

2. **Reconstruction Error as Anomaly Score**: 
   - Normal traffic → Low reconstruction error (model has seen similar patterns)
   - Anomalous traffic → High reconstruction error (unfamiliar patterns)

3. **Bottleneck Forces Compression**: The latent space (8 dimensions) forces the model to learn essential features of normal traffic, filtering out noise.

## Training Instructions

### Quick Start

```bash
# From project root directory
python -m agents.train_autoencoder --data_path data/UNSW_NB15_training-set.csv
```

### Training Options

```bash
python -m agents.train_autoencoder \
    --data_path data/UNSW_NB15_training-set.csv \
    --output_dir models/agent_one \
    --latent_dim 8 \
    --hidden_dims 32 16 \
    --dropout 0.2 \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.001 \
    --patience 10 \
    --threshold_percentile 95
```

### Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | - | Path to UNSW-NB15 CSV file |
| `--output_dir` | `models/agent_one` | Where to save model and preprocessor |
| `--latent_dim` | 8 | Bottleneck dimension (smaller = more compression) |
| `--hidden_dims` | [32, 16] | Encoder hidden layer sizes |
| `--dropout` | 0.2 | Dropout probability for regularization |
| `--batch_size` | 64 | Training batch size |
| `--epochs` | 100 | Maximum training epochs |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 10 | Early stopping patience |
| `--threshold_percentile` | 95 | Percentile for threshold calibration |
| `--all_data` | False | If set, train on all data (not just normal) |

### Training Process

1. **Data Loading**: Loads UNSW-NB15 CSV and preprocesses features
2. **Normal-Only Training**: By default, trains only on normal traffic (label=0)
3. **Early Stopping**: Monitors validation loss, stops if no improvement
4. **Threshold Calibration**: Sets threshold so 95% of normal traffic is correctly classified
5. **Evaluation**: Tests on held-out data and reports accuracy, precision, recall, F1

## Usage After Training

### Python API

```python
from agents import AgentOne, AnomalyAutoencoder
from data_pipeline import Preprocessor
import numpy as np

# Load trained agent
agent = AgentOne.from_checkpoint(
    model_path="models/agent_one/best_model.pth",
    threshold=0.05,  # Adjust based on calibration
)

# Load preprocessor for raw data
preprocessor = Preprocessor.load("models/agent_one/preprocessor.pkl")

# Detect anomaly in single flow
network_flow = np.array([...])  # Raw features
processed_flow = preprocessor.transform(network_flow.reshape(1, -1))
result = agent.detect_anomaly(processed_flow[0])

if result.is_anomaly:
    print(f"ALERT: Anomaly detected!")
    print(f"  Reconstruction error: {result.reconstruction_error:.4f}")
    print(f"  Confidence: {result.confidence:.2%}")
else:
    print(f"Normal traffic (error: {result.reconstruction_error:.4f})")

# Batch detection
results = agent.detect_anomalies(batch_of_flows)
anomaly_count = sum(1 for r in results if r.is_anomaly)
print(f"Detected {anomaly_count} anomalies in {len(results)} flows")
```

### Detection Result Fields

```python
@dataclass
class DetectionResult:
    is_anomaly: bool        # True if error > threshold
    reconstruction_error: float  # Raw MSE value
    threshold: float        # Threshold used for decision
    confidence: float       # 0-1, how likely it's an anomaly
    raw_input: np.ndarray   # Original input (if requested)
```

## Threshold Tuning

The threshold controls the sensitivity of detection:

- **Lower threshold** → More sensitive, catches more attacks, but more false positives
- **Higher threshold** → Less sensitive, fewer false positives, but may miss attacks

### Calibrating on Your Data

```python
# Load agent
agent = AgentOne.from_checkpoint("models/agent_one/best_model.pth")

# Calibrate using known normal traffic
# percentile=95 means 5% of normal traffic triggers false alarms
new_threshold = agent.calibrate_threshold(
    normal_validation_data,
    percentile=95.0  # Adjust: 99=very few false positives, 90=more sensitive
)

print(f"Recommended threshold: {new_threshold:.6f}")
```

## Architecture Choices Explained

### Why These Hyperparameters?

| Choice | Reasoning |
|--------|-----------|
| **Latent dim = 8** | Small enough to force compression, large enough to capture patterns |
| **Hidden [32, 16]** | Gradual compression prevents information loss |
| **LeakyReLU** | Prevents "dead neurons" that can occur with standard ReLU |
| **BatchNorm** | Stabilizes training, allows higher learning rates |
| **Dropout = 0.2** | Regularization prevents overfitting to training data |
| **MSE Loss** | Natural choice for reconstruction; penalizes large errors |
| **Adam optimizer** | Adaptive learning rates, works well for most deep learning |

### Why Train on Normal Only?

Training on only normal traffic has key advantages:

1. **No Label Leakage**: Model doesn't learn attack signatures, only normal patterns
2. **Generalization**: Can detect novel attacks it hasn't seen during training
3. **Simplicity**: No class imbalance issues

## Performance Expectations

On UNSW-NB15 dataset, typical results:

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 70-85% |
| Precision | 65-80% |
| Recall | 60-85% |
| F1 Score | 65-80% |

**Note**: Reconstruction-based detection has natural limitations. It works best for:
- Detecting traffic that deviates significantly from normal patterns
- Catching novel/unknown attacks

It may struggle with:
- Attacks that mimic normal traffic closely
- Very subtle anomalies

## Files After Training

```
models/agent_one/
├── best_model.pth      # Best model weights (by validation loss)
├── final_model.pth     # Final epoch model weights
└── preprocessor.pkl    # Fitted preprocessor (normalization, encoding)
```

## Troubleshooting

### High False Positive Rate
- Increase `threshold_percentile` (e.g., 99 instead of 95)
- Train longer (more epochs)
- Increase latent dimension

### Missing Attacks (Low Recall)
- Decrease `threshold_percentile` (e.g., 90 instead of 95)
- Check if attacks are similar to normal traffic in your data
- Consider combining with other detection methods

### Training Unstable / NaN Loss
- Reduce learning rate
- Check for missing values in data
- Ensure data is properly normalized
