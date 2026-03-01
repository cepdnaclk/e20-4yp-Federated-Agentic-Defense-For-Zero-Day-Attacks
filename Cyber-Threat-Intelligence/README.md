# Cyber-Threat-Intelligence Data Pipeline

A modular, object-oriented data pipeline for the UNSW-NB15 network intrusion detection dataset.

## Project Structure

```
Cyber-Threat-Intelligence/
├── data_pipeline/
│   ├── __init__.py           # Module exports
│   ├── config.py             # Dataset configuration and schema
│   ├── data_loader.py        # Data loading and cleaning
│   ├── preprocessor.py       # Feature normalization and encoding
│   └── batch_generator.py    # Batch iteration and data splitting
├── examples/
│   └── pipeline_demo.py      # Complete usage example
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from data_pipeline import DataLoader, Preprocessor, DatasetConfig
from data_pipeline.batch_generator import create_train_test_generators

# 1. Configure
config = DatasetConfig(
    normalization_method="minmax",
    default_batch_size=64,
)

# 2. Load and clean data
loader = DataLoader(config)
loader.load("path/to/UNSW_NB15.csv").clean()

# 3. Get features and labels
X, y = loader.get_features_and_labels(label_type="binary")

# 4. Preprocess
preprocessor = Preprocessor(config)
X_processed, y_encoded = preprocessor.fit_transform(X, y)

# 5. Create batch generators
train_gen, test_gen, val_gen = create_train_test_generators(
    X_processed, y_encoded, config=config
)

# 6. Train your model
for epoch in range(10):
    for X_batch, y_batch in train_gen:
        # model.train_on_batch(X_batch, y_batch)
        pass
    train_gen.on_epoch_end()
```

## Components

### DatasetConfig

Configuration class for dataset parameters:

- **numerical_features**: List of numerical column names
- **categorical_features**: List of categorical column names  
- **normalization_method**: `'minmax'`, `'standard'`, or `'robust'`
- **numerical_fill_strategy**: `'mean'`, `'median'`, or `'zero'`
- **categorical_fill_strategy**: `'mode'` or `'unknown'`
- **default_batch_size**: Batch size for training
- **test_split_ratio**: Fraction for test set (default: 0.2)
- **validation_split_ratio**: Fraction for validation set (default: 0.1)

### DataLoader

Handles loading and initial data cleaning:

```python
loader = DataLoader(config)

# Load single file
loader.load("data.csv")

# Load multiple files
loader.load_multiple(["data1.csv", "data2.csv"])

# Clean data
loader.clean(drop_duplicates=True, drop_id_columns=True)

# Validate schema
is_valid, issues = loader.validate_schema()

# Get statistics
stats = loader.get_data_statistics()

# Detect quality issues
issues = loader.detect_data_quality_issues()

# Extract features and labels
X, y = loader.get_features_and_labels(label_type="binary")
```

### Preprocessor

Handles feature transformation:

```python
preprocessor = Preprocessor(config)

# Fit and transform training data
X_train_processed, y_train = preprocessor.fit_transform(
    X_train, y_train,
    categorical_encoding="label"  # or "onehot"
)

# Transform test data (using fitted parameters)
X_test_processed = preprocessor.transform(X_test)

# Save/load preprocessor
preprocessor.save("preprocessor.pkl")
preprocessor = Preprocessor.load("preprocessor.pkl")

# Get feature names after transformation
feature_names = preprocessor.get_feature_names()
```

### BatchGenerator

Efficient batch iteration:

```python
from data_pipeline.batch_generator import (
    BatchGenerator, 
    DataSplitter,
    create_train_test_generators
)

# Create generator
generator = BatchGenerator(X, y, batch_size=64, shuffle=True)

# Iterate
for X_batch, y_batch in generator:
    pass

# Epoch handling
generator.on_epoch_end()

# Random sampling
X_sample, y_sample = generator.sample_batch()

# Data splitting
splitter = DataSplitter(test_ratio=0.2, val_ratio=0.1)
splits = splitter.split(X, y, stratify=True)
```

## UNSW-NB15 Dataset

The pipeline is designed for the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), which contains:

- **49 features** including flow-based, basic, content, and time features
- **10 attack categories**: Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms
- **Binary label**: 0 (normal) or 1 (attack)

### Features Handled

| Type | Examples |
|------|----------|
| Numerical | `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, etc. |
| Categorical | `proto`, `service`, `state` |
| Binary | `is_ftp_login`, `is_sm_ips_ports` |

## Running the Demo

```bash
python examples/pipeline_demo.py
```

## License

MIT License
