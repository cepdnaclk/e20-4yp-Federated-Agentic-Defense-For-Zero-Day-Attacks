# Federated Learning Simulation

A Python simulation demonstrating the **Federated Update** mechanism for Zero-Day attack detection using the UNSW_NB15 dataset.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GLOBAL SERVER (Aggregator)                    │
│                      localhost:8000                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Global Knowledge Base                       │    │
│  │  • Attack Signatures    • Mitigation Policies           │    │
│  │  • Zero-Day Registry    • Agent Statistics              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────┬───────────────────────┘
                      │                   │
           HTTP POST  │                   │  HTTP POST
         /upload_update                 /upload_update
                      │                   │
┌─────────────────────┴──┐    ┌──────────┴─────────────────────┐
│       AGENT A           │    │          AGENT B               │
│   (Rows 0-100)          │    │       (Rows 100-200)           │
│  ┌─────────────────┐    │    │    ┌─────────────────┐         │
│  │ Local Knowledge │    │    │    │ Local Knowledge │         │
│  │     Base        │    │    │    │     Base        │         │
│  └─────────────────┘    │    │    └─────────────────┘         │
│           │             │    │            │                    │
│  ┌────────┴────────┐    │    │   ┌────────┴────────┐          │
│  │ Dataset Loader  │    │    │   │ Dataset Loader  │          │
│  │  (UNSW_NB15)    │    │    │   │  (UNSW_NB15)    │          │
│  └─────────────────┘    │    │   └─────────────────┘          │
└─────────────────────────┘    └────────────────────────────────┘
```

## 📁 Project Structure

```
federation_simulation/
├── __init__.py
├── requirements.txt
├── README.md
├── run_simulation.py          # Main simulation orchestrator
├── data/
│   └── UNSW_NB15_training-set.csv  # Place dataset here
├── server/
│   ├── __init__.py
│   └── global_server.py       # FastAPI aggregator server
└── client/
    ├── __init__.py
    ├── dataset_loader.py      # UNSW_NB15 data streaming
    └── agent_node.py          # Local federated agent
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd federation_simulation
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the UNSW_NB15 training dataset and place it in the `data/` directory:

```
federation_simulation/data/UNSW_NB15_training-set.csv
```

You can download it from: [UNSW_NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### 3. Run the Simulation

```bash
python run_simulation.py
```

#### Command Line Options

```bash
# Use threading mode (default, recommended)
python run_simulation.py --mode thread

# Use multiprocessing mode
python run_simulation.py --mode process

# Custom data path
python run_simulation.py --data-path /path/to/data

# Custom server URL
python run_simulation.py --server-url http://localhost:8000
```

## 📡 API Endpoints

### Global Server (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/upload_update` | POST | Receive agent intelligence updates |
| `/get_global_model` | GET | Retrieve aggregated attack signatures |
| `/statistics` | GET | View knowledge base statistics |
| `/reset` | DELETE | Reset the knowledge base |

### Example: Upload Update

```bash
curl -X POST http://localhost:8000/upload_update \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "Agent_A",
    "attack_signature": [0.1, 0.2, 0.3],
    "mitigation_policy": "Block suspicious traffic",
    "is_zero_day": true,
    "attack_category": "Fuzzers"
  }'
```

### Example: Get Global Model

```bash
curl http://localhost:8000/get_global_model
```

## 🔄 Simulation Flow

1. **Server Startup**: Global server initializes on port 8000
2. **Agent Initialization**: Agent_A and Agent_B start with empty local knowledge bases
3. **Data Streaming**: Each agent processes its assigned rows from UNSW_NB15
4. **Zero-Day Detection**: When an agent encounters an unseen attack category:
   - Marks it as Zero-Day
   - Generates mitigation policy
   - Sends update to global server
   - Adds to local knowledge base
5. **Knowledge Sharing**: Global server aggregates all intelligence
6. **Summary**: Final statistics displayed

## 📊 Expected Output

```
🌐 FEDERATED LEARNING SIMULATION - THREADING MODE
=====================================================

📡 Starting Global Server thread...
⏳ Waiting for server to initialize...
🤖 Starting Agent_A...
🤖 Starting Agent_B...

🚨 Agent_A: ZERO-DAY DETECTED!
   Attack Category: Fuzzers
   ✅ Intelligence shared with global server

🔔 Global Server: Received new Zero-Day intel from Agent_A
   Attack Category: Fuzzers

🚨 Agent_B: ZERO-DAY DETECTED!
   Attack Category: Reconnaissance
   ✅ Intelligence shared with global server

...

📊 FINAL SIMULATION SUMMARY
=====================================================
Agent_A:
  • Packets Processed: 100
  • Attacks Detected: 45
  • Zero-Days Discovered: 3
  • Updates Sent: 15

Agent_B:
  • Packets Processed: 100
  • Attacks Detected: 52
  • Zero-Days Discovered: 2
  • Updates Sent: 18

GLOBAL SERVER KNOWLEDGE BASE:
  • Total Updates Received: 33
  • Zero-Days Registered: 5
  • Attack Categories: ['Fuzzers', 'Reconnaissance', 'DoS', 'Generic', 'Exploits']

✅ SIMULATION COMPLETE
```

## 🧪 Running Individual Components

### Start Server Only

```bash
python -m federation_simulation.server.global_server
```

### Test Dataset Loader

```bash
python -m federation_simulation.client.dataset_loader
```

### Run Single Agent

```bash
python -m federation_simulation.client.agent_node
```

## 🔧 Configuration

### Agent Configuration

In `agent_node.py`, you can customize:

- `MITIGATION_POLICIES`: Templates for different attack types
- Connection retry settings in `_wait_for_server()`
- Feature vector hashing in `_generate_signature_hash()`

### Dataset Loader Configuration

In `dataset_loader.py`, modify:

- `FEATURE_COLUMNS`: Which columns to use as attack signature
- Fallback behavior for missing columns

## 📚 Integration with Your System

This simulation is designed to integrate with your existing Autoencoder and RAG components:

1. **Autoencoder Integration**: Replace `features` in `process_packet()` with encoded representations
2. **RAG Integration**: Replace `_generate_mitigation_policy()` with RAG-based policy generation
3. **Real Network Traffic**: Replace dataset loader with actual packet capture

## ⚠️ Troubleshooting

### Server Connection Errors

If agents can't connect to the server:
1. Ensure port 8000 is available
2. Check firewall settings
3. Increase `retry_delay` in agent configuration

### Dataset Not Found

Ensure the CSV file is placed correctly:
```
federation_simulation/data/UNSW_NB15_training-set.csv
```

### Memory Issues

For large datasets, use the generator approach:
```python
for features, label in loader.yield_packet(0, 1000):
    # Process one packet at a time
```

## 📄 License

This project is part of the Federated Learning research for Zero-Day attack detection.
