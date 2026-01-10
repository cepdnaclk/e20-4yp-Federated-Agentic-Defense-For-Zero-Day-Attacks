# Network Packet Streamer

A python tool for streaming UNSW-NB15 dataset features to a detection API. This project uses `uv` for dependency management and is fully Dockerized.

##  Quick Start (Local)

1. **Install uv**:
   ```bash
   uv sync
   ```


2. **Setup Environment**: 
    
    Modify a config.env file in the root of pkt-streamer directory.
    ```bash
    CSV_PATH=./data/UNSW_NB15/UNSW-NB15_1.csv
    FEATURES_METADATA=./data/UNSW_NB15/dataset_features.json
    API_URL=http://localhost:5000/detect
    ```

3. **Run**:
    ```bash
    uv run main.py
    ```

## Docker Deployment

1. **Build the Image**

    ```bash
    docker build -t pkt-streamer .
    ```

2. **Using the Automation Script (Recommended)**

    Because Windows absolute paths are long, use a run.cmd file to launch the container.

    - important: update /data path in ```run.cmd```.

    then, 

    ```cmd
    run
    ```

