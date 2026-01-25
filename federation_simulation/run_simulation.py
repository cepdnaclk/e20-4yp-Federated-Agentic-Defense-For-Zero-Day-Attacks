"""
Federated Learning Simulation Runner
Orchestrates the global server and multiple local agents for Zero-Day detection simulation.

Usage:
    python run_simulation.py
    
    Or with custom data path:
    python run_simulation.py --data-path /path/to/data
"""

import threading
import multiprocessing
import time
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def start_server_process(host: str = "0.0.0.0", port: int = 8000):
    """Start the global server in a separate process"""
    from federation_simulation.server.global_server import start_server
    start_server(host=host, port=port)


def start_server_thread(host: str = "0.0.0.0", port: int = 8000):
    """Start the global server in a background thread"""
    import uvicorn
    from federation_simulation.server.global_server import app
    
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    
    # Run server in a thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def run_agent_worker(
    agent_id: str,
    start_idx: int,
    end_idx: int,
    server_url: str,
    data_path: str,
    result_queue: multiprocessing.Queue,
    use_rag: bool = False,
    rag_provider: str = "mock"
):
    """
    Worker function for running an agent in a separate process.
    """
    from federation_simulation.client.agent_node import LocalAgent
    
    agent = LocalAgent(
        agent_id=agent_id,
        server_url=server_url,
        data_path=data_path,
        use_rag=use_rag,
        rag_provider=rag_provider
    )
    
    stats = agent.run_simulation(start_idx=start_idx, end_idx=end_idx, delay=0.01)
    result_queue.put((agent_id, stats))


def run_simulation_multiprocess(data_path: str, server_url: str = "http://localhost:8000"):
    """
    Run the federated learning simulation using multiprocessing.
    
    - Starts global server in a separate process
    - Runs Agent_A and Agent_B in parallel processes
    - Agent_A processes rows 0-100
    - Agent_B processes rows 100-200
    """
    print("\n" + "="*70)
    print("🌐 FEDERATED LEARNING SIMULATION - MULTIPROCESSING MODE")
    print("="*70 + "\n")
    
    # Create result queue for collecting agent statistics
    result_queue = multiprocessing.Queue()
    
    # Start the global server process
    print("📡 Starting Global Server process...")
    server_process = multiprocessing.Process(
        target=start_server_process,
        args=("0.0.0.0", 8000),
        daemon=True
    )
    server_process.start()
    
    # Wait for server to initialize
    print("⏳ Waiting for server to initialize...")
    time.sleep(3)
    
    # Define agent configurations - attacks start around row 243
    agents_config = [
        {"agent_id": "Agent_A", "start_idx": 200, "end_idx": 400},
        {"agent_id": "Agent_B", "start_idx": 400, "end_idx": 600},
    ]
    
    # Start agent processes
    agent_processes = []
    
    for config in agents_config:
        print(f"🤖 Starting {config['agent_id']}...")
        p = multiprocessing.Process(
            target=run_agent_worker,
            args=(
                config["agent_id"],
                config["start_idx"],
                config["end_idx"],
                server_url,
                data_path,
                result_queue
            )
        )
        agent_processes.append(p)
        p.start()
        time.sleep(0.5)  # Stagger starts slightly
    
    # Wait for all agents to complete
    print("\n⏳ Waiting for agents to complete their simulations...\n")
    
    for p in agent_processes:
        p.join(timeout=120)  # 2 minute timeout per agent
    
    # Collect results
    results = {}
    while not result_queue.empty():
        agent_id, stats = result_queue.get()
        results[agent_id] = stats
    
    # Print final summary
    print_final_summary(results, server_url)
    
    # Cleanup
    server_process.terminate()
    server_process.join(timeout=5)
    
    return results


def run_simulation_threaded(
    data_path: str,
    server_url: str = "http://localhost:8000",
    use_rag: bool = False,
    rag_provider: str = "mock"
):
    """
    Run the federated learning simulation using threading.
    
    This is an alternative to multiprocessing that works better on some systems.
    
    Args:
        data_path: Path to the dataset directory
        server_url: URL of the global server
        use_rag: If True, use RAG/LLM for policy generation
        rag_provider: LLM provider ("mock", "openai", "anthropic", "ollama")
    """
    rag_status = f" + RAG ({rag_provider})" if use_rag else ""
    print("\n" + "="*70)
    print(f"🌐 FEDERATED LEARNING SIMULATION - THREADING MODE{rag_status}")
    print("="*70 + "\n")
    
    from federation_simulation.client.agent_node import LocalAgent
    
    results = {}
    results_lock = threading.Lock()
    
    def agent_thread_worker(agent_id: str, start_idx: int, end_idx: int):
        """Thread worker for running an agent"""
        agent = LocalAgent(
            agent_id=agent_id,
            server_url=server_url,
            data_path=data_path,
            use_rag=use_rag,
            rag_provider=rag_provider
        )
        stats = agent.run_simulation(start_idx=start_idx, end_idx=end_idx, delay=0.01)
        
        with results_lock:
            results[agent_id] = stats
    
    # Start the server in a background thread
    print("📡 Starting Global Server thread...")
    server_thread = start_server_thread("0.0.0.0", 8000)
    
    # Wait for server to initialize
    print("⏳ Waiting for server to initialize...")
    time.sleep(2)
    
    # Define agent configurations - attacks start around row 243
    agents_config = [
        {"agent_id": "Agent_A", "start_idx": 200, "end_idx": 400},
        {"agent_id": "Agent_B", "start_idx": 400, "end_idx": 600},
    ]
    
    # Start agent threads
    agent_threads = []
    
    for config in agents_config:
        print(f"🤖 Starting {config['agent_id']}...")
        t = threading.Thread(
            target=agent_thread_worker,
            args=(config["agent_id"], config["start_idx"], config["end_idx"])
        )
        agent_threads.append(t)
        t.start()
        time.sleep(0.5)  # Stagger starts slightly
    
    # Wait for all agents to complete
    print("\n⏳ Waiting for agents to complete their simulations...\n")
    
    for t in agent_threads:
        t.join(timeout=120)
    
    # Print final summary
    print_final_summary(results, server_url)
    
    return results


def print_final_summary(results: dict, server_url: str):
    """Print the final simulation summary"""
    import requests
    
    print("\n" + "="*70)
    print("FINAL SIMULATION SUMMARY")
    print("="*70)
    
    total_packets = 0
    total_zero_days = 0
    total_attacks = 0
    all_known_attacks = set()
    
    for agent_id, stats in results.items():
        print(f"\n{agent_id}:")
        print(f"  • Packets Processed: {stats.get('packets_processed', 0)}")
        print(f"  • Attacks Detected: {stats.get('attacks_detected', 0)}")
        print(f"  • Zero-Days Discovered: {stats.get('zero_days_discovered', 0)}")
        print(f"  • Updates Sent: {stats.get('updates_sent', 0)}")
        
        total_packets += stats.get('packets_processed', 0)
        total_zero_days += stats.get('zero_days_discovered', 0)
        total_attacks += stats.get('attacks_detected', 0)
    
    print(f"\n{'─'*40}")
    print(f"TOTALS:")
    print(f"  • Total Packets Processed: {total_packets}")
    print(f"  • Total Attacks Detected: {total_attacks}")
    print(f"  • Total Zero-Days Discovered: {total_zero_days}")
    
    # Try to get global server statistics
    try:
        response = requests.get(f"{server_url}/statistics", timeout=5)
        if response.status_code == 200:
            server_stats = response.json().get("statistics", {})
            print(f"\n{'─'*40}")
            print(f"GLOBAL SERVER KNOWLEDGE BASE:")
            print(f"  • Total Updates Received: {server_stats.get('total_updates', 0)}")
            print(f"  • Zero-Days Registered: {server_stats.get('zero_day_count', 0)}")
            print(f"  • Unique Agents: {server_stats.get('unique_agents', 0)}")
            print(f"  • Attack Categories: {server_stats.get('attack_categories', [])}")
    except Exception as e:
        print(f"\nCould not fetch server statistics: {e}")
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70 + "\n")


def main():
    """Main entry point for the simulation"""
    parser = argparse.ArgumentParser(
        description="Federated Learning Simulation for Zero-Day Attack Detection"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to directory containing UNSW_NB15_training-set.csv"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["thread", "process"],
        default="thread",
        help="Execution mode: 'thread' (default) or 'process'"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the global server"
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable RAG/LLM-powered policy generation"
    )
    parser.add_argument(
        "--rag-provider",
        type=str,
        choices=["mock", "openai", "anthropic", "ollama"],
        default="mock",
        help="LLM provider for RAG: 'mock' (default), 'openai', 'anthropic', 'ollama'"
    )
    
    args = parser.parse_args()
    
    # Resolve data path
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        # Make relative to this script's directory
        data_path = Path(__file__).parent / data_path
    
    data_path = str(data_path.resolve())
    
    print(f"\n📁 Data path: {data_path}")
    
    # Check if dataset exists
    csv_path = Path(data_path) / "UNSW_NB15_training-set.csv"
    if not csv_path.exists():
        print(f"\nWARNING: Dataset not found at {csv_path}")
        print("   Please ensure UNSW_NB15_training-set.csv is in the data directory.")
        print("   The simulation will attempt to proceed anyway.\n")
    
    # Run simulation
    if args.mode == "process":
        results = run_simulation_multiprocess(data_path, args.server_url)
    else:
        results = run_simulation_threaded(
            data_path,
            args.server_url,
            use_rag=args.use_rag,
            rag_provider=args.rag_provider
        )
    
    return results


if __name__ == "__main__":
    # Handle Windows multiprocessing
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    
    main()
