"""
Integration Test: Full Multi-Agent IDS Pipeline.

This script demonstrates the complete intrusion detection pipeline:
    1. Load and preprocess data
    2. Agent One: Anomaly detection with Autoencoder  
    3. Agent Two: Classification and reasoning with XGBoost + LLM
    4. Agent Three: RL-based mitigation decisions

Usage:
    python tests/test_integration.py --data-path data/UNSW_NB15_training-set.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_agent_one(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_path: str = "models/agent_one",
) -> List[int]:
    """
    Tests Agent One anomaly detection.
    
    Returns:
        List of indices flagged as anomalies.
    """
    from agents import AgentOne
    
    print("\n" + "=" * 60)
    print("Agent One: Anomaly Detection")
    print("=" * 60)
    
    # Load agent
    model_dir = Path(model_path)
    model_file = model_dir / "best_model.pth"
    if not model_file.exists():
        model_file = model_dir / "final_model.pth"
    
    if not model_file.exists():
        print("   [!] Agent One model not found. Run train_autoencoder.py first.")
        print("   Using random labels for demo...")
        # For demo, randomly flag ~10% as anomalies
        n_samples = len(X_test)
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        return list(anomaly_indices)
    
    # Load with optimized threshold from training
    agent = AgentOne.from_checkpoint(model_file, threshold=0.0396)
    print(f"   Loaded model from: {model_file}")
    print(f"   Threshold: {agent.threshold:.6f}")
    
    # Handle dimension mismatch (fine-tuned model may expect 42 features)
    model_input_dim = agent._model.input_dim
    data_dim = X_test.shape[1]
    if data_dim < model_input_dim:
        print(f"   Padding features: {data_dim} -> {model_input_dim}")
        padding = np.zeros((X_test.shape[0], model_input_dim - data_dim), dtype=X_test.dtype)
        X_test = np.hstack([X_test, padding])
    elif data_dim > model_input_dim:
        print(f"   Truncating features: {data_dim} -> {model_input_dim}")
        X_test = X_test[:, :model_input_dim]
    
    # Detect anomalies  
    print(f"\n   Processing {len(X_test)} samples...")
    results = agent.detect_anomalies(X_test)
    
    # Extract predictions and scores from results
    predictions = np.array([1 if r.is_anomaly else 0 for r in results])
    scores = np.array([r.reconstruction_error for r in results])
    
    anomaly_indices = np.where(predictions == 1)[0]
    normal_count = np.sum(predictions == 0)
    anomaly_count = np.sum(predictions == 1)
    
    print(f"\n   Results:")
    print(f"     Normal:   {normal_count} ({normal_count/len(predictions)*100:.1f}%)")
    print(f"     Anomaly:  {anomaly_count} ({anomaly_count/len(predictions)*100:.1f}%)")
    
    # Calculate accuracy if we have ground truth
    if y_test is not None:
        # y_test should be binary (0 = normal, 1 = attack)
        actual_anomalies = np.sum(y_test == 1)
        detected_anomalies = np.sum((predictions == 1) & (y_test == 1))
        false_negatives = np.sum((predictions == 0) & (y_test == 1))
        
        recall = detected_anomalies / actual_anomalies if actual_anomalies > 0 else 0
        
        print(f"\n   Detection Performance:")
        print(f"     Actual anomalies: {actual_anomalies}")
        print(f"     Detected:         {detected_anomalies}")
        print(f"     Missed (FN):      {false_negatives}")
        print(f"     Recall:           {recall:.4f}")
    
    return list(anomaly_indices)


def test_agent_two(
    X_test: np.ndarray,
    anomaly_indices: List[int],
    y_categories: np.ndarray = None,
    model_path: str = "models/agent_two",
    use_llm: bool = False,
    llm_provider: str = "groq",
) -> List[Dict[str, Any]]:
    """
    Tests Agent Two classification and reasoning.
    
    Args:
        llm_provider: LLM provider - 'groq' (free), 'openai', or 'ollama'
    
    Returns:
        List of analysis results for each anomaly.
    """
    from agents.agent_two import AgentTwo, ThreatAnalysisResult
    from agents.interfaces import MockLLM, FAISSVectorDB
    
    print("\n" + "=" * 60)
    print("Agent Two: Classification and Reasoning")
    print("=" * 60)
    
    model_dir = Path(model_path)
    classifier_path = model_dir / "classifier"
    kb_path = model_dir / "knowledge_base"
    
    # Check if model exists
    if not classifier_path.exists():
        print("   [!] Agent Two classifier not found. Run train_xgboost.py first.")
        return []
    
    # Create LLM (mock for testing, real for production)
    if use_llm:
        try:
            # Load .env file for API key
            from dotenv import load_dotenv
            load_dotenv()
            import os
            
            # Use CLI provider or fall back to env variable
            provider = llm_provider or os.environ.get("LLM_PROVIDER", "groq")
            provider = provider.lower()
            
            if provider == "groq":
                from agents.interfaces import GroqLLM
                llm = GroqLLM(model="llama-3.3-70b-versatile")
                print("   Using Groq LLM (llama-3.3-70b-versatile) - FREE")
            elif provider == "openai":
                from agents.interfaces import OpenAILLM
                llm = OpenAILLM(model="gpt-3.5-turbo")
                print("   Using OpenAI LLM (gpt-3.5-turbo)")
            elif provider == "ollama":
                from agents.interfaces import OllamaLLM
                llm = OllamaLLM(model="llama3")
                print("   Using Ollama LLM (llama3) - Local")
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")
        except Exception as e:
            print(f"   [!] LLM not available: {e}")
            llm = MockLLM()
    else:
        print("   Using MockLLM (set --use-llm for real reasoning)")
        llm = MockLLM()
    
    # Create vector DB
    vector_db = None
    if kb_path.exists():
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            # FAISSVectorDB auto-loads from persist_directory in __init__
            vector_db = FAISSVectorDB(embeddings, persist_directory=str(kb_path))
            print(f"   Loaded knowledge base with {vector_db.count} documents")
        except Exception as e:
            print(f"   [!] Could not load knowledge base: {e}")
    
    # Load Agent Two (pass base model_dir, not classifier_path)
    agent = AgentTwo.from_pretrained(
        model_dir=str(model_dir),
        llm=llm,
        vector_db=vector_db,
    )
    print(f"   Loaded classifier with {agent.classifier.n_classes} classes")
    print(f"   Classes: {agent.classifier.ATTACK_CATEGORIES}")
    
    # Analyze flagged anomalies
    if len(anomaly_indices) == 0:
        print("\n   No anomalies to analyze!")
        return []
    
    # Take subset if too many
    max_analyze = 20
    if len(anomaly_indices) > max_analyze:
        print(f"\n   Analyzing first {max_analyze} of {len(anomaly_indices)} anomalies...")
        analyze_indices = anomaly_indices[:max_analyze]
    else:
        print(f"\n   Analyzing {len(anomaly_indices)} anomalies...")
        analyze_indices = anomaly_indices
    
    # Get features for anomalies
    X_anomalies = X_test[analyze_indices]
    
    # Analyze batch (LLM reasoning happens automatically for zero-days if LLM is provided)
    results = agent.analyze_threats_batch(X_anomalies)
    
    # Print summary
    print("\n   Analysis Summary:")
    print("-" * 50)
    
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    category_counts = {}
    zero_day_count = 0
    
    for i, result in enumerate(results):
        predicted_cat = result.classification.predicted_category
        category_counts[predicted_cat] = category_counts.get(predicted_cat, 0) + 1
        severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
        if result.is_zero_day:
            zero_day_count += 1
    
    print("\n   By Category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"     {cat}: {count}")
    
    print("\n   By Severity:")
    for sev in ["critical", "high", "medium", "low"]:
        if severity_counts[sev] > 0:
            print(f"     {sev.upper()}: {severity_counts[sev]}")
    
    print(f"\n   Potential Zero-days: {zero_day_count}")
    
    # Compare with ground truth if available
    if y_categories is not None:
        print("\n   Accuracy Check:")
        correct = 0
        for i, idx in enumerate(analyze_indices):
            actual = y_categories[idx]
            predicted = results[i].classification.predicted_category
            if actual == predicted:
                correct += 1
        print(f"     Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    # Show detailed results for first few
    print("\n   Sample Analyses:")
    print("-" * 50)
    
    for i, result in enumerate(results[:5]):
        cat = result.classification.predicted_category
        conf = result.classification.confidence
        print(f"\n   [{i+1}] {cat} ({conf:.1%}) - Severity: {result.severity.upper()}")
        if result.is_zero_day and result.llm_reasoning:
            print(f"       Zero-day reason: {result.llm_reasoning[:100]}...")
    
    return results


def test_agent_three(
    analysis_results: List[Any],
    model_path: str = "models/agent_three",
) -> List[Dict[str, Any]]:
    """
    Tests Agent Three mitigation decisions.
    
    Args:
        analysis_results: List of ThreatAnalysisResult from Agent Two.
        model_path: Path to trained Agent Three model.
    
    Returns:
        List of mitigation decisions.
    """
    from agents import AgentThree, MitigationAction
    
    print("\n" + "=" * 60)
    print("Agent Three: Mitigation Decisions (RL)")
    print("=" * 60)
    
    model_dir = Path(model_path)
    
    # Check if model exists
    if not (model_dir / "final_model").exists() and not (model_dir / "ppo_model.zip").exists():
        print("   [!] Agent Three model not found. Run train_rl.py first.")
        print("   Using fallback rule-based policy...")
        # Create untrained agent (uses fallback policy)
        agent = AgentThree()
    else:
        # Load trained agent
        try:
            agent = AgentThree.from_pretrained(str(model_dir / "final_model"))
            print(f"   Loaded trained model from: {model_dir / 'final_model'}")
        except Exception as e:
            try:
                agent = AgentThree.from_pretrained(str(model_dir))
                print(f"   Loaded trained model from: {model_dir}")
            except Exception as e2:
                print(f"   [!] Could not load Agent Three model: {e2}")
                print("   Using fallback rule-based policy...")
                agent = AgentThree()
    
    if len(analysis_results) == 0:
        print("\n   No threats to mitigate!")
        return []
    
    print(f"\n   Processing {len(analysis_results)} threat analyses...")
    
    # Make mitigation decisions
    decisions = []
    action_counts = {
        MitigationAction.DO_NOTHING: 0,
        MitigationAction.ALERT_ADMIN: 0,
        MitigationAction.BLOCK_IP: 0,
        MitigationAction.ISOLATE_SUBNET: 0,
    }
    
    for result in analysis_results:
        decision = agent.take_action(result)
        decisions.append(decision)
        action_counts[decision.action] += 1
    
    # Print summary
    print("\n   Mitigation Summary:")
    print("-" * 50)
    
    action_names = {
        MitigationAction.DO_NOTHING: "Do Nothing",
        MitigationAction.ALERT_ADMIN: "Alert Admin",
        MitigationAction.BLOCK_IP: "Block IP",
        MitigationAction.ISOLATE_SUBNET: "Isolate Subnet",
    }
    
    print("\n   Action Distribution:")
    for action, count in action_counts.items():
        if count > 0:
            pct = count / len(decisions) * 100
            print(f"     {action_names[action]}: {count} ({pct:.1f}%)")
    
    # Group by severity and show decisions
    print("\n   Sample Decisions:")
    print("-" * 50)
    
    for i, (result, decision) in enumerate(zip(analysis_results[:5], decisions[:5])):
        cat = result.classification.predicted_category
        sev = result.severity
        action = decision.action_name
        conf = decision.confidence
        print(f"\n   [{i+1}] {cat} ({sev.upper()})")
        print(f"       → Action: {action} ({conf:.1%} confidence)")
        
        # Show brief reasoning for critical/zero-day
        if result.severity == "critical" or result.is_zero_day:
            probs = decision.action_probabilities
            top_actions = sorted(probs.items(), key=lambda x: -x[1])[:2]
            alts = ", ".join([f"{a}={p:.0%}" for a, p in top_actions])
            print(f"       → Probabilities: {alts}")
    
    return decisions


def main():
    parser = argparse.ArgumentParser(description="Test full IDS pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/UNSW_NB15_training-set.csv",
        help="Path to test data",
    )
    parser.add_argument(
        "--agent-one-model",
        type=str,
        default="models/agent_one",
        help="Path to Agent One model",
    )
    parser.add_argument(
        "--agent-two-model",
        type=str,
        default="models/agent_two",
        help="Path to Agent Two model",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use real LLM for reasoning",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="groq",
        choices=["openai", "groq", "ollama"],
        help="LLM provider: groq (free), openai, or ollama (local)",
    )
    parser.add_argument(
        "--agent-three-model",
        type=str,
        default="models/agent_three",
        help="Path to Agent Three model",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to test",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Agent IDS Integration Test")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading test data...")
    
    try:
        from data_pipeline import DataLoader, DatasetConfig, Preprocessor
        
        config = DatasetConfig()
        loader = DataLoader(config)
        loader.load(args.data_path).clean()
        
        # Get features and binary labels for Agent One
        X, y_binary = loader.get_features_and_labels(label_type="binary")
        
        # Get category labels for Agent Two evaluation
        _, y_category = loader.get_features_and_labels(label_type="multiclass")
        
        # Preprocess (y=None returns only X_processed)
        preprocessor = Preprocessor(config)
        X_processed = preprocessor.fit_transform(X, y=None, categorical_encoding="label")
        
        # Sample subset
        n = min(args.n_samples, len(X_processed))
        indices = np.random.choice(len(X_processed), size=n, replace=False)
        X_test = X_processed[indices]
        y_binary_test = np.array(y_binary)[indices]
        y_category_test = np.array(y_category)[indices]
        
        print(f"   Loaded {n} samples for testing")
        print(f"   Attack ratio: {np.mean(y_binary_test):.2%}")
        
    except Exception as e:
        print(f"   [!] Error loading data: {e}")
        print("   Using synthetic data for demo...")
        
        # Create synthetic data
        X_test = np.random.randn(args.n_samples, 41).astype(np.float32)
        y_binary_test = np.random.binomial(1, 0.3, args.n_samples)
        y_category_test = np.array(["Normal"] * args.n_samples)
    
    # Test Agent One
    print("\n[2/5] Running Agent One (Anomaly Detection)...")
    anomaly_indices = test_agent_one(
        X_test, y_binary_test, 
        model_path=args.agent_one_model
    )
    
    # Test Agent Two
    print("\n[3/5] Running Agent Two (Classification & Reasoning)...")
    results = test_agent_two(
        X_test, anomaly_indices,
        y_categories=y_category_test,
        model_path=args.agent_two_model,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
    )
    
    # Test Agent Three
    print("\n[4/5] Running Agent Three (Mitigation Decisions)...")
    decisions = test_agent_three(
        results,
        model_path=args.agent_three_model,
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("[5/5] Full Pipeline Summary")
    print("=" * 60)
    
    print(f"\n   Total samples processed: {len(X_test)}")
    print(f"   Agent One flagged: {len(anomaly_indices)} anomalies")
    print(f"   Agent Two classified: {len(results)} threats")
    print(f"   Agent Three mitigated: {len(decisions)} decisions")
    
    if results:
        critical = sum(1 for r in results if r.severity == "critical")
        zero_days = sum(1 for r in results if r.is_zero_day)
        print(f"\n   Critical threats: {critical}")
        print(f"   Potential zero-days: {zero_days}")
    
    if decisions:
        from agents import MitigationAction
        blocked = sum(1 for d in decisions if d.action in [MitigationAction.BLOCK_IP, MitigationAction.ISOLATE_SUBNET])
        alerts = sum(1 for d in decisions if d.action == MitigationAction.ALERT_ADMIN)
        ignored = sum(1 for d in decisions if d.action == MitigationAction.DO_NOTHING)
        print(f"\n   Mitigation Actions:")
        print(f"     Blocked/Isolated: {blocked}")
        print(f"     Alerts sent: {alerts}")
        print(f"     Ignored (normal): {ignored}")
    
    print("\n   Full pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
