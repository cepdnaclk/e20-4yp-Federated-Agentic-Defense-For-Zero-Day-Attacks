# agents/validation/agent.py

class ValidationAgent:
    """
    Validation Agent for network flows.
    Confirms whether a flow flagged as anomalous is a true attack
    or a false positive.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        :param confidence_threshold: Minimum anomaly score to confirm attack
        """
        self.confidence_threshold = confidence_threshold

    def evaluate(self, flow: dict, details: dict) -> dict:
        """
        Evaluate an anomalous flow.

        :param flow: Original flow data
        :param details: Inference details from autoencoder (reconstruction error, etc.)
        :return: dict with verdict and confidence
        """
        # Example logic: simple threshold check
        anomaly_score = details.get("score", 0.0)

        if anomaly_score >= self.confidence_threshold:
            verdict = "confirmed_attack"
        else:
            verdict = "false_positive"

        # You can also add additional rules here:
        # - Check specific features
        # - Historical context of source IP
        # - Frequency-based heuristics

        return {
            "verdict": verdict,
            "score": anomaly_score,
            "details": details,
            "flow_id": flow.get("id")
        }
