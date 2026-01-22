from pydantic import BaseModel
from typing import Dict
from datetime import datetime

class AnomalySignature(BaseModel):
    signature_id: str
    feature_deviation: Dict[str, float]  # z-score deltas
    confidence: float
    frequency: int
    time_window: str
    agent_id: str
    timestamp: datetime
