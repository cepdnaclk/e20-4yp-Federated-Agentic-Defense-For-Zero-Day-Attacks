from pydantic import BaseModel
from typing import Dict

class GlobalPattern(BaseModel):
    pattern_id: str
    centroid: Dict[str, float]
    agent_count: int
    total_frequency: int
    confidence: float
