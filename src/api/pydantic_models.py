from pydantic import BaseModel
from typing import Dict, Any
import numpy as np


class PredictionRequest(BaseModel):
    """Input: Customer features matching processed dataset (exclude CustomerId/target)."""
    features: Dict[str,
                   # e.g., {"Amount_sum": 100.0, "ProductCategory_airtime": 1.0, ...}
                   Any]


class PredictionResponse(BaseModel):
    """Output: Risk prob, score (0-100), category."""
    risk_probability: float
    credit_score: int
    risk_category: str  # "Low" or "High"
