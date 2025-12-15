from .pydantic_models import PredictionRequest, PredictionResponse  # Relative import
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import mlflow.sklearn
import pandas as pd
import numpy as np
import os

# Setup
app = FastAPI(title="Credit Risk Prediction API", version="1.0.0")
os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'  # Local; adjust for prod

# Load model (from Task 5 registry; fallback local)
try:
    model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")
except:
    from joblib import load
    model = load('models/best_model.pkl')  # Local fallback


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):
    try:
        # Validate/convert features to DF
        features_df = pd.DataFrame([request.features])
        # Ensure numeric (handle any str)
        features_df = features_df.apply(
            pd.to_numeric, errors='coerce').fillna(0)

        # Predict
        prob = model.predict_proba(features_df)[:, 1][0]
        score = int(prob * 100)
        category = "High" if prob > 0.5 else "Low"

        return PredictionResponse(
            risk_probability=round(prob, 4),
            credit_score=score,
            risk_category=category
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
