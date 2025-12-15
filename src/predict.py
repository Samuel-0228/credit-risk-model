import argparse
import mlflow.sklearn
import pandas as pd
import joblib
import numpy as np


def predict_risk(customer_features):
    """Predict risk prob + score (simple: prob * 100)."""
    # Load model (local fallback)
    try:
        model = mlflow.sklearn.load_model(
            "models:/CreditRiskModel/Production")  # Registered
    except:
        model = joblib.load('models/best_model.pkl')

    # Ensure DF shape
    if isinstance(customer_features, dict):
        df = pd.DataFrame([customer_features])
    else:
        df = customer_features

    prob = model.predict_proba(df)[:, 1][0]
    score = int(prob * 100)  # 0-100 scale
    return {'risk_probability': prob, 'credit_score': score, 'risk_category': 'High' if prob > 0.5 else 'Low'}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--customer_id', type=int,
                        required=True, help='CustomerId for prediction')
    args = parser.parse_args()

    # Load features (assume from labeled_features.csv)
    df_features = pd.read_csv('data/processed/labeled_features.csv')
    customer_row = df_features[df_features['CustomerId'] == args.customer_id].drop(
        ['CustomerId', 'is_high_risk'], axis=1)

    if customer_row.empty:
        print(f"No data for CustomerId {args.customer_id}")
    else:
        result = predict_risk(customer_row.iloc[0])
        print(f"Prediction for Customer {args.customer_id}: {result}")
