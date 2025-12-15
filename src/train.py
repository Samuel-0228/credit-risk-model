import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import logging
import pytest  # For tests, but run separately

# Setup
os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'
mlflow.set_experiment("credit_risk_modeling")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data (from Task 4)
data_path = 'data/processed/labeled_features.csv'
df = pd.read_csv(data_path)
logger.info(
    f"Loaded {len(df)} samples | High-risk rate: {df['is_high_risk'].mean():.2%}")

# Prep: Features (drop ID/target), split (stratified)
X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
y = df['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Models & Params
models = {
    'logistic': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {'classifier__C': [0.1, 1, 10]}
    },
    'random_forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [5, 10]}
    },
    'xgboost': {
        'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 6]}
    }
}


def evaluate_model(y_true, y_prob, y_pred):
    """Compute metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }


best_model = None
best_auc = 0

with mlflow.start_run():
    for name, config in models.items():
        # Pipeline
        pipe = Pipeline([('classifier', config['model'])])

        # Tune
        grid = GridSearchCV(
            pipe, config['params'], cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Predict
        y_pred = grid.best_estimator_.predict(X_test)
        y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_prob, y_pred)

        # Log
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_param('model_type', name)
        mlflow.log_param('cv_score', grid.best_score_)

        # ROC plot artifact
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        fig.savefig('roc.png')
        mlflow.log_artifact('roc.png')

        logger.info(f"{name} - Test AUC: {metrics['roc_auc']:.3f}")

        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = grid.best_estimator_

    # Register best
    if best_model:
        signature = infer_signature(X_test, best_model.predict(X_test))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        model_uri = mlflow.register_model(
            "runs:/" + mlflow.active_run().info.run_id + "/model", "CreditRiskModel")
        logger.info(f"Best model (AUC {best_auc:.3f}) registered: {model_uri}")

# Save best locally too
joblib.dump(best_model, 'models/best_model.pkl')
logger.info("Training complete. View in MLflow UI.")
