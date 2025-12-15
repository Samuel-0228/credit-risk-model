import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_rfm(df_raw, snapshot_date=None):
    """Compute RFM metrics per CustomerId."""
    if snapshot_date is None:
        # Dynamic snapshot: max date + 1 day
        df_raw['TransactionDate'] = pd.to_datetime(
            df_raw['TransactionStartTime'])
        snapshot_date = df_raw['TransactionDate'].max() + timedelta(days=1)
        logger.info(f"Snapshot date set to {snapshot_date.date()}")

    # Group for RFM
    rfm = df_raw.groupby('CustomerId').agg({
        'TransactionDate': ['max', 'count'],  # Last txn for R, count for F
        'Value': 'sum'  # M: total monetary (positive)
    }).reset_index()

    rfm.columns = ['CustomerId', 'last_txn_date', 'frequency', 'monetary']

    # Recency: days since last txn
    rfm['recency'] = (snapshot_date - rfm['last_txn_date']).dt.days
    rfm['monetary'] = np.log1p(rfm['monetary'])  # Handle skew

    # Drop date col
    rfm = rfm.drop('last_txn_date', axis=1)

    logger.info(f"RFM computed for {len(rfm)} customers")
    return rfm, snapshot_date


def cluster_rfm(rfm_df, n_clusters=3, random_state=42):
    """K-Means on scaled RFM; return cluster labels."""
    # Select RFM cols
    rfm_cols = ['recency', 'frequency', 'monetary']
    X_rfm = rfm_df[rfm_cols].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rfm)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Analyze clusters: Centroids (scaled); identify high-risk (high R/low F/low M)
    centroids = pd.DataFrame(scaler.inverse_transform(
        kmeans.cluster_centers_), columns=rfm_cols)
    centroids['cluster'] = range(n_clusters)
    print("Cluster Centroids (RFM):")
    print(centroids)

    # Manual label: Assume cluster with max recency + min freq/monetary = high-risk
    # (Adjust based on print: e.g., if cluster 1 has R=high, F/M=low)
    high_risk_cluster = centroids.loc[centroids['recency'].idxmax() & (
        centroids['frequency'] + centroids['monetary']).idxmin(), 'cluster']
    logger.info(f"High-risk cluster identified: {high_risk_cluster}")

    rfm_df['cluster'] = clusters
    return rfm_df, high_risk_cluster


def create_proxy_target(rfm_clustered, high_risk_cluster):
    """Binary label: 1 if high-risk cluster."""
    rfm_clustered['is_high_risk'] = (
        rfm_clustered['cluster'] == high_risk_cluster).astype(int)
    logger.info(
        f"High-risk labels assigned: {rfm_clustered['is_high_risk'].sum()} / {len(rfm_clustered)} ({rfm_clustered['is_high_risk'].mean():.2%})")
    return rfm_clustered


if __name__ == "__main__":
    # Paths
    raw_path = 'data/raw/Train.csv'
    processed_path = 'data/processed/features.csv'

    if not all(os.path.exists(p) for p in [raw_path, processed_path]):
        raise FileNotFoundError(
            f"Run Task 3 first: Need {raw_path} and {processed_path}")

    # Load
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_features = pd.read_csv(processed_path)

    # Compute RFM (needs raw for dates)
    rfm, snapshot = calculate_rfm(df_raw)

    # Cluster
    rfm_clustered, high_risk_id = cluster_rfm(rfm)

    # Label
    rfm_labeled = create_proxy_target(rfm_clustered, high_risk_id)

    # Merge with features (add RFM cols too for modeling)
    df_labeled = pd.merge(df_features, rfm_labeled,
                          on='CustomerId', how='left')
    df_labeled = df_labeled.drop('cluster', axis=1)  # Clean up

    # Save
    output_path = 'data/processed/labeled_features.csv'
    os.makedirs('data/processed', exist_ok=True)
    df_labeled.to_csv(output_path, index=False)
    logger.info(
        f"Labeled data saved: {df_labeled.shape} | High-risk rate: {df_labeled['is_high_risk'].mean():.2%}")

    # Quick EDA on proxy
    print("\nProxy Distribution:")
    print(df_labeled['is_high_risk'].value_counts(normalize=True))
    print("\nRFM by Risk (means):")
    print(rfm_labeled.groupby('is_high_risk')[
          ['recency', 'frequency', 'monetary']].mean())
