import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import check_array
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from TransactionStartTime."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(
            X['TransactionStartTime'], errors='coerce')
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        X = X.drop('TransactionStartTime', axis=1)  # Clean up
        logger.info("Temporal features extracted")
        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction data to customer level."""

    def __init__(self):
        self.agg_dict = {
            'Amount': ['sum', 'mean', 'std', 'count'],  # Raw amounts
            'log_amount': ['sum', 'mean', 'std'],  # Log-abs for monetary proxy
            'FraudResult': ['sum', 'mean'],  # Fraud exposure
            'TransactionHour': ['mean', 'std'],
            'TransactionDay': ['mean'],
            'TransactionMonth': ['mean'],
            'TransactionYear': ['max']  # Most recent year for recency proxy
        }
        self.cat_cols = ['ProductCategory', 'ChannelId',
                         'CountryCode', 'PricingStrategy']
        self.high_card_cols = ['ProductId', 'ProviderId']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Log-transform for positive monetary handling
        X['log_amount'] = np.log1p(np.abs(X['Amount']))

        # Numerical aggs
        agg_num = X.groupby('CustomerId').agg(self.agg_dict).reset_index()
        agg_num.columns = ['CustomerId'] + [f"{col[0]}_{col[1]}" if isinstance(
            col[1], str) else f"{col[0]}_{col[1]}" for col in agg_num.columns[1:]]

        # Categorical mode (most frequent)
        agg_cat = X.groupby('CustomerId')[self.cat_cols].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan).reset_index()

        # High-card frequency encoding (avg freq per customer)
        for col in self.high_card_cols:
            if col in X.columns:
                freq_map = X[col].value_counts().to_dict()
                X[f'{col}_freq'] = X[col].map(freq_map)
                agg_num[f'{col}_mean_freq'] = X.groupby(
                    'CustomerId')[f'{col}_freq'].transform('mean').reset_index()[f'{col}_freq']

        # Merge
        df_agg = pd.merge(agg_num, agg_cat, on='CustomerId', how='left')
        df_agg = df_agg.fillna(0)  # Simple fill for aggs

        logger.info(
            f"Aggregated to {len(df_agg)} customers with {len(df_agg.columns)} features")
        return df_agg


class IVSelector(BaseEstimator, TransformerMixin):
    """Compute Information Value (IV) for feature selection. Handles array input."""

    def __init__(self, iv_threshold=0.02):
        self.iv_threshold = iv_threshold
        self.selected_features = []
        self.feature_names = None

    def fit(self, X, y=None, feature_names=None):
        # Handle array input: Convert to DF
        if isinstance(X, np.ndarray):
            if feature_names is not None:
                X = pd.DataFrame(X, columns=feature_names)
            else:
                X = pd.DataFrame(
                    X, columns=[f'feat_{i}' for i in range(X.shape[1])])

        # Select numeric cols
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col !=
                        'CustomerId']  # Exclude ID

        # Placeholder IV: Use variance as proxy (replace with real woe.IV in Task 5)
        iv_proxy = {col: X[col].var() for col in numeric_cols}
        self.selected_features = [
            col for col, iv in iv_proxy.items() if iv > self.iv_threshold]

        self.feature_names = X.columns.tolist()  # For transform
        logger.info(
            f"Selected {len(self.selected_features)} features with IV proxy > {self.iv_threshold}")
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return X[self.selected_features + ['CustomerId']]


def build_preprocessing_pipeline():
    """Dynamic preprocessor: Applied after agg."""
    # This will be called on DF from agg
    def dynamic_preprocessor(df):
        # Auto-detect cols
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [col for col in num_cols if col !=
                    'CustomerId']  # Exclude ID

        cat_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

        logger.info(
            f"Preprocessing: {len(num_cols)} num cols, {len(cat_cols)} cat cols")

        if not num_cols and not cat_cols:
            return df

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), num_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(sparse_output=False,
                     handle_unknown='ignore', drop='first'))
                ]), cat_cols)
            ],
            remainder='passthrough'
        )

        transformed = preprocessor.fit_transform(df)
        feature_names = preprocessor.get_feature_names_out()
        df_transformed = pd.DataFrame(
            transformed, columns=feature_names, index=df.index)
        df_transformed['CustomerId'] = df['CustomerId']  # Preserve ID

        return df_transformed

    return dynamic_preprocessor


if __name__ == "__main__":
    raw_path = 'data/raw/Train.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Please download Train.csv to {raw_path} from Kaggle.")

    # Load and prep raw
    df_raw = pd.read_csv(raw_path, low_memory=False)
    if len(df_raw) < 1000000:
        logger.warning(
            f"Dataset small ({len(df_raw)} rows)—consider full load for production.")

    logger.info(f"Loaded {len(df_raw)} transactions")

    # Drop unused (per EDA: Value redundant, but keep for now if needed)
    drop_cols = ['TransactionId', 'BatchId',
                 'SubscriptionId', 'AccountId', 'CurrencyCode']
    df_raw = df_raw.drop(
        [col for col in drop_cols if col in df_raw.columns], axis=1, errors='ignore')

    # Pre-impute major missing (ProductCategory)
    if 'ProductCategory' in df_raw.columns:
        mode_val = df_raw['ProductCategory'].mode(
        ).iloc[0] if not df_raw['ProductCategory'].mode().empty else 'unknown'
        df_raw['ProductCategory'] = df_raw['ProductCategory'].fillna(mode_val)

    # Partial pipeline: Up to agg
    agg_pipeline = Pipeline([
        ('temporal', TemporalExtractor()),
        ('aggregate', CustomerAggregator())
    ])

    df_agg = agg_pipeline.fit_transform(df_raw)

    # Dynamic preprocess
    preprocess_func = build_preprocessing_pipeline()
    df_preprocessed = preprocess_func(df_agg)

    # IV selection
    selector = IVSelector(iv_threshold=0.02)
    df_processed = selector.fit_transform(
        df_preprocessed, feature_names=df_preprocessed.columns)

    # Save
    os.makedirs('data/processed', exist_ok=True)
    df_processed.to_csv('data/processed/features.csv', index=False)
    logger.info(f"Processed features saved: {df_processed.shape}")
    print(df_processed.head())
