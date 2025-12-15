import pytest
import pandas as pd
from src.data_processing import TemporalExtractor, CustomerAggregator  # Adjust imports


def test_temporal_extractor():
    df_test = pd.DataFrame({'TransactionStartTime': ['2023-01-01 12:00:00']})
    extractor = TemporalExtractor()
    df_out = extractor.transform(df_test)
    assert 'TransactionHour' in df_out.columns
    assert df_out['TransactionHour'].iloc[0] == 12


def test_aggregator_output_cols():
    df_test = pd.DataFrame({
        'CustomerId': [1, 1], 'Amount': [10, 20], 'TransactionStartTime': ['2023-01-01', '2023-01-02']
    })
    agg = CustomerAggregator()
    df_agg = agg.transform(df_test)
    expected_cols = ['CustomerId', 'Amount_sum',
                     'Amount_mean', 'TransactionHour_mean']
    for col in expected_cols:
        assert col in df_agg.columns
