"""
Credit Risk Model - Unit Tests
===============================
Unit tests for data_processing.py module.

"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing import (
    extract_temporal_features,
    create_aggregate_features,
    calculate_rfm_features,
    create_rfm_based_target,
    get_feature_names,
    get_target_name,
    split_features_target,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    np.random.seed(42)

    data = {
        'TransactionId': [f'Txn_{i}' for i in range(100)],
        'CustomerId': [f'CustomerId_{i % 10}' for i in range(100)],
        'AccountId': [f'AccountId_{i % 10}' for i in range(100)],
        'Amount': np.random.randn(100) * 1000,
        'Value': np.abs(np.random.randn(100) * 1000),
        'TransactionStartTime': pd.date_range('2024-01-01', periods=100, freq='D'),
        'ChannelId': np.random.choice(['ChannelId_1', 'ChannelId_2', 'ChannelId_3'], 100),
        'ProductCategory': np.random.choice(['airtime', 'financial_services', 'utility_bill'], 100),
        'ProviderId': np.random.choice(['ProviderId_1', 'ProviderId_2'], 100),
        'PricingStrategy': np.random.choice([0, 1, 2, 4], 100),
        'FraudResult': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
    }

    return pd.DataFrame(data)


@pytest.fixture
def customer_features():
    """Create sample customer-level features for testing."""
    return pd.DataFrame({
        'CustomerId': [f'CustomerId_{i}' for i in range(10)],
        'TotalAmount': np.random.randn(10) * 10000,
        'AvgAmount': np.random.randn(10) * 500,
        'StdAmount': np.abs(np.random.randn(10) * 200),
        'MinAmount': np.abs(np.random.randn(10) * 100),
        'MaxAmount': np.abs(np.random.randn(10) * 1000),
        'TransactionCount': np.random.randint(5, 50, 10),
        'Recency': np.random.randint(1, 90, 10),
        'Frequency': np.random.randint(5, 50, 10),
        'Monetary': np.random.randn(10) * 10000,
    })


# =============================================================================
# TESTS FOR TEMPORAL FEATURE EXTRACTION
# =============================================================================

class TestExtractTemporalFeatures:
    """Tests for extract_temporal_features function."""

    def test_extracts_hour(self, sample_transactions):
        """Test that hour is extracted correctly."""
        result = extract_temporal_features(sample_transactions)
        assert 'TransactionHour' in result.columns
        assert result['TransactionHour'].min() >= 0
        assert result['TransactionHour'].max() <= 23

    def test_extracts_day(self, sample_transactions):
        """Test that day of month is extracted correctly."""
        result = extract_temporal_features(sample_transactions)
        assert 'TransactionDay' in result.columns
        assert result['TransactionDay'].min() >= 1
        assert result['TransactionDay'].max() <= 31

    def test_extracts_month(self, sample_transactions):
        """Test that month is extracted correctly."""
        result = extract_temporal_features(sample_transactions)
        assert 'TransactionMonth' in result.columns
        assert result['TransactionMonth'].min() >= 1
        assert result['TransactionMonth'].max() <= 12

    def test_extracts_day_of_week(self, sample_transactions):
        """Test that day of week is extracted correctly."""
        result = extract_temporal_features(sample_transactions)
        assert 'DayOfWeek' in result.columns
        assert result['DayOfWeek'].min() >= 0
        assert result['DayOfWeek'].max() <= 6

    def test_is_weekend_correct(self, sample_transactions):
        """Test that weekend flag is correctly computed."""
        result = extract_temporal_features(sample_transactions)
        assert 'IsWeekend' in result.columns
        assert set(result['IsWeekend'].unique()).issubset({0, 1})

    def test_returns_dataframe(self, sample_transactions):
        """Test that function returns a DataFrame."""
        result = extract_temporal_features(sample_transactions)
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# TESTS FOR AGGREGATE FEATURES
# =============================================================================

class TestCreateAggregateFeatures:
    """Tests for create_aggregate_features function."""

    def test_returns_dataframe(self, sample_transactions):
        """Test that function returns a DataFrame."""
        result = create_aggregate_features(sample_transactions)
        assert isinstance(result, pd.DataFrame)

    def test_has_customer_id(self, sample_transactions):
        """Test that CustomerId is present in output."""
        result = create_aggregate_features(sample_transactions)
        assert 'CustomerId' in result.columns

    def test_has_transaction_count(self, sample_transactions):
        """Test that TransactionCount is created."""
        result = create_aggregate_features(sample_transactions)
        assert 'TransactionCount' in result.columns

    def test_correct_row_count(self, sample_transactions):
        """Test that output has correct number of unique customers."""
        result = create_aggregate_features(sample_transactions)
        expected_customers = sample_transactions['CustomerId'].nunique()
        assert len(result) == expected_customers

    def test_has_total_amount(self, sample_transactions):
        """Test that TotalAmount aggregate is created."""
        result = create_aggregate_features(sample_transactions)
        assert 'TotalAmount' in result.columns

    def test_has_fraud_metrics(self, sample_transactions):
        """Test that fraud count and rate are created."""
        result = create_aggregate_features(sample_transactions)
        assert 'FraudCount' in result.columns
        assert 'FraudRate' in result.columns


# =============================================================================
# TESTS FOR RFM FEATURES
# =============================================================================

class TestCalculateRfmFeatures:
    """Tests for calculate_rfm_features function."""

    def test_returns_dataframe(self, sample_transactions):
        """Test that function returns a DataFrame."""
        result = calculate_rfm_features(sample_transactions)
        assert isinstance(result, pd.DataFrame)

    def test_has_recency(self, sample_transactions):
        """Test that Recency feature is created."""
        result = calculate_rfm_features(sample_transactions)
        assert 'Recency' in result.columns

    def test_has_frequency(self, sample_transactions):
        """Test that Frequency feature is created."""
        result = calculate_rfm_features(sample_transactions)
        assert 'Frequency' in result.columns

    def test_has_monetary(self, sample_transactions):
        """Test that Monetary feature is created."""
        result = calculate_rfm_features(sample_transactions)
        assert 'Monetary' in result.columns

    def test_frequency_matches_transaction_count(self, sample_transactions):
        """Test that Frequency matches actual transaction count."""
        result = calculate_rfm_features(sample_transactions)

        for _, row in result.iterrows():
            customer_id = row['CustomerId']
            expected_freq = len(sample_transactions[sample_transactions['CustomerId'] == customer_id])
            assert row['Frequency'] == expected_freq

    def test_recency_is_non_negative(self, sample_transactions):
        """Test that Recency values are non-negative."""
        result = calculate_rfm_features(sample_transactions)
        assert (result['Recency'] >= 0).all()


# =============================================================================
# TESTS FOR RFM-BASED TARGET
# =============================================================================

class TestCreateRfmBasedTarget:
    """Tests for create_rfm_based_target function."""

    def test_returns_dataframe(self, customer_features):
        """Test that function returns a DataFrame."""
        result = create_rfm_based_target(customer_features)
        assert isinstance(result, pd.DataFrame)

    def test_has_is_high_risk_column(self, customer_features):
        """Test that is_high_risk column is created."""
        result = create_rfm_based_target(customer_features)
        assert 'is_high_risk' in result.columns

    def test_target_is_binary(self, customer_features):
        """Test that is_high_risk is binary (0 or 1)."""
        result = create_rfm_based_target(customer_features)
        assert set(result['is_high_risk'].unique()).issubset({0, 1})

    def test_target_distribution(self, customer_features):
        """Test that target has reasonable distribution."""
        result = create_rfm_based_target(customer_features)
        assert result['is_high_risk'].sum() > 0
        assert (result['is_high_risk'] == 0).sum() > 0

    def test_has_rfm_cluster(self, customer_features):
        """Test that RFM_Cluster column is created."""
        result = create_rfm_based_target(customer_features)
        assert 'RFM_Cluster' in result.columns

    def test_cluster_values_valid(self, customer_features):
        """Test that cluster values are 0, 1, or 2."""
        result = create_rfm_based_target(customer_features)
        assert set(result['RFM_Cluster'].unique()).issubset({0, 1, 2})


# =============================================================================
# TESTS FOR UTILITY FUNCTIONS
# =============================================================================

class TestGetFeatureNames:
    """Tests for get_feature_names function."""

    def test_returns_list(self, customer_features):
        """Test that function returns a list."""
        customer_features['is_high_risk'] = 0
        result = get_feature_names(customer_features)
        assert isinstance(result, list)

    def test_excludes_target(self, customer_features):
        """Test that target column is excluded."""
        customer_features['is_high_risk'] = 0
        result = get_feature_names(customer_features)
        assert 'is_high_risk' not in result

    def test_includes_features(self, customer_features):
        """Test that feature columns are included."""
        customer_features['is_high_risk'] = 0
        result = get_feature_names(customer_features)
        assert 'TotalAmount' in result


class TestGetTargetName:
    """Tests for get_target_name function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        result = get_target_name()
        assert isinstance(result, str)

    def test_returns_correct_target_name(self):
        """Test that target name is correct."""
        result = get_target_name()
        assert result == 'is_high_risk'


class TestSplitFeaturesTarget:
    """Tests for split_features_target function."""

    def test_returns_tuple(self, customer_features):
        """Test that function returns a tuple."""
        customer_features['is_high_risk'] = np.random.randint(0, 2, len(customer_features))
        result = split_features_target(customer_features)
        assert isinstance(result, tuple)

    def test_split_correct(self, customer_features):
        """Test that features and target are correctly split."""
        customer_features['is_high_risk'] = np.random.randint(0, 2, len(customer_features))
        X, y = split_features_target(customer_features)

        expected_cols = [c for c in customer_features.columns if c != 'is_high_risk']
        assert set(X.columns) == set(expected_cols)
        assert list(y) == list(customer_features['is_high_risk'])

    def test_y_is_series(self, customer_features):
        """Test that target is a pandas Series."""
        customer_features['is_high_risk'] = np.random.randint(0, 2, len(customer_features))
        X, y = split_features_target(customer_features)
        assert isinstance(y, pd.Series)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])