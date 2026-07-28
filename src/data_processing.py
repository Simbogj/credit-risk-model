"""
Feature Engineering Pipeline
==============================================
- Aggregate features (Total, Avg, Std of amounts per customer)
- Extract Features (hour, day, month, year)
- Categorical encoding (One-hot, Label encoding)
- Missing value handling (Imputation, Removal)
- Normalization/Standardization
"""

import logging
import os
import warnings
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw transaction data from CSV.

    Args:
        filepath: Path to the raw data CSV file.

    Returns:
        DataFrame with raw transaction data.
    """
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)

    # Parse datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    logger.info(f"Loaded {len(df):,} transactions with {df['CustomerId'].nunique():,} unique customers")
    return df


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from transaction timestamps.

    Features extracted:
    - TransactionHour: The hour of the day when the transaction occurred
    - TransactionDay: The day of the month when the transaction occurred
    - TransactionMonth: The month when the transaction occurred
    - TransactionYear: The year when the transaction occurred
    - DayOfWeek: Day of week (0=Monday, 6=Sunday)
    - IsWeekend: Binary flag for weekend transactions
    - Quarter: Quarter of the year
    - WeekOfYear: Week number of the year

    Args:
        df: Transaction DataFrame with TransactionStartTime

    Returns:
        DataFrame with temporal features added
    """
    logger.info("Extracting temporal features")

    df = df.copy()

    # Extract temporal components
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    df['IsWeekend'] = (df['TransactionStartTime'].dt.dayofweek >= 5).astype(int)
    df['Quarter'] = df['TransactionStartTime'].dt.quarter
    df['WeekOfYear'] = df['TransactionStartTime'].dt.isocalendar().week

    logger.info(f"Extracted temporal features: hour, day, month, year, dayofweek, weekend, quarter, weekofyear")
    return df


# =============================================================================
# CREATE AGGREGATE FEATURES
# =============================================================================

def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate features at customer level.

    Features created:
    - TotalAmount: Sum of all transaction amounts per customer
    - AvgAmount: Average transaction amount per customer
    - TransactionCount: Number of transactions per customer
    - StdAmount: Standard deviation of transaction amounts per customer
    - MinAmount, MaxAmount: Min/max transaction amounts
    - TotalValue, AvgValue, StdValue: Value field aggregates
    - FraudCount, FraudRate: Fraud metrics

    Args:
        df: Transaction DataFrame

    Returns:
        DataFrame with aggregate features per customer
    """
    logger.info("Creating aggregate customer-level features")

    df = df.copy()

    # Basic aggregations
    agg_funcs = {
        'Amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
        'Value': ['sum', 'mean', 'std'],
        'TransactionStartTime': ['min', 'max'],
        'FraudResult': ['sum', 'mean'],
    }

    customer_agg = df.groupby('CustomerId').agg(agg_funcs)
    customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns]
    customer_agg = customer_agg.reset_index()

    # Rename for clarity
    customer_agg = customer_agg.rename(columns={
        'Amount_sum': 'TotalAmount',
        'Amount_mean': 'AvgAmount',
        'Amount_std': 'StdAmount',
        'Amount_min': 'MinAmount',
        'Amount_max': 'MaxAmount',
        'Amount_count': 'TransactionCount',
        'Value_sum': 'TotalValue',
        'Value_mean': 'AvgValue',
        'Value_std': 'StdValue',
        'TransactionStartTime_min': 'FirstTransaction',
        'TransactionStartTime_max': 'LastTransaction',
        'FraudResult_sum': 'FraudCount',
        'FraudResult_mean': 'FraudRate',
    })

    # Credit/Debit separation (Amount < 0 is credit/refund, Amount > 0 is debit/purchase)
    credits = df[df['Amount'] < 0].groupby('CustomerId').agg(
        CreditCount=('Amount', 'count'),
        CreditTotal=('Amount', 'sum')
    ).reset_index()

    debits = df[df['Amount'] > 0].groupby('CustomerId').agg(
        DebitCount=('Amount', 'count'),
        DebitTotal=('Amount', 'sum')
    ).reset_index()

    customer_agg = customer_agg.merge(credits, on='CustomerId', how='left')
    customer_agg = customer_agg.merge(debits, on='CustomerId', how='left')

    # Fill NaN for customers with no credits or debits
    customer_agg['CreditCount'] = customer_agg['CreditCount'].fillna(0)
    customer_agg['CreditTotal'] = customer_agg['CreditTotal'].fillna(0)
    customer_agg['DebitCount'] = customer_agg['DebitCount'].fillna(0)
    customer_agg['DebitTotal'] = customer_agg['DebitTotal'].fillna(0)

    # Credit/Debit ratios
    customer_agg['CreditRatio'] = customer_agg['CreditCount'] / customer_agg['TransactionCount']
    customer_agg['DebitRatio'] = customer_agg['DebitCount'] / customer_agg['TransactionCount']

    # Amount range
    customer_agg['AmountRange'] = customer_agg['MaxAmount'] - customer_agg['MinAmount']

    logger.info(f"Created {len(customer_agg.columns) - 1} aggregate features for {len(customer_agg):,} customers")
    return customer_agg


# =============================================================================
# CHANNEL AND CATEGORY FEATURES
# =============================================================================

def create_channel_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on transaction channel and product category.

    Args:
        df: Transaction DataFrame

    Returns:
        DataFrame with channel and category count features
    """
    logger.info("Creating channel and category features")

    df = df.copy()

    # Count transactions by channel per customer
    channel_counts = df.groupby(['CustomerId', 'ChannelId']).size().unstack(fill_value=0)
    channel_counts.columns = [f'Channel_{col.replace("ChannelId_", "")}_Count' for col in channel_counts.columns]
    channel_counts = channel_counts.reset_index()

    # Count transactions by product category per customer
    category_counts = df.groupby(['CustomerId', 'ProductCategory']).size().unstack(fill_value=0)
    category_counts.columns = [f'Category_{col}_Count' for col in category_counts.columns]
    category_counts = category_counts.reset_index()

    # Amount statistics by channel per customer
    channel_amounts = df.groupby(['CustomerId', 'ChannelId'])['Amount'].sum().unstack(fill_value=0)
    channel_amounts.columns = [f'Channel_{col.replace("ChannelId_", "")}_Amount' for col in channel_amounts.columns]
    channel_amounts = channel_amounts.reset_index()

    # Amount statistics by product category per customer
    category_amounts = df.groupby(['CustomerId', 'ProductCategory'])['Amount'].sum().unstack(fill_value=0)
    category_amounts.columns = [f'Category_{col}_Amount' for col in category_amounts.columns]
    category_amounts = category_amounts.reset_index()

    # Merge all
    customer_features = df.groupby('CustomerId').size().reset_index(name='_dummy')
    customer_features = customer_features.drop('_dummy', axis=1)

    customer_features = customer_features.merge(channel_counts, on='CustomerId', how='left')
    customer_features = customer_features.merge(category_counts, on='CustomerId', how='left')
    customer_features = customer_features.merge(channel_amounts, on='CustomerId', how='left')
    customer_features = customer_features.merge(category_amounts, on='CustomerId', how='left')

    logger.info(f"Created channel and category features")
    return customer_features


# =============================================================================
# PROVIDER AND PRICING FEATURES
# =============================================================================

def create_provider_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on provider and pricing strategy.

    Args:
        df: Transaction DataFrame

    Returns:
        DataFrame with provider and pricing features
    """
    logger.info("Creating provider and pricing features")

    df = df.copy()

    # Unique providers per customer
    provider_counts = df.groupby('CustomerId')['ProviderId'].nunique().reset_index()
    provider_counts.columns = ['CustomerId', 'UniqueProviders']

    # Pricing strategy distribution per customer
    pricing_counts = df.groupby(['CustomerId', 'PricingStrategy']).size().unstack(fill_value=0)
    pricing_counts.columns = [f'PricingStrategy_{col}_Count' for col in pricing_counts.columns]
    pricing_counts = pricing_counts.reset_index()

    # Average pricing strategy per customer
    avg_pricing = df.groupby('CustomerId')['PricingStrategy'].mean().reset_index()
    avg_pricing.columns = ['CustomerId', 'AvgPricingStrategy']

    customer_features = df.groupby('CustomerId').size().reset_index(name='_dummy')
    customer_features = customer_features.drop('_dummy', axis=1)

    customer_features = customer_features.merge(provider_counts, on='CustomerId', how='left')
    customer_features = customer_features.merge(pricing_counts, on='CustomerId', how='left')
    customer_features = customer_features.merge(avg_pricing, on='CustomerId', how='left')

    logger.info(f"Created provider and pricing features")
    return customer_features


# =============================================================================
# CATEGORICAL ENCODING
# =============================================================================

def encode_categorical_features(df: pd.DataFrame, customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for model training.

    Creates ratio features (count / total) for channel and category distributions.

    Args:
        df: Original transaction DataFrame
        customer_df: Customer-level feature DataFrame

    Returns:
        DataFrame with encoded categorical features
    """
    logger.info("Encoding categorical features")

    customer_df = customer_df.copy()

    # Channel ratios (count / total transactions)
    channel_cols = [col for col in customer_df.columns if col.startswith('Channel_') and col.endswith('_Count')]

    for col in channel_cols:
        # Count ratio
        customer_df[col.replace('_Count', '_Ratio')] = customer_df[col] / customer_df['TransactionCount'].replace(0, 1)

    # Category ratios
    category_cols = [col for col in customer_df.columns if col.startswith('Category_') and col.endswith('_Count')]
    for col in category_cols:
        customer_df[col.replace('_Count', '_Ratio')] = customer_df[col] / customer_df['TransactionCount'].replace(0, 1)

    logger.info("Categorical features encoded")
    return customer_df


# =============================================================================
# NORMALIZE/SCALE FEATURES
# =============================================================================

def normalize_features(df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
    """
    Normalize/standardize numerical features.

    Args:
        df: DataFrame with features to normalize
        columns: List of column names to normalize
        method: 'standard' (mean=0, std=1) or 'robust' (IQR-based)

    Returns:
        DataFrame with normalized features
    """
    logger.info(f"Normalizing {len(columns)} features using {method} method")

    df = df.copy()

    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    # Only normalize columns that exist
    valid_columns = [c for c in columns if c in df.columns]

    if valid_columns:
        df[valid_columns] = scaler.fit_transform(df[valid_columns])

    return df


# =============================================================================
# HANDLE MISSING VALUES
# =============================================================================

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: DataFrame with potential missing values
        strategy: 'median', 'mean', 'zero', or 'drop'

    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values with strategy: {strategy}")

    df = df.copy()

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    elif strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

    elif strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)

    elif strategy == 'drop':
        # Drop rows with any missing values
        df = df.dropna()

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    logger.info(f"Missing values handled. Remaining NaN: {df.isnull().sum().sum()}")
    return df


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================

def build_features(raw_data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    This function executes Task 3 (Feature Engineering):
    1. Load raw data
    2. Extract temporal features
    3. Create aggregate features
    4. Create channel/category features
    5. Create provider/pricing features
    6. Encode categorical features
    7. Handle missing values

    Args:
        raw_data_path: Path to raw transaction CSV
        output_path: Optional path to save processed data

    Returns:
        DataFrame with engineered features (ready for Task 4 - proxy target)
    """
    logger.info("="*60)
    logger.info("TASK 3: FEATURE ENGINEERING PIPELINE")
    logger.info("="*60)

    # Step 1: Load data
    logger.info("\n[Step 1/7] Loading raw data...")
    df = load_raw_data(raw_data_path)

    # Step 2: Temporal features
    logger.info("\n[Step 2/7] Extracting temporal features...")
    df = extract_temporal_features(df)

    # Step 3: Aggregate features
    logger.info("\n[Step 3/7] Creating aggregate features...")
    customer_features = create_aggregate_features(df)

    # Step 4: Channel/Category features
    logger.info("\n[Step 4/7] Creating channel and category features...")
    channel_category = create_channel_category_features(df)
    customer_features = customer_features.merge(channel_category, on='CustomerId', how='left')

    # Step 5: Provider/Pricing features
    logger.info("\n[Step 5/7] Creating provider and pricing features...")
    provider_pricing = create_provider_pricing_features(df)
    customer_features = customer_features.merge(provider_pricing, on='CustomerId', how='left')

    # Step 6: Encode categorical features
    logger.info("\n[Step 6/7] Encoding categorical features...")
    customer_features = encode_categorical_features(df, customer_features)

    # Step 7: Handle missing values
    logger.info("\n[Step 7/7] Handling missing values...")
    customer_features = handle_missing_values(customer_features, strategy='median')

    # Drop non-feature columns (CustomerId, timestamps)
    cols_to_drop = ['CustomerId', 'FirstTransaction', 'LastTransaction']
    customer_features = customer_features.drop(columns=[c for c in cols_to_drop if c in customer_features.columns], errors='ignore')

    logger.info("\n" + "="*60)
    logger.info("TASK 3 COMPLETE: Feature Engineering")
    logger.info(f"Final dataset: {customer_features.shape[0]} customers, {customer_features.shape[1]} features")
    logger.info("="*60)

    # Save if output path provided
    if output_path:
        logger.info(f"Saving processed features to {output_path}")
        customer_features.to_csv(output_path, index=False)

    return customer_features


# =============================================================================
# PROXY TARGET VARIABLE ENGINEERING
# =============================================================================

def calculate_rfm_features(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    TASK 4.1: Calculate RFM (Recency, Frequency, Monetary) features.

    For each CustomerId, calculate:
    - Recency: Days since last transaction
    - Frequency: Number of transactions
    - Monetary: Total transaction amount

    Args:
        df: Transaction DataFrame with CustomerId and TransactionStartTime
        snapshot_date: Reference date for calculating recency.
                       If None, uses max date + 1 day.

    Returns:
        DataFrame with RFM features per customer
    """
    logger.info("Calculating RFM features for proxy target variable")

    # Reload transactions if CustomerId not in the feature df
    # Note: This function should be called on the original transaction data, not features
    transactions = load_raw_data(df) if isinstance(df, str) else df

    if snapshot_date is None:
        snapshot_date = transactions['TransactionStartTime'].max() + pd.Timedelta(days=1)

    logger.info(f"Snapshot date: {snapshot_date}")

    # Calculate Recency (days since last transaction)
    last_tx = transactions.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    last_tx.columns = ['CustomerId', 'LastTransactionDate']
    last_tx['Recency'] = (snapshot_date - last_tx['LastTransactionDate']).dt.days

    # Calculate Frequency (transaction count)
    frequency = transactions.groupby('CustomerId').size().reset_index()
    frequency.columns = ['CustomerId', 'Frequency']

    # Calculate Monetary (total amount)
    monetary = transactions.groupby('CustomerId')['Amount'].sum().reset_index()
    monetary.columns = ['CustomerId', 'Monetary']

    # Combine RFM
    rfm = last_tx.merge(frequency, on='CustomerId').merge(monetary, on='CustomerId')

    # Additional RFM metrics
    avg_amount = transactions.groupby('CustomerId')['Amount'].mean().reset_index()
    avg_amount.columns = ['CustomerId', 'AvgTransactionAmount']

    # Transactions per active day
    tx_per_day = transactions.groupby('CustomerId').apply(
        lambda x: (x['TransactionStartTime'].max() - x['TransactionStartTime'].min()).days + 1
    ).reset_index()
    tx_per_day.columns = ['CustomerId', 'ActiveDays']
    tx_per_day['TransactionsPerDay'] = rfm['Frequency'] / tx_per_day['ActiveDays'].replace(0, 1)

    rfm = rfm.merge(avg_amount, on='CustomerId')
    rfm = rfm.merge(tx_per_day[['CustomerId', 'TransactionsPerDay']], on='CustomerId')

    logger.info(f"Calculated RFM for {len(rfm):,} customers")
    logger.info(f"  - Recency range: {rfm['Recency'].min()} to {rfm['Recency'].max()} days")
    logger.info(f"  - Frequency range: {rfm['Frequency'].min()} to {rfm['Frequency'].max()} transactions")
    logger.info(f"  - Monetary range: {rfm['Monetary'].min():.2f} to {rfm['Monetary'].max():.2f}")

    return rfm


def create_rfm_based_target(rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    TASK 4.2: Create proxy target variable using K-Means clustering on RFM features.

    The assumption is that disengaged customers (low R, F, M) are high credit risk.
    We cluster customers into 3 groups and label the least engaged as high-risk.

    Args:
        rfm_df: DataFrame with RFM features (Recency, Frequency, Monetary)
        n_clusters: Number of clusters (default 3: High/Medium/Low risk)
        random_state: Random state for reproducibility

    Returns:
        DataFrame with cluster assignments and is_high_risk target variable
    """
    logger.info(f"Creating RFM-based proxy target variable using K-Means (k={n_clusters})")

    from sklearn.cluster import KMeans

    df = rfm_df.copy()

    # Prepare features for clustering (use R, F, M only)
    rfm_features = df[['Recency', 'Frequency', 'Monetary']].copy()

    # Log transform for monetary (handle negative values)
    rfm_features['Monetary'] = rfm_features['Monetary'].apply(lambda x: np.log1p(abs(x)) * np.sign(x))

    # Scale features using RobustScaler (handles outliers better)
    scaler = RobustScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    df['RFM_Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Analyze clusters to determine which is high-risk
    cluster_stats = df.groupby('RFM_Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerId': 'count'
    }).rename(columns={'CustomerId': 'Count'})

    logger.info("Cluster analysis:")
    logger.info(f"\n{cluster_stats}")

    # Score clusters: higher score = higher risk
    # Normalize metrics for scoring
    cluster_stats['Recency_norm'] = (cluster_stats['Recency'] - cluster_stats['Recency'].min()) / (cluster_stats['Recency'].max() - cluster_stats['Recency'].min() + 1e-9)
    cluster_stats['Frequency_norm'] = (cluster_stats['Frequency'] - cluster_stats['Frequency'].min()) / (cluster_stats['Frequency'].max() - cluster_stats['Frequency'].min() + 1e-9)
    cluster_stats['Monetary_norm'] = (cluster_stats['Monetary'] - cluster_stats['Monetary'].min()) / (cluster_stats['Monetary'].max() - cluster_stats['Monetary'].min() + 1e-9)

    # Risk score: high recency (inactive) + low frequency + low monetary = high risk
    cluster_stats['RiskScore'] = cluster_stats['Recency_norm'] - (cluster_stats['Frequency_norm'] + cluster_stats['Monetary_norm']) / 2

    # Identify high-risk cluster (highest risk score)
    high_risk_cluster = cluster_stats['RiskScore'].idxmax()
    logger.info(f"High-risk cluster identified: Cluster {high_risk_cluster}")

    # Create binary target
    df['is_high_risk'] = (df['RFM_Cluster'] == high_risk_cluster).astype(int)

    # Log target distribution
    target_counts = df['is_high_risk'].value_counts()
    logger.info(f"Target distribution:")
    logger.info(f"  - High Risk (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0) / len(df) * 100:.2f}%)")
    logger.info(f"  - Low Risk (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0) / len(df) * 100:.2f}%)")

    return df


def build_complete_pipeline(raw_data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Complete pipeline: Task 3 (Feature Engineering) + Task 4 (Proxy Target)

    Executes:
    1. Task 3: Feature engineering from raw transactions
    2. Task 4: RFM calculation and K-Means clustering for proxy target

    Args:
        raw_data_path: Path to raw transaction CSV
        output_path: Optional path to save final dataset with target

    Returns:
        DataFrame with features AND is_high_risk target variable
    """
    logger.info("="*60)
    logger.info("COMPLETE PIPELINE: Tasks 3 + 4")
    logger.info("Feature Engineering + Proxy Target Variable")
    logger.info("="*60)

    # Step 1: Task 3 - Feature Engineering
    logger.info("\n" + "="*40)
    logger.info("TASK 3: FEATURE ENGINEERING")
    logger.info("="*40)

    # Load raw data first
    df = load_raw_data(raw_data_path)

    # Create temporal features
    df = extract_temporal_features(df)

    # Create aggregate features
    customer_features = create_aggregate_features(df)

    # Create channel/category features
    channel_category = create_channel_category_features(df)
    customer_features = customer_features.merge(channel_category, on='CustomerId', how='left')

    # Create provider/pricing features
    provider_pricing = create_provider_pricing_features(df)
    customer_features = customer_features.merge(provider_pricing, on='CustomerId', how='left')

    # Encode categorical features
    customer_features = encode_categorical_features(df, customer_features)

    # Step 2: Task 4 - Proxy Target Variable
    logger.info("\n" + "="*40)
    logger.info("TASK 4: PROXY TARGET VARIABLE (RFM Clustering)")
    logger.info("="*40)

    # Calculate RFM features using original transaction data
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm_features = calculate_rfm_features(df, snapshot_date)

    # Create proxy target via K-Means clustering
    rfm_with_target = create_rfm_based_target(rfm_features)

    # Merge RFM features and target into customer features
    customer_id_col = customer_features['CustomerId'] if 'CustomerId' in customer_features.columns else None

    # Get RFM and target columns (excluding duplicate CustomerId)
    rfm_cols = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'AvgTransactionAmount', 'TransactionsPerDay', 'RFM_Cluster', 'is_high_risk']
    rfm_and_target = rfm_with_target[[c for c in rfm_cols if c in rfm_with_target.columns]]

    customer_features = customer_features.merge(rfm_and_target, on='CustomerId', how='left')

    # Handle missing values
    customer_features = handle_missing_values(customer_features, strategy='median')

    # Create additional derived features
    customer_features['AmountToIncomeRatio'] = customer_features['TotalAmount'] / (customer_features['TransactionCount'] + 1)
    customer_features['StdToMeanRatio'] = customer_features['StdAmount'] / (customer_features['AvgAmount'].abs() + 1)

    # Drop non-feature columns
    cols_to_drop = ['CustomerId', 'FirstTransaction', 'LastTransaction']
    customer_features = customer_features.drop(columns=[c for c in cols_to_drop if c in customer_features.columns], errors='ignore')

    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE: Tasks 3 + 4")
    logger.info(f"Final dataset: {customer_features.shape[0]} customers, {customer_features.shape[1]} features")
    logger.info(f"Target distribution: {customer_features['is_high_risk'].value_counts().to_dict()}")
    logger.info("="*60)

    # Save if output path provided
    if output_path:
        logger.info(f"Saving final dataset to {output_path}")
        customer_features.to_csv(output_path, index=False)

    return customer_features


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_names(df: pd.DataFrame) -> list:
    """Get list of feature names (excluding target)."""
    return [col for col in df.columns if col != 'is_high_risk']


def get_target_name() -> str:
    """Get the target variable name."""
    return 'is_high_risk'


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).

    Args:
        df: Processed DataFrame with features and target

    Returns:
        X: DataFrame with feature columns only
        y: Series with target variable
    """
    target_col = get_target_name()
    feature_cols = get_feature_names(df)

    X = df[feature_cols]
    y = df[target_col]

    return X, y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Credit Risk Feature Engineering')
    parser.add_argument(
    '--data',
    type=str,
    default='data/raw/data.csv',
    help='Path to raw transaction data'
)
    parser.add_argument('--output', type=str, default='data/processed/processed_customers.csv', help='Path to save processed data')

    args = parser.parse_args()

    # Run complete pipeline (Tasks 3 + 4)
    result = build_complete_pipeline(args.data, args.output)

    print(f"\nFinal dataset shape: {result.shape}")
    print(f"Target distribution:\n{result['is_high_risk'].value_counts()}")