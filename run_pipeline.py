"""
Credit Risk Model - Data Pipeline Runner
========================================
Executes the full data processing pipeline to generate training data.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from paths import RAW_DATA_PATH, PROCESSED_TRAIN_PATH
from data_processing import (
    load_raw_data,
    extract_temporal_features,
    create_aggregate_features,
    create_channel_category_features,
    create_provider_pricing_features,
    encode_categorical_features,
    handle_missing_values,
    calculate_rfm_features,
    create_rfm_based_target,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = PROCESSED_TRAIN_PATH


def run_pipeline():
    """Run the complete data processing pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING DATA PROCESSING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load raw data
    logger.info("\n[Step 1/8] Loading raw data...")
    df = load_raw_data(str(RAW_DATA_PATH))

    # Step 2: Extract temporal features
    logger.info("\n[Step 2/8] Extracting temporal features...")
    df = extract_temporal_features(df)

    # Step 3: Create aggregate features
    logger.info("\n[Step 3/8] Creating aggregate features...")
    customer_features = create_aggregate_features(df)

    # Step 4: Create channel/category features
    logger.info("\n[Step 4/8] Creating channel and category features...")
    channel_category = create_channel_category_features(df)
    customer_features = customer_features.merge(channel_category, on='CustomerId', how='left')

    # Step 5: Create provider/pricing features
    logger.info("\n[Step 5/8] Creating provider and pricing features...")
    provider_pricing = create_provider_pricing_features(df)
    customer_features = customer_features.merge(provider_pricing, on='CustomerId', how='left')

    # Step 6: Encode categorical features
    logger.info("\n[Step 6/8] Encoding categorical features...")
    customer_features = encode_categorical_features(df, customer_features)

    # Step 7: Calculate RFM features for target variable
    logger.info("\n[Step 7/8] Calculating RFM features for target variable...")
    rfm_features = calculate_rfm_features(df)
    customer_features = customer_features.merge(rfm_features, on='CustomerId', how='left')

    # Add derived features before creating target
    logger.info("Adding derived features...")
    customer_features['AmountToIncomeRatio'] = customer_features['AvgAmount'] / (customer_features['AvgAmount'].mean() + 1)
    customer_features['StdToMeanRatio'] = customer_features['StdAmount'] / (customer_features['AvgAmount'] + 1)

    # Step 8: Create RFM-based target variable
    logger.info("\n[Step 8/8] Creating RFM-based target variable...")
    customer_features = create_rfm_based_target(customer_features)

    # Handle missing values
    logger.info("\n[Final] Handling missing values...")
    customer_features = handle_missing_values(customer_features, strategy='median')

    # Drop non-feature columns
    cols_to_drop = ['CustomerId', 'FirstTransaction', 'LastTransaction', 
                    'LastTransactionDate', 'RFM_Cluster']
    customer_features = customer_features.drop(
        columns=[c for c in cols_to_drop if c in customer_features.columns],
        errors='ignore'
    )

    # Save processed data
    logger.info(f"\nSaving processed data to {PROCESSED_DATA_PATH}")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    customer_features.to_csv(PROCESSED_DATA_PATH, index=False)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Dataset shape: {customer_features.shape}")
    logger.info(f"Target distribution:")
    logger.info(f"  - is_high_risk=0 (Low Risk): {(customer_features['is_high_risk'] == 0).sum()}")
    logger.info(f"  - is_high_risk=1 (High Risk): {(customer_features['is_high_risk'] == 1).sum()}")
    logger.info(f"Features: {len(customer_features.columns) - 1}")
    logger.info("=" * 60)

    return customer_features


if __name__ == '__main__':
    df = run_pipeline()
    print(f"\nSample of processed data:")
    print(df.head())
