"""
Credit Risk Model - Inference Pipeline
=======================================
Load trained models and make predictions on new data.

Author: Bati Bank Analytics Team
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


# =============================================================================
# SECTION 1: MODEL LOADING
# =============================================================================

def load_model_from_registry(
    model_name: str = 'CreditRiskModel',
    stage: str = 'Production',
    version: Optional[int] = None
) -> Pipeline:
    """
    Load a trained model from MLflow Model Registry or local storage.

    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, etc.)
        version: Specific model version (if None, uses latest)

    Returns:
        Loaded sklearn Pipeline
    """
    logger.info(f"Loading model: {model_name} (stage={stage}, version={version})")

    # Try MLflow Registry first
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

        client = mlflow.MlflowClient()

        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"

        logger.info(f"Loading from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded from MLflow Registry successfully")
        return model

    except Exception as e:
        logger.warning(f"Could not load from MLflow: {e}")
        logger.info("Trying local storage...")

    # Fallback to local storage
    local_model_path = MODEL_DIR / f"{model_name}_best.joblib"

    if local_model_path.exists():
        logger.info(f"Loading from local path: {local_model_path}")
        model = joblib.load(local_model_path)
        logger.info("Model loaded from local storage successfully")
        return model

    raise FileNotFoundError(f"Model not found: {model_name}")


def load_model_local(model_path: str) -> Pipeline:
    """
    Load a model from a local file path.

    Args:
        model_path: Path to the model file (.joblib)

    Returns:
        Loaded sklearn Pipeline
    """
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model


# =============================================================================
# SECTION 2: PREDICTION
# =============================================================================

def predict(
    model: Pipeline,
    X: Union[pd.DataFrame, np.ndarray],
    return_proba: bool = True
) -> Dict[str, Any]:
    """
    Make predictions using a trained model.

    Args:
        model: Trained sklearn Pipeline
        X: Features (DataFrame or numpy array)
        return_proba: Whether to return probability scores

    Returns:
        Dictionary with predictions and probabilities
    """
    logger.info(f"Making predictions on {X.shape[0] if hasattr(X, 'shape') else len(X)} samples")

    # Ensure numpy array
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # Predictions
    y_pred = model.predict(X_array)

    result = {
        'predictions': y_pred,
        'n_predictions': len(y_pred),
    }

    if return_proba:
        y_proba = model.predict_proba(X_array)
        result['probabilities'] = y_proba
        result['risk_probability'] = y_proba[:, 1]
        result['confidence'] = np.max(y_proba, axis=1)

    return result


def predict_single(
    model: Pipeline,
    features: Dict[str, float],
    feature_names: list
) -> Dict[str, Any]:
    """
    Make prediction for a single customer.

    Args:
        model: Trained sklearn Pipeline
        features: Dictionary of feature name -> value
        feature_names: Ordered list of feature names expected by model

    Returns:
        Dictionary with prediction and probability
    """
    X = np.array([[features.get(f, 0) for f in feature_names]])

    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]

    result = {
        'predicted_class': int(y_pred),
        'risk_probability': float(y_proba[1]),
        'confidence': float(max(y_proba)),
    }

    if result['risk_probability'] < 0.3:
        result['risk_level'] = 'Low'
    elif result['risk_probability'] < 0.6:
        result['risk_level'] = 'Medium'
    else:
        result['risk_level'] = 'High'

    result['credit_score'] = int(300 + (1 - result['risk_probability']) * 550)

    return result


# =============================================================================
# SECTION 3: CREDIT SCORE CALCULATION
# =============================================================================

def calculate_credit_score(
    risk_probability: float,
    min_score: int = 300,
    max_score: int = 850
) -> int:
    """
    Convert risk probability to credit score.

    Uses inverse scaling (lower risk = higher score)
    Similar to FICO score range (300-850)

    Args:
        risk_probability: Probability of default (0-1)
        min_score: Minimum credit score
        max_score: Maximum credit score

    Returns:
        Credit score (integer)
    """
    score = int(min_score + (1 - risk_probability) * (max_score - min_score))
    return score


def get_credit_decision(
    risk_probability: float,
    threshold_low: float = 0.3,
    threshold_high: float = 0.6
) -> Dict[str, Any]:
    """
    Determine credit decision based on risk probability.

    Args:
        risk_probability: Probability of default
        threshold_low: Below this = Approve
        threshold_high: Above this = Decline

    Returns:
        Dictionary with decision details
    """
    if risk_probability <= threshold_low:
        decision = 'Approve'
        terms = {
            'max_amount': 10000,
            'max_duration_months': 12,
            'interest_rate': 0.12
        }
    elif risk_probability <= threshold_high:
        decision = 'Review'
        terms = {
            'max_amount': 5000,
            'max_duration_months': 6,
            'interest_rate': 0.18
        }
    else:
        decision = 'Decline'
        terms = {
            'max_amount': 0,
            'max_duration_months': 0,
            'interest_rate': None
        }

    credit_score = calculate_credit_score(risk_probability)

    return {
        'decision': decision,
        'risk_probability': risk_probability,
        'credit_score': credit_score,
        'recommended_terms': terms
    }


# =============================================================================
# SECTION 4: BATCH PREDICTION
# =============================================================================

def predict_batch(
    model: Pipeline,
    data_path: str,
    output_path: Optional[str] = None,
    return_proba: bool = True
) -> pd.DataFrame:
    """
    Make predictions on a batch of data from CSV.

    Args:
        model: Trained sklearn Pipeline
        data_path: Path to input CSV
        output_path: Optional path to save results
        return_proba: Whether to include probabilities

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    id_cols = ['CustomerId', 'Customer_ID', 'customer_id', 'Id', 'ID']
    id_col = None

    for col in id_cols:
        if col in df.columns:
            id_col = col
            break

    if id_col:
        identifiers = df[id_col]
        features = df.drop(columns=[id_col])
    else:
        identifiers = pd.Series(range(len(df)), name='Index')
        features = df

    results = predict(model, features, return_proba=return_proba)

    results_df = pd.DataFrame({
        'CustomerId': identifiers if id_col else identifiers,
        'PredictedRisk': results['predictions'],
        'RiskProbability': results['risk_probability'],
        'Confidence': results['confidence'],
    })

    results_df['CreditScore'] = results_df['RiskProbability'].apply(calculate_credit_score)

    def get_risk_level(prob):
        if prob < 0.3:
            return 'Low'
        elif prob < 0.6:
            return 'Medium'
        else:
            return 'High'

    results_df['RiskLevel'] = results_df['RiskProbability'].apply(get_risk_level)

    if output_path:
        logger.info(f"Saving predictions to {output_path}")
        results_df.to_csv(output_path, index=False)

    return results_df


# =============================================================================
# SECTION 5: FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model: Pipeline,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained sklearn Pipeline
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance scores
    """
    logger.info("Extracting feature importance")

    classifier = model.named_steps['classifier']

    importance_df = pd.DataFrame({'Feature': feature_names})

    if hasattr(classifier, 'feature_importances_'):
        importance_df['Importance'] = classifier.feature_importances_
        importance_df = importance_df.sort_values('Importance', ascending=False)
    elif hasattr(classifier, 'coef_'):
        importance_df['Importance'] = np.abs(classifier.coef_[0])
        importance_df = importance_df.sort_values('Importance', ascending=False)
    else:
        logger.warning("Model does not support feature importance extraction")
        importance_df['Importance'] = 0.0

    return importance_df.reset_index(drop=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Credit Risk Model Inference')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file or model name in registry')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions')
    parser.add_argument('--registry', action='store_true',
                        help='Load from MLflow registry')

    args = parser.parse_args()

    if args.registry:
        model = load_model_from_registry(args.model or 'CreditRiskModel')
    elif args.model:
        model = load_model_local(args.model)
    else:
        possible_paths = [
            MODEL_DIR / "CreditRiskModel_best.joblib",
            PROJECT_ROOT / "models" / "CreditRiskModel_best.joblib",
        ]

        model = None
        for path in possible_paths:
            if path.exists():
                model = load_model_local(str(path))
                break

        if model is None:
            print("Error: No model found. Please specify --model path")
            exit(1)

    results = predict_batch(model, args.data, args.output)

    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Total predictions: {len(results)}")
    print(f"\nRisk distribution:")
    print(results['RiskLevel'].value_counts())
    print(f"\nSample predictions:")
    print(results.head(10))