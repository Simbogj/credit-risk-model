"""
Credit Risk Model - Model Training Pipeline
============================================
Model training, hyperparameter tuning, and MLflow experiment tracking.
Implements TASK 5: Model Training and Tracking.

Author: Bati Bank Analytics Team
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

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

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)  # Will use local storage if not set

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_processed_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data and split into features and target.

    Args:
        filepath: Path to processed CSV file

    Returns:
        X: Features DataFrame
        y: Target Series
    """
    logger.info(f"Loading processed data from {filepath}")
    df = pd.read_csv(filepath)

    target_col = 'is_high_risk'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


# =============================================================================
# SECTION 2: DATA PREPARATION
# =============================================================================

def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets with stratification.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data: {test_size*100:.0f}% for testing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]:,} samples")
    logger.info(f"Test set: {X_test.shape[0]:,} samples")
    logger.info(f"Training target distribution: {y_train.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 3: MODEL DEFINITIONS
# =============================================================================

def get_models() -> Dict[str, Any]:
    """
    Get dictionary of candidate models with their hyperparameter grids.

    Returns:
        Dictionary of model name -> (model instance, param grid)
    """
    models = {
        'LogisticRegression': (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l2'],
                'classifier__class_weight': ['balanced', None],
            }
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': ['balanced', None],
            }
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__class_weight': ['balanced', None],
            }
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
            }
        ),
    }

    return models


# =============================================================================
# SECTION 4: PIPELINE CREATION
# =============================================================================

def create_pipeline(model) -> Pipeline:
    """
    Create a sklearn Pipeline with preprocessing and model.

    The pipeline:
    1. Imputes missing values with median
    2. Scales features using StandardScaler
    3. Applies the classifier

    Args:
        model: sklearn classifier

    Returns:
        sklearn Pipeline
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model),
    ])

    return pipeline


# =============================================================================
# SECTION 5: MODEL TRAINING WITH MLFLOW
# =============================================================================

def train_and_evaluate(
    model_name: str,
    model,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: int = 5,
    search_method: str = 'random',
    n_iter: int = 20,
    experiment_name: str = 'CreditRiskModel'
) -> Tuple[Optional[Pipeline], Dict[str, Any]]:
    """
    Train a model with hyperparameter tuning and log to MLflow.

    Args:
        model_name: Name of the model
        model: sklearn classifier
        param_grid: Hyperparameter grid
        X_train, X_test, y_train, y_test: Data splits
        cv: Number of cross-validation folds
        search_method: 'grid' or 'random' search
        n_iter: Number of iterations for random search
        experiment_name: MLflow experiment name

    Returns:
        best_pipeline: Best trained pipeline
        best_params: Best hyperparameters
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Create pipeline
    pipeline = create_pipeline(model)

    # Select search method
    if search_method == 'grid':
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )

    # Start MLflow run
    with mlflow.start_run(run_name=model_name) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Train model
        search.fit(X_train, y_train)

        # Get best model
        best_pipeline = search.best_estimator_
        best_params = search.best_params_

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score (ROC-AUC): {search.best_score_:.4f}")

        # Predictions
        y_pred = best_pipeline.predict(X_test)
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }

        # Log parameters
        mlflow.log_params({
            'model_name': model_name,
            'best_params': str(best_params),
            'cv_folds': cv,
            'search_method': search_method,
        })

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        # Log model
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path=f"{model_name}_model",
            signature=infer_signature(X_train, y_train)
        )

        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for class_key, class_values in report.items():
            if isinstance(class_values, dict):
                for metric_key, metric_value in class_values.items():
                    mlflow.log_metric(f"{class_key}_{metric_key}", metric_value)

        # Confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1'])
        cm_path = MODEL_DIR / f"{model_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=True)
        mlflow.log_artifact(str(cm_path))

        logger.info(f"\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

    return best_pipeline, metrics


# =============================================================================
# SECTION 6: COMPARE AND SELECT BEST MODEL
# =============================================================================

def compare_models(results: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, Any]]:
    """
    Compare all trained models and select the best based on ROC-AUC.

    Args:
        results: Dictionary of model_name -> metrics

    Returns:
        best_model_name: Name of the best model
        best_metrics: Metrics of the best model
    """
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)

    # Create comparison DataFrame
    comparison = []
    for model_name, metrics in results.items():
        row = {'Model': model_name, **metrics}
        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

    logger.info(f"\n{comparison_df.to_string(index=False)}")

    # Select best model (by ROC-AUC)
    best_model_name = comparison_df.iloc[0]['Model']
    best_metrics = comparison_df.iloc[0].to_dict()
    best_metrics.pop('Model', None)

    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")

    return best_model_name, best_metrics


# =============================================================================
# SECTION 7: REGISTER BEST MODEL
# =============================================================================

def register_best_model(
    model_name: str,
    pipeline,
    metrics: Dict[str, float],
    model_registry_name: str = 'CreditRiskModel'
) -> str:
    """
    Register the best model in MLflow Model Registry.

    Args:
        model_name: Name of the model
        pipeline: Trained sklearn pipeline
        metrics: Dictionary of evaluation metrics
        model_registry_name: Name for the model registry

    Returns:
        model_version: Registered model version
    """
    logger.info("\n" + "="*60)
    logger.info("REGISTERING BEST MODEL")
    logger.info("="*60)

    # Set registry
    mlflow.set_registry_uri(mlflow.tracking.get_tracking_uri())

    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}_model"

    try:
        # Try to get existing model version
        client = mlflow.MlflowClient()
        try:
            latest_version = client.get_latest_versions(name=model_registry_name, stages=["Production", "Staging"])
            version_num = len(latest_version) + 1
        except Exception:
            version_num = 1

        # Register new model
        model_version = mlflow.register_model(model_uri, model_registry_name)

        logger.info(f"Registered model: {model_registry_name}")
        logger.info(f"Version: {model_version.version}")
        logger.info(f"Stage: {model_version.current_stage}")

        # Transition to Production
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_registry_name,
            version=model_version.version,
            stage="Production"
        )

        logger.info(f"Model transitioned to Production")

        return model_version.version

    except Exception as e:
        logger.warning(f"Could not register model: {e}")
        logger.info("Saving model locally instead")

        # Save locally
        model_path = MODEL_DIR / f"{model_name}_best.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to {model_path}")

        return None


# =============================================================================
# SECTION 8: MAIN TRAINING PIPELINE
# =============================================================================

def run_training_pipeline(
    data_path: str,
    experiment_name: str = 'CreditRiskModel',
    test_size: float = 0.2,
    cv: int = 5,
    search_method: str = 'random',
    n_iter: int = 20
) -> Dict[str, Any]:
    """
    Execute the full training pipeline.

    Args:
        data_path: Path to processed CSV
        experiment_name: MLflow experiment name
        test_size: Test set proportion
        cv: Cross-validation folds
        search_method: 'grid' or 'random' hyperparameter search
        n_iter: Number of iterations for random search

    Returns:
        Dictionary with training results and best model info
    """
    logger.info("="*60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("="*60)

    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Load data
    X, y = load_processed_data(data_path)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=test_size)

    # Get models
    models = get_models()

    # Train and evaluate each model
    results = {}
    trained_pipelines = {}

    for model_name, (model, param_grid) in models.items():
        try:
            pipeline, metrics = train_and_evaluate(
                model_name=model_name,
                model=model,
                param_grid=param_grid,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                cv=cv,
                search_method=search_method,
                n_iter=n_iter,
                experiment_name=experiment_name
            )

            results[model_name] = metrics
            trained_pipelines[model_name] = pipeline

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue

    # Compare models
    best_model_name, best_metrics = compare_models(results)

    # Register best model
    best_pipeline = trained_pipelines[best_model_name]
    model_version = register_best_model(best_model_name, best_pipeline, best_metrics)

    # Save summary
    summary = {
        'best_model_name': best_model_name,
        'best_metrics': best_metrics,
        'model_version': model_version,
        'all_results': results,
        'comparison': pd.DataFrame([
            {'Model': k, **v} for k, v in results.items()
        ]).sort_values('roc_auc', ascending=False).to_dict('records')
    }

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    logger.info("="*60)

    return summary


# =============================================================================
# SECTION 9: STANDALONE EXECUTION
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Credit Risk Models')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to processed CSV data')
    parser.add_argument('--experiment', type=str, default='CreditRiskModel',
                        help='MLflow experiment name')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--cv', type=int, default=5,
                        help='Cross-validation folds')
    parser.add_argument('--search', type=str, default='random',
                        choices=['grid', 'random'],
                        help='Hyperparameter search method')
    parser.add_argument('--n-iter', type=int, default=20,
                        help='Number of iterations for random search')

    args = parser.parse_args()

    # Default data path
    if args.data is None:
        possible_paths = [
            DATA_DIR / "processed_customers.csv",
            PROJECT_ROOT / "data" / "processed_customers.csv",
        ]

        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = str(path)
                break

        if data_path is None:
            logger.error("No processed data found. Run data_processing.py first or provide --data path")
            sys.exit(1)
    else:
        data_path = args.data

    # Run training
    results = run_training_pipeline(
        data_path=data_path,
        experiment_name=args.experiment,
        test_size=args.test_size,
        cv=args.cv,
        search_method=args.search,
        n_iter=args.n_iter
    )

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Metrics: {results['best_metrics']}")