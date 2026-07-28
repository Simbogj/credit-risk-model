"""
Credit Risk Model - Simplified Training Pipeline
================================================
Train models and save locally without MLflow registry issues.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(filepath):
    """Load processed data."""
    df = pd.read_csv(filepath)
    target_col = 'is_high_risk'
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def create_pipeline(model):
    """Create sklearn pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model),
    ])


def train_and_evaluate(model_name, model, param_grid, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    
    # Randomized search
    search = RandomizedSearchCV(
        create_pipeline(model),
        param_grid,
        n_iter=5,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    
    # Predictions
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }
    
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return best_pipeline, metrics


def main():
    # Load data
    logger.info("Loading data...")
    X, y = load_data(DATA_PATH)
    logger.info(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Models
    models = {
        'LogisticRegression': (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {'classifier__C': [0.01, 0.1, 1.0], 'classifier__class_weight': ['balanced']}
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {'classifier__max_depth': [3, 5, 7], 'classifier__class_weight': ['balanced']}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1),
            {'classifier__max_depth': [5, 10], 'classifier__min_samples_split': [2, 5]}
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100),
            {'classifier__max_depth': [3, 5], 'classifier__learning_rate': [0.1, 0.2]}
        ),
    }
    
    # Train all models
    results = {}
    trained_pipelines = {}
    
    for name, (model, params) in models.items():
        try:
            pipeline, metrics = train_and_evaluate(name, model, params, X_train, X_test, y_train, y_test)
            results[name] = metrics
            trained_pipelines[name] = pipeline
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    
    # Compare and select best
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    comparison = pd.DataFrame([{'Model': k, **v} for k, v in results.items()])
    comparison = comparison.sort_values('roc_auc', ascending=False)
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    # Save best model
    best_model_name = comparison.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = {k: v for k, v in comparison.iloc[0].items() if k != 'Model'}
    
    model_path = MODEL_DIR / f"{best_model_name}_best.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"Saved to: {model_path}")
    
    # Save feature names
    feature_names_path = MODEL_DIR / "feature_names.joblib"
    joblib.dump(list(X.columns), feature_names_path)
    
    # Save comparison results
    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    
    # Save metrics summary
    metrics_summary = {
        'best_model': best_model_name,
        'best_roc_auc': best_metrics['roc_auc'],
        'best_accuracy': best_metrics['accuracy'],
        'best_precision': best_metrics['precision'],
        'best_recall': best_metrics['recall'],
        'best_f1': best_metrics['f1'],
    }
    
    import json
    metrics_path = MODEL_DIR / "metrics_summary.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    logger.info("="*60)
    
    return best_model_name, best_metrics


if __name__ == '__main__':
    main()
