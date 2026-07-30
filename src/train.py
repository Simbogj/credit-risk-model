"""
Credit Risk Model - Model Training Pipeline
==========================================
Model training, hyperparameter tuning, and MLflow experiment tracking.
Implements formal sklearn Pipeline with WoE transformation.
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
import joblib

# Import from data_processing
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import (
    WoETransformer,
    CategoricalWoETransformer,
    get_feature_names,
    get_target_name,
    RANDOM_STATE
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)  # Will use local storage if not set

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# : DATA LOADING
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

    target_col = get_target_name()
    feature_cols = get_feature_names(df)

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


# =============================================================================
# DATA PREPARATION
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
# SKLEARN PIPELINE WITH WOE
# =============================================================================

class CreditScoringPipeline(BaseEstimator, ClassifierMixin):
    """
    Complete Credit Scoring Pipeline with WoE transformation.
    
    This pipeline implements the full credit scoring workflow:
    1. WoE transformation for numerical features
    2. Imputation of missing values
    3. Scaling of features
    4. Classification model
    
    The pipeline is fully sklearn-compatible and can be used with
    GridSearchCV, cross_val_score, etc.
    """
    
    def __init__(
        self,
        model_type: str = 'gradient_boosting',
        apply_woe: bool = True,
        woe_bins: int = 10,
        impute_strategy: str = 'median',
        scale_method: str = 'robust',
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = RANDOM_STATE
    ):
        self.model_type = model_type
        self.apply_woe = apply_woe
        self.woe_bins = woe_bins
        self.impute_strategy = impute_strategy
        self.scale_method = scale_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # These will be set during fit
        self.woe_transformer_ = None
        self.scaler_ = None
        self.model_ = None
        self.numerical_features_ = None
        self.categorical_features_ = None
        self.classes_ = None
        self._pipeline = None
    
    def _get_model(self):
        """Create the underlying model based on model_type."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=self.max_depth,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate
            )
        else:
            return GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate
            )
    
    def _get_scaler(self):
        """Get the scaler based on scale_method."""
        if self.scale_method == 'standard':
            return StandardScaler()
        else:
            return RobustScaler()
    
    def _identify_features(self, X: pd.DataFrame):
        """Identify numerical and categorical features."""
        self.numerical_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Identified {len(self.numerical_features_)} numerical, {len(self.categorical_features_)} categorical features")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CreditScoringPipeline':
        """
        Fit the complete pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target variable
        
        Returns:
            self
        """
        self._identify_features(X)
        self.classes_ = np.unique(y)
        
        # Step 1: Apply WoE transformation if requested
        if self.apply_woe and self.numerical_features_:
            logger.info("Applying WoE transformation...")
            self.woe_transformer_ = WoETransformer(
                variables=self.numerical_features_,
                bins=self.woe_bins,
                strategy='quantile',
                smoothing=0.5
            )
            X_transformed = self.woe_transformer_.fit_transform(X, y)
            # Update numerical features list after WoE (same column names)
        else:
            self.woe_transformer_ = None
            X_transformed = X.copy()
        
        # Step 2: Impute missing values
        logger.info(f"Imputing missing values with {self.impute_strategy}...")
        imputer = SimpleImputer(strategy=self.impute_strategy)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_transformed),
            columns=X_transformed.columns,
            index=X_transformed.index
        )
        
        # Step 3: Scale features
        logger.info(f"Scaling features with {self.scale_method}...")
        self.scaler_ = self._get_scaler()
        X_scaled = pd.DataFrame(
            self.scaler_.fit_transform(X_imputed[self.numerical_features_]),
            columns=self.numerical_features_,
            index=X_imputed.index
        )
        
        # Add categorical features if any
        if self.categorical_features_:
            for col in self.categorical_features_:
                X_scaled[col] = X_imputed[col].values
        
        # Step 4: Fit the model
        logger.info(f"Training {self.model_type} model...")
        self.model_ = self._get_model()
        self.model_.fit(X_scaled, y)
        
        logger.info("Pipeline fitting complete")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predicted class labels
        """
        X_transformed = self._transform(X)
        return self.model_.predict(X_transformed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of class probabilities
        """
        X_transformed = self._transform(X)
        return self.model_.predict_proba(X_transformed)
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data through the pipeline."""
        if self.woe_transformer_ is not None:
            X = self.woe_transformer_.transform(X)
        
        imputer = SimpleImputer(strategy=self.impute_strategy)
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        X_scaled = pd.DataFrame(
            self.scaler_.transform(X[self.numerical_features_]),
            columns=self.numerical_features_,
            index=X.index
        )
        
        if self.categorical_features_:
            for col in self.categorical_features_:
                X_scaled[col] = X[col].values
        
        return X_scaled
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Return ROC-AUC score."""
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)
    
    def get_woe_iv_summary(self) -> Optional[pd.DataFrame]:
        """Get WoE/IV summary if WoE was applied."""
        if self.woe_transformer_ is not None:
            return self.woe_transformer_.get_iv_summary()
        return None


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get dictionary of model configurations for comparison.
    
    Each config includes the pipeline class and hyperparameter grid.

    Returns:
        Dictionary of model name -> (config dict)
    """
    configs = {
        'LogisticRegression': {
            'pipeline': CreditScoringPipeline(model_type='logistic_regression'),
            'params': {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__class_weight': ['balanced', None],
            }
        },
        'DecisionTree': {
            'pipeline': CreditScoringPipeline(model_type='decision_tree'),
            'params': {
                'model__max_depth': [3, 5, 7, 10],
                'model__min_samples_split': [2, 5, 10],
            }
        },
        'RandomForest': {
            'pipeline': CreditScoringPipeline(model_type='random_forest'),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 15],
            }
        },
        'GradientBoosting': {
            'pipeline': CreditScoringPipeline(model_type='gradient_boosting'),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
            }
        },
        'GradientBoosting_WoE': {
            'pipeline': CreditScoringPipeline(model_type='gradient_boosting', apply_woe=True),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.05, 0.1],
                'woe_bins': [5, 10, 20],
            }
        },
    }

    return configs


def get_pipeline_with_hyperparameters(
    model_type: str = 'gradient_boosting',
    apply_woe: bool = True,
    random_state: int = RANDOM_STATE
) -> CreditScoringPipeline:
    """
    Create a CreditScoringPipeline with specified configuration.

    Args:
        model_type: Type of model to use
        apply_woe: Whether to apply WoE transformation
        random_state: Random state for reproducibility

    Returns:
        Configured CreditScoringPipeline
    """
    return CreditScoringPipeline(
        model_type=model_type,
        apply_woe=apply_woe,
        random_state=random_state
    )


# =============================================================================
# MODEL TRAINING WITH MLFLOW
# =============================================================================

def train_and_evaluate(
    model_name: str,
    pipeline: CreditScoringPipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: int = 5,
    search_method: str = 'random',
    n_iter: int = 20,
    experiment_name: str = 'CreditRiskModel'
) -> Tuple[Optional[CreditScoringPipeline], Dict[str, Any]]:
    """
    Train a model with hyperparameter tuning and log to MLflow.

    Args:
        model_name: Name of the model
        pipeline: CreditScoringPipeline instance
        param_grid: Hyperparameter grid
        X_train, X_test, y_train, y_test: Data splits
        cv: Number of cross-validation folds
        search_method: 'grid' or 'random' search
        n_iter: Number of iterations for random search
        experiment_name: MLflow experiment name

    Returns:
        best_pipeline: Best trained pipeline
        metrics: Dictionary of evaluation metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

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

        try:
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
                'model_type': best_pipeline.model_type,
                'apply_woe': best_pipeline.apply_woe,
                'best_params': str(best_params),
                'cv_folds': cv,
                'search_method': search_method,
            })

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"  {metric_name}: {metric_value:.4f}")

            # Log WoE/IV summary if available
            woe_iv_summary = best_pipeline.get_woe_iv_summary()
            if woe_iv_summary is not None:
                woe_path = MODEL_DIR / f"{model_name}_woe_iv.csv"
                woe_iv_summary.to_csv(woe_path, index=False)
                mlflow.log_artifact(str(woe_path))
                logger.info("Logged WoE/IV summary")

            # Log model using sklearn's log_model
            mlflow.sklearn.log_model(
                best_pipeline,
                artifact_path=f"{model_name}_model",
                signature=infer_signature(X_train.head(100), best_pipeline.predict(X_train.head(100)))
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

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    return best_pipeline, metrics


# =============================================================================
# COMPARE AND SELECT BEST MODEL
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
# REGISTER BEST MODEL
# =============================================================================

def register_best_model(
    model_name: str,
    pipeline: CreditScoringPipeline,
    metrics: Dict[str, float],
    model_registry_name: str = 'CreditRiskModel'
) -> Optional[str]:
    """
    Register the best model in MLflow Model Registry.

    Args:
        model_name: Name of the model
        pipeline: Trained sklearn pipeline
        metrics: Dictionary of evaluation metrics
        model_registry_name: Name for the model registry

    Returns:
        model_version: Registered model version or None if failed
    """
    logger.info("\n" + "="*60)
    logger.info("REGISTERING BEST MODEL")
    logger.info("="*60)

    # Set registry
    mlflow.set_registry_uri(mlflow.tracking.get_tracking_uri())

    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}_model"

    try:
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

        # Save feature names for later use
        feature_names_path = MODEL_DIR / f"{model_registry_name}_feature_names.joblib"
        joblib.dump(pipeline.numerical_features_, feature_names_path)
        logger.info(f"Saved feature names to {feature_names_path}")

        return str(model_version.version)

    except Exception as e:
        logger.warning(f"Could not register model: {e}")
        logger.info("Saving model locally instead")

        # Save locally
        model_path = MODEL_DIR / f"{model_name}_best.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to {model_path}")

        return None


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_training_pipeline(
    data_path: str,
    experiment_name: str = 'CreditRiskModel',
    test_size: float = 0.2,
    cv: int = 5,
    search_method: str = 'random',
    n_iter: int = 20,
    models_to_train: List[str] = None
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
        models_to_train: List of model names to train (default: all)

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

    # Get model configs
    model_configs = get_model_configs()
    
    if models_to_train:
        model_configs = {k: v for k, v in model_configs.items() if k in models_to_train}

    # Train and evaluate each model
    results = {}
    trained_pipelines = {}

    for model_name, config in model_configs.items():
        try:
            pipeline, metrics = train_and_evaluate(
                model_name=model_name,
                pipeline=config['pipeline'],
                param_grid=config['params'],
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
# STANDALONE EXECUTION
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Credit Risk Models with sklearn Pipeline')
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
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of models to train')

    args = parser.parse_args()

    # Default data path
    if args.data is None:
        from paths import resolve_processed_data

        resolved = resolve_processed_data()
        if not resolved.is_file():
            logger.error(
                "No processed data found at %s. Run `python run_pipeline.py` "
                "or provide --data path.",
                resolved,
            )
            sys.exit(1)
        data_path = str(resolved)
    else:
        data_path = args.data

    # Parse models
    models_to_train = args.models.split(',') if args.models else None

    # Run training
    results = run_training_pipeline(
        data_path=data_path,
        experiment_name=args.experiment,
        test_size=args.test_size,
        cv=args.cv,
        search_method=args.search,
        n_iter=args.n_iter,
        models_to_train=models_to_train
    )

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Metrics: {results['best_metrics']}")
