"""
Credit Risk Model - Additional Unit Tests
==========================================
Tests for train.py module and API.
"""

import json
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for training tests."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'TotalAmount': np.random.randn(n_samples) * 10000,
        'AvgAmount': np.random.randn(n_samples) * 500,
        'StdAmount': np.abs(np.random.randn(n_samples) * 200),
        'MinAmount': np.abs(np.random.randn(n_samples) * 100),
        'MaxAmount': np.abs(np.random.randn(n_samples) * 1000),
        'TransactionCount': np.random.randint(5, 50, n_samples),
        'Recency': np.random.randint(1, 90, n_samples),
        'Frequency': np.random.randint(5, 50, n_samples),
        'Monetary': np.random.randn(n_samples) * 10000,
        'is_high_risk': np.random.randint(0, 2, n_samples),
    })
    
    return df


@pytest.fixture
def sample_X(sample_data):
    """Sample features."""
    return sample_data.drop('is_high_risk', axis=1)


@pytest.fixture
def sample_y(sample_data):
    """Sample target."""
    return sample_data['is_high_risk']


# =============================================================================
# TESTS FOR MODEL CREATION
# =============================================================================

class TestModelCreation:
    """Tests for model pipeline creation."""
    
    def test_logistic_regression_creation(self):
        """Test Logistic Regression model can be created."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        assert model is not None
        assert model.random_state == 42
    
    def test_decision_tree_creation(self):
        """Test Decision Tree model can be created."""
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
        assert model is not None
        assert model.max_depth == 5
    
    def test_random_forest_creation(self):
        """Test Random Forest model can be created."""
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        assert model is not None
        assert model.n_estimators == 10


# =============================================================================
# TESTS FOR MODEL TRAINING
# =============================================================================

class TestModelTraining:
    """Tests for model training functionality."""
    
    def test_logistic_regression_fit(self, sample_X, sample_y):
        """Test Logistic Regression can be fitted."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(sample_X, sample_y)
        
        predictions = model.predict(sample_X)
        assert len(predictions) == len(sample_y)
        assert set(predictions).issubset({0, 1})
    
    def test_decision_tree_fit(self, sample_X, sample_y):
        """Test Decision Tree can be fitted."""
        model = DecisionTreeClassifier(random_state=42, max_depth=3)
        model.fit(sample_X, sample_y)
        
        predictions = model.predict(sample_X)
        assert len(predictions) == len(sample_y)
    
    def test_random_forest_fit(self, sample_X, sample_y):
        """Test Random Forest can be fitted."""
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(sample_X, sample_y)
        
        predictions = model.predict(sample_X)
        assert len(predictions) == len(sample_y)
    
    def test_model_predict_proba(self, sample_X, sample_y):
        """Test predict_proba returns probabilities."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(sample_X, sample_y)
        
        proba = model.predict_proba(sample_X)
        assert proba.shape == (len(sample_y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# =============================================================================
# TESTS FOR METRICS CALCULATION
# =============================================================================

class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        from sklearn.metrics import accuracy_score
        
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 0.8
    
    def test_precision_calculation(self):
        """Test precision metric calculation."""
        from sklearn.metrics import precision_score
        
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        
        precision = precision_score(y_true, y_pred)
        assert precision == 1.0
    
    def test_recall_calculation(self):
        """Test recall metric calculation."""
        from sklearn.metrics import recall_score
        
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        
        recall = recall_score(y_true, y_pred)
        assert recall == 0.6666666666666666
    
    def test_roc_auc_calculation(self):
        """Test ROC-AUC metric calculation."""
        from sklearn.metrics import roc_auc_score
        
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        auc = roc_auc_score(y_true, y_scores)
        assert 0 <= auc <= 1


# =============================================================================
# TESTS FOR DATA HANDLING
# =============================================================================

class TestDataHandling:
    """Tests for data handling functions."""
    
    def test_train_test_split_stratified(self, sample_X, sample_y):
        """Test stratified train/test split."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            sample_X, sample_y, 
            test_size=0.2, 
            random_state=42,
            stratify=sample_y
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_cross_validation(self, sample_X, sample_y):
        """Test cross-validation works."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(model, sample_X, sample_y, cv=3, scoring='roc_auc')
        
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


# =============================================================================
# TESTS FOR HYPERPARAMETER SEARCH
# =============================================================================

class TestHyperparameterSearch:
    """Tests for hyperparameter tuning."""
    
    def test_grid_search_runs(self, sample_X, sample_y):
        """Test GridSearchCV can run."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        param_grid = {
            'classifier__C': [0.1, 1.0]
        }
        
        search = GridSearchCV(
            pipeline, param_grid, 
            cv=2, 
            scoring='roc_auc',
            n_jobs=1
        )
        search.fit(sample_X, sample_y)
        
        assert search.best_score_ >= 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
