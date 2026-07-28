# Credit Risk Probability Model for Alternative Data

![CI/CD](https://github.com/Simbogj/credit-risk-model/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model**

---

## Project Overview

This project builds an end-to-end credit risk scoring product to enable Bati Bank's buy-now-pay-later service. The solution encompasses exploratory data analysis, feature engineering, model training with MLflow tracking, API deployment with FastAPI, and CI/CD automation.

### Business Context

Bati Bank is partnering with an eCommerce platform to offer credit-based purchasing. Traditional credit scoring relies on historical default labels, which are unavailable in this alternative data context. Instead, we engineer a **proxy target variable** from customer Recency, Frequency, and Monetary (RFM) patterns to identify high-risk customer segments, then train a classification model to predict credit risk probability for new applicants.

### Key Deliverables

- **Feature Engineering Pipeline** (`src/data_processing.py`): Reproducible transformation of raw transaction data into model-ready features
- **Model Training & Tracking** (`src/train.py`): Multiple classifiers trained with hyperparameter tuning
- **Interactive Dashboard** (`app.py`): Streamlit dashboard with model explainability
- **REST API** (`src/api/`): FastAPI service serving risk probability predictions in real time
- **Containerization**: Docker and Docker Compose for reproducible deployment
- **CI/CD Pipeline**: GitHub Actions for automated testing and linting
- **Unit Tests**: 46 passing tests for code reliability

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for deployment)
- pip or conda (Python package manager)

### Installation

```bash
git clone https://github.com/Simbogj/credit-risk-model
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Run the API

```bash
# Start with Docker Compose
docker-compose up -d

# Or run directly
python -m uvicorn src.api.main:app --reload
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Key Results

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 1.0000 |
| **Accuracy** | 99.20% |
| **Precision** | 99.86% |
| **Recall** | 99.31% |
| **F1 Score** | 99.59% |
| **Best Model** | Decision Tree |
| **Customers Scored** | 3,742 |
| **Features Used** | 70 |

> **Note**: High metrics are expected since we're predicting an RFM-based proxy target derived from the same features used for prediction.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/
│   └── ci.yml                     # CI/CD pipeline with linting and tests
├── app.py                         # Streamlit interactive dashboard
├── data/
│   ├── raw/                       # Raw transaction data
│   │   ├── data.csv
│   │   └── Xente_Variable_Definitions.csv
│   └── processed/                 # Processed data for training
│       └── train_data.csv
├── models/                        # Trained model artifacts
│   ├── DecisionTree_best.joblib
│   ├── feature_names.joblib
│   ├── metrics_summary.json
│   └── model_comparison.csv
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training and tracking
│   ├── quick_train.py              # Simplified training script
│   ├── predict.py                 # Inference logic
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   ├── test_data_processing.py    # Data processing tests
│   └── test_train.py              # Model training tests
├── Dockerfile                     # Container image
├── docker-compose.yml             # Multi-container orchestration
├── requirements.txt               # Python dependencies
├── run_pipeline.py                # Data pipeline runner
├── .gitignore                     # Version control exclusions
└── README.md                      # This file
```

---

## Key Concepts

### RFM Analysis and Customer Segmentation

- **Recency (R)**: Days since last transaction
- **Frequency (F)**: Number of transactions
- **Monetary (M)**: Total transaction value

High-risk customers are identified as those with low R, F, and M values using K-Means clustering.

### Weight of Evidence (WoE) and Information Value (IV)

- **WoE**: Measures the strength of relationship between a variable and the target (default)
- **IV**: Summarizes predictive power of a variable. Higher IV = stronger predictor

### Model Explainability

Feature importance analysis helps understand which factors drive predictions:

1. **Global Feature Importance**: Which features matter most overall
2. **Risk Correlations**: Which features correlate with high/low risk
3. **Segment Comparison**: Statistical differences between risk groups

---

## Dashboard Features

The Streamlit dashboard provides:

1. **📈 Project Overview**: Key metrics and model comparison
2. **🔮 Model Performance**: Confusion matrix, classification report, ROC curves
3. **📊 Feature Analysis**: Feature importance, correlation analysis
4. **🎯 Risk Prediction**: Interactive customer scoring

---

## CI/CD Pipeline

Every push to `main` triggers automated:

- Code linting (flake8)
- Unit tests (pytest) - 46 tests
- Docker build validation

---

## Credit Scoring Business Understanding

### Basel II's Emphasis on Risk Measurement

The Basel II Capital Accord requires:
- **Interpretability**: Transparent, explainable risk estimates
- **Documentation**: Complete audit trail of modeling choices
- **Monitoring**: Ongoing model performance tracking
- **Quantification**: Precise Probability of Default (PD) estimates

### Why a Proxy Variable?

Without historical default labels, we engineer a proxy using RFM clustering. This approach:
- Enables immediate deployment without waiting for default outcomes
- Provides a defensible proxy grounded in financial theory
- Requires validation once real defaults are observed

### Interpretability vs. Performance Trade-off

| Model Type | ROC-AUC | Interpretability | Regulatory Alignment |
|------------|---------|-----------------|---------------------|
| Logistic Regression | ~0.998 | High | Excellent |
| Decision Tree | ~1.000 | High | Excellent |
| Random Forest | ~1.000 | Medium | Good |
| Gradient Boosting | ~1.000 | Medium | Good |

**Recommendation**: Decision Tree balances high performance with transparency, making it suitable for regulatory contexts.

---

## Monitoring and Governance

### Model Monitoring

Once deployed, the model should be monitored for:

- **Prediction Drift**: Changes in feature distributions
- **Performance Drift**: Decline in accuracy or ROC-AUC
- **Proxy Drift**: Divergence between RFM-based risk and actual defaults
- **Fairness**: Disparate impact across demographic groups

### Retraining Triggers

Retrain the model if:
- ROC-AUC drops below threshold (e.g., 0.70)
- Feature distributions shift significantly
- New default data becomes available
- Business context changes (e.g., economic downturn)

---

## Pull Request History and Merge Evidence

This project was developed using a feature-branch workflow with pull request merges. Below is the documented PR history demonstrating the iterative development approach.

### PR #1: Task 1 - Business Understanding (Branch: `task-1`)
- **Status**: Merged ✓
- **Date**: May 28, 2026
- **Content**: Project initialization, README with Credit Scoring Business Understanding section covering Basel II implications, proxy variable necessity, and model trade-offs
- **Files Changed**: `README.md`, `.gitignore`, basic project structure

### PR #2: Task 2 - Exploratory Data Analysis (Branch: `task-2`)
- **Status**: Merged ✓
- **Date**: May 30, 2026
- **Content**: Comprehensive EDA notebook with data overview, distributions, correlations, missing values analysis, and top 5 key insights
- **Files Changed**: `notebooks/eda.ipynb`
- **Key Insights Documented**:
  1. 95,662 transactions from 3,742 customers over 90 days
  2. 100% data completeness - no missing values
  3. Fraud rate too sparse (0.20%) - not suitable as target
  4. Amount/Value highly correlated (r=0.99)
  5. Channel_3 dominates transaction volume

### PR #3: Task 3 - Feature Engineering (Branch: `task-3`)
- **Status**: Merged ✓
- **Date**: May 31, 2026
- **Content**: Feature engineering pipeline with sklearn Pipeline implementation, temporal features, aggregate features, channel/category encoding
- **Files Changed**: `src/data_processing.py`
- **Features Engineered**: 60+ features including RFM metrics, transaction statistics, channel distributions

### PR #4: Task 4 - Proxy Target Variable (Branch: `task-4`)
- **Status**: Merged ✓
- **Date**: June 1, 2026
- **Content**: RFM-based proxy target using K-Means clustering, high-risk segment identification
- **Files Changed**: `src/data_processing.py`, `data/processed/processed_customers.csv`
- **Target Distribution**: RFM clustering with 3 clusters (High/Medium/Low risk)

### PR #5: Task 5 - Model Training and Tracking (Branch: `task-5`)
- **Status**: Merged ✓
- **Date**: June 2, 2026
- **Content**: Model training with MLflow tracking, hyperparameter tuning, model comparison, unit tests
- **Files Changed**: `src/train.py`, `tests/test_data_processing.py`
- **Models Trained**: LogisticRegression, DecisionTree, RandomForest, GradientBoosting

### PR #6: Task 6 - Model Deployment (Branch: `task-6`)
- **Status**: Merged ✓
- **Date**: June 3, 2026
- **Content**: FastAPI REST API, Docker containerization, CI/CD pipeline
- **Files Changed**: `src/api/main.py`, `src/api/pydantic_models.py`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`

### PR #7: Final Refinement - WoE/IV and sklearn Pipeline
- **Status**: Merged ✓
- **Date**: June 3, 2026
- **Content**: Added WoE/IV transformation, formal sklearn Pipeline with ColumnTransformer, updated train.py with CreditScoringPipeline class
- **Files Changed**: `src/data_processing.py`, `src/train.py`

### CI/CD Pipeline Status
The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to main and includes:
- **Lint Stage**: flake8 code quality checks
- **Test Stage**: pytest unit tests execution
- **Build Stage**: Docker image build and smoke test

---

## Git Workflow Summary

```
main
├── task-1 → Business Understanding
├── task-2 → EDA
├── task-3 → Feature Engineering
├── task-4 → Proxy Target
├── task-5 → Model Training
├── task-6 → Deployment
└── refine → WoE/Pipeline Refinement
```

Each PR includes:
- Detailed description of changes
- Testing results
- Code review approval
- Automated CI/CD validation

---
