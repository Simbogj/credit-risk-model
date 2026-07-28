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
