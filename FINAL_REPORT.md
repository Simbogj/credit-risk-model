# Credit Risk Probability Model for Alternative Data
## An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

**Bati Bank Analytics Team**
*Final Project Report*

---

## Executive Summary

This report documents the development, deployment, and automation of a credit risk scoring model for Bati Bank's buy-now-pay-later service. The project addresses a fundamental challenge in alternative credit scoring: **building a predictive risk model when no historical default labels exist**.

Our solution transforms behavioral transaction data into a credit risk signal using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to create a proxy target variable. We then train, tune, and compare multiple classification models, tracking all experiments with MLflow. The final model is packaged as a containerized REST API with automated CI/CD testing.

### Key Deliverables

| Component | Technology | Status |
|-----------|------------|--------|
| Feature Engineering Pipeline | scikit-learn Pipeline | ✅ Complete |
| Proxy Target Variable | RFM + K-Means Clustering | ✅ Complete |
| Model Training & Tracking | MLflow + Multiple Models | ✅ Complete |
| REST API Service | FastAPI + Pydantic | ✅ Complete |
| Containerization | Docker + Docker Compose | ✅ Complete |
| CI/CD Pipeline | GitHub Actions | ✅ Complete |

---

## 1. Business Problem and Solution Overview

### 1.1 The Challenge

Bati Bank is partnering with an eCommerce platform to enable credit-based purchasing. Traditional credit scoring relies on historical loan performance data—specifically, whether borrowers defaulted on their loans. However, the eCommerce transaction dataset contains **no default labels**.

Without a direct target variable, we cannot train a standard supervised classifier. The critical innovation in this project lies in **transforming behavioral data into a predictive risk signal**.

### 1.2 Our Approach

We address the missing default labels through a multi-step methodology:

1. **RFM Analysis**: Calculate Recency, Frequency, and Monetary metrics for each customer from their transaction history
2. **Customer Segmentation**: Use K-Means clustering to segment customers into risk tiers based on their RFM profiles
3. **Proxy Target Creation**: Label the least engaged cluster (low frequency, low monetary, high recency of inactivity) as "high-risk"
4. **Model Training**: Train classification models to predict this proxy target
5. **Deployment**: Package the best model as a REST API for real-time scoring

### 1.3 Why This Approach Works

Financial theory supports using engagement metrics as risk proxies:
- **Frequency** indicates customer commitment and account utilization
- **Monetary** reflects revenue generation potential and ability to service debt
- **Recency** captures active engagement vs. dormancy (a precursor to churn and potential default)

Customers who are disengaged (low F, low M, high R) statistically demonstrate:
- Lower utilization of financial products
- Higher probability of account dormancy
- Reduced skin-in-the-game for repaying new credit

---

## 2. Data Understanding and Exploratory Analysis

### 2.1 Dataset Overview

The Xente eCommerce transaction dataset contains **95,662 transactions** across **3,742 unique customers** over a **90-day observation window** (November 15, 2018 – February 13, 2019).

| Metric | Value |
|--------|-------|
| Total Transactions | 95,662 |
| Unique Customers | 3,742 |
| Total Transaction Value | 642.6M UGX |
| Average Transaction | 6,717.85 UGX |
| Fraud Rate | 0.20% |
| Data Completeness | 100% |

### 2.2 Key Data Characteristics

**Transaction Structure:**
- Single currency (UGX - Uganda Shilling) and single country (CountryCode 256)
- Bidirectional transactions: 39.92% credits (refunds) and 60.08% debits (purchases)
- 4 transaction channels with ChannelId_3 dominating (59.5%)
- 9 product categories dominated by financial_services (47.4%) and airtime (47.0%)

**Data Quality:**
- No missing values across all 16 features
- Extreme outliers present (25.55% flagged by IQR method) but represent legitimate high-value transactions
- Amount field highly skewed (skewness: 51.10) with values ranging from -1,000,000 to +9,880,000 UGX

**Important Finding:**
The FraudResult field (0.20% fraud rate) is **not suitable** as a target variable for credit risk. Fraud is distinct from credit default—it represents malicious activity, not inability to repay. Using fraud as a proxy for credit risk would create a fundamentally incorrect model.

### 2.3 Feature Correlations

| Relationship | Correlation | Implication |
|--------------|-------------|-------------|
| Amount ↔ Value | 0.9897 | **Redundant** - drop one feature |
| Value ↔ FraudResult | 0.5667 | Moderate correlation |
| Amount ↔ FraudResult | 0.5574 | Moderate correlation |
| CountryCode | Constant (256) | **No variance** - drop feature |

---

## 3. Feature Engineering Pipeline

### 3.1 Engineering Approach

We built a comprehensive feature engineering pipeline that transforms raw transaction data into **57 customer-level features** organized into the following categories:

#### Temporal Features (8 features)
- TransactionHour, TransactionDay, TransactionMonth, TransactionYear
- DayOfWeek, IsWeekend, Quarter, WeekOfYear

#### Aggregate Transaction Features (18 features)
- TotalAmount, AvgAmount, StdAmount, MinAmount, MaxAmount, TransactionCount
- TotalValue, AvgValue, StdValue
- FraudCount, FraudRate
- CreditCount, CreditTotal, DebitCount, DebitTotal, CreditRatio, DebitRatio
- AmountRange

#### Channel Distribution Features (10 features)
- Per-channel transaction counts and amounts
- Channel ratios relative to total transactions

#### Product Category Features (9 features)
- Per-category transaction counts and amounts
- Category ratios relative to total transactions

#### Provider and Pricing Features (6 features)
- UniqueProviders count
- AveragePricingStrategy
- PricingStrategy distribution counts

#### RFM Features (6 features)
- Recency (days since last transaction)
- Frequency (transaction count)
- Monetary (total amount)
- AvgTransactionAmount
- TransactionsPerDay

### 3.2 Pipeline Architecture

The feature engineering is implemented as a modular, reproducible scikit-learn Pipeline:

```python
# Simplified pipeline structure
Pipeline([
    ('load_data', LoadRawData()),
    ('extract_temporal', TemporalFeatureExtractor()),
    ('aggregate_features', CustomerAggregator()),
    ('create_channel_features', ChannelFeatureCreator()),
    ('create_category_features', CategoryFeatureCreator()),
    ('calculate_rfm', RFMCalculator()),
    ('encode_categorical', CategoricalEncoder()),
])
```

All stochastic operations use `random_state=42` for reproducibility.

---

## 4. Proxy Target Variable Engineering

### 4.1 RFM-Based Clustering Methodology

The proxy target variable is created through a rigorous process:

**Step 1: RFM Calculation**
For each customer, we calculate:
- **Recency (R)**: Days since last transaction relative to snapshot date
- **Frequency (F)**: Total number of transactions
- **Monetary (M)**: Sum of all transaction amounts

**Step 2: Feature Preprocessing**
- Log transform of Monetary to handle negative values
- RobustScaler to handle extreme outliers (IQR-based scaling)

**Step 3: K-Means Clustering**
- k=3 clusters (High/Medium/Low risk segments)
- Random state: 42 for reproducibility

**Step 4: Risk Score Assignment**
We compute a risk score for each cluster using normalized RFM values:

```python
RiskScore = Recency_norm - (Frequency_norm + Monetary_norm) / 2
```

High Recency (inactive) + Low Frequency + Low Monetary = High Risk

**Step 5: Binary Target Creation**
The cluster with the highest risk score is labeled as `is_high_risk = 1`; all others are `is_high_risk = 0`.

### 4.2 Target Distribution

After clustering, we observe the following distribution:

| Segment | Count | Percentage | Interpretation |
|---------|-------|------------|----------------|
| High Risk (1) | ~1,200 | ~32% | Disengaged customers |
| Low Risk (0) | ~2,542 | ~68% | Active customers |

This distribution is reasonably balanced, avoiding the severe class imbalance that would complicate model training.

### 4.3 Business Risks of Proxy-Based Prediction

We explicitly document the limitations of this approach:

1. **Label Validity Risk**: RFM-derived labels may not correlate with actual default behavior
2. **Selection Bias**: Observed customers may differ from new applicants
3. **Target Leakage**: Denying credit to predicted high-risk customers may confirm predictions
4. **Feedback Loops**: Model decisions influence future RFM values

**Mitigation Strategies:**
- Deploy as first-pass risk estimator, not final arbiter
- Combine with other signals (expert judgment, credit bureau data)
- Implement continuous monitoring for proxy drift
- Track actual default rates post-deployment and adjust thresholds

---

## 5. Model Training and Experiment Tracking

### 5.1 Model Selection

We trained and compared four classification models:

| Model | Rationale |
|-------|-----------|
| **Logistic Regression** | Industry standard for credit scoring; interpretable coefficients; WoE-compatible |
| **Decision Tree** | Simple interpretability; easy to explain to stakeholders |
| **Random Forest** | Ensemble of trees; handles non-linear relationships; robust to outliers |
| **Gradient Boosting** | State-of-the-art performance; captures complex patterns |

### 5.2 Hyperparameter Tuning

For each model, we performed hyperparameter search using:
- **Search Method**: RandomizedSearchCV (20 iterations)
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Primary Metric**: ROC-AUC (optimal for imbalanced classification)

**Hyperparameter Grids:**

```python
LogisticRegression:
  - C: [0.01, 0.1, 1.0, 10.0]
  - penalty: ['l2']
  - class_weight: ['balanced', None]

DecisionTree:
  - max_depth: [3, 5, 7, 10]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
  - class_weight: ['balanced', None]

RandomForest:
  - n_estimators: [50, 100, 200]
  - max_depth: [5, 10, 15]
  - min_samples_split: [2, 5]
  - min_samples_leaf: [1, 2]

GradientBoosting:
  - n_estimators: [50, 100, 200]
  - max_depth: [3, 5, 7]
  - learning_rate: [0.01, 0.1, 0.2]
```

### 5.3 MLflow Experiment Tracking

All experiments are logged to MLflow with:
- **Model parameters** (hyperparameters, random seed)
- **Evaluation metrics** (accuracy, precision, recall, F1, ROC-AUC)
- **Model artifacts** (trained pipeline)
- **Classification reports** and confusion matrices

### 5.4 Model Comparison Results

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| Logistic Regression | 0.78 | 0.74 | 0.71 | 0.73 | 0.72 |
| Decision Tree | 0.75 | 0.71 | 0.68 | 0.70 | 0.69 |
| Random Forest | 0.82 | 0.78 | 0.75 | 0.77 | 0.76 |
| Gradient Boosting | 0.84 | 0.80 | 0.77 | 0.79 | 0.78 |

**Best Model**: Gradient Boosting (ROC-AUC: 0.84)

### 5.5 Model Selection Decision

For Basel II compliance, we face a trade-off:
- **Interpretability**: Logistic Regression (coefficients directly explain risk drivers)
- **Performance**: Gradient Boosting (2-5% higher ROC-AUC)

**Our Decision**: Use Gradient Boosting as the primary model with SHAP explanations for individual predictions. This balances regulatory requirements (explainability via SHAP) with predictive performance.

---

## 6. API Development and Model Deployment

### 6.1 FastAPI REST API

The deployed API provides real-time credit risk scoring:

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch predictions |
| `/features` | GET | List of model features |

**Prediction Response:**

```json
{
  "success": true,
  "prediction": {
    "customer_id": "CUST_001",
    "predicted_class": 0,
    "risk_probability": 0.25,
    "confidence": 0.85,
    "risk_level": "Low",
    "credit_score": 663
  },
  "decision": {
    "decision": "Approve",
    "risk_probability": 0.25,
    "credit_score": 663,
    "recommended_terms": {
      "max_amount": 10000,
      "max_duration_months": 12,
      "interest_rate": 0.12
    }
  },
  "model_version": "mlflow-CreditRiskModel-Production",
  "timestamp": "2026-06-03T09:50:00Z"
}
```

### 6.2 Credit Score Calculation

We convert risk probability to a FICO-style credit score (300-850):

```python
CreditScore = 300 + (1 - RiskProbability) * 550
```

This provides a familiar, interpretable metric for credit decisions.

### 6.3 Credit Decision Thresholds

| Risk Probability | Decision | Max Amount | Max Duration | Interest Rate |
|------------------|----------|------------|--------------|---------------|
| ≤ 0.30 | Approve | 10,000 | 12 months | 12% |
| 0.30 - 0.60 | Review | 5,000 | 6 months | 18% |
| > 0.60 | Decline | 0 | 0 months | N/A |

---

## 7. Containerization and CI/CD

### 7.1 Docker Implementation

The service is containerized using a multi-stage Dockerfile:

**Stage 1 (Builder):**
- Python 3.9 slim base
- Install build dependencies
- Copy and install requirements.txt

**Stage 2 (Production):**
- Python 3.9 slim base
- Copy built artifacts from builder
- Create non-root user for security
- Expose port 8000
- Health check configuration

### 7.2 Docker Compose Orchestration

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --backend-store-uri sqlite:///mlflow_store/mlflow.db
```

### 7.3 GitHub Actions CI/CD Pipeline

The CI/CD pipeline triggers on every push to main and pull requests:

```yaml
jobs:
  lint:
    - Runs flake8 linter on src/
    - Checks for code style issues
  
  test:
    - Installs dependencies
    - Runs pytest on tests/
  
  build:
    - Sets up Docker Buildx
    - Builds Docker image
    - Runs container smoke test
```

**Pipeline Status**: ✅ All checks passing

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Proxy Target Validation**: The RFM-based proxy has not been validated against actual default outcomes. Once historical default data becomes available, the proxy should be benchmarked against true labels.

2. **Single Country/Currency**: The model is trained on Ugandan data (UGX). Generalization to other markets requires recalibration.

3. **Feature Engineering Scope**: Temporal features are limited to extraction; no time-series analysis or trend features are included.

4. **Model Update Frequency**: The model is trained on a fixed 90-day window. Production deployment requires regular retraining.

### 8.2 Recommended Improvements

1. **Proxy Validation**: Once default data accumulates, compare proxy-based predictions against actual outcomes.

2. **Additional Features**:
   - Customer tenure (days since first transaction)
   - Transaction velocity trends (increasing/decreasing engagement)
   - Cross-channel behavior patterns

3. **Model Monitoring**:
   - Implement prediction drift detection
   - Set up alerts for feature distribution shifts
   - Establish retraining triggers based on performance degradation

4. **Regulatory Alignment**:
   - Document model card for Basel II compliance
   - Implement explainability requirements (SHAP)
   - Establish model governance processes

---

## 9. Technical Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT APPLICATION                        │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ POST /predict
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI REST API                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  Pydantic   │  │   Model      │  │    Credit Decision     │ │
│  │  Validation │  │   Loader     │  │    Engine              │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRADIENT BOOSTING MODEL                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Pipeline: Imputer → StandardScaler → GradientBoosting    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Getting Started

### Quick Start

```bash
# Clone repository
git clone https://github.com/Simbogj/credit-risk-model
cd credit-risk-model

# Build and run with Docker Compose
docker-compose up -d

# Access API documentation
open http://localhost:8000/docs

# Run training
python -m src.train --data data/processed/processed_customers.csv
```

### API Usage Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "features": {
      "TotalAmount": 25000.0,
      "AvgAmount": 1250.0,
      "TransactionCount": 20,
      "Recency": 5,
      "Frequency": 20,
      "Monetary": 25000.0
    }
  }'
```

---

## Appendix: Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/
│   ├── raw/                      # Raw transaction data
│   └── processed/                # Processed features
├── notebooks/
│   └── eda.ipynb                 # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # Feature engineering
│   ├── train.py                  # Model training
│   ├── predict.py                # Inference
│   └── api/
│       ├── main.py               # FastAPI application
│       └── pydantic_models.py    # Request/response schemas
├── tests/
│   └── test_data_processing.py   # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

**Author**: Bati Bank Analytics Team
**Date**: June 3, 2026
**Version**: 1.0.0

---

*This report serves as a self-contained artifact for Bati Bank's leadership and risk team, documenting the methodology, justification, and operational guidance for the credit risk scoring model.*