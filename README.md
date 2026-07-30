# Credit Risk Probability Model for Alternative Data

![CI/CD](https://github.com/Simbogj/credit-risk-model/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A production-grade credit risk scoring system for Bati Bank's buy-now-pay-later platform — built for reliability, transparency, and finance-sector compliance.**

---

## Business Problem

Bati Bank partners with an eCommerce platform to offer credit-based purchasing. Traditional credit scoring requires historical default labels, which are **unavailable** for new digital customers using alternative transaction data.

Without a defensible risk model, the bank faces:

- **Credit losses** from approving high-risk applicants
- **Lost revenue** from declining low-risk customers
- **Regulatory exposure** under Basel II interpretability requirements

This project delivers an end-to-end scoring product: feature engineering, model training, REST API, interactive dashboard with SHAP explainability, and automated CI/CD — so risk teams can score customers immediately while maintaining audit-ready documentation.

---

## Solution Overview

1. **Engineer 70 features** from 95,662 transactions across 3,742 customers
2. **Create an RFM-based proxy target** (K-Means clustering) when default labels are absent
3. **Train and compare** four classifiers; select Decision Tree for interpretability
4. **Deploy** via FastAPI + Docker with health checks and batch scoring
5. **Explain** decisions with SHAP global and local plots in a Streamlit dashboard

---

## Key Results

| Metric | Score | Business meaning |
|--------|-------|------------------|
| **ROC-AUC** | 1.0000 | Perfect rank-ordering of risk (proxy target) |
| **Accuracy** | 99.20% | Correct risk classification rate |
| **Precision** | 99.86% | Low false-approval rate |
| **Recall** | 99.31% | High-risk customers identified |
| **Customers scored** | 3,742 | Full portfolio coverage |
| **Features engineered** | 70 | Rich behavioral profile per customer |
| **Unit tests** | 53 passing | Automated correctness checks |

> Metrics reflect proxy-target performance. Validation against actual defaults is planned once outcome data matures.

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip
- Docker (optional, for API deployment)

### Installation

```bash
git clone https://github.com/Simbogj/credit-risk-model
cd credit-risk-model

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### Data Setup

Raw transaction data is **not committed** (gitignored). To reproduce the full pipeline:

1. Download the [Xente Fraud Detection dataset](https://www.kaggle.com/datasets/xente-challenge/xente-fraud-detection)
2. Place `data.csv` at `data/raw/data.csv`
3. Run:

```bash
python setup_data.py --run    # or: python run_pipeline.py
python quick_train.py         # optional: retrain models
```

Check status anytime:

```bash
python setup_data.py
```

The dashboard and API work on a fresh clone using committed model artifacts; full charts and SHAP require processed data.

### Run the Dashboard

```bash
streamlit run app.py
```

### Run the API

```bash
docker-compose up -d
# or
python -m uvicorn src.api.main:app --reload --port 8000
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml    # Lint, test, Docker build
├── app.py                      # Streamlit dashboard (+ SHAP)
├── setup_data.py               # Data check & pipeline runner
├── run_pipeline.py             # Feature engineering pipeline
├── src/
│   ├── paths.py                # Centralized path configuration
│   ├── data_processing.py      # Feature engineering + RFM target
│   ├── train.py                # MLflow training pipeline
│   ├── predict.py              # Inference utilities
│   └── api/                    # FastAPI REST service
├── tests/                      # 53 unit & path tests
├── models/                     # Trained artifacts (committed)
├── data/                       # Gitignored — raw & processed CSVs
├── Dockerfile
├── docker-compose.yml
├── REPORT.md                   # Technical report (finance audience)
└── requirements.txt
```

---

## Dashboard Features

| Page | Purpose |
|------|---------|
| **Overview** | Portfolio metrics, model comparison, risk distribution |
| **Model Performance** | Confusion matrix, classification report, probability histogram |
| **Feature Analysis** | Importance rankings and risk correlations |
| **SHAP Explainability** | Global impact + per-customer decision drivers |
| **Prediction** | Interactive applicant scoring with approve/review/decline |
| **About** | Business context and Basel II alignment |

---

## CI/CD Pipeline

Every push to `main` / `develop` and every PR triggers:

1. **Lint** — flake8 on `src/`
2. **Test** — pytest (53 tests)
3. **Build** — Docker image build + container smoke test

---

## Technical Details

### Data

- **Source**: Xente alternative transaction data (95,662 rows, 3,742 customers)
- **Processing**: Temporal, aggregate, channel/category, RFM features → `data/processed/train_data.csv`
- **Target**: `is_high_risk` from RFM K-Means clustering (proxy for default risk)

### Model

- **Algorithm**: Decision Tree (max_depth=3, balanced class weights)
- **Alternatives compared**: Logistic Regression, Random Forest, Gradient Boosting
- **Tracking**: MLflow experiment logging

### Evaluation

- Stratified train/test split, 5-fold cross-validation
- Metrics: ROC-AUC, accuracy, precision, recall, F1

---

## Regulatory Alignment (Basel II)

| Requirement | Implementation |
|-------------|----------------|
| Interpretability | Decision Tree rules + SHAP force plots |
| Documentation | `REPORT.md`, README, inline docstrings |
| Audit trail | Versioned models in `models/`, MLflow tracking |
| Monitoring | Health endpoint, drift/retraining guidance in report |

---

## Future Improvements

- Validate proxy target against realized defaults
- Fairness metrics across demographic segments
- Automated retraining when drift exceeds thresholds
- Shadow deployment for A/B testing credit policies

---

## Author

**Simbogj** — [GitHub](https://github.com/Simbogj/credit-risk-model)

Week 12 Capstone — 10 Academy AI Mastery Program

---

## References

See `REPORT.md` for the full technical report, gap analysis, and presentation narrative.
