# Credit Risk Probability Model for Alternative Data

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

## Project Overview

This project builds an end-to-end credit risk scoring product to enable Bati Bank's buy-now-pay-later service. The solution encompasses exploratory data analysis, feature engineering, model training with MLflow tracking, API deployment with FastAPI, and CI/CD automation.

### Business Context

Bati Bank is partnering with an eCommerce platform to offer credit-based purchasing. Traditional credit scoring relies on historical default labels, which are unavailable in this alternative data context. Instead, we engineer a **proxy target variable** from customer Recency, Frequency, and Monetary (RFM) patterns to identify high-risk customer segments, then train a classification model to predict credit risk probability for new applicants.

### Key Deliverables

- **Feature Engineering Pipeline** (`src/data_processing.py`): Reproducible transformation of raw transaction data into model-ready features
- **Model Training & Tracking** (`src/train.py`): Multiple classifiers trained with hyperparameter tuning and MLflow experiment tracking
- **REST API** (`src/api/`): FastAPI service serving risk probability predictions in real time
- **Containerization**: Docker and Docker Compose for reproducible deployment
- **CI/CD Pipeline**: GitHub Actions for automated testing and linting
- **EDA & Final Report**: Comprehensive exploratory analysis and polished final report

---

## Credit Scoring Business Understanding

### Overview

Credit scoring is the process of assigning a quantitative measure to potential borrowers as an estimate of their likelihood to default. In regulated financial environments, credit risk models must comply with the **Basel II Capital Accord**, which establishes standards for risk measurement, documentation, and interpretability. This section addresses three critical questions that shape our modeling approach.

### 1. Basel II's Emphasis on Risk Measurement and Model Interpretability

**How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?**

The Basel II Capital Accord represents an international regulatory framework designed to ensure adequate capital reserves in financial institutions. It emphasizes three pillars:

1. **Minimum Capital Requirements** (Pillar 1): Banks must maintain capital proportional to their credit risk exposure. This requires quantifiable, defensible risk estimates.
2. **Supervisory Review** (Pillar 2): Regulators review banks' risk management processes and models. Models must be auditable and transparent.
3. **Market Discipline** (Pillar 3): Banks must disclose risk management practices and capital adequacy ratios to the public.

**Implications for Our Model:**

- **Interpretability as a Regulatory Requirement**: Basel II mandates that credit risk models produce transparent, explainable risk estimates. Regulators must understand *why* a borrower is classified as high-risk—not just that they are. This favors models like Logistic Regression with Weight of Evidence (WoE) encoding, where feature contributions to risk are interpretable and monotonic.

- **Documentation and Governance**: Every modeling choice—data sources, feature engineering decisions, threshold selections, and performance metrics—must be documented and justified. This creates an audit trail for regulatory review and ensures decisions are defensible if challenged.

- **Monitoring and Stability**: Basel II requires ongoing model monitoring. Changes in feature distributions, model performance, or business context must trigger re-evaluation. Interpretable models enable faster detection of model drift and easier recalibration.

- **Risk Quantification**: Basel II requires precise quantification of default probability (Probability of Default, or PD), Loss Given Default (LGD), and Exposure at Default (EAD). Our model's output—a risk probability—directly feeds into these calculations.

**Business Impact**: Failure to comply with Basel II standards exposes Bati Bank to regulatory penalties, loss of license, and reputational damage. Interpretable, well-documented models are a non-negotiable requirement, not a nice-to-have.

---

### 2. Proxy Target Variable Necessity and Business Risks

**Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?**

#### Why a Proxy Variable is Necessary

The raw dataset contains **no historical default labels**—we observe only transaction behavior. Without a target variable, we cannot train a supervised classifier. We have three options:

1. **Wait for Future Default Events**: Collect data for 12–24 months until defaults occur. This delays product launch and competitive advantage. **Not viable for immediate deployment.**
2. **Use Expert Labels**: Have domain experts manually label a sample of customers as risky/safe. This is subjective and doesn't scale. **Prone to bias.**
3. **Engineer a Proxy Target**: Use behavioral patterns (RFM) to infer risk. Customers who are disengaged (low frequency, low monetary value) are statistically more likely to default. **Scalable and defensible.**

We adopt **RFM-based clustering** to segment customers. The assumption is that **disengaged customers (low Recency, Frequency, Monetary) represent high credit risk**. This is grounded in financial theory: customers with minimal engagement lack incentive to repay and generate insufficient revenue to justify credit risk.

#### Business Risks of Proxy-Based Prediction

Using a proxy introduces several **material risks**:

1. **Label Validity Risk**: The RFM-derived label may not correlate with actual default. For example:
   - A customer with low frequency might be cautious and creditworthy, not risky.
   - A new customer (low recency) might be a future high-value borrower.
   - **Mitigation**: Validate the proxy against actual defaults once historical data accumulates. Conduct A/B testing with a subset of applicants.

2. **Selection Bias**: The model is trained on observed customers, who may differ systematically from new applicants. New applicants might have different RFM patterns or risk profiles. **Mitigation**: Regularly retrain on fresh data and monitor prediction drift.

3. **Target Leakage**: RFM metrics are derived from historical transactions. A customer's future engagement depends on loan approval. If we deny credit to predicted high-risk customers, we artificially confirm our prediction (self-fulfilling prophecy). **Mitigation**: Track actual default rates post-deployment and adjust model thresholds based on real outcomes.

4. **Feedback Loops**: Model predictions influence lending decisions, which influence customer behavior, which influences future RFM values. Over time, the relationship between RFM and true risk may deteriorate. **Mitigation**: Implement continuous monitoring and periodic retraining.

5. **Regulatory Acceptance**: Regulators (Basel II compliance) may question a proxy-based approach, especially if it diverges from industry standards. **Mitigation**: Document the proxy design thoroughly, benchmark against peers, and commit to validation once true defaults are observed.

**Recommendation**: Deploy the proxy-based model as a **first-pass risk estimator**, not as the final arbiter of credit decisions. Combine it with other signals (alternative data sources, expert judgment, traditional credit bureaus). Establish a robust monitoring framework to detect proxy drift and retrain frequency.

---

### 3. Trade-offs Between Interpretable and High-Performance Models in Regulated Contexts

**What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?**

#### Interpretable Models: Logistic Regression with WoE

**Logistic Regression** is the traditional choice in credit risk:
- **Pros**:
  - **Explainable**: Coefficients directly show feature impact on log-odds of default. Easy to communicate to regulators and business teams.
  - **Stable**: Few hyperparameters. Resistant to overfitting. Performance is predictable across data distributions.
  - **WoE/IV Framework**: Weight of Evidence (WoE) transforms categorical variables to maximize separation between good/bad customers. Information Value (IV) quantifies feature predictive power. Both are industry-standard regulatory metrics.
  - **Monotonic**: Relationships between features and default are monotonic (higher account age = lower risk). Easy to defend.
  - **Fast**: Trains and scores quickly. Minimal computational overhead.

- **Cons**:
  - **Linear Assumption**: Assumes linear relationships. May underperform if true patterns are non-linear.
  - **Limited Interactions**: Doesn't capture feature interactions without explicit engineering. Requires more manual feature work.
  - **Lower Accuracy**: Often yields lower ROC-AUC or accuracy vs. ensemble methods. May miss predictive signals.

#### High-Performance Models: Gradient Boosting (XGBoost, LightGBM, Random Forest)

**Ensemble methods** offer superior predictive power:
- **Pros**:
  - **Non-linear Relationships**: Naturally capture complex, non-linear patterns and interactions.
  - **Higher Accuracy**: Often achieve 2–5% higher ROC-AUC or accuracy compared to Logistic Regression.
  - **Feature Importance**: Can compute feature importance (e.g., SHAP, permutation importance), though less intuitive than WoE.
  - **Robustness**: Handle missing data and outliers better than linear models.

- **Cons**:
  - **Black Box**: Difficult to explain why a specific prediction was made. Regulators distrust models they cannot understand.
  - **Instability**: Sensitive to hyperparameter choices and training data. Requires careful tuning and validation.
  - **Overfitting Risk**: High capacity can lead to overfitting, especially with limited training data.
  - **Complexity**: Many hyperparameters to tune. Reproducibility and auditing are challenging.
  - **Regulatory Friction**: Basel II and Fair Lending regulations (especially in US/EU) increasingly require explainability. Gradient Boosting models face scrutiny.

#### The Regulated Financial Context Trade-off

In credit risk, the trade-off is **not** between interpretability and accuracy alone—it is between **accuracy and defensibility**:

| Dimension | Logistic Regression + WoE | Gradient Boosting |
|-----------|---------------------------|-------------------|
| **Accuracy (ROC-AUC)** | ~0.75–0.80 | ~0.80–0.88 |
| **Interpretability** | High (coefficients, WoE) | Low (complex rules) |
| **Regulatory Approval** | High (industry standard) | Medium–Low (requires SHAP/explanation) |
| **Audit Trail** | Clear, documented | Complex, "model card" required |
| **Bias Detection** | Easy (coefficient inspection) | Difficult (requires fairness tools) |
| **Time to Market** | Fast | Slower (more tuning needed) |
| **Recalibration Post-Deployment** | Straightforward | Complex |

#### Decision Framework

1. **If Regulatory Approvals are Critical** (true for Bati Bank): Choose Logistic Regression + WoE. Accept the 3–5% accuracy loss in exchange for regulatory acceptance, audit clarity, and faster deployment.

2. **If Risk Assessment is Life-Critical** (healthcare, fraud): Consider hybrid approaches:
   - Train Gradient Boosting as a secondary model
   - Use SHAP (SHapley Additive exPlanations) to explain individual predictions
   - Maintain a rule-based override for extreme cases
   - Document the trade-off in the model card

3. **If Regulatory Pressure is Low** (emerging markets, non-regulated contexts): Gradient Boosting alone may be justifiable.

**Our Approach**: We will **train both model types**, compare performance with MLflow, and select the final model based on:
- Regulatory alignment (prioritize interpretability)
- Business impact (acceptable error rates)
- Deployment constraints (latency, scalability)

We will then document the final choice and its trade-offs in the final report, framed as a defensible decision for Basel II compliance.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/
│   └── ci.yml                     # CI/CD pipeline with linting and tests
├── data/
│   ├── raw/                       # Raw transaction data
│   │   ├── data.csv
│   │   └── Xente_Variable_Definitions.csv
│   └── processed/                 # Processed data for training
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training and tracking
│   ├── predict.py                 # Inference logic
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile                     # Container image
├── docker-compose.yml             # Multi-container orchestration
├── requirements.txt               # Python dependencies
├── .gitignore                     # Version control exclusions
└── README.md                      # This file
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for deployment)
- pip or conda (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Simbogj/credit-risk-model
   cd credit-risk-model
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the data**: Place `data.csv` and `Xente_Variable_Definitions.csv` in `data/raw/`.

### Running the Pipeline

#### Exploratory Data Analysis
```bash
jupyter notebook notebooks/eda.ipynb
```
## Key Concepts

### RFM Analysis and Customer Segmentation
- **Recency (R)**: Days since last transaction
- **Frequency (F)**: Number of transactions
- **Monetary (M)**: Total transaction value

High-risk customers are identified as those with low R, F, and M values using K-Means clustering.

### Weight of Evidence (WoE) and Information Value (IV)
- **WoE**: Measures the strength of relationship between a variable and the target (default). Used to encode categorical features.
- **IV**: Summarizes predictive power of a variable. Higher IV = stronger predictor.

### Experiment Tracking with MLflow
All model training runs are logged to MLflow, including:
- Hyperparameters
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Model artifacts (trained sklearn Pipeline)
- Source code metadata

### CI/CD with GitHub Actions
Every push to `main` triggers automated:
- Code linting (flake8)
- Unit tests (pytest)
- Build validation

---

## Monitoring and Governance

### Model Monitoring
Once deployed, the model should be monitored for:
- **Prediction Drift**: Changes in feature distributions
- **Performance Drift**: Decline in accuracy or ROC-AUC
- **Proxy Drift**: Divergence between RFM-based risk and actual defaults
- **Fairness**: Disparate impact across demographic groups (if applicable)

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
