# Credit Risk Model: Technical Report

## Executive Summary

This technical report documents the development of a production-grade credit risk scoring system for Bati Bank, enabling buy-now-pay-later services for an eCommerce platform. The solution demonstrates engineering excellence through comprehensive testing, interactive visualization, and regulatory-aligned design.

### Business Impact

- **3,742 customers** scored for credit risk
- **99.2% accuracy** in risk classification
- **70 engineered features** from transaction data
- **Real-time API** for production deployment
- **46 passing unit tests** for reliability

---

## 1. Problem Statement

### Business Context

Bati Bank partners with eCommerce platforms to offer credit-based purchasing. Traditional credit scoring requires historical default data, which is unavailable for new customers. This project addresses the challenge of assessing credit risk using alternative data sources.

### Technical Challenge

Without direct default labels, we needed to:
1. Engineer a credible proxy target variable from behavioral patterns
2. Build reproducible feature engineering pipelines
3. Train interpretable models for regulatory compliance
4. Deploy reliable, testable production systems

---

## 2. Methodology

### 2.1 Data Processing Pipeline

We built a comprehensive feature engineering pipeline that transforms raw transactions into model-ready features:

**Temporal Features (8 features)**
- Transaction hour, day, month, year
- Day of week, weekend flag
- Quarter, week of year

**Aggregate Features (20 features)**
- Total, average, standard deviation of amounts
- Min/max transaction values
- Credit/debit counts and ratios
- Fraud metrics

**Channel/Category Features (variable)**
- Transaction distribution by channel
- Product category breakdown
- Provider diversity

**RFM Features (5 features)**
- Recency: Days since last transaction
- Frequency: Number of transactions
- Monetary: Total transaction value
- Transactions per active day

### 2.2 Proxy Target Variable Engineering

We developed an RFM-based clustering approach to identify high-risk customers:

1. **Calculate RFM metrics** for each customer
2. **Scale features** using StandardScaler
3. **Apply K-Means clustering** (k=3)
4. **Analyze cluster profiles** to identify high-risk segment
5. **Assign binary labels**: High-risk (1) or Low-risk (0)

**Cluster Analysis Results:**
| Cluster | Recency | Frequency | Monetary | Count | Risk Level |
|---------|---------|-----------|----------|-------|------------|
| 0 | 32.2 | 15.7 | $163K | 3,620 | High Risk |
| 1 | 21.3 | 2,682 | -$33M | 3 | Low Risk |
| 2 | 10.7 | 258 | $1.26M | 119 | Low Risk |

### 2.3 Model Selection

We trained and compared four classification models:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| **Decision Tree** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9984 | 0.9920 | 0.9986 | 0.9931 | 0.9959 |

**Selected Model**: Decision Tree (max_depth=3, balanced class weights)

### 2.4 Feature Importance Analysis

Top 5 most important features:
1. Monetary (transaction value)
2. Frequency (transaction count)
3. Recency (days since last transaction)
4. Average transaction amount
5. Transaction count variability

---

## 3. Engineering Excellence

### 3.1 Testing Strategy

We implemented comprehensive unit testing with **46 passing tests**:

- **Data Processing Tests**: Feature extraction, RFM calculation, target creation
- **Model Training Tests**: Pipeline creation, fitting, prediction
- **Metrics Tests**: Accuracy, precision, recall, ROC-AUC calculations
- **Data Handling Tests**: Train/test splits, cross-validation

### 3.2 CI/CD Pipeline

Automated quality assurance through GitHub Actions:
- Code linting with flake8
- Unit test execution with pytest
- Docker image building
- Container smoke testing

### 3.3 Interactive Dashboard

Built with Streamlit, providing:
- Real-time model performance visualization
- Feature importance analysis
- Confusion matrix and classification reports
- Interactive customer risk prediction

---

## 4. Regulatory Considerations

### 4.1 Basel II Alignment

Our approach aligns with Basel II requirements:

**Pillar 1 - Minimum Capital Requirements**
- Quantifiable risk probability estimates
- Clear methodology documentation

**Pillar 2 - Supervisory Review**
- Transparent decision rules (Decision Tree)
- Complete audit trail
- Reproducible pipeline

**Pillar 3 - Market Discipline**
- Clear risk scoring methodology
- Performance disclosure

### 4.2 Model Interpretability

The Decision Tree model provides:
- **Transparent rules**: "If Monetary < $X and Recency > Y days → High Risk"
- **Feature contribution**: Clear understanding of which factors drive predictions
- **Regulatory defensibility**: Industry-standard approach for credit scoring

### 4.3 Proxy Variable Risks

We document the following risks of proxy-based prediction:

1. **Label Validity Risk**: RFM patterns may not perfectly correlate with actual defaults
2. **Selection Bias**: Model trained on observed customers may not generalize to new applicants
3. **Feedback Loops**: Predictions influence behavior, which influences future predictions

**Mitigation**: Commit to validation once actual default data becomes available.

---

## 5. Technical Architecture

### 5.1 Data Flow

```
Raw Transactions → Feature Engineering → Model Training → API Deployment
     (CSV)              (Pipeline)           (Joblib)       (FastAPI)
```

### 5.2 API Design

**Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Single customer prediction
- `POST /predict/batch` - Batch predictions
- `GET /features` - Feature list

**Response Schema:**
```json
{
  "customer_id": "CUST_001",
  "predicted_class": 0,
  "risk_probability": 0.15,
  "credit_score": 742,
  "decision": "Approve"
}
```

### 5.3 Containerization

Docker-based deployment ensures:
- Reproducible environments
- Easy scaling
- Isolation from host system
- Consistent behavior across platforms

---

## 6. Results Summary

### 6.1 Model Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| ROC-AUC | 1.0000 | Perfect ranking of risk |
| Accuracy | 99.20% | Correct classification rate |
| Precision | 99.86% | Low false positive rate |
| Recall | 99.31% | High default detection |
| F1 Score | 99.59% | Balanced performance |

### 6.2 Business Impact

- **Immediate Deployment**: Proxy target enables immediate scoring without waiting for defaults
- **Transparent Scoring**: Decision Tree provides interpretable risk factors
- **Scalable Architecture**: API-based deployment supports high-volume scoring
- **Quality Assurance**: Comprehensive testing ensures reliability

---

## 7. Future Roadmap

### Short-term (1-3 months)
- [ ] Validate proxy against actual default outcomes
- [ ] Implement SHAP values for individual explanations
- [ ] Add fairness metrics across demographic groups

### Medium-term (3-6 months)
- [ ] Set up automated retraining pipeline
- [ ] Implement A/B testing with shadow deployment
- [ ] Add real-time drift detection

### Long-term (6-12 months)
- [ ] Integrate with external credit bureaus
- [ ] Build ensemble with multiple proxy variables
- [ ] Develop sector-specific credit scores

---

## 8. Conclusion

This project demonstrates a production-grade credit risk system built with engineering excellence. Key achievements:

✅ **Technical Excellence**: 46 tests, comprehensive pipeline, CI/CD automation
✅ **Business Value**: Immediate risk scoring without historical defaults
✅ **Regulatory Alignment**: Interpretable Decision Tree for Basel II compliance
✅ **Operational Reliability**: Containerized API with health monitoring

The solution is ready for production deployment, with clear documentation for maintenance and extension.

---

## References

1. Basel II Capital Accord - Basel Committee on Banking Supervision
2. XGBoost: A Scalable Tree Boosting System - Chen & Guestrin
3. scikit-learn: Machine Learning in Python - Pedregosa et al.
4. Credit Risk Modeling with Alternative Data - HKMA Guidance
