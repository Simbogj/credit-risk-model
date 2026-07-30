"""
Credit Risk Model - Streamlit Dashboard
=======================================
Interactive dashboard for credit risk model visualization and explainability.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Safe import of Streamlit
try:
    import streamlit as st
except ModuleNotFoundError:
    class _MockSt:
        def warning(self, *args, **kwargs):
            pass
        def cache_data(self, func):
            return func
        def cache_resource(self, func):
            return func
        def set_page_config(self, **kwargs):
            pass
        def markdown(self, *args, **kwargs):
            pass
        def metric(self, *args, **kwargs):
            pass
        def subheader(self, *args, **kwargs):
            pass
        def header(self, *args, **kwargs):
            pass
        def write(self, *args, **kwargs):
            pass
        def columns(self, n):
            return [_MockSt() for _ in range(n)]
        def plotly_chart(self, *args, **kwargs):
            pass
        def sidebar(self):
            return self
        def selectbox(self, *args, **kwargs):
            return None
        def radio(self, *args, **kwargs):
            return None
        def button(self, *args, **kwargs):
            return False
    st = _MockSt()

# Safe import of Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    st.warning("Plotly not installed; visualizations will be disabled.")
    px = None
    go = None
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Page config
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "DecisionTree_best.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "feature_names.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_data.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics_summary.json"
COMPARISON_PATH = PROJECT_ROOT / "models" / "model_comparison.csv"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data. If the default data file is missing, create a small synthetic dataset.
    This ensures the dashboard runs even on a fresh clone without data.
    """
    if DATA_PATH.is_file():
        return pd.read_csv(DATA_PATH)
    else:
        st.warning("Processed data not found. Using synthetic demo data.")
        # Define minimal required columns for the dashboard and prediction pages
        cols = [
            "is_high_risk",
            "AvgAmount",
            "TransactionCount",
            "TotalAmount",
            "Recency",
            "Frequency",
            "Monetary",
            "FraudRate",
            "StdAmount",
            "CreditRatio",
            "DebitRatio",
        ]
        data = {
            "is_high_risk": [0, 1],
            "AvgAmount": [1200.0, 3000.0],
            "TransactionCount": [10, 30],
            "TotalAmount": [12000.0, 90000.0],
            "Recency": [5, 20],
            "Frequency": [15, 40],
            "Monetary": [18000.0, 120000.0],
            "FraudRate": [0.02, 0.15],
            "StdAmount": [200.0, 600.0],
            "CreditRatio": [0.2, 0.8],
            "DebitRatio": [0.8, 0.2],
        }
        df = pd.DataFrame(data)
        # Ensure column order matches expected
        return df[cols]


@st.cache_resource
def load_model():
    """Load trained model."""
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_resource
def load_features():
    """Load feature names."""
    features = joblib.load(FEATURES_PATH)
    return features


@st.cache_data
def load_metrics():
    """Load metrics summary."""
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)


@st.cache_data
def load_comparison():
    """Load model comparison."""
    return pd.read_csv(COMPARISON_PATH)


def main():
    # Header
    st.markdown('<p class="main-header">📊 Credit Risk Probability Model Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    model = load_model()
    features = load_features()
    metrics = load_metrics()
    comparison = load_comparison()
    
    # Sidebar
    st.sidebar.header("🔧 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📈 Overview", "🔮 Model Performance", "📊 Feature Analysis", "🎯 Prediction", "ℹ️ About"]
    )
    
    if page == "📈 Overview":
        overview_page(df, metrics, comparison, features)
    elif page == "🔮 Model Performance":
        performance_page(df, model, metrics)
    elif page == "📊 Feature Analysis":
        feature_analysis_page(df)
    elif page == "🎯 Prediction":
        prediction_page(model, features)
    elif page == "ℹ️ About":
        about_page()


def overview_page(df, metrics, comparison, features):
    """Overview dashboard page."""
    st.header("📈 Project Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        high_risk = (df['is_high_risk'] == 1).sum()
        st.metric("High Risk", f"{high_risk:,}", f"{high_risk/len(df)*100:.1f}%")
    
    with col3:
        low_risk = (df['is_high_risk'] == 0).sum()
        st.metric("Low Risk", f"{low_risk:,}", f"{low_risk/len(df)*100:.1f}%")
    
    with col4:
        st.metric("Features", len(features))
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("🏆 Model Comparison")
    
    fig = px.bar(
        comparison.melt(id_vars='Model', var_name='Metric', value_name='Score'),
        x='Model',
        y='Score',
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        height=400
    )
    fig.update_layout(
        yaxis_title='Score',
        xaxis_title='Model',
        legend_title='Metric'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Best model summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎖️ Best Model")
        st.success(f"**{metrics['best_model']}**")
        st.write(f"ROC-AUC: **{metrics['best_roc_auc']:.4f}**")
    
    with col2:
        st.subheader("📋 All Metrics")
        for key, value in metrics.items():
            if key != 'best_model':
                st.write(f"- {key.replace('_', ' ').title()}: **{value:.4f}**")
    
    st.markdown("---")
    
    # Target distribution
    st.subheader("🎯 Target Distribution")
    
    fig = px.pie(
        df['is_high_risk'].value_counts().reset_index(),
        values='is_high_risk',
        names=['Low Risk (0)', 'High Risk (1)'],
        title='Risk Distribution',
        hole=0.4
    )
    fig.update_traces(marker=dict(colors=['#2ecc71', '#e74c3c']))
    st.plotly_chart(fig, use_container_width=True)


def performance_page(df, model, metrics):
    """Model performance page."""
    st.header("🔮 Model Performance")
    
    # Split data
    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']
    
    # Calculate predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['best_accuracy']:.2%}")
    with col2:
        st.metric("ROC-AUC", f"{metrics['best_roc_auc']:.4f}")
    with col3:
        st.metric("Precision", f"{metrics['best_precision']:.2%}")
    with col4:
        st.metric("Recall", f"{metrics['best_recall']:.2%}")
    with col5:
        st.metric("F1 Score", f"{metrics['best_f1']:.2%}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    
    cm = confusion_matrix(y, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Low Risk', 'Predicted High Risk'],
        y=['Actual Low Risk', 'Actual High Risk'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        showscale=False
    ))
    fig.update_layout(
        title='Confusion Matrix',
        height=400,
        width=500
    )
    st.plotly_chart(fig, use_container_width=False)
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("📋 Classification Report")
    
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2%}"))
    
    st.markdown("---")
    
    # ROC Curve visualization
    st.subheader("📈 Risk Score Distribution")
    
    fig = px.histogram(
        df.assign(risk_score=y_proba),
        x='risk_score',
        color='is_high_risk',
        nbins=50,
        title='Risk Probability Distribution',
        labels={'is_high_risk': 'Actual Risk', 'risk_score': 'Predicted Risk Probability'}
    )
    fig.update_layout(
        barmode='overlay',
        height=400
    )
    fig.update_traces(opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)


def feature_analysis_page(df):
    """Feature analysis page with SHAP-like visualizations."""
    st.header("📊 Feature Analysis")
    
    # Calculate feature importance (using tree-based feature importance)
    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']
    
    # Fit a quick model for feature importance
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 20 features
    top_20 = importance_df.head(20)
    
    st.subheader("🔝 Top 20 Most Important Features")
    
    fig = px.bar(
        top_20,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance (Random Forest)',
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=600,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature correlation with target
    st.subheader("📈 Feature Correlation with Risk")
    
    correlations = df.corr()['is_high_risk'].drop('is_high_risk').sort_values(key=abs, ascending=False)
    
    fig = px.bar(
        x=correlations.head(20).values,
        y=correlations.head(20).index,
        orientation='h',
        title='Top 20 Features by Correlation with Risk',
        color=correlations.head(20).values,
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(
        height=600,
        xaxis_title='Correlation',
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature statistics
    st.subheader("📊 Feature Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### High Risk Customers")
        high_risk_df = df[df['is_high_risk'] == 1]
        st.write(f"Count: {len(high_risk_df):,}")
        st.write(f"Avg Transaction Amount: ${high_risk_df['AvgAmount'].mean():,.2f}")
        st.write(f"Avg Transaction Count: {high_risk_df['TransactionCount'].mean():.1f}")
    
    with col2:
        st.write("### Low Risk Customers")
        low_risk_df = df[df['is_high_risk'] == 0]
        st.write(f"Count: {len(low_risk_df):,}")
        st.write(f"Avg Transaction Amount: ${low_risk_df['AvgAmount'].mean():,.2f}")
        st.write(f"Avg Transaction Count: {low_risk_df['TransactionCount'].mean():.1f}")


def prediction_page(model, features):
    """Customer prediction page."""
    st.header("🎯 Risk Prediction")
    
    st.markdown("""
    Enter customer features below to predict their credit risk probability.
    """)
    
    # Sample data for defaults
    df = load_data()
    
    # Input fields
    st.subheader("📝 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_amount = st.number_input("Total Amount", value=25000.0, step=100.0)
        avg_amount = st.number_input("Average Amount", value=1250.0, step=50.0)
        transaction_count = st.number_input("Transaction Count", value=20, step=1)
        recency = st.number_input("Recency (days)", value=5, step=1)
    
    with col2:
        frequency = st.number_input("Frequency", value=20, step=1)
        monetary = st.number_input("Monetary Value", value=25000.0, step=100.0)
        fraud_rate = st.slider("Fraud Rate", 0.0, 1.0, 0.05)
    
    with col3:
        std_amount = st.number_input("Std Amount", value=500.0, step=50.0)
        credit_ratio = st.slider("Credit Ratio", 0.0, 1.0, 0.1)
        debit_ratio = st.slider("Debit Ratio", 0.0, 1.0, 0.9)
    
    # Create feature vector
    if st.button("🔮 Predict Risk", type="primary"):
        # Create feature dict with all zeros
        feature_values = {f: 0.0 for f in features}
        
        # Update with user inputs
        feature_values['TotalAmount'] = total_amount
        feature_values['AvgAmount'] = avg_amount
        feature_values['TransactionCount'] = transaction_count
        feature_values['Recency'] = recency
        feature_values['Frequency'] = frequency
        feature_values['Monetary'] = monetary
        feature_values['FraudRate'] = fraud_rate
        feature_values['StdAmount'] = std_amount
        feature_values['CreditRatio'] = credit_ratio
        feature_values['DebitRatio'] = debit_ratio
        
        # Create feature vector in correct order
        X_pred = pd.DataFrame([feature_values])[features]
        
        # Predict
        prediction = model.predict(X_pred)[0]
        probability = model.predict_proba(X_pred)[0]
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_class = "HIGH RISK" if prediction == 1 else "LOW RISK"
            color = "🔴" if prediction == 1 else "🟢"
            st.markdown(f"### {color} {risk_class}")
        
        with col2:
            st.metric("High Risk Probability", f"{probability[1]*100:.1f}%")
        
        with col3:
            st.metric("Low Risk Probability", f"{probability[0]*100:.1f}%")
        
        # Credit score (FICO-style)
        credit_score = int(300 + (1 - probability[1]) * 550)
        
        if probability[1] < 0.3:
            decision = "✅ APPROVE"
            decision_color = "success"
        elif probability[1] < 0.6:
            decision = "⚠️ REVIEW"
            decision_color = "warning"
        else:
            decision = "❌ DECLINE"
            decision_color = "error"
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Credit Score (FICO-style)", credit_score)
        
        with col2:
            if decision == "✅ APPROVE":
                st.success(f"### {decision}")
            elif decision == "⚠️ REVIEW":
                st.warning(f"### {decision}")
            else:
                st.error(f"### {decision}")


def about_page():
    """About page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Credit Risk Probability Model for Alternative Data
    
    This dashboard showcases an end-to-end credit risk scoring solution developed for **Bati Bank**,
    a leading financial service provider partnering with eCommerce platforms to enable
    buy-now-pay-later services.
    
    ### Key Features
    
    - **RFM-Based Risk Proxy**: Uses Recency, Frequency, and Monetary analysis to identify
      high-risk customer segments without historical default labels
    
    - **Multiple Model Comparison**: Trains and compares Logistic Regression, Decision Tree,
      Random Forest, and Gradient Boosting models
    
    - **SHAP-like Explainability**: Feature importance analysis to understand model predictions
    
    - **Real-time API**: FastAPI-based REST API for production deployment
    
    ### Business Impact
    
    - Enables Bati Bank to assess credit risk for new customers using alternative data
    - Provides interpretable risk scores for regulatory compliance (Basel II)
    - Supports automated credit decisions with configurable thresholds
    
    ### Technology Stack
    
    - **Python 3.9+** - Core programming language
    - **scikit-learn** - Machine learning pipeline
    - **pandas/NumPy** - Data processing
    - **Streamlit** - Interactive dashboard
    - **FastAPI** - REST API
    - **Docker** - Containerization
    
    ### Model Performance
    
    The best model achieves:
    - **ROC-AUC**: ~1.0 (on proxy target)
    - **Accuracy**: ~99%
    - **Precision**: ~99%
    
    > ⚠️ **Note**: High performance metrics are expected since we're predicting the RFM-based
    > proxy target, which is derived from the same features used for prediction.
    > Real-world validation would require actual default outcomes.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Regulatory Considerations (Basel II)
    
    This model aligns with Basel II requirements for:
    
    1. **Interpretability**: Decision Tree provides transparent decision rules
    2. **Documentation**: All modeling choices are documented
    3. **Monitoring**: Framework for ongoing model performance tracking
    4. **Validation**: Clear methodology for proxy target validation
    
    ### Future Improvements
    
    - Validate proxy against actual default outcomes once available
    - Implement SHAP values for individual prediction explanations
    - Add fairness metrics for demographic groups
    - Set up automated retraining pipeline
    """)


if __name__ == "__main__":
    main()
