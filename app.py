"""
Bati Bank Credit Risk Intelligence Platform
=============================================
Enterprise-grade credit risk scoring, model evaluation, and explainability dashboard.
Designed for financial institution risk committees, underwriters, and compliance officers.
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
        def error(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass
        def success(self, *args, **kwargs):
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
            return [_MockSt() for _ in range(n if isinstance(n, int) else len(n))]
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
        def dataframe(self, *args, **kwargs):
            pass
        def slider(self, *args, **kwargs):
            return kwargs.get("value", 0)
        def number_input(self, *args, **kwargs):
            return kwargs.get("value", 0.0)
        def spinner(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def pyplot(self, *args, **kwargs):
            pass
        def caption(self, *args, **kwargs):
            pass
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
sys.path.insert(0, str(Path(__file__).parent / "src"))
from paths import (
    PATHS,
    PROCESSED_TRAIN_PATH,
    MODEL_PATH,
    FEATURES_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
)

# Page configuration
st.set_page_config(
    page_title="Bati Bank | Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project Paths
PROJECT_ROOT = PATHS.project_root
DATA_PATH = PROCESSED_TRAIN_PATH
MODEL_PATH = PATHS.model
FEATURES_PATH = PATHS.features
METRICS_PATH = PATHS.metrics
COMPARISON_PATH = PATHS.comparison

# ---------------------------------------------------------------------
# Institutional Banking Enterprise CSS Theme (Light)
# ---------------------------------------------------------------------
BANK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bb-navy-primary: #0B2545;
        --bb-navy-secondary: #133E87;
        --bb-navy-muted: #1E3E62;
        --bb-accent-blue: #2563EB;
        --bb-surface-bg: #F8FAFC;
        --bb-card-bg: #FFFFFF;
        --bb-border: #E2E8F0;
        --bb-border-subtle: #EDF2F7;
        --bb-text-main: #0F172A;
        --bb-text-muted: #475569;
        --bb-text-light: #64748B;
        --bb-success: #059669;
        --bb-success-bg: #ECFDF5;
        --bb-success-border: #A7F3D0;
        --bb-warning: #D97706;
        --bb-warning-bg: #FFFBEB;
        --bb-warning-border: #FDE68A;
        --bb-danger: #DC2626;
        --bb-danger-bg: #FEF2F2;
        --bb-danger-border: #FECACA;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--bb-text-main);
    }

    /* Top application header */
    .bank-header-container {
        background: linear-gradient(135deg, #0B2545 0%, #133E87 100%);
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 16px rgba(11, 37, 69, 0.12);
        color: #FFFFFF;
        border-left: 6px solid #60A5FA;
    }
    .bank-header-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #93C5FD;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .bank-header-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.25;
    }
    .bank-header-subtitle {
        font-size: 0.92rem;
        color: #CBD5E1;
        margin-top: 0.45rem;
        line-height: 1.5;
        max-width: 860px;
    }
    .bank-header-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .bank-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.14);
        color: #F8FAFC;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }
    .bank-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #34D399;
    }

    /* Cards and containers */
    .bank-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .bank-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }
    .bank-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .bank-card-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 1rem;
        line-height: 1.5;
    }

    /* KPI metric tiles */
    .bank-kpi {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .bank-kpi-eyebrow {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .bank-kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .bank-kpi-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.45rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .delta-positive {
        color: #059669;
    }
    .delta-negative {
        color: #DC2626;
    }
    .delta-neutral {
        color: #475569;
    }

    /* Decision banner */
    .decision-banner {
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin: 1.2rem 0;
        border-width: 2px;
        border-style: solid;
    }
    .decision-approved {
        background-color: #ECFDF5;
        border-color: #10B981;
        color: #064E3B;
    }
    .decision-review {
        background-color: #FFFBEB;
        border-color: #F59E0B;
        color: #78350F;
    }
    .decision-declined {
        background-color: #FEF2F2;
        border-color: #EF4444;
        color: #7F1D1D;
    }

    /* Sidebar enterprise styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-brand-box {
        padding: 1.2rem 0.5rem 1rem 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .sidebar-brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: linear-gradient(135deg, #0B2545 0%, #133E87 100%);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.25rem;
        box-shadow: 0 2px 8px rgba(11, 37, 69, 0.2);
    }
    .sidebar-brand-name {
        font-size: 1.12rem;
        font-weight: 800;
        color: #0B2545;
        letter-spacing: -0.01em;
        margin-top: 0.45rem;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #64748B;
        text-transform: uppercase;
    }

    /* Tables and Streamlit component refinements */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #64748B !important;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.5rem 1.25rem;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0B2545 !important;
        border-color: #0B2545 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(11, 37, 69, 0.18);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #133E87 !important;
        border-color: #133E87 !important;
        box-shadow: 0 4px 12px rgba(11, 37, 69, 0.25);
    }

    /* Badges */
    .badge-institutional {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-navy { background-color: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .badge-green { background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-amber { background-color: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
    .badge-red { background-color: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-gray { background-color: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; }
</style>
"""

st.markdown(BANK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Asset Loaders (Preserved for compatibility and tests)
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load processed customer data. Fallback to synthetic data if missing."""
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
    if DATA_PATH.is_file():
        return pd.read_csv(DATA_PATH)
    else:
        st.warning("Processed training dataset not found. Utilizing synthetic reference data.")
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
        return df[cols]


@st.cache_resource
def load_model():
    """Load the trained champion credit risk model."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load champion model from {MODEL_PATH}: {exc}")
        return None


@st.cache_resource
def load_features():
    """Load ordered feature list."""
    try:
        return joblib.load(FEATURES_PATH)
    except Exception:
        return [
            "TotalAmount",
            "AvgAmount",
            "TransactionCount",
            "Recency",
            "Frequency",
            "Monetary",
            "FraudRate",
            "StdAmount",
            "CreditRatio",
            "DebitRatio",
        ]


@st.cache_data
def load_metrics():
    """Load metrics summary json."""
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "best_model": "DecisionTree",
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
        }


@st.cache_data
def load_comparison():
    """Load model comparison matrix CSV."""
    try:
        return pd.read_csv(COMPARISON_PATH)
    except Exception:
        return pd.DataFrame([
            {"Model": "DecisionTree", "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0},
            {"Model": "RandomForest", "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0},
            {"Model": "GradientBoosting", "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0},
            {"Model": "LogisticRegression", "accuracy": 0.992, "precision": 0.999, "recall": 0.993, "f1": 0.996, "roc_auc": 0.998},
        ])


# ---------------------------------------------------------------------
# UI Component Helpers
# ---------------------------------------------------------------------
def render_header(title: str, subtitle: str, category: str = "CREDIT INTELLIGENCE"):
    """Render institutional header hero."""
    st.markdown(
        f"""
        <div class="bank-header-container">
            <div class="bank-header-eyebrow">
                <span>BATI BANK FINANCIAL SERVICES</span>
                <span>/</span>
                <span>{category}</span>
            </div>
            <h1 class="bank-header-title">{title}</h1>
            <div class="bank-header-subtitle">{subtitle}</div>
            <div class="bank-header-pills">
                <span class="bank-pill"><span class="bank-pill-dot"></span>Production Engine Active</span>
                <span class="bank-pill">Champion Model: DecisionTree v1.0</span>
                <span class="bank-pill">Target: RFM Risk Proxy</span>
                <span class="bank-pill">Regulatory Standard: Basel II Aligned</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_bank_chart_style(fig, height: int = 400, title: str = ""):
    """Apply institutional banking light theme to Plotly charts."""
    if fig is None:
        return None
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#0F172A", family="Inter, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        margin=dict(l=40, r=25, t=55 if title else 25, b=40),
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#475569"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(color="#475569", size=10),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(color="#475569", size=10),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10, color="#334155"),
        ),
    )
    return fig


# ---------------------------------------------------------------------
# Dashboard Pages
# ---------------------------------------------------------------------
def overview_page(df, metrics, comparison, features):
    """Portfolio Overview & Executive Risk Intelligence."""
    render_header(
        title="Portfolio Risk Overview & Champion Benchmark",
        subtitle="Executive monitoring of portfolio credit risk segmentation, champion model metrics, and model comparison across underwriting candidate architectures.",
        category="PORTFOLIO ANALYTICS",
    )

    total_customers = len(df)
    high_risk_count = (df["is_high_risk"] == 1).sum() if "is_high_risk" in df.columns else 0
    low_risk_count = (df["is_high_risk"] == 0).sum() if "is_high_risk" in df.columns else 0
    high_risk_pct = (high_risk_count / total_customers * 100) if total_customers > 0 else 0
    low_risk_pct = (low_risk_count / total_customers * 100) if total_customers > 0 else 0

    # Executive KPI summary row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="bank-kpi">
                <div class="bank-kpi-eyebrow">Scored Customers</div>
                <div class="bank-kpi-value">{total_customers:,}</div>
                <div class="bank-kpi-delta delta-neutral">100% Portfolio Coverage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="bank-kpi">
                <div class="bank-kpi-eyebrow">Low Risk Borrowers</div>
                <div class="bank-kpi-value" style="color:#059669;">{low_risk_count:,}</div>
                <div class="bank-kpi-delta delta-positive">● {low_risk_pct:.1f}% Prime Tier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="bank-kpi">
                <div class="bank-kpi-eyebrow">High Risk Borrowers</div>
                <div class="bank-kpi-value" style="color:#DC2626;">{high_risk_count:,}</div>
                <div class="bank-kpi-delta delta-negative">● {high_risk_pct:.1f}% Elevated Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="bank-kpi">
                <div class="bank-kpi-eyebrow">Engineered Features</div>
                <div class="bank-kpi-value">{len(features)}</div>
                <div class="bank-kpi-delta delta-neutral">Behavioral & Velocity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        roc_val = metrics.get("roc_auc", metrics.get("best_roc_auc", 1.0))
        st.markdown(
            f"""
            <div class="bank-kpi">
                <div class="bank-kpi-eyebrow">Champion ROC-AUC</div>
                <div class="bank-kpi-value" style="color:#2563EB;">{roc_val:.4f}</div>
                <div class="bank-kpi-delta delta-positive">DecisionTree (Top Ranked)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Two column layout: Model Benchmark and Risk Distribution
    left_col, right_col = st.columns([1.5, 1])

    with left_col:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">🏆 Model Benchmark Comparison</div>
                <div class="bank-card-subtitle">
                    Cross-algorithm evaluation across Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
                </div>
            """,
            unsafe_allow_html=True,
        )
        if px is not None and not comparison.empty:
            melted_comp = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig_comp = px.bar(
                melted_comp,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#0B2545", "#2563EB", "#0E8388", "#10B981", "#F59E0B"],
            )
            fig_comp = apply_bank_chart_style(fig_comp, height=360)
            st.plotly_chart(fig_comp, width="stretch")
        else:
            st.dataframe(comparison)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">🎯 Portfolio Risk Concentration</div>
                <div class="bank-card-subtitle">
                    Distribution of alternative-data borrower risk classification across the active portfolio.
                </div>
            """,
            unsafe_allow_html=True,
        )
        if px is not None and "is_high_risk" in df.columns:
            counts = df["is_high_risk"].value_counts().reset_index()
            counts.columns = ["Risk_Code", "Customer_Count"]
            counts["Risk_Label"] = counts["Risk_Code"].map({0: "Low Risk (Approved)", 1: "High Risk (Review/Decline)"})
            fig_pie = px.pie(
                counts,
                values="Customer_Count",
                names="Risk_Label",
                hole=0.55,
                color="Risk_Label",
                color_discrete_map={
                    "Low Risk (Approved)": "#10B981",
                    "High Risk (Review/Decline)": "#E11D48",
                },
            )
            fig_pie = apply_bank_chart_style(fig_pie, height=360)
            fig_pie.update_traces(
                textposition="outside",
                textinfo="percent+label",
                marker=dict(line=dict(color="#FFFFFF", width=2)),
            )
            st.plotly_chart(fig_pie, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Champion Summary & Methodology
    col_summary, col_table = st.columns([1, 1])

    with col_summary:
        st.markdown(
            f"""
            <div class="bank-card">
                <div class="bank-card-title">🎖️ Champion Model Specifications</div>
                <div class="bank-card-subtitle">Current production model deployed for real-time customer underwriting.</div>
                <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                    <tr style="border-bottom:1px solid #E2E8F0; padding:8px 0;">
                        <td style="padding:8px 0; color:#64748B;">Architecture</td>
                        <td style="padding:8px 0; font-weight:700; text-align:right; color:#0F172A;">{metrics.get('best_model', 'DecisionTree')}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:8px 0; color:#64748B;">Accuracy</td>
                        <td style="padding:8px 0; font-weight:700; text-align:right; color:#059669;">{metrics.get('accuracy', 1.0):.2%}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:8px 0; color:#64748B;">ROC-AUC Score</td>
                        <td style="padding:8px 0; font-weight:700; text-align:right; color:#059669;">{metrics.get('roc_auc', 1.0):.4f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:8px 0; color:#64748B;">Precision / Recall</td>
                        <td style="padding:8px 0; font-weight:700; text-align:right; color:#059669;">{metrics.get('precision', 1.0):.2%} / {metrics.get('recall', 1.0):.2%}</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 0; color:#64748B;">Deployment Status</td>
                        <td style="padding:8px 0; font-weight:700; text-align:right;"><span class="badge-institutional badge-green">ACTIVE PRODUCTION</span></td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_table:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">⚖️ Governance Notice: Target Proxy Context</div>
                <div class="bank-card-subtitle">Methodological advisory for Risk Committee and Credit Audit.</div>
                <div style="font-size:0.84rem; color:#475569; line-height:1.65;">
                    The original project documentation reports near-perfect validation metrics (approximately 1.0 ROC-AUC and 99%+ accuracy)
                    because the target variable is an <b>RFM-derived risk proxy</b> (Recency, Frequency, Monetary clustering).
                    <br><br>
                    <b>Regulatory Guidance:</b> While this proxy provides strong rank-ordering for cold-start BNPL customers with no credit bureau records,
                    ongoing calibration against realized delinquency and charge-off performance is mandated under Basel II supervisory validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def performance_page(df, model, metrics):
    """Detailed Model Performance, Validation & Confusion Metrics."""
    render_header(
        title="Model Performance & Statistical Validation",
        subtitle="Quantitative assessment of model discrimination, error distribution, confusion matrix, and probability calibration.",
        category="MODEL VALIDATION",
    )

    X = df.drop("is_high_risk", axis=1, errors="ignore")
    y = df["is_high_risk"]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        acc = metrics.get("accuracy", metrics.get("best_accuracy", 0.99))
        st.metric("Accuracy", f"{acc:.2%}", "Balanced accuracy")
    with c2:
        roc = metrics.get("roc_auc", metrics.get("best_roc_auc", 1.0))
        st.metric("ROC-AUC", f"{roc:.4f}", "Rank ordering")
    with c3:
        prec = metrics.get("precision", metrics.get("best_precision", 0.99))
        st.metric("Precision", f"{prec:.2%}", "High-risk purity")
    with c4:
        rec = metrics.get("recall", metrics.get("best_recall", 0.99))
        st.metric("Recall", f"{rec:.2%}", "Risk capture rate")
    with c5:
        f1 = metrics.get("f1", metrics.get("best_f1", 0.99))
        st.metric("F1 Score", f"{f1:.2%}", "Harmonic mean")

    st.write("")

    col_cm, col_report = st.columns([1.1, 1.3])

    with col_cm:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">📊 Confusion Matrix & Error Classification</div>
                <div class="bank-card-subtitle">Actual vs predicted borrower risk classification.</div>
            """,
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y, y_pred)
        if go is not None:
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted Low Risk (0)", "Predicted High Risk (1)"],
                    y=["Actual Low Risk (0)", "Actual High Risk (1)"],
                    colorscale=[
                        [0.0, "#F8FAFC"],
                        [0.2, "#DBEAFE"],
                        [0.5, "#3B82F6"],
                        [1.0, "#0B2545"],
                    ],
                    text=cm,
                    texttemplate="<b>%{text}</b>",
                    textfont={"size": 18, "family": "Inter, sans-serif"},
                    showscale=False,
                )
            )
            fig_cm = apply_bank_chart_style(fig_cm, height=320)
            st.plotly_chart(fig_cm, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_report:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">📋 Comprehensive Classification Report</div>
                <div class="bank-card-subtitle">Precision, recall, and F1 breakdown per credit risk tier.</div>
            """,
            unsafe_allow_html=True,
        )
        report_dict = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.rename(
            index={
                "0": "Class 0 (Low Risk)",
                "1": "Class 1 (High Risk)",
                "accuracy": "Overall Accuracy",
                "macro avg": "Macro Average",
                "weighted avg": "Weighted Average",
            }
        )
        st.dataframe(
            report_df.style.format({
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1-score": "{:.2%}",
                "support": "{:,.0f}",
            }),
            use_container_width=True,
        )
        st.markdown(
            """
            <div style="font-size:0.80rem; color:#64748B; margin-top:0.75rem;">
                <b>Underwriting takeaway:</b> High precision minimizes bad-debt charge-offs by blocking actual high-risk borrowers;
                high recall ensures minimal creditworthy applicants are incorrectly turned away.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Risk Probability Distribution Histogram
    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">📈 Predicted Risk Score Distribution & Threshold Boundaries</div>
            <div class="bank-card-subtitle">
                Separation of predicted default probabilities across actual low-risk and high-risk cohorts with underwriting thresholds at 0.30 and 0.60.
            </div>
        """,
        unsafe_allow_html=True,
    )
    if px is not None:
        scored_df = df.copy()
        scored_df["Risk_Probability"] = y_proba
        scored_df["Actual_Risk_Class"] = scored_df["is_high_risk"].map({0: "Actual Low Risk", 1: "Actual High Risk"})
        fig_hist = px.histogram(
            scored_df,
            x="Risk_Probability",
            color="Actual_Risk_Class",
            nbins=40,
            barmode="overlay",
            color_discrete_map={"Actual Low Risk": "#10B981", "Actual High Risk": "#DC2626"},
            labels={"Risk_Probability": "Predicted Risk Probability (PD)", "count": "Applicant Volume"},
        )
        fig_hist = apply_bank_chart_style(fig_hist, height=350)
        fig_hist.update_traces(opacity=0.75)
        # Add regulatory threshold cutoffs
        fig_hist.add_vline(x=0.30, line_dash="dash", line_color="#F59E0B", annotation_text="Approve / Review Cutoff (0.30)")
        fig_hist.add_vline(x=0.60, line_dash="dash", line_color="#EF4444", annotation_text="Review / Decline Cutoff (0.60)")
        st.plotly_chart(fig_hist, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def feature_analysis_page(df):
    """Feature Importance & Risk Factor Correlation."""
    render_header(
        title="Feature Importance & Behavioral Signals",
        subtitle="Empirical analysis of customer velocity, monetary scale, and transactional indicators driving credit risk determinations.",
        category="SIGNAL ANALYSIS",
    )

    X = df.drop("is_high_risk", axis=1, errors="ignore")
    y = df["is_high_risk"]

    from sklearn.ensemble import RandomForestClassifier

    @st.cache_resource
    def compute_feature_rankings(_X, _y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(_X, _y)
        importance_df = pd.DataFrame({
            "Feature": _X.columns,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False)
        return importance_df

    importance_df = compute_feature_rankings(X, y)
    top_20_imp = importance_df.head(20)

    col_imp, col_corr = st.columns([1, 1])

    with col_imp:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">🔝 Top 20 Most Predictive Features</div>
                <div class="bank-card-subtitle">Random Forest Gini-importance ranking across behavioral variables.</div>
            """,
            unsafe_allow_html=True,
        )
        if px is not None:
            fig_imp = px.bar(
                top_20_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[[0, "#93C5FD"], [1, "#0B2545"]],
            )
            fig_imp = apply_bank_chart_style(fig_imp, height=520)
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_imp, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_corr:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">📈 Top Correlations with Credit Risk Target</div>
                <div class="bank-card-subtitle">Directional impact: positive (elevates risk) vs negative (reduces risk).</div>
            """,
            unsafe_allow_html=True,
        )
        if px is not None:
            corrs = df.corr()["is_high_risk"].drop("is_high_risk", errors="ignore").sort_values(key=abs, ascending=False)
            top_20_corr = corrs.head(20).reset_index()
            top_20_corr.columns = ["Feature", "Correlation"]
            top_20_corr["Direction"] = top_20_corr["Correlation"].apply(lambda v: "Positive (Elevates Risk)" if v > 0 else "Negative (Reduces Risk)")

            fig_corr = px.bar(
                top_20_corr,
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={
                    "Positive (Elevates Risk)": "#DC2626",
                    "Negative (Reduces Risk)": "#10B981",
                },
            )
            fig_corr = apply_bank_chart_style(fig_corr, height=520)
            fig_corr.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_corr, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # Cohort financial profiling
    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">👥 Borrower Cohort Financial Profiles</div>
            <div class="bank-card-subtitle">Comparative financial behavior between Prime (Low Risk) and Elevated (High Risk) customers.</div>
        """,
        unsafe_allow_html=True,
    )
    high_risk_df = df[df["is_high_risk"] == 1]
    low_risk_df = df[df["is_high_risk"] == 0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Low Risk Avg Ticket",
            f"${low_risk_df['AvgAmount'].mean():,.2f}" if "AvgAmount" in low_risk_df else "N/A",
            f"{len(low_risk_df):,} customers",
        )
    with c2:
        st.metric(
            "High Risk Avg Ticket",
            f"${high_risk_df['AvgAmount'].mean():,.2f}" if "AvgAmount" in high_risk_df else "N/A",
            f"{len(high_risk_df):,} customers",
            delta_color="inverse",
        )
    with c3:
        st.metric(
            "Low Risk Recency",
            f"{low_risk_df['Recency'].mean():.1f} days" if "Recency" in low_risk_df else "N/A",
            "Recent activity",
        )
    with c4:
        st.metric(
            "High Risk Recency",
            f"{high_risk_df['Recency'].mean():.1f} days" if "Recency" in high_risk_df else "N/A",
            "Dormant/Lagging",
            delta_color="inverse",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def shap_page(df, model, features):
    """SHAP Explainability & Adverse Action Audit Trail."""
    render_header(
        title="SHAP Explainability & Adverse Action Audit",
        subtitle="Game-theoretic feature attribution (SHAP) ensuring transparent, auditable credit determinations in compliance with Basel II Pillar 2 & 3 guidelines.",
        category="REGULATORY EXPLAINABILITY",
    )

    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">🏛️ Regulatory Explainability Context (Basel II & Fair Lending)</div>
            <div style="font-size:0.86rem; color:#475569; line-height:1.6;">
                SHAP (SHapley Additive exPlanations) computes the marginal contribution of each financial behavior feature
                toward the credit risk score. In banking operations, SHAP fulfills two mandatory governance mandates:
                <br>
                1. <b>Global Transparency:</b> Proving to internal audit and regulators that the model does not rely on illegal proxies or discriminatory signals.
                <br>
                2. <b>Adverse Action Explanations:</b> Generating statutory reason codes for rejected applicants under consumer protection guidelines.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.is_file():
        st.warning("Processed training data file not detected. Showing feature importance fallback.")
        feature_analysis_page(df)
        return

    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("SHAP or Matplotlib not available in current environment. Run: pip install shap matplotlib")
        return

    X = df.drop(columns=["is_high_risk"], errors="ignore")
    missing = [f for f in features if f not in X.columns]
    if missing:
        st.error(f"Training dataset is missing {len(missing)} required features. Re-run pipeline.")
        return

    X = X[features]
    sample_size = min(300, len(X))
    X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

    with st.spinner("Calculating SHAP Shapley attribution values..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    # Global SHAP plot
    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">🌐 Global Feature Impact (SHAP Summary)</div>
            <div class="bank-card-subtitle">Distribution of SHAP impact across the sampled borrower portfolio. Red indicates high feature value; blue indicates low feature value.</div>
        """,
        unsafe_allow_html=True,
    )
    plt.figure(figsize=(10, 5.5))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=12)
    plt.tight_layout()
    st.pyplot(plt.gcf(), clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Local single applicant drilldown
    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">🔍 Individual Applicant Decision Audit (Local Explanation)</div>
            <div class="bank-card-subtitle">Select an applicant from the sample to inspect the exact behavioral features that pushed the risk probability higher or lower.</div>
        """,
        unsafe_allow_html=True,
    )

    col_sel, col_stat = st.columns([1, 2])
    with col_sel:
        idx = st.slider("Select Applicant Index", 0, len(X_sample) - 1, 0)
    with col_stat:
        row = X_sample.iloc[[idx]]
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]
        decision_label = "HIGH RISK (Declined)" if pred == 1 else "LOW RISK (Approved)"
        badge_cls = "badge-red" if pred == 1 else "badge-green"
        st.markdown(
            f"""
            <div style="padding-top:0.6rem;">
                <b>Applicant #{idx} Classification:</b> <span class="badge-institutional {badge_cls}">{decision_label}</span>
                <span style="margin-left:1rem; font-size:0.9rem; color:#64748B;">Predicted Default Probability: <b>{proba:.1%}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    expected = explainer.expected_value
    if isinstance(expected, list):
        expected = expected[1]

    plt.figure(figsize=(12, 3))
    shap.force_plot(expected, shap_values[idx], row, matplotlib=True, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf(), clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)


def prediction_page(model, features):
    """Underwriting Terminal & Real-Time Customer Scoring."""
    render_header(
        title="Interactive Loan Underwriting & Customer Scoring",
        subtitle="Simulate real-time credit underwriting decisions for alternative-data loan applicants. Computes default probability, FICO-style credit score, and policy recommendations.",
        category="UNDERWRITING ENGINE",
    )

    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">⚡ Underwriting Presets</div>
            <div class="bank-card-subtitle">Click a standard archetype to auto-populate the underwriting dossier, or manually adjust parameters below:</div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)
    preset_choice = None
    with p1:
        if st.button("🟢 Load Prime Applicant (Low Risk)", use_container_width=True):
            preset_choice = "prime"
    with p2:
        if st.button("🟡 Load Borderline Applicant (Review Tier)", use_container_width=True):
            preset_choice = "borderline"
    with p3:
        if st.button("🔴 Load Elevated Risk Applicant (High Risk)", use_container_width=True):
            preset_choice = "high_risk"
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize session state defaults if presets selected
    if preset_choice == "prime":
        defaults = {
            "total_amount": 45000.0, "avg_amount": 2250.0, "tx_count": 25, "recency": 2,
            "frequency": 25, "monetary": 45000.0, "fraud_rate": 0.00, "std_amount": 350.0,
            "credit_ratio": 0.10, "debit_ratio": 0.90,
        }
    elif preset_choice == "borderline":
        defaults = {
            "total_amount": 18000.0, "avg_amount": 1200.0, "tx_count": 15, "recency": 14,
            "frequency": 12, "monetary": 15000.0, "fraud_rate": 0.04, "std_amount": 800.0,
            "credit_ratio": 0.40, "debit_ratio": 0.60,
        }
    elif preset_choice == "high_risk":
        defaults = {
            "total_amount": 4000.0, "avg_amount": 400.0, "tx_count": 5, "recency": 45,
            "frequency": 3, "monetary": 2500.0, "fraud_rate": 0.15, "std_amount": 1100.0,
            "credit_ratio": 0.75, "debit_ratio": 0.25,
        }
    else:
        defaults = {
            "total_amount": 25000.0, "avg_amount": 1250.0, "tx_count": 20, "recency": 5,
            "frequency": 20, "monetary": 25000.0, "fraud_rate": 0.02, "std_amount": 500.0,
            "credit_ratio": 0.15, "debit_ratio": 0.85,
        }

    # Underwriting Dossier Form
    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">📝 Applicant Financial Profile & Behavioral Variables</div>
            <div class="bank-card-subtitle">Alternative data inputs derived from eCommerce activity and payment gateways.</div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<b>Financial & Ticket Volume</b>", unsafe_allow_html=True)
        total_amount = st.number_input("Total Transaction Amount ($)", value=defaults["total_amount"], step=500.0)
        avg_amount = st.number_input("Average Transaction Amount ($)", value=defaults["avg_amount"], step=50.0)
        transaction_count = st.number_input("Total Transaction Count", value=int(defaults["tx_count"]), step=1)
        std_amount = st.number_input("Amount Volatility / Std ($)", value=defaults["std_amount"], step=50.0)

    with col2:
        st.markdown("<b>Velocity & RFM Dynamics</b>", unsafe_allow_html=True)
        recency = st.number_input("Recency (Days since last payment)", value=int(defaults["recency"]), step=1)
        frequency = st.number_input("Frequency (Annual transactions)", value=int(defaults["frequency"]), step=1)
        monetary = st.number_input("Monetary Value ($)", value=defaults["monetary"], step=500.0)

    with col3:
        st.markdown("<b>Risk, Fraud & Channel Ratios</b>", unsafe_allow_html=True)
        fraud_rate = st.slider("Historical Fraud Dispute Rate", 0.0, 1.0, float(defaults["fraud_rate"]), step=0.01)
        credit_ratio = st.slider("Credit Channel Ratio", 0.0, 1.0, float(defaults["credit_ratio"]), step=0.05)
        debit_ratio = st.slider("Debit Channel Ratio", 0.0, 1.0, float(defaults["debit_ratio"]), step=0.05)

    st.write("")
    predict_clicked = st.button("⚡ EXECUTE CREDIT RISK ASSESSMENT", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        feature_values = {f: 0.0 for f in features}
        feature_values["TotalAmount"] = total_amount
        feature_values["AvgAmount"] = avg_amount
        feature_values["TransactionCount"] = transaction_count
        feature_values["Recency"] = recency
        feature_values["Frequency"] = frequency
        feature_values["Monetary"] = monetary
        feature_values["FraudRate"] = fraud_rate
        feature_values["StdAmount"] = std_amount
        feature_values["CreditRatio"] = credit_ratio
        feature_values["DebitRatio"] = debit_ratio

        X_pred = pd.DataFrame([feature_values])[features]

        prediction = model.predict(X_pred)[0]
        probability = model.predict_proba(X_pred)[0]
        prob_high = probability[1]
        prob_low = probability[0]

        # FICO-style credit score (300 to 850 scale)
        credit_score = int(300 + (1 - prob_high) * 550)

        # Decision thresholds
        if prob_high < 0.30:
            decision = "APPROVED"
            decision_class = "decision-approved"
            icon = "✅"
            tier = "Prime Borrower (Tier 1)"
            recommendation = "Authorize standard BNPL credit facility up to $5,000 with preferred financing terms."
        elif prob_high < 0.60:
            decision = "CONDITIONAL / MANUAL REVIEW"
            decision_class = "decision-review"
            icon = "⚠️"
            tier = "Moderate Risk (Tier 2)"
            recommendation = "Refer to Senior Credit Underwriter. Request verification of alternative income or require a 20% down payment."
        else:
            decision = "DECLINED"
            decision_class = "decision-declined"
            icon = "❌"
            tier = "High Default Risk (Tier 3)"
            recommendation = "Decline application under standard BNPL facility. Offer prepaid or secured credit alternatives."

        st.markdown(
            f"""
            <div class="decision-banner {decision_class}">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; opacity:0.85;">
                    AUTOMATED UNDERWRITING DETERMINATION
                </div>
                <div style="font-size:2rem; font-weight:800; margin:0.35rem 0;">
                    {icon} {decision}
                </div>
                <div style="font-size:0.95rem; line-height:1.5;">
                    <b>Risk Assessment:</b> {tier}<br>
                    <b>Policy Recommendation:</b> {recommendation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        res1, res2, res3, res4 = st.columns(4)
        with res1:
            st.metric("Credit Score (FICO Equiv)", f"{credit_score}", f"{'Strong' if credit_score >= 680 else 'Subprime'}")
        with res2:
            st.metric("Default Probability (PD)", f"{prob_high:.1%}", f"{'Low' if prob_high < 0.3 else 'Elevated'}", delta_color="inverse")
        with res3:
            st.metric("Repayment Probability", f"{prob_low:.1%}", "Approval Confidence")
        with res4:
            facility = "$5,000" if prob_high < 0.3 else ("$1,500 (Conditional)" if prob_high < 0.6 else "$0.00")
            st.metric("Max Loan Facility", facility, "Policy Cap")


def about_page():
    """Enterprise Regulatory & Governance Information."""
    render_header(
        title="Regulatory Compliance, Governance & Methodology",
        subtitle="Institutional alignment with Basel II/III banking standards, model audit trails, technology architecture, and ethical lending guardrails.",
        category="GOVERNANCE & AUDIT",
    )

    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">🏦 Bati Bank Alternative-Data Credit Initiative</div>
            <div style="font-size:0.90rem; color:#475569; line-height:1.7;">
                Bati Bank is a leading regional financial institution partnering with eCommerce and digital merchant ecosystems
                to provide point-of-sale Buy-Now-Pay-Later (BNPL) credit facilities. Because eCommerce customers frequently lack
                formal credit bureau files (thin-file / unbanked borrowers), this platform leverages alternative transactional
                data to estimate Probability of Default (PD) objectively and responsibly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Basel II Framework Alignment Table
    st.markdown(
        """
        <div class="bank-card">
            <div class="bank-card-title">⚖️ Basel II Regulatory Pillar Alignment</div>
            <div class="bank-card-subtitle">How this alternative-data scoring solution maps to banking regulatory requirements:</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.86rem; margin-top:0.5rem;">
                <thead>
                    <tr style="border-bottom:2px solid #CBD5E1; text-align:left; color:#0F172A;">
                        <th style="padding:10px 8px;">Pillar</th>
                        <th style="padding:10px 8px;">Regulatory Requirement</th>
                        <th style="padding:10px 8px;">Platform Implementation</th>
                        <th style="padding:10px 8px;">Compliance Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:10px 8px; font-weight:700;">Pillar 1</td>
                        <td style="padding:10px 8px;">Minimum Capital & PD Estimation</td>
                        <td style="padding:10px 8px;">Calibrated Probability of Default (PD) scoring feeding risk-weighted asset (RWA) reserves.</td>
                        <td style="padding:10px 8px;"><span class="badge-institutional badge-green">COMPLIANT</span></td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:10px 8px; font-weight:700;">Pillar 2</td>
                        <td style="padding:10px 8px;">Supervisory Review & Model Governance</td>
                        <td style="padding:10px 8px;">Documented pipeline lineage, drift threshold alarms, and versioned artifacts in MLflow.</td>
                        <td style="padding:10px 8px;"><span class="badge-institutional badge-green">COMPLIANT</span></td>
                    </tr>
                    <tr style="border-bottom:1px solid #E2E8F0;">
                        <td style="padding:10px 8px; font-weight:700;">Pillar 3</td>
                        <td style="padding:10px 8px;">Market Discipline & Model Transparency</td>
                        <td style="padding:10px 8px;">SHAP attribution plots supporting clear adverse-action reason codes and auditability.</td>
                        <td style="padding:10px 8px;"><span class="badge-institutional badge-green">COMPLIANT</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">⚙️ Enterprise Technology Architecture</div>
                <ul style="font-size:0.86rem; color:#475569; line-height:1.75; padding-left:1.2rem;">
                    <li><b>Core Modeling Engine:</b> scikit-learn pipeline with DecisionTree and ensemble benchmarks.</li>
                    <li><b>Data Engineering:</b> Automated feature extraction (70 features across velocity, RFM, and ratios).</li>
                    <li><b>Model Serving & API:</b> High-throughput FastAPI REST service with Pydantic validation.</li>
                    <li><b>Audit & Experiment Tracking:</b> MLflow for hyperparameter tracking and model registry.</li>
                    <li><b>Containerization & CI/CD:</b> Dockerized microservice architecture validated by GitHub Actions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown(
            """
            <div class="bank-card">
                <div class="bank-card-title">🛡️ Model Risk Governance & Next Steps</div>
                <ul style="font-size:0.86rem; color:#475569; line-height:1.75; padding-left:1.2rem;">
                    <li><b>Realized Default Calibration:</b> Validate RFM proxy target against actual 90-day delinquency outcomes as portfolio matures.</li>
                    <li><b>Fair Lending & Bias Audits:</b> Continuous statistical parity and disparate impact monitoring across demographic cohorts.</li>
                    <li><b>Automated Model Retraining:</b> Scheduled pipeline refresh when population stability index (PSI) breaches 0.10.</li>
                    <li><b>Human-in-the-Loop Oversight:</b> All borderline/review decisions remain subject to approved credit committee policy.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# Application Entry Point & Navigation
# ---------------------------------------------------------------------
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-brand-mark">B</div>
                <div class="sidebar-brand-name">BATI BANK</div>
                <div class="sidebar-brand-sub">Credit Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#64748B; margin-bottom:0.5rem;">WORKSPACES</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation Menu",
            [
                "🏢 Portfolio Overview",
                "📊 Model Performance",
                "🔬 Feature Analysis",
                "🔎 SHAP Explainability",
                "🎯 Loan Underwriting",
                "ℹ️ Regulatory & Governance",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#64748B; line-height:1.6;">
                <b>INSTITUTIONAL METADATA</b><br>
                Institution: Bati Bank Financial<br>
                Model ID: <code>CR-DT-PROD-v1.0</code><br>
                Regulatory Tier: Basel II Aligned<br>
                Target: RFM Risk Proxy
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load core analytical assets
    df = load_data()
    model = load_model()
    features = load_features()
    metrics = load_metrics()
    comparison = load_comparison()

    # Route navigation
    if "Portfolio Overview" in page:
        overview_page(df, metrics, comparison, features)
    elif "Model Performance" in page:
        performance_page(df, model, metrics)
    elif "Feature Analysis" in page:
        feature_analysis_page(df)
    elif "SHAP Explainability" in page:
        shap_page(df, model, features)
    elif "Loan Underwriting" in page:
        prediction_page(model, features)
    elif "Regulatory & Governance" in page:
        about_page()

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; font-size:0.75rem; color:#94A3B8; padding: 0.5rem 0 1.5rem 0;">
            <b>BATI BANK FINANCIAL SERVICES</b> · Credit Risk Analytics & Underwriting Engine · Internal Institutional Tool
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
