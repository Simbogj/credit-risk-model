"""
Credit Risk Model - Pydantic Models
====================================
Request/Response schemas for FastAPI service.
Implements Task 6: Model Deployment.

Author: Bati Bank Analytics Team
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# =============================================================================
# REQUEST MODELS
# =============================================================================

class CustomerFeatures(BaseModel):
    """Feature set for a single customer prediction request."""

    # Transaction statistics
    TotalAmount: float = Field(..., description="Total transaction amount sum")
    AvgAmount: float = Field(..., description="Average transaction amount")
    StdAmount: float = Field(0.0, description="Standard deviation of amounts")
    MinAmount: float = Field(0.0, description="Minimum transaction amount")
    MaxAmount: float = Field(0.0, description="Maximum transaction amount")
    TransactionCount: int = Field(..., description="Number of transactions")

    # Credit/Debit metrics
    CreditCount: float = Field(0.0, description="Count of credit (refund) transactions")
    DebitCount: float = Field(0.0, description="Count of debit transactions")
    CreditTotal: float = Field(0.0, description="Total credit amount")
    DebitTotal: float = Field(0.0, description="Total debit amount")
    CreditRatio: float = Field(0.0, description="Ratio of credits to total transactions")
    DebitRatio: float = Field(0.0, description="Ratio of debits to total transactions")

    # Value metrics
    TotalValue: float = Field(0.0, description="Sum of absolute values")
    AvgValue: float = Field(0.0, description="Average absolute value")
    StdValue: float = Field(0.0, description="Std dev of absolute values")

    # Fraud metrics
    FraudCount: float = Field(0.0, description="Count of fraud cases")
    FraudRate: float = Field(0.0, description="Fraud rate (proportion)")

    # Amount range
    AmountRange: float = Field(0.0, description="Range between max and min amount")

    # Channel distribution (counts)
    Channel_3_Count: int = Field(0, description="Channel 3 transaction count")
    Channel_2_Count: int = Field(0, description="Channel 2 transaction count")
    Channel_5_Count: int = Field(0, description="Channel 5 transaction count")
    Channel_1_Count: int = Field(0, description="Channel 1 transaction count")

    # Category distribution (counts)
    Category_financial_services_Count: int = Field(0, description="Financial services count")
    Category_airtime_Count: int = Field(0, description="Airtime count")
    Category_utility_bill_Count: int = Field(0, description="Utility bill count")
    Category_data_bundles_Count: int = Field(0, description="Data bundles count")
    Category_tv_Count: int = Field(0, description="TV count")
    Category_ticket_Count: int = Field(0, description="Ticket count")
    Category_movies_Count: int = Field(0, description="Movies count")
    Category_transport_Count: int = Field(0, description="Transport count")
    Category_other_Count: int = Field(0, description="Other category count")

    # Channel amounts
    Channel_3_Amount: float = Field(0.0, description="Channel 3 total amount")
    Channel_2_Amount: float = Field(0.0, description="Channel 2 total amount")

    # Category amounts
    Category_financial_services_Amount: float = Field(0.0, description="Financial services amount")
    Category_airtime_Amount: float = Field(0.0, description="Airtime amount")
    Category_utility_bill_Amount: float = Field(0.0, description="Utility bill amount")

    # Provider metrics
    UniqueProviders: int = Field(0, description="Number of unique providers")
    AvgPricingStrategy: float = Field(0.0, description="Average pricing strategy")

    # Pricing strategy counts
    PricingStrategy_2_Count: int = Field(0, description="Pricing strategy 2 count")
    PricingStrategy_4_Count: int = Field(0, description="Pricing strategy 4 count")
    PricingStrategy_1_Count: int = Field(0, description="Pricing strategy 1 count")
    PricingStrategy_0_Count: int = Field(0, description="Pricing strategy 0 count")

    # RFM features
    Recency: int = Field(..., description="Days since last transaction")
    Frequency: int = Field(..., description="Number of transactions")
    Monetary: float = Field(..., description="Total monetary value")
    AvgTransactionAmount: float = Field(0.0, description="Average transaction amount")
    TransactionsPerDay: float = Field(0.0, description="Transactions per active day")

    # Temporal features
    TransactionHour: int = Field(0, description="Hour of transaction (0-23)")
    TransactionDay: int = Field(0, description="Day of month (1-31)")
    TransactionMonth: int = Field(0, description="Month (1-12)")
    TransactionYear: int = Field(0, description="Year")
    DayOfWeek: int = Field(0, description="Day of week (0=Monday)")
    IsWeekend: int = Field(0, description="Is weekend (0/1)")
    Quarter: int = Field(0, description="Quarter (1-4)")
    WeekOfYear: int = Field(0, description="Week of year")

    # Derived features
    AmountToIncomeRatio: float = Field(0.0, description="Amount to income ratio proxy")
    StdToMeanRatio: float = Field(0.0, description="Coefficient of variation")

    class Config:
        schema_extra = {
            "example": {
                "TotalAmount": 25000.0,
                "AvgAmount": 1250.0,
                "StdAmount": 500.0,
                "MinAmount": 100.0,
                "MaxAmount": 5000.0,
                "TransactionCount": 20,
                "Recency": 5,
                "Frequency": 20,
                "Monetary": 25000.0,
            }
        }


class PredictionRequest(BaseModel):
    """Request schema for single customer prediction."""

    customer_id: str = Field(..., description="Unique customer identifier")
    features: CustomerFeatures = Field(..., description="Customer feature set")

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "features": {
                    "TotalAmount": 25000.0,
                    "AvgAmount": 1250.0,
                    "StdAmount": 500.0,
                    "MinAmount": 100.0,
                    "MaxAmount": 5000.0,
                    "TransactionCount": 20,
                    "Recency": 5,
                    "Frequency": 20,
                    "Monetary": 25000.0,
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    customers: List[Dict[str, Any]] = Field(..., description="List of customer feature dictionaries")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class RiskPrediction(BaseModel):
    """Individual risk prediction result."""

    customer_id: str = Field(..., description="Customer identifier")
    predicted_class: int = Field(..., description="Predicted risk class (0=Low, 1=High)")
    risk_probability: float = Field(..., description="Probability of high risk (0-1)")
    confidence: float = Field(..., description="Model confidence in prediction (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    credit_score: int = Field(..., description="Credit score (300-850, FICO-style)")


class CreditTerms(BaseModel):
    """Recommended credit terms based on risk assessment."""

    max_amount: float = Field(..., description="Maximum recommended credit amount")
    max_duration_months: int = Field(..., description="Maximum loan duration in months")
    interest_rate: Optional[float] = Field(None, description="Recommended interest rate")


class CreditDecision(BaseModel):
    """Complete credit decision with terms."""

    customer_id: str = Field(..., description="Customer identifier")
    decision: str = Field(..., description="Decision: Approve, Review, or Decline")
    risk_probability: float = Field(..., description="Probability of default")
    credit_score: int = Field(..., description="Credit score (300-850)")
    recommended_terms: CreditTerms = Field(..., description="Recommended credit terms")


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""

    success: bool = Field(..., description="Whether prediction succeeded")
    prediction: RiskPrediction = Field(..., description="Risk prediction details")
    decision: Optional[CreditDecision] = Field(None, description="Credit decision details")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction response."""

    customer_id: str
    predicted_class: int
    risk_probability: float
    confidence: float
    risk_level: str
    credit_score: int


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    success: bool = Field(..., description="Whether batch prediction succeeded")
    predictions: List[BatchPredictionItem] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Response timestamp (ISO format)")


# =============================================================================
# ERROR MODELS
# =============================================================================

class ErrorDetail(BaseModel):
    """Error detail information."""

    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error")


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = Field(False, description="Always false for errors")
    error: ErrorDetail = Field(..., description="Error details")
    timestamp: str = Field(..., description="Error timestamp (ISO format)")


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")