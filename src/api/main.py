"""
Credit Risk Model - FastAPI Application
========================================
REST API for credit risk predictions.
Implements Task 6: Model Deployment.

Author: Bati Bank Analytics Team
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic_models import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CreditDecision,
    CreditTerms,
    HealthCheckResponse,
    PredictionRequest,
    PredictionResponse,
    RiskPrediction,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_MODEL_NAME = 'CreditRiskModel'
DEFAULT_MODEL_STAGE = 'Production'

THRESHOLD_LOW = 0.3
THRESHOLD_HIGH = 0.6

APP_START_TIME = time.time()

model = None
model_version = None
feature_names = None


# =============================================================================
# SECTION 1: MODEL LOADING
# =============================================================================

def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    stage: str = DEFAULT_MODEL_STAGE
) -> Any:
    """Load the trained model from MLflow registry or local storage."""
    global model, model_version, feature_names

    logger.info(f"Loading model: {model_name} (stage: {stage})")

    try:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        client = mlflow.MlflowClient()
        model_uri = f"models:/{model_name}/{stage}"

        logger.info(f"Loading from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        model_version = f"mlflow-{model_name}-{stage}"
        logger.info("Model loaded from MLflow Registry successfully")

    except Exception as e:
        logger.warning(f"Could not load from MLflow: {e}")
        logger.info("Trying local storage...")

        local_model_path = MODEL_DIR / f"{model_name}_best.joblib"

        if local_model_path.exists():
            logger.info(f"Loading from: {local_model_path}")
            model = joblib.load(local_model_path)
            model_version = f"local-{local_model_path.stem}"
            logger.info("Model loaded from local storage")
        else:
            raise RuntimeError(f"Model not found: {model_name}")

    feature_names_path = MODEL_DIR / f"{model_name}_feature_names.joblib"
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
        logger.info(f"Loaded {len(feature_names)} feature names")
    else:
        logger.warning("Feature names not found")
        feature_names = None

    return model


def get_feature_order() -> List[str]:
    """Get the expected feature order for prediction."""
    return [
        'TotalAmount', 'AvgAmount', 'StdAmount', 'MinAmount', 'MaxAmount',
        'TransactionCount', 'TotalValue', 'AvgValue', 'StdValue',
        'FraudCount', 'FraudRate', 'CreditCount', 'CreditTotal',
        'DebitCount', 'DebitTotal', 'CreditRatio', 'DebitRatio', 'AmountRange',
        'Channel_3_Count', 'Channel_2_Count', 'Channel_5_Count', 'Channel_1_Count',
        'Category_financial_services_Count', 'Category_airtime_Count',
        'Category_utility_bill_Count', 'Category_data_bundles_Count',
        'Category_tv_Count', 'Category_ticket_Count', 'Category_movies_Count',
        'Category_transport_Count', 'Category_other_Count',
        'Channel_3_Amount', 'Channel_2_Amount',
        'Category_financial_services_Amount', 'Category_airtime_Amount',
        'Category_utility_bill_Amount',
        'UniqueProviders', 'AvgPricingStrategy',
        'PricingStrategy_2_Count', 'PricingStrategy_4_Count',
        'PricingStrategy_1_Count', 'PricingStrategy_0_Count',
        'Recency', 'Frequency', 'Monetary', 'AvgTransactionAmount',
        'TransactionsPerDay', 'TransactionHour', 'TransactionDay',
        'TransactionMonth', 'TransactionYear', 'DayOfWeek', 'IsWeekend',
        'Quarter', 'WeekOfYear', 'AmountToIncomeRatio', 'StdToMeanRatio',
    ]


# =============================================================================
# SECTION 2: PREDICTION FUNCTIONS
# =============================================================================

def make_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Make a single prediction using the loaded model."""
    global model, feature_names

    if model is None:
        raise RuntimeError("Model not loaded")

    feature_order = feature_names if feature_names else get_feature_order()

    try:
        feature_vector = [[features.get(f, 0.0) for f in feature_order]]
    except Exception as e:
        logger.error(f"Error creating feature vector: {e}")
        raise ValueError(f"Invalid features: {e}")

    try:
        y_pred = model.predict(feature_vector)[0]
        y_proba = model.predict_proba(feature_vector)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise RuntimeError(f"Model prediction failed: {e}")

    risk_probability = float(y_proba[1])
    confidence = float(max(y_proba))
    predicted_class = int(y_pred)

    if risk_probability < 0.3:
        risk_level = 'Low'
    elif risk_probability < 0.6:
        risk_level = 'Medium'
    else:
        risk_level = 'High'

    credit_score = int(300 + (1 - risk_probability) * 550)

    return {
        'predicted_class': predicted_class,
        'risk_probability': risk_probability,
        'confidence': confidence,
        'risk_level': risk_level,
        'credit_score': credit_score,
    }


def get_credit_decision(risk_probability: float) -> Dict[str, Any]:
    """Determine credit decision based on risk probability."""
    if risk_probability <= THRESHOLD_LOW:
        decision = 'Approve'
        terms = {
            'max_amount': 10000.0,
            'max_duration_months': 12,
            'interest_rate': 0.12
        }
    elif risk_probability <= THRESHOLD_HIGH:
        decision = 'Review'
        terms = {
            'max_amount': 5000.0,
            'max_duration_months': 6,
            'interest_rate': 0.18
        }
    else:
        decision = 'Decline'
        terms = {
            'max_amount': 0.0,
            'max_duration_months': 0,
            'interest_rate': None
        }

    credit_score = int(300 + (1 - risk_probability) * 550)

    return {
        'decision': decision,
        'risk_probability': risk_probability,
        'credit_score': credit_score,
        'recommended_terms': terms
    }


# =============================================================================
# SECTION 3: FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Credit Risk API",
    description="REST API for Bati Bank Credit Risk Scoring Model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, model_version
    try:
        model = load_model()
        logger.info("Application startup complete - model loaded")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - basic info."""
    return {
        "service": "Credit Risk API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"], response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    global model, model_version

    uptime = time.time() - APP_START_TIME

    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": model_version,
        "uptime_seconds": uptime
    }


@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a risk prediction for a single customer."""
    global model, model_version

    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not available: {str(e)}"
            )

    start_time = time.time()

    try:
        features_dict = request.features.dict()
        prediction = make_prediction(features_dict)
        decision = get_credit_decision(prediction['risk_probability'])

        response = PredictionResponse(
            success=True,
            prediction=RiskPrediction(
                customer_id=request.customer_id,
                predicted_class=prediction['predicted_class'],
                risk_probability=prediction['risk_probability'],
                confidence=prediction['confidence'],
                risk_level=prediction['risk_level'],
                credit_score=prediction['credit_score'],
            ),
            decision=CreditDecision(
                customer_id=request.customer_id,
                decision=decision['decision'],
                risk_probability=decision['risk_probability'],
                credit_score=decision['credit_score'],
                recommended_terms=CreditTerms(**decision['recommended_terms']),
            ),
            model_version=model_version or "unknown",
            timestamp=datetime.utcnow().isoformat(),
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/predict/batch", tags=["Prediction"], response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make risk predictions for multiple customers."""
    global model, model_version

    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not available: {str(e)}"
            )

    start_time = time.time()
    predictions = []

    try:
        for customer in request.customers:
            try:
                features = customer.get('features', {})
                customer_id = customer.get('customer_id', 'unknown')

                result = make_prediction(features)

                pred_item = BatchPredictionItem(
                    customer_id=customer_id,
                    predicted_class=result['predicted_class'],
                    risk_probability=result['risk_probability'],
                    confidence=result['confidence'],
                    risk_level=result['risk_level'],
                    credit_score=result['credit_score'],
                )
                predictions.append(pred_item)

            except Exception as e:
                logger.warning(f"Error predicting for customer: {e}")
                predictions.append(BatchPredictionItem(
                    customer_id=customer.get('customer_id', 'unknown'),
                    predicted_class=-1,
                    risk_probability=-1.0,
                    confidence=0.0,
                    risk_level='Error',
                    credit_score=0,
                ))

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            success=True,
            predictions=predictions,
            total_count=len(predictions),
            processing_time_ms=processing_time,
            model_version=model_version or "unknown",
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/features", tags=["Information"])
async def get_features():
    """Get list of features used by the model."""
    feature_order = feature_names if feature_names else get_feature_order()

    return {
        "success": True,
        "count": len(feature_order),
        "features": feature_order
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "error_code": "INTERNAL_ERROR",
                "message": str(exc),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


# =============================================================================
# SECTION 4: MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit Risk API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Log level")

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info(f"Starting Credit Risk API on {args.host}:{args.port}")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )