"""
Sentinel Payment Fraud Risk Engine - FastAPI Service

Provides RESTful endpoints for real-time transaction fraud scoring:
- GET /health: Health check, model status, and operating thresholds
- POST /predict: Score transaction, determine risk decision (ALLOW/REVIEW/BLOCK),
                 generate structured explainability evidence, and record audit trail.
- GET /audit/recent: Retrieve recent scored transactions from SQLite.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FraudDetector
from src.audit import AuditLogger

# Global instances initialized on startup
detector: Optional[FraudDetector] = None
audit_logger: Optional[AuditLogger] = None


def get_detector() -> FraudDetector:
    """Retrieve or initialize the singleton FraudDetector."""
    global detector
    if detector is None:
        try:
            detector = FraudDetector(model_dir="data/processed/model")
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Fraud detection model not loaded: {str(e)}. Run 'python run_pipeline.py' first.",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error initializing fraud detector: {str(e)}",
            )
    return detector


def get_audit_logger() -> AuditLogger:
    """Retrieve or initialize the singleton AuditLogger."""
    global audit_logger
    if audit_logger is None:
        audit_logger = AuditLogger()
    return audit_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: initialize model and audit logger on startup without retraining."""
    try:
        get_detector()
        get_audit_logger()
    except Exception as e:
        print(f"[WARN] Startup model loading deferred: {e}")
    yield


app = FastAPI(
    title="Sentinel - Payment Fraud Risk Engine",
    description="Defense-only ML engine routing payment transactions into ALLOW, REVIEW, or BLOCK.",
    version="1.0.0",
    lifespan=lifespan,
)


# =====================================================================
# REQUEST & RESPONSE SCHEMAS
# =====================================================================

class TransactionInput(BaseModel):
    """Transaction schema for fraud risk prediction."""
    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(
        ...,
        gt=0,
        description="Transaction amount in monetary units (must be > 0)",
        examples=[150.00],
    )
    TransactionID: Optional[Any] = Field(
        None, description="Optional identifier for the transaction", examples=[2987001]
    )
    TransactionDT: Optional[int] = Field(
        None, description="Seconds from reference timestamp", examples=[86400]
    )
    ProductCD: Optional[str] = Field("W", description="Product code (W, C, H, R, S)", examples=["W"])
    card1: Optional[int] = Field(None, description="Card payment identifier 1", examples=[12345])
    card2: Optional[float] = Field(None, description="Card payment identifier 2", examples=[360.0])
    card3: Optional[float] = Field(None, description="Card payment identifier 3", examples=[150.0])
    card4: Optional[str] = Field(None, description="Card brand (visa, mastercard, discover, etc.)", examples=["visa"])
    card5: Optional[float] = Field(None, description="Card payment identifier 5", examples=[166.0])
    card6: Optional[str] = Field(None, description="Card type (debit, credit)", examples=["debit"])
    addr1: Optional[float] = Field(None, description="Purchaser billing region / state", examples=[315.0])
    addr2: Optional[float] = Field(None, description="Purchaser billing country code", examples=[87.0])
    dist1: Optional[float] = Field(None, description="Distance from billing address / zip", examples=[15.0])
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain", examples=["gmail.com"])
    C1: Optional[float] = Field(None, description="Count of phone numbers associated with card", examples=[1.0])
    C2: Optional[float] = Field(None, description="Count of email addresses associated with card", examples=[1.0])
    C5: Optional[float] = Field(None, description="Count of transactions for this card / IP", examples=[0.0])
    C11: Optional[float] = Field(None, description="Count feature 11", examples=[1.0])
    C13: Optional[float] = Field(None, description="Count feature 13", examples=[1.0])
    C14: Optional[float] = Field(None, description="Count feature 14", examples=[1.0])
    D1: Optional[float] = Field(None, description="Timedelta from first transaction", examples=[0.0])
    D2: Optional[float] = Field(None, description="Timedelta from last transaction", examples=[0.0])
    D3: Optional[float] = Field(None, description="Timedelta 3", examples=[0.0])
    D4: Optional[float] = Field(None, description="Timedelta 4", examples=[0.0])
    D10: Optional[float] = Field(None, description="Timedelta 10", examples=[0.0])
    D15: Optional[float] = Field(None, description="Timedelta 15", examples=[0.0])


class EvidenceOutput(BaseModel):
    transaction_amount: float
    important_features: List[str]
    transaction_signals: List[str] = []
    top_model_factors: List[str] = []
    model_name: str
    engine: str


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Estimated probability of fraud [0.0, 1.0]")
    risk_level: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    decision: str = Field(..., description="Risk action: ALLOW, REVIEW, or BLOCK")
    confidence: float = Field(..., description="Decision confidence [0.5, 1.0]")
    evidence: EvidenceOutput
    audit_id: Optional[int] = Field(None, description="SQLite audit log ID")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    thresholds: Dict[str, float]
    defense_only: bool


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Reports API readiness, loaded model name, and active decision thresholds.
    """
    try:
        det = get_detector()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=det.model_name,
            thresholds={
                "review_threshold": det.th_review,
                "block_threshold": det.th_block,
            },
            defense_only=True,
        )
    except HTTPException:
        return HealthResponse(
            status="unhealthy - model not loaded",
            model_loaded=False,
            model_name="none",
            thresholds={},
            defense_only=True,
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_fraud(transaction: TransactionInput):
    """
    Evaluate fraud risk for an inbound payment transaction.

    - Preprocesses input using the exact saved ColumnTransformer
    - Predicts fraud probability using the trained model
    - Routes into ALLOW, REVIEW, or BLOCK decision using validation-tuned thresholds
    - Generates structured explainability evidence (amount, velocity, feature importances)
    - Records decision in SQLite audit log
    """
    det = get_detector()
    logger = get_audit_logger()

    txn_dict = transaction.model_dump()

    try:
        score_result = det.score(txn_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error during inference: {str(e)}",
        )

    # Record to SQLite audit log
    audit_id = None
    try:
        audit_id = logger.log(
            fraud_probability=score_result["fraud_probability"],
            risk_level=score_result["risk_level"],
            decision=score_result["decision"],
            transaction_id=str(txn_dict.get("TransactionID", "")),
            transaction_amount=score_result["evidence"]["transaction_amount"],
            evidence=score_result["evidence"],
        )
    except Exception as e:
        print(f"[WARN] Failed to write audit log: {e}")

    return PredictionResponse(
        fraud_probability=score_result["fraud_probability"],
        risk_level=score_result["risk_level"],
        decision=score_result["decision"],
        confidence=score_result["confidence"],
        evidence=EvidenceOutput(**score_result["evidence"]),
        audit_id=audit_id,
    )


@app.get("/audit/recent", tags=["Audit"])
def get_recent_audit_logs(limit: int = 20):
    """Retrieve recent transaction prediction records from SQLite audit log."""
    logger = get_audit_logger()
    records = logger.get_recent(limit=min(limit, 100))
    return {"count": len(records), "records": records}