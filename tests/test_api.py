"""
Tests for Sentinel FastAPI Application and Risk Engine

Validates:
- GET /health endpoint and model readiness
- POST /predict endpoint with schema validation
- 3-tier risk decision consistency (ALLOW / REVIEW / BLOCK)
- Structured explainability evidence structure
- Validation errors for invalid or negative inputs
- SQLite audit logging integration
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app, get_detector, get_audit_logger
from src.audit import AuditLogger


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check_returns_200(self, client):
        """Verify /health returns HTTP 200 and healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "RandomForest"
        assert data["defense_only"] is True
        assert "review_threshold" in data["thresholds"]
        assert "block_threshold" in data["thresholds"]
        assert data["thresholds"]["review_threshold"] < data["thresholds"]["block_threshold"]


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_legitimate_transaction(self, client):
        """Test scoring a typical legitimate transaction."""
        payload = {
            "TransactionAmt": 45.00,
            "TransactionID": "TEST-LEGIT-001",
            "ProductCD": "W",
            "card1": 1004,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "C1": 1.0,
            "C5": 0.0,
            "D1": 14.0,
            "D3": 5.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "fraud_probability" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert data["decision"] in ["ALLOW", "REVIEW", "BLOCK"]
        assert data["confidence"] >= 0.5

        # Check structured evidence
        evidence = data["evidence"]
        assert "transaction_amount" in evidence
        assert evidence["transaction_amount"] == 45.00
        assert "important_features" in evidence
        assert len(evidence["important_features"]) > 0
        assert "transaction_signals" in evidence
        assert "engine" in evidence
        assert "audit_id" in data
        assert data["audit_id"] is not None

    def test_predict_suspicious_transaction(self, client):
        """Test scoring a suspicious transaction with elevated velocity counts."""
        payload = {
            "TransactionAmt": 2500.00,
            "TransactionID": "TEST-FRAUD-001",
            "ProductCD": "C",
            "card1": 9876,
            "card4": "mastercard",
            "card6": "credit",
            "P_emaildomain": "anonymous.com",
            "C1": 35.0,
            "C5": 12.0,
            "C13": 20.0,
            "D1": 0.0,
            "D3": 0.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["fraud_probability"] > 0.30
        assert data["decision"] in ["REVIEW", "BLOCK"]
        assert data["risk_level"] in ["MEDIUM", "HIGH"]

        signals = data["evidence"]["transaction_signals"]
        assert any("High-value" in s or "Above-average" in s for s in signals)
        assert any("velocity" in s.lower() for s in signals)

    def test_predict_invalid_negative_amount(self, client):
        """Verify negative transaction amount fails validation with 422."""
        payload = {
            "TransactionAmt": -50.00,
            "card1": 1234,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_amount_field(self, client):
        """Verify missing TransactionAmt fails validation with 422."""
        payload = {
            "card1": 1234,
            "ProductCD": "W",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_amount_type(self, client):
        """Verify string amount fails validation with 422."""
        payload = {
            "TransactionAmt": "not-a-number",
            "card1": 1234,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestRiskDecisionConsistency:
    """Verify decision logic adheres strictly to validation thresholds."""

    def test_threshold_decision_rules(self):
        """Verify ALLOW, REVIEW, and BLOCK partition probabilities accurately."""
        detector = get_detector()
        th_rev = detector.th_review
        th_blk = detector.th_block

        assert 0.0 < th_rev < th_blk < 1.0

        res_low = detector.score({"TransactionAmt": 10.0, "C1": 0, "D1": 100})
        if res_low["fraud_probability"] < th_rev:
            assert res_low["decision"] == "ALLOW"
            assert res_low["risk_level"] == "LOW"

        sim_probs = [th_rev - 0.05, th_rev + 0.05, th_blk + 0.05]
        for p in sim_probs:
            if p >= th_blk:
                dec, risk = "BLOCK", "HIGH"
            elif p >= th_rev:
                dec, risk = "REVIEW", "MEDIUM"
            else:
                dec, risk = "ALLOW", "LOW"

            if p < th_rev:
                assert dec == "ALLOW" and risk == "LOW"
            elif p < th_blk:
                assert dec == "REVIEW" and risk == "MEDIUM"
            else:
                assert dec == "BLOCK" and risk == "HIGH"


class TestAuditLogging:
    """Tests for SQLite audit logging."""

    def test_audit_log_persists_and_queries(self, tmp_path):
        """Verify AuditLogger writes to and reads from SQLite."""
        db_path = tmp_path / "test_audit.db"
        logger = AuditLogger(db_path=str(db_path))

        assert logger.count() == 0

        log_id = logger.log(
            fraud_probability=0.75,
            risk_level="HIGH",
            decision="BLOCK",
            transaction_id="TXN-999",
            transaction_amount=500.0,
            evidence={"signals": ["High amount"]},
        )
        assert log_id == 1
        assert logger.count() == 1

        recent = logger.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0]["transaction_id"] == "TXN-999"
        assert recent[0]["fraud_probability"] == 0.75
        assert recent[0]["decision"] == "BLOCK"
        assert recent[0]["risk_level"] == "HIGH"
        assert recent[0]["evidence"]["signals"] == ["High amount"]

    def test_api_audit_recent_endpoint(self, client):
        """Verify GET /audit/recent returns records created by /predict."""
        client.post("/predict", json={"TransactionAmt": 88.0, "TransactionID": "AUDIT-CHECK"})
        response = client.get("/audit/recent?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "records" in data
        assert data["count"] > 0