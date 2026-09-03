"""
Tests for Batch CSV Risk Analysis, Validation, and Batch Audit Logging

Validates:
- Valid CSV DataFrame processing via FraudDetector.score_dataframe()
- Presence of required output columns: fraud_probability, risk_level, decision
- Decision routing adhering strictly to validation thresholds (0.2977, 0.8023)
- Detection of missing required columns (TransactionAmt)
- Safe handling of empty DataFrames
- Row-level error isolation (non-positive / corrupt values skipped without batch failure)
- Batch SQLite audit logging
- Unchanged single-transaction scoring behavior
"""

import pytest
import pandas as pd
import numpy as np
from src.inference import FraudDetector
from src.audit import AuditLogger


@pytest.fixture
def detector():
    """Load the trained FraudDetector."""
    return FraudDetector()


@pytest.fixture
def sample_batch_df():
    """Create a sample valid transaction DataFrame for batch scoring."""
    return pd.DataFrame([
        {
            "TransactionID": "BATCH-001",
            "TransactionAmt": 45.00,
            "ProductCD": "W",
            "card1": 1004,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "C1": 1.0,
            "C5": 0.0,
            "D1": 14.0,
            "D3": 5.0,
        },
        {
            "TransactionID": "BATCH-002",
            "TransactionAmt": 1250.00,
            "ProductCD": "C",
            "card1": 9876,
            "card4": "mastercard",
            "card6": "credit",
            "P_emaildomain": "anonymous.com",
            "C1": 30.0,
            "C5": 8.0,
            "D1": 0.0,
            "D3": 0.0,
        },
        {
            "TransactionID": "BATCH-003",
            "TransactionAmt": 320.00,
            "ProductCD": "W",
            "card1": 4567,
            "card4": "visa",
            "card6": "credit",
            "P_emaildomain": "yahoo.com",
            "C1": 4.0,
            "C5": 1.0,
            "D1": 1.0,
            "D3": 1.0,
        },
    ])


class TestBatchCSVValidationAndScoring:
    """Test suite for batch CSV validation and scoring."""

    def test_score_valid_dataframe(self, detector, sample_batch_df):
        """Verify valid batch scores completely and contains all required columns."""
        scored_df, failed_rows, summary = detector.score_dataframe(sample_batch_df)

        assert len(scored_df) == 3
        assert len(failed_rows) == 0

        # Check required columns
        for col in ["fraud_probability", "risk_level", "decision"]:
            assert col in scored_df.columns

        # Verify probabilities in [0.0, 1.0]
        for prob in scored_df["fraud_probability"]:
            assert 0.0 <= prob <= 1.0

        # Verify decisions match risk tiers
        for _, row in scored_df.iterrows():
            p = row["fraud_probability"]
            d = row["decision"]
            r = row["risk_level"]
            if p >= detector.th_block:
                assert d == "BLOCK" and r == "HIGH"
            elif p >= detector.th_review:
                assert d == "REVIEW" and r == "MEDIUM"
            else:
                assert d == "ALLOW" and r == "LOW"

        # Check KPIs
        assert summary["total_transactions"] == 3
        assert summary["successfully_scored"] == 3
        assert summary["failed_transactions"] == 0
        assert summary["allow_count"] + summary["review_count"] + summary["block_count"] == 3

    def test_missing_required_amount_column_raises_error(self, detector):
        """Verify DataFrame missing TransactionAmt raises clear ValueError."""
        bad_df = pd.DataFrame([{"card1": 1000, "ProductCD": "W"}])
        with pytest.raises(ValueError, match="Missing required column"):
            detector.score_dataframe(bad_df)

    def test_empty_dataframe_raises_error(self, detector):
        """Verify empty DataFrame raises clear ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty dataframe"):
            detector.score_dataframe(empty_df)

    def test_row_level_error_isolation(self, detector):
        """Verify bad rows are isolated without crashing the entire batch."""
        mixed_df = pd.DataFrame([
            {"TransactionID": "GOOD-1", "TransactionAmt": 100.0, "card1": 1000},
            {"TransactionID": "BAD-NEG", "TransactionAmt": -50.0, "card1": 1000},
            {"TransactionID": "BAD-STR", "TransactionAmt": "not_numeric", "card1": 1000},
            {"TransactionID": "GOOD-2", "TransactionAmt": 250.0, "card1": 2000},
        ])

        scored_df, failed_rows, summary = detector.score_dataframe(mixed_df)

        assert summary["total_transactions"] == 4
        assert summary["successfully_scored"] == 2
        assert summary["failed_transactions"] == 2
        assert len(scored_df) == 2
        assert len(failed_rows) == 2

        failed_reasons = [f["reason"] for f in failed_rows]
        assert any("Non-positive" in r for r in failed_reasons)
        assert any("Invalid numeric" in r for r in failed_reasons)

    def test_batch_audit_logging(self, tmp_path):
        """Verify AuditLogger.log_batch writes multiple records efficiently."""
        db_path = tmp_path / "test_batch_audit.db"
        logger = AuditLogger(db_path=str(db_path))

        records = [
            {"transaction_id": "B-1", "fraud_probability": 0.15, "risk_level": "LOW", "decision": "ALLOW", "transaction_amount": 50.0},
            {"transaction_id": "B-2", "fraud_probability": 0.45, "risk_level": "MEDIUM", "decision": "REVIEW", "transaction_amount": 150.0},
            {"transaction_id": "B-3", "fraud_probability": 0.85, "risk_level": "HIGH", "decision": "BLOCK", "transaction_amount": 900.0},
        ]

        n_inserted = logger.log_batch(records)
        assert n_inserted == 3
        assert logger.count() == 3

        recent = logger.get_recent(limit=5)
        assert len(recent) == 3
        assert recent[0]["transaction_id"] == "B-3"
        assert recent[0]["decision"] == "BLOCK"

    def test_existing_single_transaction_scoring_unchanged(self, detector):
        """Verify single transaction scoring still functions identically."""
        single_txn = {"TransactionAmt": 75.0, "card1": 1004, "ProductCD": "W"}
        res = detector.score(single_txn)
        assert "fraud_probability" in res
        assert res["decision"] in ["ALLOW", "REVIEW", "BLOCK"]
        assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert "evidence" in res