"""
Fraud Detection Inference Module

Provides production-ready inference interface for the trained Sentinel fraud detection engine:
- Loads full pipeline bundle (model + ColumnTransformer preprocessor + thresholds + metadata)
- Supports single transaction and batch scoring
- Translates fraud probabilities into 3-tier risk decisions: ALLOW / REVIEW / BLOCK
- Provides structured explainability evidence based on model factors and transaction signals
- Defense-only; strictly deterministic ML (no LLM in the decision path)
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.features import (
    engineer_features,
    ENGINEERED_NUMERIC_COLS,
    ENGINEERED_CATEGORICAL_COLS,
)


class FraudDetector:
    """Load and use trained fraud detection pipeline for inference and explainability."""

    def __init__(self, model_dir: str = "data/processed/model"):
        """
        Initialize detector by loading model pipeline artifacts.

        Args:
            model_dir: Path to directory containing trained artifacts
        """
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        bundle_path = self.model_dir / "model_pipeline.pkl"
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
            self.model = bundle["model"]
            self.preprocessor = bundle["preprocessor"]
            self.feature_names = bundle.get("feature_names_out", [])
            self.thresholds = bundle.get("thresholds", {})
            self.top_features = bundle.get("top_features", [])
            self.model_name = bundle.get("model_name", "TrainedModel")
        else:
            # Fallback to individual components
            model_path = self.model_dir / "fraud_model.pkl"
            prep_path = self.model_dir / "preprocessor.pkl"
            thresh_path = self.model_dir / "thresholds.pkl"

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found in: {self.model_dir}")

            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(prep_path) if prep_path.exists() else None
            self.thresholds = joblib.load(thresh_path) if thresh_path.exists() else {
                "review": 0.20, "block": 0.70, "cost_optimal": 0.35, "default": 0.50
            }
            self.feature_names = []
            self.top_features = []
            self.model_name = type(self.model).__name__

        # Normalize threshold keys
        self.th_review = float(
            self.thresholds.get("review", self.thresholds.get("REVIEW", 0.20))
        )
        self.th_block = float(
            self.thresholds.get("block", self.thresholds.get("BLOCK", 0.70))
        )

    def predict_proba(self, transaction: Dict[str, Any]) -> float:
        """
        Predict fraud probability for a single transaction dictionary.

        Args:
            transaction: Dict with transaction features

        Returns:
            float: Fraud probability between 0.0 and 1.0
        """
        df = pd.DataFrame([transaction])
        df_eng = engineer_features(df)

        feature_cols = ENGINEERED_NUMERIC_COLS + ENGINEERED_CATEGORICAL_COLS
        X_trans = self.preprocessor.transform(df_eng[feature_cols])

        proba = float(self.model.predict_proba(X_trans)[0, 1])
        return proba

    def score(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a transaction and return risk decision with structured evidence.

        Args:
            transaction: Dict with transaction features

        Returns:
            Dict containing:
              - fraud_probability (0.0 - 1.0)
              - risk_level (LOW, MEDIUM, HIGH)
              - decision (ALLOW, REVIEW, BLOCK)
              - confidence (0.5 - 1.0)
              - evidence (signals, top model factors)
              - thresholds applied
        """
        prob = self.predict_proba(transaction)

        # 3-tier routing rule
        if prob >= self.th_block:
            decision = "BLOCK"
            risk_level = "HIGH"
        elif prob >= self.th_review:
            decision = "REVIEW"
            risk_level = "MEDIUM"
        else:
            decision = "ALLOW"
            risk_level = "LOW"

        confidence = float(max(prob, 1.0 - prob))

        # Structured evidence (rule/model-based, strictly non-LLM)
        evidence_signals = []
        amt = float(transaction.get("TransactionAmt", transaction.get("TransactionAMT", 0.0)))
        if amt >= 1000.0:
            evidence_signals.append(f"High-value transaction (${amt:,.2f})")
        elif amt >= 300.0:
            evidence_signals.append(f"Above-average transaction amount (${amt:,.2f})")

        c1 = transaction.get("C1", None)
        if c1 is not None and float(c1) >= 10.0:
            evidence_signals.append(f"Elevated velocity count (C1={c1})")

        d1 = transaction.get("D1", None)
        if d1 is not None and float(d1) == 0.0:
            evidence_signals.append("Zero timedelta since card/account reference (D1=0)")

        email = str(transaction.get("P_emaildomain", "")).lower()
        if "anonymous" in email or "proton" in email:
            evidence_signals.append(f"Privacy-focused email provider ({email})")

        top_factors = [
            f"{feat}: {score:.3f}" for feat, score in self.top_features[:5]
        ]

        return {
            "fraud_probability": round(prob, 4),
            "risk_level": risk_level,
            "decision": decision,
            "confidence": round(confidence, 4),
            "thresholds": {
                "review_threshold": self.th_review,
                "block_threshold": self.th_block,
            },
            "evidence": {
                "transaction_amount": amt,
                "important_features": top_factors,
                "transaction_signals": evidence_signals,
                "top_model_factors": top_factors,
                "model_name": self.model_name,
                "engine": "Sentinel-v1-DefenseOnly",
            },
        }

    def score_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a list of transaction dictionaries efficiently."""
        if not transactions:
            return []

        df = pd.DataFrame(transactions)
        df_eng = engineer_features(df)
        feature_cols = ENGINEERED_NUMERIC_COLS + ENGINEERED_CATEGORICAL_COLS
        X_trans = self.preprocessor.transform(df_eng[feature_cols])

        probs = self.model.predict_proba(X_trans)[:, 1]

        results = []
        for txn, prob in zip(transactions, probs):
            p = float(prob)
            if p >= self.th_block:
                decision, risk = "BLOCK", "HIGH"
            elif p >= self.th_review:
                decision, risk = "REVIEW", "MEDIUM"
            else:
                decision, risk = "ALLOW", "LOW"

            results.append({
                "fraud_probability": round(p, 4),
                "risk_level": risk,
                "decision": decision,
                "confidence": round(max(p, 1.0 - p), 4),
            })

        return results

    def score_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Score a pandas DataFrame of transactions row-by-row using the existing
        inference pipeline and validation-derived decision thresholds.

        Validates input structure, handles per-row missing/invalid values safely
        without crashing the entire batch, and generates a compact risk summary.

        Args:
            df: Input DataFrame containing transaction records

        Returns:
            Tuple of:
              - scored_df: DataFrame with original columns plus fraud_probability,
                           risk_level, decision
              - failed_rows: List of dicts describing rows that failed validation
              - summary: Dict with KPI counts (total, scored, failed, allow, review,
                         block, high_risk_rate)

        Raises:
            ValueError: If DataFrame is empty or missing required amount column
        """
        if df is None or df.empty:
            raise ValueError("CSV contains no data (empty dataframe).")

        has_amt = "TransactionAmt" in df.columns or "TransactionAMT" in df.columns
        if not has_amt:
            raise ValueError(
                "Missing required column: 'TransactionAmt' (or 'TransactionAMT'). "
                f"Found columns: {list(df.columns)}"
            )

        id_col = "TransactionID" if "TransactionID" in df.columns else None

        scored_records = []
        failed_records = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            raw_amt = row_dict.get("TransactionAmt", row_dict.get("TransactionAMT"))

            # Validate numeric positive amount
            try:
                amt_val = float(raw_amt)
                if pd.isna(amt_val) or amt_val <= 0:
                    failed_records.append({
                        "row_index": idx + 1,
                        "transaction_id": row_dict.get(id_col) if id_col else f"ROW-{idx + 1}",
                        "reason": f"Non-positive or NaN TransactionAmt ({raw_amt})",
                    })
                    continue
            except (ValueError, TypeError):
                failed_records.append({
                    "row_index": idx + 1,
                    "transaction_id": row_dict.get(id_col) if id_col else f"ROW-{idx + 1}",
                    "reason": f"Invalid numeric TransactionAmt ('{raw_amt}')",
                })
                continue

            # Score row using existing single-transaction scoring
            try:
                score_res = self.score(row_dict)
                row_copy = dict(row_dict)
                row_copy["fraud_probability"] = score_res["fraud_probability"]
                row_copy["risk_level"] = score_res["risk_level"]
                row_copy["decision"] = score_res["decision"]
                scored_records.append(row_copy)
            except Exception as e:
                failed_records.append({
                    "row_index": idx + 1,
                    "transaction_id": row_dict.get(id_col) if id_col else f"ROW-{idx + 1}",
                    "reason": f"Scoring error: {str(e)}",
                })

        n_total = len(df)
        n_scored = len(scored_records)
        n_failed = len(failed_records)

        n_allow = sum(1 for r in scored_records if r["decision"] == "ALLOW")
        n_review = sum(1 for r in scored_records if r["decision"] == "REVIEW")
        n_block = sum(1 for r in scored_records if r["decision"] == "BLOCK")
        high_risk_rate = (n_block / n_scored) if n_scored > 0 else 0.0

        summary = {
            "total_transactions": n_total,
            "successfully_scored": n_scored,
            "failed_transactions": n_failed,
            "allow_count": n_allow,
            "review_count": n_review,
            "block_count": n_block,
            "allow_rate": (n_allow / n_scored) if n_scored > 0 else 0.0,
            "review_rate": (n_review / n_scored) if n_scored > 0 else 0.0,
            "block_rate": high_risk_rate,
            "high_risk_rate": high_risk_rate,
        }

        scored_df = pd.DataFrame(scored_records) if scored_records else pd.DataFrame()
        return scored_df, failed_records, summary


if __name__ == "__main__":
    import sys
    try:
        detector = FraudDetector()
        sample_txn = {
            "TransactionAmt": 450.00,
            "card1": 12345,
            "card2": 361,
            "card3": 150,
            "card5": 166,
            "addr1": 100,
            "addr2": 50,
            "ProductCD": "W",
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "C1": 12,
            "C2": 1,
            "C5": 0,
            "C11": 1,
            "C13": 1,
            "C14": 1,
            "D1": 0,
            "D2": 0,
            "D3": 0,
            "D4": 0,
            "D10": 0,
            "D15": 0,
        }
        res = detector.score(sample_txn)
        print("Sample Inference Result:")
        print(res)
    except FileNotFoundError:
        print("Model artifacts not yet saved. Run run_pipeline.py first.")
