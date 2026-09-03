"""
Sentinel Audit Logging Module (SQLite)

Stores prediction history for compliance, dispute analysis, and risk monitoring:
- timestamp
- transaction_id (if available)
- fraud_probability
- risk_level (LOW / MEDIUM / HIGH)
- decision (ALLOW / REVIEW / BLOCK)
- transaction_amount
- evidence summary
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Optional


DEFAULT_DB_PATH = "data/processed/audit_log.db"


class AuditLogger:
    """Manages SQLite audit trail for Sentinel fraud predictions."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yield a database connection and ensure it is cleanly closed."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Create the audit table if it does not already exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    transaction_id TEXT,
                    fraud_probability REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    transaction_amount REAL,
                    evidence_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp DESC)"
            )
            conn.commit()

    def log(
        self,
        fraud_probability: float,
        risk_level: str,
        decision: str,
        transaction_id: Optional[str] = None,
        transaction_amount: Optional[float] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a fraud prediction event into the audit log.

        Returns:
            id of the inserted audit record
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        evidence_str = json.dumps(evidence) if evidence else None

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_logs (
                    timestamp,
                    transaction_id,
                    fraud_probability,
                    risk_level,
                    decision,
                    transaction_amount,
                    evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    str(transaction_id) if transaction_id is not None else None,
                    float(fraud_probability),
                    str(risk_level),
                    str(decision),
                    float(transaction_amount) if transaction_amount is not None else None,
                    evidence_str,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        Record multiple fraud prediction events in a single SQLite transaction.

        Args:
            records: List of dictionaries with prediction results

        Returns:
            Count of successfully inserted records
        """
        if not records:
            return 0

        timestamp = datetime.now(timezone.utc).isoformat()
        rows_to_insert = []
        for r in records:
            evidence_str = json.dumps(r.get("evidence")) if r.get("evidence") else None
            rows_to_insert.append((
                timestamp,
                str(r.get("transaction_id")) if r.get("transaction_id") is not None else None,
                float(r["fraud_probability"]),
                str(r["risk_level"]),
                str(r["decision"]),
                float(r["transaction_amount"]) if r.get("transaction_amount") is not None else None,
                evidence_str,
            ))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO audit_logs (
                    timestamp,
                    transaction_id,
                    fraud_probability,
                    risk_level,
                    decision,
                    transaction_amount,
                    evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            conn.commit()
            return len(rows_to_insert)

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent prediction events.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of audit log dictionaries ordered by timestamp descending
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, transaction_id, fraud_probability,
                       risk_level, decision, transaction_amount, evidence_json
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                if item.get("evidence_json"):
                    try:
                        item["evidence"] = json.loads(item["evidence_json"])
                    except Exception:
                        item["evidence"] = None
                else:
                    item["evidence"] = None
                del item["evidence_json"]
                results.append(item)
            return results

    def count(self) -> int:
        """Return total count of logged predictions."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM audit_logs")
            return int(cursor.fetchone()[0])

