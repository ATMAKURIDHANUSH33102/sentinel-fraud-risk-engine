"""
Data Loader for IEEE-CIS Fraud Detection Dataset

Loads and validates transaction and identity datasets.
Performs basic data quality checks and safe merging.
Safely handles absence of train_identity.csv without invalid fallbacks.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class DataLoader:
    """Load and validate IEEE-CIS fraud detection datasets."""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the directory containing raw CSV files
        """
        self.data_dir = Path(data_dir)
        self.transaction_file = self.data_dir / "train_transaction.csv"
        self.identity_file = self.data_dir / "train_identity.csv"
        self.test_identity_file = self.data_dir / "test_identity.csv"

        self.transaction_data = None
        self.identity_data = None
        self.merged_data = None

    def _validate_file_exists(self, filepath: Path) -> bool:
        """Check if a file exists."""
        return filepath.exists()

    def load_transactions(
        self, usecols: Any = None, nrows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load transaction dataset.

        Args:
            usecols: Optional list of column names to load
            nrows: Optional number of rows to load

        Returns:
            DataFrame with transaction data

        Raises:
            FileNotFoundError: If train_transaction.csv not found
            ValueError: If isFraud column missing
        """
        if not self._validate_file_exists(self.transaction_file):
            raise FileNotFoundError(
                f"Transaction file not found: {self.transaction_file}"
            )

        print(f"Loading transactions from: {self.transaction_file}")
        self.transaction_data = pd.read_csv(
            self.transaction_file, usecols=usecols, nrows=nrows
        )

        # Validate required column
        if "isFraud" not in self.transaction_data.columns:
            raise ValueError(
                "Column 'isFraud' not found in transaction dataset"
            )

        print(f"[OK] Transactions loaded: {self.transaction_data.shape}")
        return self.transaction_data

    def load_identity(self, allow_missing: bool = True) -> Optional[pd.DataFrame]:
        """
        Load identity dataset.

        Attempts to load train_identity.csv. If missing and allow_missing=True,
        returns None and logs limitation (does NOT use test_identity to avoid leakage/mismatch).

        Args:
            allow_missing: If True, return None when train_identity.csv is absent

        Returns:
            DataFrame with identity data, or None if not available

        Raises:
            FileNotFoundError: If identity file not found and allow_missing=False
        """
        if self._validate_file_exists(self.identity_file):
            print(f"Loading identity from: {self.identity_file}")
            self.identity_data = pd.read_csv(self.identity_file)
            print(f"[OK] Identity loaded: {self.identity_data.shape}")
        elif allow_missing:
            print(
                f"[INFO] train_identity.csv not found in {self.data_dir}. "
                "Identity data is unavailable for training transactions. "
                "Proceeding with transaction features only (documented limitation)."
            )
            self.identity_data = None
        else:
            raise FileNotFoundError(
                f"Identity file not found: {self.identity_file}"
            )

        return self.identity_data

    def validate_transactions(self) -> Dict[str, Any]:
        """
        Validate transaction data quality.

        Returns:
            Dictionary with validation results
        """
        if self.transaction_data is None:
            raise ValueError("Transaction data not loaded. Call load_transactions() first.")

        validation_report = {}

        # Row and column counts
        validation_report["shape"] = self.transaction_data.shape
        validation_report["n_rows"] = len(self.transaction_data)
        validation_report["n_columns"] = len(self.transaction_data.columns)

        # TransactionID checks
        validation_report["n_unique_transaction_ids"] = (
            self.transaction_data["TransactionID"].nunique()
        )
        validation_report["n_duplicate_transaction_ids"] = (
            validation_report["n_rows"] - validation_report["n_unique_transaction_ids"]
        )

        # Target variable checks
        if "isFraud" in self.transaction_data.columns:
            validation_report["n_fraud"] = (
                self.transaction_data["isFraud"] == 1
            ).sum()
            validation_report["n_legit"] = (
                self.transaction_data["isFraud"] == 0
            ).sum()
            validation_report["fraud_prevalence"] = (
                validation_report["n_fraud"] / validation_report["n_rows"]
            )
            fraud_pct = 100 * validation_report["fraud_prevalence"]
            print(
                f"[OK] Target variable: {validation_report['n_fraud']} fraud, "
                f"{validation_report['n_legit']} legitimate ({fraud_pct:.2f}% fraud)"
            )

        # Data types
        validation_report["dtypes"] = self.transaction_data.dtypes.to_dict()
        validation_report["n_numeric_columns"] = (
            self.transaction_data.select_dtypes(include=["number"]).shape[1]
        )
        validation_report["n_object_columns"] = (
            self.transaction_data.select_dtypes(include=["object", "string"]).shape[1]
        )

        # Missing values
        missing_counts = self.transaction_data.isnull().sum()
        validation_report["missing_values"] = missing_counts[missing_counts > 0].to_dict()
        validation_report["n_columns_with_missing"] = len(validation_report["missing_values"])

        # Check for duplicates
        if validation_report["n_duplicate_transaction_ids"] > 0:
            print(
                f"[WARN] WARNING: {validation_report['n_duplicate_transaction_ids']} "
                f"duplicate TransactionID values found"
            )

        return validation_report

    def validate_identity(self) -> Dict[str, Any]:
        """
        Validate identity data quality.

        Returns:
            Dictionary with validation results
        """
        if self.identity_data is None:
            raise ValueError("Identity data not loaded. Call load_identity() first.")

        validation_report = {}

        # Row and column counts
        validation_report["shape"] = self.identity_data.shape
        validation_report["n_rows"] = len(self.identity_data)
        validation_report["n_columns"] = len(self.identity_data.columns)

        # TransactionID checks
        validation_report["n_unique_transaction_ids"] = (
            self.identity_data["TransactionID"].nunique()
        )
        validation_report["n_duplicate_transaction_ids"] = (
            validation_report["n_rows"] - validation_report["n_unique_transaction_ids"]
        )

        # Data types
        validation_report["dtypes"] = self.identity_data.dtypes.to_dict()

        # Missing values
        missing_counts = self.identity_data.isnull().sum()
        validation_report["missing_values"] = missing_counts[missing_counts > 0].to_dict()
        validation_report["n_columns_with_missing"] = len(validation_report["missing_values"])

        return validation_report

    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge transaction and identity datasets.

        Performs a left join on TransactionID if identity data is available.
        Does NOT drop rows with missing identity data. If identity data is
        absent, returns a copy of transaction data.

        Returns:
            Merged DataFrame

        Raises:
            ValueError: If transaction dataset is not loaded
        """
        if self.transaction_data is None:
            raise ValueError("Transaction data not loaded. Call load_transactions() first.")

        if self.identity_data is None:
            print("[INFO] Identity data is None. Retaining transaction dataset directly (no identity merge).")
            self.merged_data = self.transaction_data.copy()
            return self.merged_data

        print(f"\nMerging datasets on TransactionID...")
        print(f"  Transaction rows: {len(self.transaction_data)}")
        print(f"  Identity rows: {len(self.identity_data)}")

        self.merged_data = self.transaction_data.merge(
            self.identity_data,
            on="TransactionID",
            how="left"
        )

        print(f"[OK] Merged shape: {self.merged_data.shape}")
        print(f"  All {len(self.transaction_data)} transaction rows retained")

        # Check how many transactions have identity data
        n_with_identity = (
            self.merged_data["DeviceType"].notna().sum()
            if "DeviceType" in self.merged_data.columns else 0
        )
        identity_coverage = 100 * n_with_identity / len(self.merged_data)
        print(f"  Identity coverage: {n_with_identity} rows ({identity_coverage:.1f}%)")

        return self.merged_data

    def get_summary(self) -> str:
        """
        Get a summary report of loaded data.

        Returns:
            Summary string
        """
        if self.transaction_data is None:
            return "No data loaded yet."

        summary = []
        summary.append("=" * 60)
        summary.append("DATA LOADING SUMMARY")
        summary.append("=" * 60)

        summary.append(f"\nTransactions: {self.transaction_data.shape}")
        summary.append(f"  Columns: {self.transaction_data.shape[1]}")
        summary.append(f"  Rows: {self.transaction_data.shape[0]}")

        if self.identity_data is not None:
            summary.append(f"\nIdentity: {self.identity_data.shape}")
            summary.append(f"  Columns: {self.identity_data.shape[1]}")
            summary.append(f"  Rows: {self.identity_data.shape[0]}")

        if self.merged_data is not None:
            summary.append(f"\nMerged: {self.merged_data.shape}")
            summary.append(f"  Columns: {self.merged_data.shape[1]}")
            summary.append(f"  Rows: {self.merged_data.shape[0]}")

        summary.append("\n" + "=" * 60)

        return "\n".join(summary)


def load_fraud_data(
    data_dir: str = "data/raw",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Convenience function to load all fraud detection data.

    Args:
        data_dir: Path to raw data directory

    Returns:
        Tuple of (transactions, identity, merged) DataFrames
    """
    loader = DataLoader(data_dir=data_dir)

    transactions = loader.load_transactions()
    identity = loader.load_identity(allow_missing=True)
    merged = loader.merge_datasets()

    print(loader.get_summary())

    return transactions, identity, merged
