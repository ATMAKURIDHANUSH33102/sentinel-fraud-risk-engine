"""
Tests for src/data/loader.py

Validates data loading, merging, and validation functions.
Uses fast synthetic fixtures and targeted sample checks to ensure rapid test execution.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader, load_fraud_data


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with synthetic transaction and identity files."""
    data_dir = tmp_path / "data_raw"
    data_dir.mkdir()

    # Synthetic transactions
    trans_df = pd.DataFrame({
        "TransactionID": [1001, 1002, 1003, 1004],
        "isFraud": [0, 1, 0, 0],
        "TransactionDT": [86400, 86450, 86500, 86600],
        "TransactionAmt": [50.0, 120.5, 999.0, 15.0],
        "ProductCD": ["W", "C", "W", "H"],
        "card1": [100, 200, 300, 400],
    })
    trans_df.to_csv(data_dir / "train_transaction.csv", index=False)

    # Synthetic identity
    ident_df = pd.DataFrame({
        "TransactionID": [1001, 1002],
        "DeviceType": ["mobile", "desktop"],
        "DeviceInfo": ["iOS", "Windows"],
    })
    ident_df.to_csv(data_dir / "train_identity.csv", index=False)

    return data_dir


class TestDataLoaderWithFixtures:
    """Test suite using fast fixtures."""

    def test_loader_initialization(self, sample_data_dir):
        """Test DataLoader initialization paths."""
        loader = DataLoader(data_dir=str(sample_data_dir))
        assert loader.data_dir == sample_data_dir
        assert loader.transaction_file == sample_data_dir / "train_transaction.csv"
        assert loader.identity_file == sample_data_dir / "train_identity.csv"

    def test_load_transactions_success(self, sample_data_dir):
        """Test successful loading of transaction data."""
        loader = DataLoader(data_dir=str(sample_data_dir))
        df = loader.load_transactions()

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "isFraud" in df.columns
        assert "TransactionID" in df.columns

    def test_load_transactions_usecols(self, sample_data_dir):
        """Test loading transactions with column filter."""
        loader = DataLoader(data_dir=str(sample_data_dir))
        df = loader.load_transactions(usecols=["TransactionID", "isFraud", "TransactionAmt"])

        assert list(df.columns) == ["TransactionID", "isFraud", "TransactionAmt"]
        assert len(df) == 4

    def test_load_transactions_missing_target_raises_error(self, tmp_path):
        """Test error when isFraud column is missing."""
        bad_dir = tmp_path / "bad_data"
        bad_dir.mkdir()
        df = pd.DataFrame({"TransactionID": [1, 2], "Amount": [10, 20]})
        df.to_csv(bad_dir / "train_transaction.csv", index=False)

        loader = DataLoader(data_dir=str(bad_dir))
        with pytest.raises(ValueError, match="Column 'isFraud' not found"):
            loader.load_transactions()

    def test_load_identity_when_present(self, sample_data_dir):
        """Test loading identity data when file exists."""
        loader = DataLoader(data_dir=str(sample_data_dir))
        df = loader.load_identity()

        assert df is not None
        assert len(df) == 2
        assert "TransactionID" in df.columns

    def test_load_identity_when_absent_allows_missing(self, tmp_path):
        """Test that missing identity file returns None when allow_missing=True."""
        no_ident_dir = tmp_path / "no_ident"
        no_ident_dir.mkdir()
        trans = pd.DataFrame({"TransactionID": [1], "isFraud": [0]})
        trans.to_csv(no_ident_dir / "train_transaction.csv", index=False)

        loader = DataLoader(data_dir=str(no_ident_dir))
        ident = loader.load_identity(allow_missing=True)
        assert ident is None

    def test_merge_datasets_with_identity(self, sample_data_dir):
        """Test left-join merging of transactions and identity."""
        loader = DataLoader(data_dir=str(sample_data_dir))
        loader.load_transactions()
        loader.load_identity()
        merged = loader.merge_datasets()

        assert len(merged) == 4
        assert "DeviceType" in merged.columns
        assert merged["DeviceType"].notna().sum() == 2

    def test_merge_datasets_without_identity(self, tmp_path):
        """Test merging behavior when identity data is absent."""
        no_ident_dir = tmp_path / "no_ident"
        no_ident_dir.mkdir()
        trans = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1]})
        trans.to_csv(no_ident_dir / "train_transaction.csv", index=False)

        loader = DataLoader(data_dir=str(no_ident_dir))
        loader.load_transactions()
        loader.load_identity(allow_missing=True)
        merged = loader.merge_datasets()

        assert len(merged) == 2
        assert "isFraud" in merged.columns

    def test_validate_transactions(self, sample_data_dir):
        """Test transaction validation report."""
        loader = DataLoader(data_dir=str(sample_data_dir))
        loader.load_transactions()
        report = loader.validate_transactions()

        assert report["n_rows"] == 4
        assert report["n_fraud"] == 1
        assert report["n_legit"] == 3
        assert report["fraud_prevalence"] == 0.25

    def test_convenience_function(self, sample_data_dir):
        """Test load_fraud_data convenience function."""
        trans, ident, merged = load_fraud_data(data_dir=str(sample_data_dir))
        assert len(trans) == 4
        assert len(merged) == 4


class TestActualRawDataIntegrity:
    """Lightweight integration tests on the actual data directory."""

    def test_actual_transaction_file_readable(self):
        """Verify the actual raw transaction file exists and has correct columns."""
        loader = DataLoader(data_dir="data/raw")
        df_sample = loader.load_transactions(nrows=10)

        assert len(df_sample) == 10
        assert "isFraud" in df_sample.columns
        assert "TransactionID" in df_sample.columns
        assert "TransactionDT" in df_sample.columns
        assert "TransactionAmt" in df_sample.columns

    def test_actual_identity_limitation_handled(self):
        """Verify actual data directory handles missing train_identity safely."""
        loader = DataLoader(data_dir="data/raw")
        ident = loader.load_identity(allow_missing=True)
        # train_identity.csv is absent in data/raw; verify None is returned
        assert ident is None
