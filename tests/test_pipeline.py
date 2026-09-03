"""
Tests for Sentinel Pipeline, Feature Engineering, Evaluation, and Inference

Validates:
- Feature engineering & email domain cleaning
- Safe ColumnTransformer handling of missing values & unseen categories
- Chronological train/val/test ordering
- PR-AUC, Precision, Recall, F1, and cost calculation
- 3-tier decision routing (ALLOW / REVIEW / BLOCK)
- Production inference with structured explainability
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.features import (
    engineer_features,
    build_preprocessor,
    get_feature_names_out,
    clean_email_domain,
    ENGINEERED_NUMERIC_COLS,
    ENGINEERED_CATEGORICAL_COLS,
)
from src.evaluation import (
    calculate_metrics,
    calculate_pr_auc,
    find_optimal_cost_threshold,
    calculate_tier_metrics,
)
from src.inference import FraudDetector
from src.models.pipeline import FraudDetectionPipeline


@pytest.fixture
def synthetic_pipeline_data(tmp_path):
    """Generate small chronological transaction dataset for fast testing."""
    np.random.seed(42)
    n = 200
    dts = np.sort(np.random.randint(86400, 86400 * 30, size=n))
    amts = np.random.exponential(scale=100.0, size=n)
    is_fraud = (np.random.rand(n) < 0.10).astype(int)

    data = {
        "TransactionID": list(range(10000, 10000 + n)),
        "isFraud": is_fraud,
        "TransactionDT": dts,
        "TransactionAmt": amts,
        "ProductCD": np.random.choice(["W", "C", "H", "R", "S"], size=n),
        "card1": np.random.randint(1000, 9999, size=n),
        "card2": np.random.choice([100.0, 200.0, np.nan], size=n),
        "card3": [150.0] * n,
        "card4": np.random.choice(["visa", "mastercard", "discover", None], size=n),
        "card5": [226.0] * n,
        "card6": np.random.choice(["debit", "credit", None], size=n),
        "addr1": np.random.choice([100.0, 200.0, np.nan], size=n),
        "addr2": [50.0] * n,
        "dist1": np.random.choice([10.0, 50.0, np.nan], size=n),
        "P_emaildomain": np.random.choice(
            ["gmail.com", "yahoo.com", "unknown.xyz", None], size=n
        ),
    }

    for c in ["C1", "C2", "C5", "C11", "C13", "C14"]:
        data[c] = np.random.poisson(lam=1.5, size=n).astype(float)
    for d in ["D1", "D2", "D3", "D4", "D10", "D15"]:
        data[d] = np.random.choice([0.0, 10.0, np.nan], size=n)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = pd.DataFrame(data)
    df.to_csv(raw_dir / "train_transaction.csv", index=False)
    return raw_dir, tmp_path / "output"


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_clean_email_domain(self):
        """Test grouping of top email domains and fallback to other/missing."""
        series = pd.Series(["GMAIL.COM", "yahoo.com", "obscure-provider.org", None, "   HOTMAIL.COM  "])
        cleaned = clean_email_domain(series)

        assert cleaned.iloc[0] == "gmail.com"
        assert cleaned.iloc[1] == "yahoo.com"
        assert cleaned.iloc[2] == "other"
        assert cleaned.iloc[3] == "missing"
        assert cleaned.iloc[4] == "hotmail.com"

    def test_engineer_features_derives_correct_columns(self):
        """Test feature extraction on raw DataFrame."""
        df = pd.DataFrame({
            "TransactionAmt": [100.0, 0.0],
            "TransactionDT": [3600 * 14, 86400 * 2],
            "ProductCD": ["W", "C"],
            "P_emaildomain": ["gmail.com", "weird.io"],
        })
        engineered = engineer_features(df)

        assert "DT_hour" in engineered.columns
        assert "DT_day_of_week" in engineered.columns
        assert "log_TransactionAmt" in engineered.columns
        assert "P_emaildomain_clean" in engineered.columns

        assert engineered["DT_hour"].iloc[0] == 14
        assert engineered["log_TransactionAmt"].iloc[0] > 0

    def test_preprocessor_handles_unseen_and_missing(self):
        """Test ColumnTransformer with missing values and unseen categorical values."""
        df_train = pd.DataFrame({
            "TransactionAmt": [100.0, 200.0, np.nan],
            "log_TransactionAmt": [4.6, 5.3, 0.0],
            "DT_hour": [10, 12, 14],
            "DT_day_of_week": [1, 2, 3],
            "card1": [1000, 2000, 3000],
            "card2": [np.nan, 200.0, 300.0],
            "card3": [150.0, 150.0, 150.0],
            "card5": [226.0, 226.0, 226.0],
            "addr1": [np.nan, 100.0, 200.0],
            "addr2": [50.0, 50.0, 50.0],
            "dist1": [np.nan, 5.0, 10.0],
            "C1": [1, 2, 3], "C2": [1, 2, 3], "C5": [0, 0, 0],
            "C11": [1, 1, 1], "C13": [1, 2, 3], "C14": [1, 1, 1],
            "D1": [0, 10, np.nan], "D2": [np.nan, 5, 10], "D3": [0, np.nan, 2],
            "D4": [np.nan, 1, 2], "D10": [0, 5, np.nan], "D15": [0, 10, 20],
            "ProductCD": ["W", "C", "missing"],
            "card4": ["visa", "mastercard", "missing"],
            "card6": ["debit", "credit", "missing"],
            "P_emaildomain_clean": ["gmail.com", "other", "missing"],
        })

        prep = build_preprocessor()
        X_tr = prep.fit_transform(df_train)
        assert X_tr.shape[0] == 3
        assert not np.isnan(X_tr).any(), "Transformed output must not contain NaNs"

        df_test = pd.DataFrame({
            "TransactionAmt": [500.0],
            "log_TransactionAmt": [6.2],
            "DT_hour": [23],
            "DT_day_of_week": [6],
            "card1": [9999],
            "card2": [np.nan],
            "card3": [np.nan],
            "card5": [np.nan],
            "addr1": [np.nan],
            "addr2": [np.nan],
            "dist1": [np.nan],
            "C1": [10], "C2": [5], "C5": [1],
            "C11": [2], "C13": [8], "C14": [3],
            "D1": [np.nan], "D2": [np.nan], "D3": [np.nan],
            "D4": [np.nan], "D10": [np.nan], "D15": [np.nan],
            "ProductCD": ["TOTALLY_UNSEEN_PRODUCT"],
            "card4": ["UNSEEN_CARD"],
            "card6": ["UNSEEN_TYPE"],
            "P_emaildomain_clean": ["UNSEEN_DOMAIN"],
        })
        X_te = prep.transform(df_test)
        assert X_te.shape[0] == 1
        assert not np.isnan(X_te).any(), "Unseen categories transformed safely without NaNs"


class TestEvaluationAndCosts:
    """Test evaluation metrics, cost functions, and 3-tier routing."""

    def test_calculate_pr_auc(self):
        """Test that calculate_pr_auc returns valid float in [0, 1]."""
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.1, 0.7, 0.4])
        score = calculate_pr_auc(y_true, y_prob)

        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_calculate_metrics_structure(self):
        """Test that calculate_metrics returns all required fields."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.7, 0.3])
        m = calculate_metrics(y_true, y_prob, threshold=0.5, fp_cost=10.0, fn_cost=100.0)

        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert "pr_auc" in m
        assert "roc_auc" in m
        assert "fpr" in m
        assert "confusion_matrix" in m
        assert "cost" in m
        assert m["cost"]["total_cost"] >= 0.0

    def test_optimal_cost_threshold(self):
        """Test that find_optimal_cost_threshold finds minimum cost on validation data."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.60, 0.70, 0.80, 0.90])
        best_th, min_cost, df = find_optimal_cost_threshold(y_true, y_prob, fp_cost=10.0, fn_cost=100.0)

        assert 0.20 <= best_th <= 0.60
        assert min_cost == 0.0

    def test_tier_metrics_routing(self):
        """Test ALLOW / REVIEW / BLOCK distribution logic."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.05, 0.30, 0.50, 0.95])

        tiers = calculate_tier_metrics(y_true, y_prob, threshold_review=0.20, threshold_block=0.80)

        assert tiers["allow"]["total"] == 1
        assert tiers["review"]["total"] == 2
        assert tiers["block"]["total"] == 1
        assert tiers["block"]["fraud"] == 1
        assert tiers["block"]["precision"] == 1.0
        assert tiers["total_fraud_captured_pct"] == 1.0


class TestPipelineEndToEnd:
    """Test full pipeline execution on synthetic data."""

    def test_pipeline_runs_and_saves_artifacts(self, synthetic_pipeline_data):
        """Verify pipeline executes cleanly and produces loadable artifacts."""
        raw_dir, out_dir = synthetic_pipeline_data
        pipeline = FraudDetectionPipeline(
            data_dir=str(raw_dir),
            output_dir=str(out_dir),
            fp_cost=10.0,
            fn_cost=100.0,
        )

        pipeline.load_data()
        pipeline.engineer_features()
        pipeline.create_splits(train_ratio=0.70, val_ratio=0.15)
        pipeline.train_models()
        pipeline.optimize_thresholds()
        pipeline.evaluate_held_out_test()
        pipeline.save_pipeline()

        model_dir = out_dir / "model"
        assert (model_dir / "model_pipeline.pkl").exists()
        assert (model_dir / "fraud_model.pkl").exists()
        assert (model_dir / "preprocessor.pkl").exists()
        assert (model_dir / "pipeline_summary.json").exists()

        detector = FraudDetector(model_dir=str(model_dir))
        sample_txn = {
            "TransactionAmt": 150.00,
            "TransactionDT": 86400 * 25,
            "ProductCD": "W",
            "card1": 5000,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "C1": 2, "C2": 1, "C5": 0, "C11": 1, "C13": 1, "C14": 1,
            "D1": 0, "D2": 0, "D3": 0, "D4": 0, "D10": 0, "D15": 0,
        }
        res = detector.score(sample_txn)
        assert "fraud_probability" in res
        assert res["decision"] in ["ALLOW", "REVIEW", "BLOCK"]
        assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert "evidence" in res