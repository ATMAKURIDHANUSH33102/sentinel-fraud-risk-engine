"""
Sentinel Fraud Detection Pipeline

End-to-end production ML pipeline:
1. Dataset loading (handles absence of train_identity safely)
2. Feature engineering & preprocessing (ColumnTransformer)
3. Chronological train/validation/test split (TransactionDT, no leakage)
4. Model training: Logistic Regression baseline vs Random Forest
5. Validation evaluation & model selection based on PR-AUC
6. Threshold selection on validation data (cost-optimal & 3-tier ALLOW/REVIEW/BLOCK)
7. Final held-out test evaluation
8. Model and pipeline serialization
9. Transaction scoring & explainability (defense-only, no LLM)
"""

import os
import sys
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.data.loader import DataLoader
from src.features import (
    engineer_features,
    build_preprocessor,
    get_feature_names_out,
    RAW_NUMERIC_COLS,
    RAW_CATEGORICAL_COLS,
    ENGINEERED_NUMERIC_COLS,
    ENGINEERED_CATEGORICAL_COLS,
)
from src.evaluation import (
    calculate_metrics,
    calculate_pr_auc,
    find_optimal_cost_threshold,
    calculate_tier_metrics,
)

warnings.filterwarnings("ignore")


class FraudDetectionPipeline:
    """End-to-end fraud detection machine learning pipeline."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        fp_cost: float = 10.0,
        fn_cost: float = 100.0,
    ):
        """
        Initialize pipeline parameters.

        Args:
            data_dir: Directory containing raw dataset files
            output_dir: Directory to save processed models and evaluation artifacts
            fp_cost: Cost assigned to false positives (project assumption: customer friction)
            fn_cost: Cost assigned to false negatives (project assumption: chargeback loss)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = self.output_dir / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.fp_cost = float(fp_cost)
        self.fn_cost = float(fn_cost)

        self.df_raw: Optional[pd.DataFrame] = None
        self.df_engineered: Optional[pd.DataFrame] = None
        self.preprocessor = None
        self.feature_names_out: List[str] = []

        # Splits
        self.X_train_raw = None
        self.y_train = None
        self.X_val_raw = None
        self.y_val = None
        self.X_test_raw = None
        self.y_test = None

        self.X_train = None
        self.X_val = None
        self.X_test = None

        # Models & results
        self.models: Dict[str, Any] = {}
        self.model_val_metrics: Dict[str, Dict[str, Any]] = {}
        self.selected_model_name: Optional[str] = None
        self.selected_model: Optional[Any] = None
        self.thresholds: Dict[str, float] = {}
        self.validation_evaluation: Dict[str, Any] = {}
        self.test_evaluation: Dict[str, Any] = {}
        self.feature_importances: Dict[str, float] = {}
        self.training_time_seconds: float = 0.0

    # =====================================================================
    # STEP 1: DATA LOADING
    # =====================================================================

    def load_data(self) -> "FraudDetectionPipeline":
        """
        Load transaction data from data_dir.
        Checks for train_identity.csv; if absent, documents limitation and
        proceeds with transaction data only (does NOT fall back to test_identity).
        """
        print("\n" + "=" * 75)
        print("STEP 1: LOADING DATASET")
        print("=" * 75)

        loader = DataLoader(data_dir=str(self.data_dir))
        
        # Load identity safely
        identity_df = loader.load_identity(allow_missing=True)
        if identity_df is None:
            print("  [Data Notice] train_identity.csv not present in data/raw.")
            print("  [Data Notice] Proceeding with transaction data only. Identity features omitted.")
        
        # Load transactions with required practical columns
        load_cols = (
            ["TransactionID", "isFraud", "TransactionDT"]
            + RAW_NUMERIC_COLS
            + RAW_CATEGORICAL_COLS
        )
        # Deduplicate column names just in case
        load_cols = list(dict.fromkeys(load_cols))

        print(f"  Loading columns ({len(load_cols)} total) from train_transaction.csv...")
        t0 = time.time()
        self.df_raw = loader.load_transactions(usecols=load_cols)
        load_time = time.time() - t0
        print(f"  Loaded {len(self.df_raw):,} rows in {load_time:.2f}s")
        print(f"  Memory usage: {self.df_raw.memory_usage().sum() / 1e6:.1f} MB")

        fraud_cnt = int(self.df_raw["isFraud"].sum())
        total_cnt = len(self.df_raw)
        fraud_rate = (fraud_cnt / total_cnt) * 100
        print(f"  Prevalence: {fraud_cnt:,} fraud / {total_cnt:,} total ({fraud_rate:.2f}%)")

        return self

    # =====================================================================
    # STEP 2: FEATURE ENGINEERING
    # =====================================================================

    def engineer_features(self) -> "FraudDetectionPipeline":
        """Engineer temporal, amount-based, and categorical features."""
        print("\n" + "=" * 75)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 75)

        t0 = time.time()
        print("  Extracting temporal signals (DT_hour, DT_day_of_week)...")
        print("  Computing log-transformed amount (log_TransactionAmt)...")
        print("  Cleaning and grouping email domains (top 10 + other + missing)...")

        self.df_engineered = engineer_features(self.df_raw)
        print(f"  Engineered features completed in {time.time() - t0:.2f}s")
        print(f"  Numeric features: {len(ENGINEERED_NUMERIC_COLS)}")
        print(f"  Categorical features: {len(ENGINEERED_CATEGORICAL_COLS)}")

        return self

    # =====================================================================
    # STEP 3: CHRONOLOGICAL TRAIN / VAL / TEST SPLIT & PREPROCESSING
    # =====================================================================

    def create_splits(
        self, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> "FraudDetectionPipeline":
        """
        Chronologically split dataset using TransactionDT.
        Strict temporal ordering: Train -> Validation -> Test.
        Fits ColumnTransformer ONLY on the train split to prevent target leakage.
        """
        print("\n" + "=" * 75)
        print("STEP 3: CHRONOLOGICAL TRAIN / VAL / TEST SPLIT")
        print("=" * 75)

        # Sort strictly by TransactionDT
        print("  Sorting chronologically by TransactionDT...")
        df_sorted = self.df_engineered.sort_values("TransactionDT").reset_index(drop=True)

        n = len(df_sorted)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        df_train = df_sorted.iloc[:n_train].copy()
        df_val = df_sorted.iloc[n_train : n_train + n_val].copy()
        df_test = df_sorted.iloc[n_train + n_val :].copy()

        # Time ranges
        t_tr_min, t_tr_max = df_train["TransactionDT"].min(), df_train["TransactionDT"].max()
        t_val_min, t_val_max = df_val["TransactionDT"].min(), df_val["TransactionDT"].max()
        t_te_min, t_te_max = df_test["TransactionDT"].min(), df_test["TransactionDT"].max()

        print(f"  Train set:      {len(df_train):,} rows ({df_train['isFraud'].mean():.4%} fraud) | DT: [{t_tr_min} - {t_tr_max}]")
        print(f"  Validation set: {len(df_val):,} rows ({df_val['isFraud'].mean():.4%} fraud) | DT: [{t_val_min} - {t_val_max}]")
        print(f"  Test set:       {len(df_test):,} rows ({df_test['isFraud'].mean():.4%} fraud) | DT: [{t_te_min} - {t_te_max}]")

        # Verify temporal separation
        assert t_tr_max <= t_val_min, "Temporal leakage detected between Train and Validation!"
        assert t_val_max <= t_te_min, "Temporal leakage detected between Validation and Test!"
        print("  [OK] Strict chronological ordering verified: No future transactions leak into training.")

        feature_cols = ENGINEERED_NUMERIC_COLS + ENGINEERED_CATEGORICAL_COLS

        self.X_train_raw = df_train[feature_cols]
        self.y_train = df_train["isFraud"].values.astype(int)

        self.X_val_raw = df_val[feature_cols]
        self.y_val = df_val["isFraud"].values.astype(int)

        self.X_test_raw = df_test[feature_cols]
        self.y_test = df_test["isFraud"].values.astype(int)

        # Build and fit ColumnTransformer ONLY on X_train_raw
        print("\n  Fitting preprocessing pipeline on training data only...")
        t0 = time.time()
        self.preprocessor = build_preprocessor()
        self.X_train = self.preprocessor.fit_transform(self.X_train_raw)
        self.X_val = self.preprocessor.transform(self.X_val_raw)
        self.X_test = self.preprocessor.transform(self.X_test_raw)

        self.feature_names_out = get_feature_names_out(self.preprocessor)
        print(f"  Preprocessed features: {self.X_train.shape[1]} columns in {time.time() - t0:.2f}s")
        print("  [OK] Median imputation and standard scaling fitted on train only.")
        print("  [OK] One-hot encoding fitted with handle_unknown='ignore' (unseen categories safe).")

        return self

    # =====================================================================
    # STEP 4: MODEL TRAINING & COMPARISON
    # =====================================================================

    def train_models(self) -> "FraudDetectionPipeline":
        """
        Train Logistic Regression baseline and Random Forest classifier.
        Evaluate both on validation set using PR-AUC (Average Precision).
        Select best practical model based on validation PR-AUC.
        """
        print("\n" + "=" * 75)
        print("STEP 4: MODEL TRAINING & VALIDATION COMPARISON")
        print("=" * 75)

        start_time = time.time()

        # 1. Logistic Regression Baseline
        print("\n  [1/2] Training Logistic Regression baseline (class_weight='balanced')...")
        t0 = time.time()
        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        lr.fit(self.X_train, self.y_train)
        lr_train_time = time.time() - t0
        print(f"  Logistic Regression trained in {lr_train_time:.2f}s")

        lr_val_prob = lr.predict_proba(self.X_val)[:, 1]
        lr_metrics = calculate_metrics(
            self.y_val, lr_val_prob, threshold=0.5, fp_cost=self.fp_cost, fn_cost=self.fn_cost
        )
        self.models["LogisticRegression"] = lr
        self.model_val_metrics["LogisticRegression"] = lr_metrics

        print(f"  Logistic Regression Val PR-AUC:  {lr_metrics['pr_auc']:.4f}")
        print(f"  Logistic Regression Val ROC-AUC: {lr_metrics['roc_auc']:.4f}")
        print(f"  Logistic Regression Val F1:      {lr_metrics['f1']:.4f}")

        # 2. Random Forest
        print("\n  [2/2] Training Random Forest classifier (class_weight='balanced', max_depth=12)...")
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(self.X_train, self.y_train)
        rf_train_time = time.time() - t0
        print(f"  Random Forest trained in {rf_train_time:.2f}s")

        rf_val_prob = rf.predict_proba(self.X_val)[:, 1]
        rf_metrics = calculate_metrics(
            self.y_val, rf_val_prob, threshold=0.5, fp_cost=self.fp_cost, fn_cost=self.fn_cost
        )
        self.models["RandomForest"] = rf
        self.model_val_metrics["RandomForest"] = rf_metrics

        print(f"  Random Forest Val PR-AUC:        {rf_metrics['pr_auc']:.4f}")
        print(f"  Random Forest Val ROC-AUC:       {rf_metrics['roc_auc']:.4f}")
        print(f"  Random Forest Val F1:            {rf_metrics['f1']:.4f}")

        self.training_time_seconds = time.time() - start_time

        # Model selection: Best validation PR-AUC (Average Precision)
        # Note: Do NOT use accuracy as primary metric.
        model_praucs = {name: m["pr_auc"] for name, m in self.model_val_metrics.items()}
        self.selected_model_name = max(model_praucs, key=model_praucs.get)
        self.selected_model = self.models[self.selected_model_name]

        print("\n" + "-" * 75)
        print(f"  MODEL COMPARISON SUMMARY (VALIDATION PR-AUC):")
        for m_name, prauc in model_praucs.items():
            roc = self.model_val_metrics[m_name]["roc_auc"]
            f1 = self.model_val_metrics[m_name]["f1"]
            print(f"    - {m_name:20s}: PR-AUC = {prauc:.4f} | ROC-AUC = {roc:.4f} | F1 (th=0.5) = {f1:.4f}")
        print(f"  SELECTED MODEL: {self.selected_model_name} (highest validation PR-AUC)")
        print("-" * 75)

        # Extract feature importances
        if hasattr(self.selected_model, "feature_importances_"):
            importances = self.selected_model.feature_importances_
        elif hasattr(self.selected_model, "coef_"):
            importances = np.abs(self.selected_model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names_out))

        feat_imp_series = pd.Series(importances, index=self.feature_names_out).sort_values(ascending=False)
        self.feature_importances = feat_imp_series.to_dict()

        print("\n  Top 10 Feature Importances:")
        for feat, score in list(self.feature_importances.items())[:10]:
            print(f"    {feat:32s}: {score:.4f}")

        return self

    # =====================================================================
    # STEP 5: THRESHOLD SELECTION & COST OPTIMIZATION (VALIDATION ONLY)
    # =====================================================================

    def optimize_thresholds(self) -> "FraudDetectionPipeline":
        """
        Optimize decision thresholds strictly on validation data.
        Never tune thresholds on the test set.

        Finds:
        1. threshold_cost_optimal: Minimizes total financial cost (FP=$10, FN=$100).
        2. threshold_review: Set at high-recall operating point (~90% recall) to route to manual review.
        3. threshold_block: Set at high-precision operating point (>= 70% precision) to auto-block.
        """
        print("\n" + "=" * 75)
        print("STEP 5: VALIDATION THRESHOLD SELECTION & COST OPTIMIZATION")
        print("=" * 75)

        val_probs = self.selected_model.predict_proba(self.X_val)[:, 1]

        # 1. Cost-optimal threshold
        best_th, min_cost, cost_df = find_optimal_cost_threshold(
            self.y_val, val_probs, fp_cost=self.fp_cost, fn_cost=self.fn_cost, n_thresholds=100
        )
        print(f"  Cost-Optimal Threshold on Validation: {best_th:.2f}")
        print(f"  Expected Validation Total Cost:       ${min_cost:,.2f}")
        print(f"  (Assumed costs: FP = ${self.fp_cost:.2f}, FN = ${self.fn_cost:.2f} — project assumptions)")

        # 2. Operating points from Precision-Recall curve
        precisions, recalls, pr_thresholds = precision_recall_curve(self.y_val, val_probs)

        # Review threshold: Target ~85-90% recall (catch fraud before it causes damage)
        # Find index where recall >= 0.85 with lowest probability threshold
        valid_rec_indices = np.where(recalls >= 0.85)[0]
        if len(valid_rec_indices) > 0 and valid_rec_indices[-1] < len(pr_thresholds):
            th_review = float(pr_thresholds[valid_rec_indices[-1]])
        else:
            th_review = 0.15
        th_review = max(0.05, min(th_review, best_th))

        # Block threshold: Target high precision (e.g. >= 65-75%) for automatic block
        valid_prec_indices = np.where(precisions >= 0.65)[0]
        if len(valid_prec_indices) > 0 and valid_prec_indices[0] < len(pr_thresholds):
            th_block = float(pr_thresholds[valid_prec_indices[0]])
        else:
            th_block = max(best_th + 0.15, 0.70)
        
        # Ensure strict ordering: 0 < review < block < 1
        th_block = max(th_review + 0.15, min(0.85, th_block))

        self.thresholds = {
            "review": round(float(th_review), 4),
            "block": round(float(th_block), 4),
            "cost_optimal": round(float(best_th), 4),
            "default": 0.50,
        }

        print(f"\n  Selected 3-Tier Operating Thresholds (Validation-tuned):")
        print(f"    ALLOW  (< {self.thresholds['review']:.2f})  : Low Risk  - Instant processing")
        print(f"    REVIEW ([{self.thresholds['review']:.2f}, {self.thresholds['block']:.2f})): Medium Risk - Stepped-up auth / human review")
        print(f"    BLOCK  (>= {self.thresholds['block']:.2f}) : High Risk - Automatic decline")

        # Validation tier evaluation
        val_tier_metrics = calculate_tier_metrics(
            self.y_val, val_probs, self.thresholds["review"], self.thresholds["block"]
        )
        val_binary_cost_metrics = calculate_metrics(
            self.y_val, val_probs, threshold=self.thresholds["cost_optimal"], fp_cost=self.fp_cost, fn_cost=self.fn_cost
        )

        self.validation_evaluation = {
            "model_name": self.selected_model_name,
            "thresholds": self.thresholds,
            "cost_optimal_metrics": val_binary_cost_metrics,
            "tier_metrics": val_tier_metrics,
            "all_models": self.model_val_metrics,
        }

        print(f"\n  Validation 3-Tier Traffic Routing:")
        print(f"    ALLOW:  {val_tier_metrics['allow']['total']:,} ({val_tier_metrics['allow']['pct_of_traffic']:.1%}) | Fraud leak: {val_tier_metrics['allow']['fraud']:,}")
        print(f"    REVIEW: {val_tier_metrics['review']['total']:,} ({val_tier_metrics['review']['pct_of_traffic']:.1%}) | Fraud captured: {val_tier_metrics['review']['fraud']:,} (Prevalence: {val_tier_metrics['review']['fraud_prevalence']:.1%})")
        print(f"    BLOCK:  {val_tier_metrics['block']['total']:,} ({val_tier_metrics['block']['pct_of_traffic']:.1%}) | Fraud blocked: {val_tier_metrics['block']['fraud']:,} (Precision: {val_tier_metrics['block']['precision']:.1%})")
        print(f"    Total Fraud Captured (Review+Block): {val_tier_metrics['total_fraud_captured_pct']:.1%}")

        return self

    # =====================================================================
    # STEP 6: FINAL HELD-OUT TEST EVALUATION
    # =====================================================================

    def evaluate_held_out_test(self) -> "FraudDetectionPipeline":
        """
        Evaluate selected model on the final held-out test set using frozen thresholds.
        Never tunes or touches thresholds on test set.
        """
        print("\n" + "=" * 75)
        print("STEP 6: FINAL HELD-OUT TEST EVALUATION (TOUCHED ONCE)")
        print("=" * 75)

        test_probs = self.selected_model.predict_proba(self.X_test)[:, 1]

        # Test metrics at cost-optimal threshold
        test_cost_metrics = calculate_metrics(
            self.y_test,
            test_probs,
            threshold=self.thresholds["cost_optimal"],
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
        )

        # Test metrics at default 0.5 threshold
        test_default_metrics = calculate_metrics(
            self.y_test,
            test_probs,
            threshold=0.50,
            fp_cost=self.fp_cost,
            fn_cost=self.fn_cost,
        )

        # Test 3-tier routing metrics
        test_tier_metrics = calculate_tier_metrics(
            self.y_test, test_probs, self.thresholds["review"], self.thresholds["block"]
        )

        self.test_evaluation = {
            "model_name": self.selected_model_name,
            "cost_optimal_metrics": test_cost_metrics,
            "default_metrics": test_default_metrics,
            "tier_metrics": test_tier_metrics,
            "cost_assumptions": {
                "fp_cost": self.fp_cost,
                "fn_cost": self.fn_cost,
                "disclaimer": "Project assumptions only, not Razorpay actuals.",
            },
        }

        print(f"\n  Final Test Performance Summary (Model: {self.selected_model_name}):")
        print(f"    PR-AUC (Primary Metric): {test_cost_metrics['pr_auc']:.4f}")
        print(f"    ROC-AUC:                {test_cost_metrics['roc_auc']:.4f}")
        print(f"\n  At Cost-Optimal Threshold ({self.thresholds['cost_optimal']:.2f}):")
        print(f"    Precision:              {test_cost_metrics['precision']:.4f}")
        print(f"    Recall:                 {test_cost_metrics['recall']:.4f}")
        print(f"    F1 Score:               {test_cost_metrics['f1']:.4f}")
        print(f"    False Positive Rate:    {test_cost_metrics['fpr']:.4f} ({test_cost_metrics['fpr']*100:.2f}%)")
        cm = test_cost_metrics["confusion_matrix"]
        print(f"    Confusion Matrix:       TN={cm['tn']:,} | FP={cm['fp']:,} | FN={cm['fn']:,} | TP={cm['tp']:,}")
        print(f"    Expected Total Cost:    ${test_cost_metrics['cost']['total_cost']:,.2f}")
        print(f"    Cost per Transaction:   ${test_cost_metrics['cost']['avg_cost_per_transaction']:.2f}")

        print(f"\n  Test 3-Tier Routing Distribution:")
        print(f"    ALLOW  (< {self.thresholds['review']:.2f})  : {test_tier_metrics['allow']['total']:,} txns ({test_tier_metrics['allow']['pct_of_traffic']:.1%}) | Fraud missed: {test_tier_metrics['allow']['fraud']:,}")
        print(f"    REVIEW ([{self.thresholds['review']:.2f}, {self.thresholds['block']:.2f})): {test_tier_metrics['review']['total']:,} txns ({test_tier_metrics['review']['pct_of_traffic']:.1%}) | Fraud caught: {test_tier_metrics['review']['fraud']:,}")
        print(f"    BLOCK  (>= {self.thresholds['block']:.2f}) : {test_tier_metrics['block']['total']:,} txns ({test_tier_metrics['block']['pct_of_traffic']:.1%}) | Precision: {test_tier_metrics['block']['precision']:.1%}")
        print(f"    Total Fraud Captured (Review + Block): {test_tier_metrics['total_fraud_captured_pct']:.1%}")

        return self

    # =====================================================================
    # STEP 7: ARTIFACT SERIALIZATION
    # =====================================================================

    def save_pipeline(self) -> "FraudDetectionPipeline":
        """
        Save trained model, preprocessor, thresholds, and metadata together.
        Ensures production application can load the exact same pipeline.
        """
        print("\n" + "=" * 75)
        print("STEP 7: SAVING TRAINED PIPELINE ARTIFACTS")
        print("=" * 75)

        bundle = {
            "model_name": self.selected_model_name,
            "model": self.selected_model,
            "preprocessor": self.preprocessor,
            "feature_names_out": self.feature_names_out,
            "engineered_numeric_cols": ENGINEERED_NUMERIC_COLS,
            "engineered_categorical_cols": ENGINEERED_CATEGORICAL_COLS,
            "thresholds": self.thresholds,
            "cost_assumptions": {
                "fp_cost": self.fp_cost,
                "fn_cost": self.fn_cost,
                "disclaimer": "Project assumptions only, not Razorpay actuals.",
            },
            "validation_evaluation": self.validation_evaluation,
            "test_evaluation": self.test_evaluation,
            "top_features": list(self.feature_importances.items())[:20],
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_rows": len(self.df_raw) if self.df_raw is not None else 0,
                "training_time_seconds": self.training_time_seconds,
                "identity_data_present": False,
                "limitation": "train_identity.csv not present; trained strictly on transaction features.",
            },
        }

        # 1. Complete bundle
        bundle_path = self.model_dir / "model_pipeline.pkl"
        joblib.dump(bundle, bundle_path)
        print(f"  [OK] Complete pipeline bundle: {bundle_path}")

        # 2. Individual artifacts for backwards compatibility
        joblib.dump(self.selected_model, self.model_dir / "fraud_model.pkl")
        joblib.dump(self.preprocessor, self.model_dir / "preprocessor.pkl")
        joblib.dump(self.thresholds, self.model_dir / "thresholds.pkl")
        joblib.dump(self.feature_names_out, self.model_dir / "feature_names.pkl")
        print(f"  [OK] Individual components saved in {self.model_dir}/")

        # 3. JSON summary for reporting and dashboards
        summary_path = self.model_dir / "pipeline_summary.json"
        json_data = {
            "selected_model": self.selected_model_name,
            "training_time_seconds": round(self.training_time_seconds, 2),
            "thresholds": self.thresholds,
            "cost_assumptions": bundle["cost_assumptions"],
            "validation_metrics": {
                "pr_auc": round(self.validation_evaluation["cost_optimal_metrics"]["pr_auc"], 4),
                "roc_auc": round(self.validation_evaluation["cost_optimal_metrics"]["roc_auc"], 4),
                "precision": round(self.validation_evaluation["cost_optimal_metrics"]["precision"], 4),
                "recall": round(self.validation_evaluation["cost_optimal_metrics"]["recall"], 4),
                "f1": round(self.validation_evaluation["cost_optimal_metrics"]["f1"], 4),
                "fpr": round(self.validation_evaluation["cost_optimal_metrics"]["fpr"], 4),
                "total_cost": round(self.validation_evaluation["cost_optimal_metrics"]["cost"]["total_cost"], 2),
            },
            "test_metrics": {
                "pr_auc": round(self.test_evaluation["cost_optimal_metrics"]["pr_auc"], 4),
                "roc_auc": round(self.test_evaluation["cost_optimal_metrics"]["roc_auc"], 4),
                "precision": round(self.test_evaluation["cost_optimal_metrics"]["precision"], 4),
                "recall": round(self.test_evaluation["cost_optimal_metrics"]["recall"], 4),
                "f1": round(self.test_evaluation["cost_optimal_metrics"]["f1"], 4),
                "fpr": round(self.test_evaluation["cost_optimal_metrics"]["fpr"], 4),
                "total_cost": round(self.test_evaluation["cost_optimal_metrics"]["cost"]["total_cost"], 2),
                "confusion_matrix": self.test_evaluation["cost_optimal_metrics"]["confusion_matrix"],
            },
            "top_10_features": list(self.feature_importances.items())[:10],
            "limitation": bundle["metadata"]["limitation"],
        }
        with open(summary_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"  [OK] Human-readable summary: {summary_path}")

        return self

    # =====================================================================
    # STEP 8: INFERENCE & EXPLAINABILITY HELPER
    # =====================================================================

    def score_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single transaction dict and return risk decision with feature evidence.
        Defense-only; strictly no LLM.
        """
        raw_df = pd.DataFrame([transaction])
        engineered_df = engineer_features(raw_df)
        X_proc = self.preprocessor.transform(
            engineered_df[ENGINEERED_NUMERIC_COLS + ENGINEERED_CATEGORICAL_COLS]
        )

        proba = float(self.selected_model.predict_proba(X_proc)[0, 1])

        # 3-tier routing
        if proba >= self.thresholds["block"]:
            decision = "BLOCK"
            risk_level = "HIGH"
        elif proba >= self.thresholds["review"]:
            decision = "REVIEW"
            risk_level = "MEDIUM"
        else:
            decision = "ALLOW"
            risk_level = "LOW"

        # Explainability evidence (top feature signals)
        evidence = []
        amt = float(transaction.get("TransactionAmt", transaction.get("TransactionAMT", 0.0)))
        if amt > 500.0:
            evidence.append(f"Elevated transaction amount (${amt:,.2f})")
        
        c1 = transaction.get("C1", 0)
        if c1 and float(c1) > 5.0:
            evidence.append(f"High transaction count signal (C1={c1})")
        
        top_importances = list(self.feature_importances.items())[:5]
        top_model_factors = [f"{feat} (weight: {val:.3f})" for feat, val in top_importances]

        return {
            "fraud_probability": round(proba, 4),
            "risk_level": risk_level,
            "decision": decision,
            "confidence": round(max(proba, 1.0 - proba), 4),
            "thresholds": self.thresholds,
            "evidence": {
                "transaction_signals": evidence,
                "top_model_factors": top_model_factors,
                "model_name": self.selected_model_name,
                "evaluation_mode": "defense_only_deterministic_ml",
            },
        }

    # =====================================================================
    # PIPELINE RUNNER
    # =====================================================================

    def run_complete_pipeline(self) -> "FraudDetectionPipeline":
        """Run the complete fraud detection pipeline from end to end."""
        t_start = time.time()
        print("\n" + "#" * 75)
        print("  SENTINEL PAYMENTS FRAUD RISK ENGINE - ML PIPELINE")
        print("#" * 75)

        self.load_data()
        self.engineer_features()
        self.create_splits()
        self.train_models()
        self.optimize_thresholds()
        self.evaluate_held_out_test()
        self.save_pipeline()

        total_elapsed = time.time() - t_start
        print("\n" + "=" * 75)
        print(f"[OK] PIPELINE COMPLETE IN {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)")
        print("=" * 75)
        return self


def main():
    pipeline = FraudDetectionPipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
