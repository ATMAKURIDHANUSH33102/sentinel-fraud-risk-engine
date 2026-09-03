"""
Sentinel Fraud Detection - Evaluation Metrics Module

Provides robust metric computation for fraud risk models:
- Precision, Recall, F1, PR-AUC (Average Precision)
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR)
- Cost-sensitive analysis with configurable FP/FN costs
- Validation threshold optimization (binary and 3-tier ALLOW/REVIEW/BLOCK)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)


def calculate_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fp_cost: float = 10.0,
    fn_cost: float = 100.0,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pr_auc = calculate_pr_auc(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))

    total_cost = float((fp * fp_cost) + (fn * fn_cost))
    avg_cost = float(total_cost / len(y_true)) if len(y_true) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "cost": {
            "fp_cost_rate": float(fp_cost),
            "fn_cost_rate": float(fn_cost),
            "total_fp_cost": float(fp * fp_cost),
            "total_fn_cost": float(fn * fn_cost),
            "total_cost": total_cost,
            "avg_cost_per_transaction": avg_cost,
            "cost_disclaimer": "Project assumptions only, not Razorpay actuals.",
        },
    }


def find_optimal_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float = 10.0,
    fn_cost: float = 100.0,
    n_thresholds: int = 100,
) -> Tuple[float, float, pd.DataFrame]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    records = []

    best_threshold = 0.5
    lowest_cost = float("inf")

    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = (fp * fp_cost) + (fn * fn_cost)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1_val = f1_score(y_true, pred, zero_division=0)

        records.append({
            "threshold": float(th),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_val),
            "total_cost": float(cost),
        })

        if cost < lowest_cost:
            lowest_cost = cost
            best_threshold = float(th)

    cost_df = pd.DataFrame(records)
    return best_threshold, lowest_cost, cost_df


def calculate_tier_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_review: float,
    threshold_block: float,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_total = len(y_true)
    n_fraud = int(np.sum(y_true == 1))
    n_legit = int(np.sum(y_true == 0))

    is_allow = y_prob < threshold_review
    is_review = (y_prob >= threshold_review) & (y_prob < threshold_block)
    is_block = y_prob >= threshold_block

    allow_fraud = int(np.sum((y_true == 1) & is_allow))
    allow_legit = int(np.sum((y_true == 0) & is_allow))

    review_fraud = int(np.sum((y_true == 1) & is_review))
    review_legit = int(np.sum((y_true == 0) & is_review))

    block_fraud = int(np.sum((y_true == 1) & is_block))
    block_legit = int(np.sum((y_true == 0) & is_block))

    return {
        "thresholds": {
            "review": float(threshold_review),
            "block": float(threshold_block),
        },
        "allow": {
            "total": int(np.sum(is_allow)),
            "pct_of_traffic": float(np.sum(is_allow) / n_total) if n_total > 0 else 0.0,
            "fraud": allow_fraud,
            "legit": allow_legit,
            "leakage_fraud_rate": float(allow_fraud / allow_legit) if allow_legit > 0 else 0.0,
        },
        "review": {
            "total": int(np.sum(is_review)),
            "pct_of_traffic": float(np.sum(is_review) / n_total) if n_total > 0 else 0.0,
            "fraud": review_fraud,
            "legit": review_legit,
            "fraud_prevalence": float(review_fraud / np.sum(is_review)) if np.sum(is_review) > 0 else 0.0,
        },
        "block": {
            "total": int(np.sum(is_block)),
            "pct_of_traffic": float(np.sum(is_block) / n_total) if n_total > 0 else 0.0,
            "fraud": block_fraud,
            "legit": block_legit,
            "precision": float(block_fraud / np.sum(is_block)) if np.sum(is_block) > 0 else 0.0,
        },
        "total_fraud_captured_pct": float((review_fraud + block_fraud) / n_fraud) if n_fraud > 0 else 0.0,
        "auto_block_fraud_captured_pct": float(block_fraud / n_fraud) if n_fraud > 0 else 0.0,
    }