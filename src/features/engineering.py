"""
Sentinel Feature Engineering Module

Extracts and prepares practical features for fraud detection:
- Correct numeric feature names (e.g. TransactionAmt)
- Key transaction attributes: card, address, distances, counts (C), timedeltas (D)
- Engineered temporal signals from TransactionDT (hour, day of week)
- Log-transformed transaction amounts
- Cleaned and grouped high-value categorical features (ProductCD, card4, card6, P_emaildomain)
- Safe preprocessor construction with sklearn ColumnTransformer
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Top email domains to keep distinct; all others grouped as 'other'
TOP_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "anonymous.com",
    "aol.com",
    "comcast.net",
    "icloud.com",
    "outlook.com",
    "msn.com",
    "att.net",
]

# Raw columns required from dataset
RAW_NUMERIC_COLS = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "C1",
    "C2",
    "C5",
    "C11",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
]

RAW_CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
]

# Engineered / transformed feature lists
ENGINEERED_NUMERIC_COLS = [
    "TransactionAmt",
    "log_TransactionAmt",
    "DT_hour",
    "DT_day_of_week",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "C1",
    "C2",
    "C5",
    "C11",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
]

ENGINEERED_CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain_clean",
]


def clean_email_domain(email_series: pd.Series) -> pd.Series:
    """Clean and group email domains into top providers, 'other', and 'missing'."""
    cleaned = email_series.fillna("missing").astype(str).str.lower().str.strip()
    return cleaned.apply(
        lambda x: x if (x in TOP_EMAIL_DOMAINS or x == "missing") else "other"
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain features from raw transaction attributes.

    Args:
        df: DataFrame containing raw transaction columns

    Returns:
        DataFrame with added derived features
    """
    df = df.copy()

    # Handle TransactionAmt & log transform
    if "TransactionAmt" in df.columns:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0)
        df["TransactionAmt"] = amt
        df["log_TransactionAmt"] = np.log1p(np.maximum(amt, 0.0))
    elif "TransactionAMT" in df.columns:
        amt = pd.to_numeric(df["TransactionAMT"], errors="coerce").fillna(0.0)
        df["TransactionAmt"] = amt
        df["log_TransactionAmt"] = np.log1p(np.maximum(amt, 0.0))
    else:
        df["TransactionAmt"] = 0.0
        df["log_TransactionAmt"] = 0.0

    # Temporal feature derivation
    if "TransactionDT" in df.columns and df["TransactionDT"].notna().any():
        dt = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0)
        df["DT_hour"] = (dt // 3600) % 24
        df["DT_day_of_week"] = (dt // 86400) % 7
    else:
        if "DT_hour" not in df.columns:
            df["DT_hour"] = 12.0  # default midday
        if "DT_day_of_week" not in df.columns:
            df["DT_day_of_week"] = 3.0  # default midweek

    # Categorical email domain grouping
    if "P_emaildomain" in df.columns:
        df["P_emaildomain_clean"] = clean_email_domain(df["P_emaildomain"])
    elif "P_emaildomain_clean" not in df.columns:
        df["P_emaildomain_clean"] = "missing"

    # Ensure all required numeric columns exist
    for col in ENGINEERED_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure all required categorical columns exist
    for col in ["ProductCD", "card4", "card6"]:
        if col not in df.columns:
            df[col] = "missing"
        else:
            df[col] = df[col].fillna("missing").astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for fraud feature preprocessing.
    
    - Numeric features: Median imputation + StandardScaler
    - Categorical features: Constant 'missing' imputation + OneHotEncoder(ignore unknown)
    
    Returns:
        Unfitted ColumnTransformer instance
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ENGINEERED_NUMERIC_COLS),
            ("cat", categorical_transformer, ENGINEERED_CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Get the names of all transformed features output by the ColumnTransformer."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(
        cat_encoder.get_feature_names_out(ENGINEERED_CATEGORICAL_COLS)
    )
    return ENGINEERED_NUMERIC_COLS + cat_feature_names
