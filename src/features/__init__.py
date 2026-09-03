# Feature engineering module
"""Feature extraction and transformation."""

from src.features.engineering import (
    engineer_features,
    build_preprocessor,
    get_feature_names_out,
    clean_email_domain,
    ENGINEERED_NUMERIC_COLS,
    ENGINEERED_CATEGORICAL_COLS,
    RAW_NUMERIC_COLS,
    RAW_CATEGORICAL_COLS,
)

__all__ = [
    "engineer_features",
    "build_preprocessor",
    "get_feature_names_out",
    "clean_email_domain",
    "ENGINEERED_NUMERIC_COLS",
    "ENGINEERED_CATEGORICAL_COLS",
    "RAW_NUMERIC_COLS",
    "RAW_CATEGORICAL_COLS",
]
