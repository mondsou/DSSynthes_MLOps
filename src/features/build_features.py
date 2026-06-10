"""Feature engineering utilities."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def drop_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Drop specified columns from a DataFrame, ignoring missing ones.

    Args:
        df: Input DataFrame.
        columns: Column names to remove.

    Returns:
        DataFrame without the specified columns.
    """
    existing = [c for c in columns if c in df.columns]
    if existing:
        logger.info("Dropping columns: %s", existing)
    return df.drop(columns=existing)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot-encode all object/category columns in a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with categorical columns replaced by one-hot dummies.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return df
    logger.info("One-hot encoding columns: %s", cat_cols)
    return pd.get_dummies(df, columns=cat_cols, dtype=np.uint8)


def scale_numerics(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardise numeric columns with zero mean and unit variance.

    Args:
        df: Input DataFrame.
        scaler: A pre-fitted ``StandardScaler`` instance, or *None* to create a new one.
        fit: Whether to fit the scaler on *df*. Set to ``False`` when transforming
            test/inference data.

    Returns:
        Tuple of (transformed DataFrame, fitted scaler).
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return df, scaler or StandardScaler()

    if scaler is None:
        scaler = StandardScaler()

    df = df.copy()
    if fit:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    logger.info("Scaled numeric columns: %s", num_cols)
    return df, scaler


def build_features(
    df: pd.DataFrame,
    drop_cols: Sequence[str] | None = None,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Run the full feature-engineering pipeline.

    Args:
        df: Raw or pre-processed DataFrame (without target column).
        drop_cols: Columns to remove before engineering.
        scaler: Optional pre-fitted scaler (for inference).
        fit_scaler: Whether to fit a new scaler on *df*.

    Returns:
        Tuple of (feature DataFrame, fitted StandardScaler).
    """
    if drop_cols:
        df = drop_columns(df, drop_cols)
    df = encode_categoricals(df)
    df, scaler = scale_numerics(df, scaler=scaler, fit=fit_scaler)
    return df, scaler


def main() -> None:
    """Entry-point for the feature-engineering stage."""
    import yaml

    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    processed_path = cfg["data"]["processed_path"]
    drop_cols = cfg["features"].get("drop_columns", [])
    target_col = cfg["features"]["target_column"]

    df = pd.read_parquet(processed_path)
    X = df.drop(columns=[target_col])
    X_feat, _ = build_features(X, drop_cols=drop_cols)
    logger.info("Feature matrix shape: %s", X_feat.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
