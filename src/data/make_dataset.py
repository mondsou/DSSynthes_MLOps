"""Data ingestion and pre-processing utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw data from a CSV or Parquet file.

    Args:
        path: Path to the raw data file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    logger.info("Loading raw data from %s", path)

    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: '{path.suffix}'. Use .csv or .parquet.")


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply basic cleaning steps to a DataFrame.

    Steps:
    - Drop fully-null rows.
    - Optionally drop duplicate rows.

    Args:
        df: Input DataFrame.
        drop_duplicates: Whether to remove duplicate rows.

    Returns:
        Cleaned DataFrame.
    """
    before = len(df)
    df = df.dropna(how="all")
    if drop_duplicates:
        df = df.drop_duplicates()
    logger.info("Cleaned data: %d → %d rows", before, len(df))
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train / test feature and target sets.

    Args:
        df: Full dataset including the target column.
        target_column: Name of the target (label) column.
        test_size: Fraction of samples to reserve for the test set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split data: train=%d, test=%d",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to a Parquet file, creating parent directories if needed.

    Args:
        df: DataFrame to persist.
        path: Destination path (must end in .parquet).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved processed data to %s", path)


def main() -> None:
    """Entry-point for the data-preparation stage."""
    import yaml

    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    raw_path = cfg["data"]["raw_path"]
    processed_path = cfg["data"]["processed_path"]

    df = load_raw_data(raw_path)
    df = clean_data(df)
    save_processed_data(df, processed_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
