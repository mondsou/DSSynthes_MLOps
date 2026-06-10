"""End-to-end training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.data.make_dataset import clean_data, load_raw_data, save_processed_data, split_data
from src.features.build_features import build_features
from src.models.train_model import evaluate, save_model, train

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_pipeline(config: dict[str, Any]) -> dict[str, float]:
    """Execute the full training pipeline.

    Steps:
    1. Load and clean raw data.
    2. Save processed data.
    3. Split into train / test sets.
    4. Engineer features.
    5. Train model and log with MLflow.
    6. Evaluate on the hold-out set.
    7. Persist the model artefact.

    Args:
        config: Hydra / YAML config dictionary (see ``configs/config.yaml``).

    Returns:
        Dictionary of evaluation metrics.
    """
    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]

    # --- Data stage ---
    df = load_raw_data(data_cfg["raw_path"])
    df = clean_data(df)
    save_processed_data(df, data_cfg["processed_path"])

    # --- Split ---
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=feat_cfg["target_column"],
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    # --- Feature engineering ---
    drop_cols = feat_cfg.get("drop_columns", [])
    X_train_feat, scaler = build_features(X_train, drop_cols=drop_cols)
    X_test_feat, _ = build_features(
        X_test, drop_cols=drop_cols, scaler=scaler, fit_scaler=False
    )

    # --- Training ---
    model, run_id = train(
        X_train_feat,
        y_train,
        model_name=model_cfg["name"],
        model_params=model_cfg.get("params", {}),
        experiment_name=mlflow_cfg["experiment_name"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )
    logger.info("Completed MLflow run: %s", run_id)

    # --- Evaluation ---
    metrics = evaluate(model, X_test_feat, y_test)

    # --- Persist model ---
    save_model(model, "models/model.joblib")

    return metrics


def main() -> None:
    """Command-line entry-point for the training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the DSSynthes MLOps training pipeline.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = run_pipeline(config)
    logger.info("Pipeline finished. Metrics: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
