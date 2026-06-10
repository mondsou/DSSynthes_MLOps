from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _try_import_mlflow():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def _mlflow_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("mlflow", {})


@contextmanager
def pipeline_run(config: dict[str, Any]) -> Iterator[None]:
    mlflow = _try_import_mlflow()
    cfg = _mlflow_cfg(config)
    enabled = cfg.get("enabled", True)

    if not enabled or mlflow is None:
        yield
        return

    experiment = cfg.get("experiment")
    run_name = cfg.get("run_name", "hr-ai-mlops")
    tags = cfg.get("tags", {})

    if experiment:
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_param("source_gold_table", config.get("source", {}).get("gold_hr_table", "unknown"))
        mlflow.log_param("write_mode", config.get("write_mode", "overwrite"))
        yield


def log_pipeline_metrics(config: dict[str, Any], metrics: dict[str, float]) -> None:
    mlflow = _try_import_mlflow()
    cfg = _mlflow_cfg(config)
    enabled = cfg.get("enabled", True)

    if not enabled or mlflow is None:
        return

    mlflow.log_metrics(metrics)
