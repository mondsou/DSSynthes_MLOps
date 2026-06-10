from __future__ import annotations

import argparse

from hr_mlops.config import load_config
from hr_mlops.pipelines.orchestrator import run_pipeline
from hr_mlops.session import get_spark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HR AI MLOps pipeline on Databricks")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spark = get_spark(app_name=config.get("app_name", "hr-ai-mlops"))
    run_pipeline(spark, config)


if __name__ == "__main__":
    main()
