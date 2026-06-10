from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "hr-ai-mlops") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()
