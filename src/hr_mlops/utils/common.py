from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, functions as F


def resolve_column(config: dict[str, Any], logical_name: str) -> str:
    return config.get("column_mappings", {}).get(logical_name, logical_name)


def ensure_columns(df: DataFrame, columns: list[str], default_value: float = 0.0) -> DataFrame:
    out = df
    for col_name in columns:
        if col_name not in out.columns:
            out = out.withColumn(col_name, F.lit(default_value))
    return out


def cast_to_double(df: DataFrame, columns: list[str]) -> DataFrame:
    out = df
    for col_name in columns:
        out = out.withColumn(col_name, F.col(col_name).cast("double"))
    return out
