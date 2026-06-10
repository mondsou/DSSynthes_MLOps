from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession


def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    return spark.table(table_name)


def load_gold_hr_data(spark: SparkSession, config: dict[str, Any]) -> DataFrame:
    source_cfg = config["source"]
    sql_query = source_cfg.get("sql_query")

    if sql_query:
        return spark.sql(sql_query)

    table_name = source_cfg["gold_hr_table"]
    return read_table(spark, table_name)
