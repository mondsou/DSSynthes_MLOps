from __future__ import annotations

from pyspark.sql import DataFrame


def write_delta_table(df: DataFrame, target_table: str, mode: str = "overwrite") -> None:
    (
        df.write.format("delta")
        .mode(mode)
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )
