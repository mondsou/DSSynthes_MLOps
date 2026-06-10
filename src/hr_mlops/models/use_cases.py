from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from hr_mlops.utils.common import cast_to_double, ensure_columns, resolve_column


def attrition_prediction(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    cfg = config["use_cases"]["attrition"]
    employee_id = resolve_column(config, "employee_id")

    features = [resolve_column(config, c) for c in cfg["features"]]
    weights = cfg.get("weights", [1.0] * len(features))

    work = ensure_columns(df, features)
    work = cast_to_double(work, features)

    weighted_exprs = [F.coalesce(F.col(c), F.lit(0.0)) * F.lit(w) for c, w in zip(features, weights)]
    score_raw = weighted_exprs[0]
    for expr in weighted_exprs[1:]:
        score_raw = score_raw + expr

    result = (
        work.select(employee_id, *features)
        .withColumn("risk_30_days", F.round(F.least(F.lit(100.0), F.greatest(F.lit(0.0), score_raw * 0.90)), 2))
        .withColumn("risk_60_days", F.round(F.least(F.lit(100.0), F.greatest(F.lit(0.0), score_raw * 1.00)), 2))
        .withColumn("risk_90_days", F.round(F.least(F.lit(100.0), F.greatest(F.lit(0.0), score_raw * 1.10)), 2))
        .withColumn(
            "attrition_risk_bucket",
            F.when(F.col("risk_60_days") >= 75, F.lit("High"))
            .when(F.col("risk_60_days") >= 40, F.lit("Medium"))
            .otherwise(F.lit("Low")),
        )
    )
    return result


def workforce_demand_forecast(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    cfg = config["use_cases"]["workforce_demand"]
    date_col = resolve_column(config, cfg["date_col"])
    headcount_col = resolve_column(config, cfg["headcount_col"])

    work = ensure_columns(df, [headcount_col])
    work = cast_to_double(work, [headcount_col])

    monthly = (
        work.withColumn("forecast_month", F.date_trunc("month", F.col(date_col)))
        .groupBy("forecast_month")
        .agg(F.sum(headcount_col).alias("actual_headcount"))
        .orderBy("forecast_month")
    )

    window = Window.orderBy("forecast_month").rowsBetween(-2, 0)
    growth = cfg.get("growth_factor", 1.01)

    return (
        monthly.withColumn("moving_avg_3m", F.avg("actual_headcount").over(window))
        .withColumn(
            "forecast_headcount",
            F.round(F.coalesce(F.col("moving_avg_3m"), F.col("actual_headcount")) * F.lit(growth), 0),
        )
        .select("forecast_month", "actual_headcount", "forecast_headcount")
    )


def flight_risk_prediction(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    cfg = config["use_cases"]["flight_risk"]
    employee_id = resolve_column(config, "employee_id")

    top_perf = resolve_column(config, cfg["top_performer_col"])
    critical_role = resolve_column(config, cfg["critical_role_col"])
    leadership = resolve_column(config, cfg["leadership_pipeline_col"])
    engagement = resolve_column(config, cfg["engagement_col"])

    cols = [top_perf, critical_role, leadership, engagement]
    work = ensure_columns(df, cols)

    return (
        work.select(employee_id, *cols)
        .withColumn(
            "flight_risk_score",
            F.round(
                F.coalesce(F.col(top_perf).cast("double"), F.lit(0.0)) * 25
                + F.coalesce(F.col(critical_role).cast("double"), F.lit(0.0)) * 25
                + F.coalesce(F.col(leadership).cast("double"), F.lit(0.0)) * 20
                + (100 - F.coalesce(F.col(engagement).cast("double"), F.lit(50.0))) * 0.3,
                2,
            ),
        )
        .withColumn(
            "flight_risk_band",
            F.when(F.col("flight_risk_score") >= 70, F.lit("High"))
            .when(F.col("flight_risk_score") >= 40, F.lit("Medium"))
            .otherwise(F.lit("Low")),
        )
    )


def internal_mobility_recommendation(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    cfg = config["use_cases"]["internal_mobility"]
    employee_id = resolve_column(config, "employee_id")

    current_role = resolve_column(config, cfg["current_role_col"])
    tenure = resolve_column(config, cfg["tenure_col"])
    performance = resolve_column(config, cfg["performance_col"])

    work = ensure_columns(df, [current_role, tenure, performance])
    work = cast_to_double(work, [tenure, performance])

    # Simple, transparent rule-based recommender for initial MLOps deployment.
    return (
        work.select(employee_id, current_role, tenure, performance)
        .withColumn(
            "recommended_role",
            F.when((F.col(performance) >= 4.5) & (F.col(tenure) >= 2), F.concat(F.col(current_role), F.lit(" - Senior")))
            .when((F.col(performance) >= 3.5) & (F.col(tenure) >= 1), F.concat(F.col(current_role), F.lit(" - Lead")))
            .otherwise(F.col(current_role)),
        )
        .withColumn(
            "mobility_confidence",
            F.round(F.least(F.lit(1.0), (F.col(performance) / 5.0) * 0.7 + (F.col(tenure) / 10.0) * 0.3), 3),
        )
    )


def diversity_trend_forecast(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    cfg = config["use_cases"]["dei_forecast"]
    date_col = resolve_column(config, cfg["date_col"])
    diversity_ratio_col = resolve_column(config, cfg["diversity_ratio_col"])

    work = ensure_columns(df, [diversity_ratio_col], default_value=0.0)
    work = cast_to_double(work, [diversity_ratio_col])

    monthly = (
        work.withColumn("forecast_month", F.date_trunc("month", F.col(date_col)))
        .groupBy("forecast_month")
        .agg(F.avg(diversity_ratio_col).alias("actual_diversity_ratio"))
        .orderBy("forecast_month")
    )

    window = Window.orderBy("forecast_month").rowsBetween(-2, 0)
    improvement = cfg.get("improvement_rate", 0.01)

    return (
        monthly.withColumn("rolling_ratio", F.avg("actual_diversity_ratio").over(window))
        .withColumn(
            "forecast_diversity_ratio",
            F.round(F.coalesce(F.col("rolling_ratio"), F.col("actual_diversity_ratio")) * F.lit(1 + improvement), 4),
        )
        .select("forecast_month", "actual_diversity_ratio", "forecast_diversity_ratio")
    )


def compensation_anomaly_detection(df: DataFrame, config: dict[str, Any]) -> DataFrame:
    cfg = config["use_cases"]["comp_anomaly"]
    employee_id = resolve_column(config, "employee_id")
    comp_col = resolve_column(config, cfg["compensation_col"])
    band_col = resolve_column(config, cfg["pay_band_col"])

    work = ensure_columns(df, [comp_col, band_col])
    work = cast_to_double(work, [comp_col])

    stats = work.groupBy(band_col).agg(
        F.avg(comp_col).alias("band_avg_comp"),
        F.stddev_pop(comp_col).alias("band_std_comp"),
    )

    joined = work.join(stats, on=band_col, how="left")

    return (
        joined.select(employee_id, band_col, comp_col, "band_avg_comp", "band_std_comp")
        .withColumn(
            "z_score",
            F.when(F.col("band_std_comp") > 0, (F.col(comp_col) - F.col("band_avg_comp")) / F.col("band_std_comp")).otherwise(F.lit(0.0)),
        )
        .withColumn("anomaly_score", F.round(F.abs(F.col("z_score")), 3))
        .withColumn(
            "anomaly_flag",
            F.when(F.col("anomaly_score") >= cfg.get("high_threshold", 2.5), F.lit("High"))
            .when(F.col("anomaly_score") >= cfg.get("medium_threshold", 1.5), F.lit("Medium"))
            .otherwise(F.lit("Low")),
        )
    )


def workforce_health_score(
    attrition_df: DataFrame,
    demand_df: DataFrame,
    flight_df: DataFrame,
    mobility_df: DataFrame,
    dei_df: DataFrame,
    comp_df: DataFrame,
    base_df: DataFrame,
    config: dict[str, Any],
) -> DataFrame:
    cfg = config["use_cases"]["workforce_health_score"]
    bu_col = resolve_column(config, cfg["business_unit_col"])

    absenteeism_col = resolve_column(config, cfg["absenteeism_col"])
    engagement_col = resolve_column(config, cfg["engagement_col"])
    hiring_velocity_col = resolve_column(config, cfg["hiring_velocity_col"])

    base = ensure_columns(base_df, [bu_col, absenteeism_col, engagement_col, hiring_velocity_col])
    base = cast_to_double(base, [absenteeism_col, engagement_col, hiring_velocity_col])

    base_agg = base.groupBy(bu_col).agg(
        F.avg(absenteeism_col).alias("avg_absenteeism"),
        F.avg(engagement_col).alias("avg_engagement"),
        F.avg(hiring_velocity_col).alias("avg_hiring_velocity"),
    )

    attrition_bu = attrition_df.join(base.select(resolve_column(config, "employee_id"), bu_col), on=resolve_column(config, "employee_id"), how="left")
    attrition_agg = attrition_bu.groupBy(bu_col).agg(F.avg("risk_60_days").alias("avg_attrition_risk"))

    flight_bu = flight_df.join(base.select(resolve_column(config, "employee_id"), bu_col), on=resolve_column(config, "employee_id"), how="left")
    flight_agg = flight_bu.groupBy(bu_col).agg(F.avg("flight_risk_score").alias("avg_flight_risk"))

    mobility_bu = mobility_df.join(base.select(resolve_column(config, "employee_id"), bu_col), on=resolve_column(config, "employee_id"), how="left")
    mobility_agg = mobility_bu.groupBy(bu_col).agg(F.avg("mobility_confidence").alias("avg_mobility_conf"))

    comp_bu = comp_df.join(base.select(resolve_column(config, "employee_id"), bu_col), on=resolve_column(config, "employee_id"), how="left")
    comp_agg = comp_bu.groupBy(bu_col).agg(F.avg("anomaly_score").alias("avg_comp_anomaly"))

    dei_latest = dei_df.agg(F.max("forecast_diversity_ratio").alias("dei_ratio"))
    demand_latest = demand_df.agg(F.max("forecast_headcount").alias("forecast_hc"))

    joined = (
        base_agg.join(attrition_agg, on=bu_col, how="left")
        .join(flight_agg, on=bu_col, how="left")
        .join(mobility_agg, on=bu_col, how="left")
        .join(comp_agg, on=bu_col, how="left")
        .crossJoin(dei_latest)
        .crossJoin(demand_latest)
    )

    weights = cfg.get(
        "weights",
        {
            "attrition": 0.25,
            "hiring_velocity": 0.15,
            "engagement": 0.15,
            "diversity": 0.15,
            "absenteeism": 0.10,
            "mobility": 0.10,
            "flight_risk": 0.05,
            "comp_anomaly": 0.05,
        },
    )

    return (
        joined.withColumn(
            "workforce_health_score",
            F.round(
                (100 - F.coalesce(F.col("avg_attrition_risk"), F.lit(50.0))) * F.lit(weights["attrition"])
                + F.least(F.lit(100.0), F.coalesce(F.col("avg_hiring_velocity"), F.lit(50.0))) * F.lit(weights["hiring_velocity"])
                + F.least(F.lit(100.0), F.coalesce(F.col("avg_engagement"), F.lit(50.0))) * F.lit(weights["engagement"])
                + F.least(F.lit(100.0), F.coalesce(F.col("dei_ratio"), F.lit(0.5)) * 100) * F.lit(weights["diversity"])
                + (100 - F.least(F.lit(100.0), F.coalesce(F.col("avg_absenteeism"), F.lit(10.0)) * 10)) * F.lit(weights["absenteeism"])
                + F.least(F.lit(100.0), F.coalesce(F.col("avg_mobility_conf"), F.lit(0.5)) * 100) * F.lit(weights["mobility"])
                + (100 - F.least(F.lit(100.0), F.coalesce(F.col("avg_flight_risk"), F.lit(40.0)))) * F.lit(weights["flight_risk"])
                + (100 - F.least(F.lit(100.0), F.coalesce(F.col("avg_comp_anomaly"), F.lit(1.0)) * 20)) * F.lit(weights["comp_anomaly"]),
                2,
            ),
        )
        .select(bu_col, "workforce_health_score", "avg_attrition_risk", "avg_engagement", "avg_absenteeism", "dei_ratio", "forecast_hc")
    )
