from __future__ import annotations

from typing import Any

from hr_mlops.io.readers import load_gold_hr_data
from hr_mlops.io.writers import write_delta_table
from hr_mlops.mlflow_tracking import log_pipeline_metrics, pipeline_run
from hr_mlops.models.use_cases import (
    attrition_prediction,
    compensation_anomaly_detection,
    diversity_trend_forecast,
    flight_risk_prediction,
    internal_mobility_recommendation,
    workforce_demand_forecast,
    workforce_health_score,
)


def run_pipeline(spark, config: dict[str, Any]) -> dict[str, float]:
    source_df = load_gold_hr_data(spark, config)

    with pipeline_run(config):
        attrition_df = attrition_prediction(source_df, config)
        demand_df = workforce_demand_forecast(source_df, config)
        flight_df = flight_risk_prediction(source_df, config)
        mobility_df = internal_mobility_recommendation(source_df, config)
        dei_df = diversity_trend_forecast(source_df, config)
        comp_df = compensation_anomaly_detection(source_df, config)
        health_df = workforce_health_score(
            attrition_df=attrition_df,
            demand_df=demand_df,
            flight_df=flight_df,
            mobility_df=mobility_df,
            dei_df=dei_df,
            comp_df=comp_df,
            base_df=source_df,
            config=config,
        )

        outputs = config["outputs"]
        write_mode = config.get("write_mode", "overwrite")

        write_delta_table(attrition_df, outputs["attrition_table"], write_mode)
        write_delta_table(demand_df, outputs["demand_forecast_table"], write_mode)
        write_delta_table(flight_df, outputs["flight_risk_table"], write_mode)
        write_delta_table(mobility_df, outputs["internal_mobility_table"], write_mode)
        write_delta_table(dei_df, outputs["dei_forecast_table"], write_mode)
        write_delta_table(comp_df, outputs["comp_anomaly_table"], write_mode)
        write_delta_table(health_df, outputs["workforce_health_table"], write_mode)

        metrics = {
            "source_row_count": float(source_df.count()),
            "attrition_row_count": float(attrition_df.count()),
            "demand_forecast_row_count": float(demand_df.count()),
            "flight_risk_row_count": float(flight_df.count()),
            "internal_mobility_row_count": float(mobility_df.count()),
            "dei_forecast_row_count": float(dei_df.count()),
            "comp_anomaly_row_count": float(comp_df.count()),
            "workforce_health_row_count": float(health_df.count()),
        }
        log_pipeline_metrics(config, metrics)

    return metrics
