# Project Architecture & Deployment Summary

This document provides a high-level overview of the HR AI MLOps project structure, deployment options, and architecture.

## Project Overview

**hr-mlops-databricks** is a config-driven, reusable Python package that:
- Reads curated HR data from Databricks Gold layer
- Executes 7 ML/statistical use cases in parallel
- Writes prediction tables back to Databricks
- Integrates with MLflow for experiment tracking
- Supports Power BI semantic models for executive dashboards

**Use Cases** (all configurable):
1. Employee Attrition Risk (30/60/90 day windows)
2. Workforce Demand Forecasting (monthly headcount trend)
3. Flight Risk Prediction for critical talent
4. Internal Mobility Recommendations
5. DEI Trend Forecasting
6. Compensation Anomaly Detection
7. Workforce Health Score (composite index by business unit)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Databricks Workspace                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                                          │
│  │   Gold Layer     │                                          │
│  ├──────────────────┤                                          │
│  │ fact_workforce   │  ◄────── Workday Extract               │
│  │ fact_compensation│         (nightly)                       │
│  │ dim_employee     │                                          │
│  │ dim_job          │                                          │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  HR MLOps Pipeline (Python Databricks Job)              │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │                                                          │ │
│  │  Config: YAML (column mappings, feature sets, weights)  │ │
│  │    │                                                     │ │
│  │    ├─► Load Gold Data (SQL or single table)             │ │
│  │    │                                                     │ │
│  │    ├─► Feature Engineering (7 use cases parallel)       │ │
│  │    │   ├─ Attrition Risk                                │ │
│  │    │   ├─ Workforce Demand                              │ │
│  │    │   ├─ Flight Risk                                   │ │
│  │    │   ├─ Internal Mobility                             │ │
│  │    │   ├─ DEI Forecast                                  │ │
│  │    │   ├─ Comp Anomaly                                  │ │
│  │    │   └─ Workforce Health Score                        │ │
│  │    │                                                     │ │
│  │    ├─► MLflow Tracking (optional)                       │ │
│  │    │                                                     │ │
│  │    └─► Write Delta Tables                               │ │
│  │                                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │   MLOps Schema (Output Tables)                          │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ pred_attrition_risk                                     │ │
│  │ pred_workforce_demand                                   │ │
│  │ pred_flight_risk                                        │ │
│  │ pred_internal_mobility                                  │ │
│  │ pred_dei_forecast                                       │ │
│  │ pred_comp_anomaly                                       │ │
│  │ pred_workforce_health                                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Power BI Semantic Model                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Executive Dashboards:                                         │
│  - Attrition Risk Heatmaps                                     │
│  - Workforce Health Scorecards                                 │
│  - Demand Forecasting Charts                                   │
│  - Talent Mobility Pathways                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Deployment Options

### Option 1: Quick Start (5 minutes) - Recommended for Testing

```
QUICKSTART.md
  ├─ Upload code to Databricks Repos
  ├─ Configure Gold table reference
  ├─ Create simple job via UI or Bundle
  └─ Run pipeline
```

**When to use**: First-time setup, testing, proof-of-concept

### Option 2: Full Production Deployment (30 minutes)

```
DEPLOYMENT.md
  ├─ Step 1: Prepare Gold Layer (create schemas, load data)
  ├─ Step 2: Set up Output Schema
  ├─ Step 3: Deploy via Databricks Asset Bundle (recommended)
  │   └─ Or: Deploy via Databricks Repos + Manual Job
  ├─ Step 4: Configure MLflow Tracking (optional)
  ├─ Step 5: Run and Monitor Pipeline
  ├─ Step 6: Connect Power BI Semantic Model
  ├─ Step 7: Schedule Daily Job (e.g., 02:00 UTC)
  ├─ Step 8: Configure Alerts and Access Controls
  └─ Step 9: Hardening (validation, partitioning, policies)
```

**When to use**: Production environments, large teams, multi-workspace setups

## Configuration Profiles

Four config templates provided for different scenarios:

| File | Use Case | Complexity | Best For |
|------|----------|-----------|----------|
| `base_config.yaml` | Generic template with logical mappings | Low | Learning, prototyping |
| `base_config_powerbi_model.yaml` | Your workbook with business names | Medium | Testing with workbook |
| `base_config_workbook_strict.yaml` | Exact workbook field names | Medium | Production (single table) |
| `base_config_multitable_join.yaml` | Multi-table Gold join with SQL | High | Production (optimal) |

**Recommendation**: Use `base_config_multitable_join.yaml` in production for:
- Guaranteed field availability across facts and dimensions
- Optimized joins with proper cardinality handling
- Future extensibility with additional dimensions

## File Structure

```
AI-MLOps/
├── README.md                             # Main documentation
├── QUICKSTART.md                         # 5-minute setup guide
├── DEPLOYMENT.md                         # Full deployment guide
├── PROJECT_ARCHITECTURE.md               # This file
│
├── pyproject.toml                        # Python package metadata
├── requirements.txt                      # Runtime dependencies
├── databricks.yml                        # Databricks Asset Bundle config
│
├── configs/
│   ├── base_config.yaml                 # Default template
│   ├── base_config_powerbi_model.yaml   # Workbook-friendly
│   ├── base_config_workbook_strict.yaml # Workbook-strict mapping
│   └── base_config_multitable_join.yaml # Multi-table join (recommended)
│
├── src/hr_mlops/
│   ├── __init__.py
│   ├── config.py                        # Config loader
│   ├── session.py                       # Spark session factory
│   ├── mlflow_tracking.py               # MLflow integration (optional)
│   │
│   ├── io/
│   │   ├── readers.py                   # Read from Gold (table or SQL)
│   │   └── writers.py                   # Write to Delta
│   │
│   ├── models/
│   │   └── use_cases.py                 # All 7 ML/stat use cases
│   │
│   ├── pipelines/
│   │   └── orchestrator.py              # Pipeline runner
│   │
│   └── utils/
│       └── common.py                    # Shared utilities
│
├── jobs/
│   └── run_databricks_job.py            # CLI entrypoint
│
├── tests/
│   ├── conftest.py                      # Pytest setup
│   └── test_config_loading.py           # Sample test
│
└── dist/
    ├── hr_mlops_databricks-0.1.0-py3-none-any.whl
    └── hr_mlops_databricks-0.1.0.tar.gz
```

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Pipeline language |
| PySpark | 3.5+ | Distributed data processing |
| PyYAML | 6.0+ | Configuration management |
| MLflow | 2.14+ | Experiment tracking (optional) |
| Databricks | 15.4+ | Execution environment |
| Delta Lake | Native | Storage format |

## Key Features

### ✅ Configuration-Driven
- All column mappings, feature sets, and weights in YAML
- No code changes needed for different source systems or models
- Easy to extend for new use cases

### ✅ Reusable Patterns
- Applies to any HR payroll/HCM source (Workday, SAP SuccessFactors, etc.)
- Flexible feature engineering functions
- Pluggable MLflow tracking

### ✅ Production Ready
- Input validation hooks
- MLflow run logging
- Graceful error handling
- Configurable output modes (overwrite, append, merge)

### ✅ Extensible
- Add new models by extending `use_cases.py`
- Add new data sources by overriding `readers.py`
- Add custom transformations in pipeline orchestration

## Run Commands by Scenario

### Local Testing (with Spark)
```bash
python -m jobs.run_databricks_job --config configs/base_config.yaml
```

### Databricks CLI (Repos)
```bash
databricks jobs run-now --job-id <job-id>
```

### Databricks Bundle Deploy
```bash
databricks bundle deploy
databricks bundle run hr_ai_mlops_pipeline
```

### Scheduled (Daily at 02:00 UTC)
Set in Databricks Job:
- Frequency: Daily
- Time: 02:00 UTC
- Cluster: 2 workers, Spark 15.4+

## Monitoring & Troubleshooting

### MLflow Experiment Tracking
- **Workspace Path**: `/Shared/hr-ai-mlops`
- **Logged Metrics**: Row counts per output table
- **Logged Params**: Source table, write mode, config path

### Output Validation
```sql
-- Check table row counts
SELECT 
  'attrition' as model, COUNT(*) as rows 
FROM hr_mlops.pred_attrition_risk
UNION ALL
SELECT 
  'workforce_health', COUNT(*) 
FROM hr_mlops.pred_workforce_health;
```

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Table not found | Wrong catalog/schema | Update config `source.sql_query` |
| Missing columns | Schema mismatch | Run `DESCRIBE TABLE` to validate |
| MLflow context inactive | MLflow not initialized | Set `mlflow.enabled: false` |
| Timeout on large data | Single-node cluster | Increase workers in cluster config |

## Future Enhancements

- [ ] Advanced ML models (XGBoost, LightGBM with hyperparameter tuning)
- [ ] Time-series cross validation for demand forecasting
- [ ] Explainability (SHAP values) for attrition/flight risk
- [ ] Real-time endpoint serving for mobile/web apps
- [ ] Data quality monitoring with Great Expectations
- [ ] Auto-generated Power BI reports via API

## Support

- **Quick help**: [QUICKSTART.md](QUICKSTART.md)
- **Full guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **API docs**: [README.md](README.md)
- **Code**: [src/hr_mlops/models/use_cases.py](src/hr_mlops/models/use_cases.py)

---

**Version**: 0.1.0  
**Last Updated**: June 2026  
**Status**: Production Ready
