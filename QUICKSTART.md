# Quick Start - 5-Minute Databricks Deployment

This guide gets you running in Databricks in minimal steps. For full details, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Prerequisites

- Databricks workspace with contributor access
- Databricks CLI installed
- HR Gold table in Databricks (example: `hr_gold.fact_workforce_snapshot`)

## Step 1: Upload Code to Databricks (1 minute)

```bash
# Authenticate
databricks auth login

# Create repo in Databricks
databricks repos create \
  --url https://github.com/<your-org>/AI-MLOps \
  --provider GITHUB \
  --path /Repos/<your-username>/AI-MLOps
```

Or upload the wheel directly:

```bash
# Upload to Databricks Files
databricks fs cp dist/hr_mlops_databricks-0.1.0-py3-none-any.whl \
  dbfs:/FileStore/hr-mlops/hr_mlops_databricks-0.1.0-py3-none-any.whl
```

## Step 2: Configure Your Gold Table (1 minute)

Edit `configs/base_config_multitable_join.yaml` and update the SQL query to match your Gold table names:

```yaml
source:
  sql_query: |
    SELECT
      w.`Snapshot_Date` AS snapshot_date,
      w.`Employee_ID` AS employee_id,
      -- Update these to match your actual column names
      CAST(w.`Attr_tenure` AS DOUBLE) AS tenure_months,
      -- ... rest of columns
    FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.fact_workforce_snapshot w
```

## Step 3: Create Databricks Job (2 minutes)

### Option A: Using Bundle (Fastest)

```bash
cd c:\Retirement_Solution\AI-MLOps

# Update databricks.yml with your config path
# Then deploy:
databricks bundle deploy
databricks bundle run hr_ai_mlops_pipeline
```

### Option B: Using UI

1. Open Databricks Workspace
2. **Jobs → Create Job**
3. **Task details**:
   - Name: `run_hr_ai_mlops`
   - Type: `Python file`
   - Path: `/Repos/<your-user>/AI-MLOps/jobs/run_databricks_job.py`
   - Parameters:
     ```
     --config /Repos/<your-user>/AI-MLOps/configs/base_config_multitable_join.yaml
     ```
4. **Cluster**: Create new
   - Type: Single node or 2 workers
   - Spark: 15.4+
   - Python: 3.10+
5. **Libraries**: Add via PyPI
   - `pyspark>=3.5.0`
   - `PyYAML>=6.0.1`
   - `mlflow>=2.14.0`
6. **Save**

## Step 4: Run Pipeline (1 minute)

```bash
# Run immediately
databricks jobs run-now --job-id <your-job-id>

# Or in UI: Jobs → hr_ai_mlops_pipeline → Run now
```

## Step 5: Verify Output (30 seconds)

In Databricks SQL:

```sql
SELECT * FROM hr_mlops.pred_attrition_risk LIMIT 100;
SELECT * FROM hr_mlops.pred_workforce_health LIMIT 100;
```

## Done! 🎉

Your pipeline is now running and producing prediction tables. These are ready for Power BI semantic models.

---

## Next Steps

- **Full deployment details**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Schedule recurring runs**: Set job schedule to daily at 02:00 UTC
- **Connect Power BI**: Use Databricks connector to query `hr_mlops.*` tables
- **Configure MLflow tracking**: Set `mlflow.enabled: true` in config
- **Customize models**: Edit feature weights in [configs/](configs/)

## Troubleshooting

**Job fails with "Table not found"**
- Verify Gold table name in your SQL query matches actual Databricks table
- Run: `SHOW TABLES IN hr_gold;`

**"Module hr_mlops not found"**
- Ensure Databricks repo path or wheel installation is correct
- Check log: `databricks jobs get-run --run-id <run-id>`

**No output tables created**
- Check `hr_mlops` schema exists: `CREATE SCHEMA IF NOT EXISTS hr_mlops;`
- Verify MLflow is disabled if not set up: `mlflow.enabled: false`

For more help, see [DEPLOYMENT.md](DEPLOYMENT.md#step-8-monitor-and-troubleshoot).
