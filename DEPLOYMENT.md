# HR AI MLOps Deployment Guide - Databricks

This guide walks you through deploying the `hr-mlops-databricks` package end-to-end in Databricks.

## Prerequisites

1. **Databricks Workspace**: Access to a Databricks workspace (at least contributor role)
2. **Databricks CLI**: Installed and configured on your local machine
3. **Python 3.10+**: On your local development machine
4. **Build Artifacts**: Wheel and source distributions from `dist/` folder
5. **Gold Layer Tables**: HR data curated in Databricks (see below)

## Step 1: Prepare Gold Layer Data in Databricks

### 1.1 Create the Required Catalog and Schema

In your Databricks workspace, run these SQL commands:

```sql
-- Create or use existing catalog
CREATE CATALOG IF NOT EXISTS hr_gold;

-- Create or use existing schema
CREATE SCHEMA IF NOT EXISTS hr_gold.default;

--  Ensure appropriate permissions
GRANT CREATE TABLE, CREATE EXTERNAL TABLE, MODIFY, SELECT ON SCHEMA hr_gold.default TO `<your-group>`;
```

### 1.2 Load Source Data (example for fact_workforce_snapshot)

Replace `<source>` with your actual data source (Workday export, CSV upload, or existing Delta table):

```sql
-- Create fact_workforce_snapshot from curated source
CREATE OR REPLACE TABLE hr_gold.fact_workforce_snapshot AS
SELECT
  -- Required columns for MLOps pipeline (map to your actual columns)
  `Snapshot_Date`,
  `Employee_ID`,
  `Attr_tenure`,
  `Attr_change_in_inflation`,
  `Attr_prmtn`,
  `Attr_rating_high`,
  `Attr_mgr_change`,
  `Attr_ltral`,
  `HC_Total_Days_On_Leave`,
  `Attr_OV_Engagement`,
  `Location_Description`,
  `Employee_Function`,
  `Potential`,
  `Eligibility_for_Scorecard_Flag`,
  `Attr_OV_Talent_Development`,
  `Position_Title`,
  `FTE_Factor`,
  `Attr_OV_Inclusion`,
  `Wage`,
  `GS`,
  `HR_Sector`,
  -- ... all other required columns from workbook model
FROM <source-table-or-external-location>
WHERE `Snapshot_Date` >= DATE_SUB(CURRENT_DATE(), 12*30);  -- Keep 12 months of history
```

### 1.3 Optional: Create Companion Fact/Dimension Tables (for multitable_join config)

```sql
-- fact_compensation_snapshot
CREATE OR REPLACE TABLE hr_gold.fact_compensation_snapshot AS
SELECT
  `Snapshot_Date`,
  `Employee_ID`,
  `Job_Level_Pay_Code`,
  `Base_Pay_Amount`,
  `Supervisory_Current_Position_Pay_Grade`,
  -- ... other required fields
FROM <compensation-source>;

-- dim_employee
CREATE OR REPLACE TABLE hr_gold.dim_employee AS
SELECT
  `Employee Id`,
  `Snapshot Date`,
  `Email Address`,
  `First Name`,
  `Last Name`,
  `Gender`,
  `Ethnicity`,
  -- ... other fields
FROM <employee-dimension>;

-- dim_job
CREATE OR REPLACE TABLE hr_gold.dim_job AS
SELECT
  `ID`,
  `Job_Profile`,
  `Job_Function`,
  `Job_Family`,
  `Compensation_Grade`,
  -- ... other fields
FROM <job-dimension>;
```

## Step 2: Set Up Output Schema

Create the output schema where prediction tables will be written:

```sql
CREATE SCHEMA IF NOT EXISTS hr_mlops.default;

GRANT CREATE TABLE, MODIFY ON SCHEMA hr_mlops.default TO `<your-group>`;
```

## Step 3: Deploy Using Databricks Asset Bundle (Recommended)

### 3.1 Update databricks.yml

Edit [databricks.yml](databricks.yml) to match your workspace:

```yaml
# Update these values:
resources:
  jobs:
    hr_ai_mlops_pipeline:
      name: hr_ai_mlops_pipeline
      tasks:
        - task_key: run_hr_ai_mlops
          spark_python_task:
            python_file: jobs/run_databricks_job.py
            parameters:
              - --config
              - /Workspace/Repos/<user>/AI-MLOps/configs/base_config_multitable_join.yaml
          new_cluster:
            spark_version: 15.4.x-scala2.12
            node_type_id: Standard_DS3_v2  # Adjust for your region/quota
            num_workers: 2
            aws_attributes:
              availability: SPOT_WITH_FALLBACK  # or ON_DEMAND
            instance_pool_id: <your-pool-id>  # Optional
          max_retries: 1
          timeout_seconds: 3600
```

### 3.2 Deploy the Bundle

```bash
# Navigate to project root
cd c:\Retirement_Solution\AI-MLOps

# Authenticate with Databricks
databricks auth login

# Deploy the bundle
databricks bundle deploy

# Run the pipeline
databricks bundle run hr_ai_mlops_pipeline
```

## Step 4: Deploy Using Databricks Repos (Alternative)

If you prefer Databricks Repos without Bundle:

### 4.1 Clone Repository to Databricks

```bash
# From your local machine, push to GitHub or use direct repo clone:
databricks repos create --url https://github.com/<your-org>/AI-MLOps --provider GITHUB --path /Repos/<user>/AI-MLOps
```

### 4.2 Create Job Manually in UI

1. **Databricks Workspace → Jobs → Create Job**
2. **Task Details**:
   - Task name: `run_hr_ai_mlops`
   - Type: `Python file`
   - Python file path: `/Repos/<user>/AI-MLOps/jobs/run_databricks_job.py`
   - Parameters:
     ```
     --config
     /Repos/<user>/AI-MLOps/configs/base_config_multitable_join.yaml
     ```

3. **Cluster**:
   - Single node or multi-node cluster
   - Spark version: 15.4 or later
   - Python version: 3.10+
   - Libraries:
     - PyPI: `pyspark>=3.5.0`
     - PyPI: `PyYAML>=6.0.1`
     - PyPI: `mlflow>=2.14.0`

4. **Schedule** (optional):
   - Daily at 02:00 UTC (after Workday data refresh)
   - Timezone: UTC

5. **Save and Run**

## Step 5: Configure Secrets for Databricks

### 5.1 Optional: Store Config in Databricks Secrets

If you want to reference config via secrets instead of workspace paths:

```bash
# Create secret scope
databricks secrets create-scope --scope hr-mlops

# Store config as secret
databricks secrets put --scope hr-mlops --key config-multitable --file configs/base_config_multitable_join.yaml
```

### 5.2 Update jobs/run_databricks_job.py to Use Secrets

```python
import dbutils

# Read config from secrets
config_yaml = dbutils.secrets.get(scope="hr-mlops", key="config-multitable")
config = yaml.safe_load(config_yaml)
```

## Step 6: Configure MLflow Tracking (Optional)

### 6.1 Enable MLflow in Experiment

Update your config to point to your workspace experiment:

```yaml
mlflow:
  enabled: true
  experiment: /Shared/hr-ai-mlops
  run_name: hr-ai-mlops-daily
  tags:
    domain: hr
    source_layer: gold
    environment: production
```

### 6.2 View Runs

In Databricks UI: **Experiments > /Shared/hr-ai-mlops**

## Step 7: Run the Pipeline

### Option A: CLI

```bash
databricks jobs run-now --job-id <job-id>
```

### Option B: Databricks UI

1. Navigate to **Jobs**
2. Click **hr_ai_mlops_pipeline**
3. Click **Run Now**

### Option C: Bundle

```bash
databricks bundle run hr_ai_mlops_pipeline
```

## Step 8: Monitor and Troubleshoot

### 8.1 View Job Run Logs

```bash
databricks jobs get-run --run-id <run-id>
```

### 8.2 Check Output Tables

```sql
-- Verify prediction tables were created
SELECT * FROM hr_mlops.pred_attrition_risk LIMIT 10;
SELECT * FROM hr_mlops.pred_workforce_demand LIMIT 10;
SELECT * FROM hr_mlops.pred_flight_risk LIMIT 10;
SELECT * FROM hr_mlops.pred_internal_mobility LIMIT 10;
SELECT * FROM hr_mlops.pred_dei_forecast LIMIT 10;
SELECT * FROM hr_mlops.pred_comp_anomaly LIMIT 10;
SELECT * FROM hr_mlops.pred_workforce_health LIMIT 10;
```

### 8.3 Common Issues and Fixes

**Issue**: `ModuleNotFoundError: No module named 'hr_mlops'`

**Fix**: Ensure the job's Python file path is correct and the package is in the repo:
```bash
databricks workspace list /Repos/<user>/AI-MLOps
```

---

**Issue**: `AnalysisException: Table not found: hr_gold.fact_workforce_snapshot`

**Fix**: Verify Gold table exists and matches config:
```sql
SHOW TABLES IN hr_gold;
```

---

**Issue**: `MLflow run context is not active`

**Fix**: If MLflow is failing, set `mlflow.enabled: false` in config:
```yaml
mlflow:
  enabled: false
```

### 8.4 View Metrics and Logs in MLflow

1. **Databricks UI → Experiments**
2. **Select `/Shared/hr-ai-mlops`**
3. **Click on latest run**
4. **View params and metrics**

## Step 9: Connect to Power BI (Post-Pipeline)

After the pipeline runs successfully:

### 9.1 Create Power BI Semantic Model

In Power BI Desktop:

1. Get Data → Databricks
2. Connect to your Databricks workspace
3. Query prediction tables:
   - `hr_mlops.pred_attrition_risk`
   - `hr_mlops.pred_workforce_health`
   - etc.

### 9.2 Build Dashboards

Create visuals for:
- **Attrition Risk**: Scatter plot (employee_id vs risk_60_days)
- **Workforce Health**: Scorecard (business_unit vs workforce_health_score)
- **Demand Forecast**: Line chart (forecast_month vs forecast_headcount)
- **Flight Risk**: Table with High/Medium/Low bands

## Step 10: Schedule Recurring Runs

Ensure the job runs on a cadence aligned with HR data refresh:

### 10.1 Set Job Schedule

In Databricks UI: **Jobs → hr_ai_mlops_pipeline → Edit → Schedule**

- **Frequency**: Daily
- **Time**: 02:00 UTC (after Workday nightly extract typically completes around 01:00)
- **Timezone**: UTC

### 10.2 Configure Alerts (Optional)

In job settings:
- Email on failure: `<hr-analytics@company.com>`
- Email on success (optional)

## Step 11: Production Hardening

### 11.1 Add Input Validation

Extend [src/hr_mlops/pipelines/orchestrator.py](src/hr_mlops/pipelines/orchestrator.py) to validate Gold table schema before processing:

```python
def validate_input_schema(df: DataFrame, config: dict[str, Any]) -> None:
    required_cols = config.get("required_columns", [])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
```

### 11.2 Add Data Quality Checks

```python
# Example: Check for null employees
if source_df.filter(F.col("employee_id").isNull()).count() > 0:
    raise ValueError("Found null employee_id values in source")
```

### 11.3 Enable Table Access Controls

```sql
-- Grant read access to Power BI service principal
GRANT SELECT ON TABLE hr_mlops.pred_attrition_risk TO `<powerbi-app-id>`;
GRANT SELECT ON TABLE hr_mlops.pred_workforce_health TO `<powerbi-app-id>`;
-- ... repeat for all output tables
```

## Step 12: Performance Optimization

### 12.1 Partition Output Tables (Large Deployments)

```sql
CREATE OR REPLACE TABLE hr_mlops.pred_attrition_risk
PARTITIONED BY (business_unit)
USING DELTA
AS
SELECT * FROM hr_mlops.pred_attrition_risk;
```

### 12.2 Add Cluster Policies

Define a cluster policy to enforce consistency:

```json
{
  "spark_conf": {
    "spark.sql.adaptive.enabled": true,
    "spark.sql.adaptive.coalescePartitions.enabled": true
  },
  "node_type_id": "Standard_DS3_v2",
  "num_workers": 2
}
```

---

## Support & Troubleshooting

For issues or questions, check the main [README.md](README.md) or review inline comments in:
- [src/hr_mlops/models/use_cases.py](src/hr_mlops/models/use_cases.py)
- [src/hr_mlops/pipelines/orchestrator.py](src/hr_mlops/pipelines/orchestrator.py)
- [configs/base_config_multitable_join.yaml](configs/base_config_multitable_join.yaml)
