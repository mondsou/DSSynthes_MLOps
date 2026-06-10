# DSSynthes MLOps

> An end-to-end MLOps template for data science synthesis projects, featuring reproducible pipelines, experiment tracking, and CI/CD.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Testing](#testing)
7. [Experiment Tracking](#experiment-tracking)
8. [Contributing](#contributing)

---

## Project Overview

DSSynthes MLOps provides a production-ready scaffold for machine-learning projects with a focus on:

- **Reproducibility** – versioned data (DVC), pinned dependencies, and tracked experiments (MLflow).
- **Modularity** – clean separation between data ingestion, feature engineering, model training, and inference.
- **Automation** – CI/CD via GitHub Actions with automated linting, testing, and optional model registration.

---

## Project Structure

```
DSSynthes_MLOps/
├── configs/                  # Hydra / OmegaConf configuration files
│   └── config.yaml
├── data/                     # (git-ignored) raw, interim, and processed data
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/                   # (git-ignored) serialised model artefacts
├── notebooks/                # Exploratory Jupyter notebooks
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py   # Data ingestion and cleaning
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py    # Model training
│   │   └── predict_model.py  # Inference / scoring
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── training_pipeline.py  # End-to-end training pipeline
│   └── __init__.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- [pip](https://pip.pypa.io/) or [uv](https://github.com/astral-sh/uv)

### Installation

```bash
# Clone the repository
git clone https://github.com/mondsou/DSSynthes_MLOps.git
cd DSSynthes_MLOps

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install runtime + dev dependencies
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt
```

---

## Configuration

All runtime settings live in `configs/config.yaml` and are managed via [Hydra](https://hydra.cc/):

```yaml
data:
  raw_path: data/raw/dataset.csv
  processed_path: data/processed/dataset.parquet
  test_size: 0.2
  random_state: 42

features:
  drop_columns: []
  target_column: target

model:
  name: random_forest
  params:
    n_estimators: 100
    max_depth: null
    random_state: 42

mlflow:
  tracking_uri: mlruns
  experiment_name: dssynthes-experiment
```

Override any value from the command line:

```bash
dssynthes-train model.params.n_estimators=200
```

---

## Running the Pipeline

### Training

```bash
dssynthes-train
```

### Prediction

```bash
dssynthes-predict --model-path models/model.joblib --input data/processed/test.parquet
```

### Individual stages

```bash
# 1. Build dataset
python -m src.data.make_dataset

# 2. Engineer features
python -m src.features.build_features

# 3. Train model
python -m src.models.train_model

# 4. Run inference
python -m src.models.predict_model
```

---

## Testing

```bash
pytest                     # run all tests with coverage
pytest tests/test_data.py  # run a specific test file
```

---

## Experiment Tracking

MLflow is used for tracking metrics, parameters, and model artefacts:

```bash
# Start the MLflow UI
mlflow ui --backend-store-uri mlruns
# Then open http://localhost:5000
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`.
3. Make your changes and add tests.
4. Run linting: `ruff check . && black --check .`
5. Run tests: `pytest`
6. Open a pull request.
