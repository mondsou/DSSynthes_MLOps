"""Tests for model training and prediction utilities."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.models.predict_model import predict, predict_proba
from src.models.train_model import build_model, evaluate


@pytest.fixture()
def classification_data():
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=10, n_classes=2, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture()
def fitted_model(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


class TestBuildModel:
    def test_builds_random_forest(self):
        model = build_model("random_forest", {"n_estimators": 5, "random_state": 0})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 5

    def test_raises_on_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("super_fancy_model", {})


class TestEvaluate:
    def test_returns_accuracy_and_f1(self, fitted_model, classification_data):
        X, y = classification_data
        metrics = evaluate(fitted_model, X, y)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics

    def test_metrics_are_floats_between_0_and_1(self, fitted_model, classification_data):
        X, y = classification_data
        metrics = evaluate(fitted_model, X, y)
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} is outside [0, 1]"


class TestPredict:
    def test_returns_series(self, fitted_model, classification_data):
        X, _ = classification_data
        result = predict(fitted_model, X)
        assert isinstance(result, pd.Series)
        assert result.name == "prediction"
        assert len(result) == len(X)

    def test_index_matches_input(self, fitted_model, classification_data):
        X, _ = classification_data
        X_subset = X.iloc[10:20]
        result = predict(fitted_model, X_subset)
        assert result.index.equals(X_subset.index)


class TestPredictProba:
    def test_returns_dataframe(self, fitted_model, classification_data):
        X, _ = classification_data
        result = predict_proba(fitted_model, X)
        assert isinstance(result, pd.DataFrame)

    def test_probabilities_sum_to_one(self, fitted_model, classification_data):
        import numpy as np

        X, _ = classification_data
        result = predict_proba(fitted_model, X)
        row_sums = result.sum(axis=1)
        assert all(abs(row_sums - 1.0) < 1e-6)

    def test_raises_for_model_without_predict_proba(self, classification_data):
        from sklearn.svm import SVC

        X, y = classification_data
        svc = SVC(probability=False)
        svc.fit(X, y)
        with pytest.raises(AttributeError):
            predict_proba(svc, X)
