"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.features.build_features import (
    build_features,
    drop_columns,
    encode_categoricals,
    scale_numerics,
)


class TestDropColumns:
    def test_drops_existing_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = drop_columns(df, ["a", "b"])
        assert list(result.columns) == ["c"]

    def test_ignores_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = drop_columns(df, ["z"])
        assert list(result.columns) == ["a"]

    def test_empty_list_is_no_op(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        pd.testing.assert_frame_equal(drop_columns(df, []), df)


class TestEncodeCategoricals:
    def test_encodes_object_columns(self):
        df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]})
        result = encode_categoricals(df)
        assert "cat_a" in result.columns or "cat" not in result.columns
        assert "num" in result.columns

    def test_no_op_on_numeric_only(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        pd.testing.assert_frame_equal(encode_categoricals(df), df)

    def test_output_dtype_is_uint8(self):
        df = pd.DataFrame({"cat": ["x", "y"]})
        result = encode_categoricals(df)
        for col in result.columns:
            assert result[col].dtype == np.uint8


class TestScaleNumerics:
    @pytest.fixture()
    def num_df(self):
        return pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [10.0, 20.0, 30.0]})

    def test_output_has_zero_mean(self, num_df):
        result, _ = scale_numerics(num_df)
        assert np.allclose(result.mean(), 0.0, atol=1e-10)

    def test_output_has_unit_std(self, num_df):
        result, _ = scale_numerics(num_df)
        assert np.allclose(result.std(ddof=0), 1.0, atol=1e-10)

    def test_returns_scaler_instance(self, num_df):
        _, scaler = scale_numerics(num_df)
        assert isinstance(scaler, StandardScaler)

    def test_transform_uses_existing_scaler(self, num_df):
        _, fitted_scaler = scale_numerics(num_df, fit=True)
        new_df = pd.DataFrame({"a": [3.0], "b": [40.0]})
        result, _ = scale_numerics(new_df, scaler=fitted_scaler, fit=False)
        expected = fitted_scaler.transform([[3.0, 40.0]])
        assert np.allclose(result.values, expected, atol=1e-10)

    def test_no_op_on_non_numeric(self):
        df = pd.DataFrame({"cat": ["a", "b"]})
        result, scaler = scale_numerics(df)
        pd.testing.assert_frame_equal(result, df)


class TestBuildFeatures:
    def test_pipeline_runs(self):
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0],
                "num2": [4.0, 5.0, 6.0],
                "cat": ["a", "b", "a"],
            }
        )
        result, scaler = build_features(df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(scaler, StandardScaler)
        assert "cat" not in result.columns

    def test_drop_cols_applied(self):
        df = pd.DataFrame({"keep": [1.0, 2.0], "drop_me": [3.0, 4.0]})
        result, _ = build_features(df, drop_cols=["drop_me"])
        assert "drop_me" not in result.columns
        assert "keep" in result.columns
