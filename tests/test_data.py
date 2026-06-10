"""Tests for the data ingestion and processing module."""

from __future__ import annotations

import pytest
import pandas as pd

from src.data.make_dataset import clean_data, split_data


class TestCleanData:
    def test_drops_fully_null_rows(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, None, 6]})
        result = clean_data(df)
        assert len(result) == 2
        assert result["a"].tolist() == [1.0, 3.0]

    def test_drops_duplicates_by_default(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = clean_data(df)
        assert len(result) == 2

    def test_keeps_duplicates_when_disabled(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = clean_data(df, drop_duplicates=False)
        assert len(result) == 3

    def test_returns_reset_index(self):
        df = pd.DataFrame({"a": [None, 1, 2], "b": [None, 3, 4]})
        result = clean_data(df)
        assert list(result.index) == list(range(len(result)))

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = clean_data(df)
        assert len(result) == 0


class TestSplitData:
    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame(
            {
                "feature_1": range(100),
                "feature_2": range(100, 200),
                "target": [i % 2 for i in range(100)],
            }
        )

    def test_split_sizes(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, "target", test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_target_column_removed_from_features(self, sample_df):
        X_train, X_test, _, _ = split_data(sample_df, "target")
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_reproducible_with_same_seed(self, sample_df):
        _, X_test_1, _, _ = split_data(sample_df, "target", random_state=0)
        _, X_test_2, _, _ = split_data(sample_df, "target", random_state=0)
        pd.testing.assert_frame_equal(X_test_1, X_test_2)

    def test_different_seeds_give_different_splits(self, sample_df):
        _, X_test_1, _, _ = split_data(sample_df, "target", random_state=1)
        _, X_test_2, _, _ = split_data(sample_df, "target", random_state=2)
        assert not X_test_1.index.equals(X_test_2.index)
