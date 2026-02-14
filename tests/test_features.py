"""Tests for feature engineering modules."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from trillium_watts.features.temporal import extract_temporal_features
from trillium_watts.features.cyclic import encode_cyclic_features, compute_cyclic_for_date


@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({"ACTIVA": [100, 200, 300, 400, 500]}, index=dates)


def test_extract_temporal_features(sample_df):
    result = extract_temporal_features(sample_df)
    assert "year" in result.columns
    assert "month" in result.columns
    assert "weekday" in result.columns
    assert "dayofyear" in result.columns
    assert result["year"].iloc[0] == 2023
    assert result["month"].iloc[0] == 1
    assert result["dayofyear"].iloc[0] == 1


def test_encode_cyclic_features(sample_df):
    df = extract_temporal_features(sample_df)
    result = encode_cyclic_features(df)
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns
    # All sin/cos values should be in [-1, 1]
    for col in ["month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"]:
        assert result[col].min() >= -1.0
        assert result[col].max() <= 1.0


def test_compute_cyclic_for_date():
    date = datetime(2023, 6, 15)
    result = compute_cyclic_for_date(date)
    assert "month_sin" in result
    assert "month_cos" in result
    assert -1.0 <= result["month_sin"] <= 1.0
    assert -1.0 <= result["month_cos"] <= 1.0
    # June is month 6, so sin(2*pi*6/12) = sin(pi) ~ 0
    assert abs(result["month_sin"]) < 0.01
