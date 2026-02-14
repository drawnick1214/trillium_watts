"""Tests for data cleaning module."""

import pandas as pd
import pytest

from trillium_watts.data.cleaning import (
    convert_numeric_columns,
    parse_dates,
    filter_by_date,
    set_date_index,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "FECHA": ["01/01/2020", "02/01/2020", "03/01/2020"],
            "ACTIVA": ["100,000", "200,000", "300,000"],
            "REACTIVA": ["50,000", "60,000", "70,000"],
        }
    )


def test_convert_numeric_columns(sample_df):
    result = convert_numeric_columns(sample_df, ["ACTIVA", "REACTIVA"])
    assert result["ACTIVA"].dtype == float
    assert result["ACTIVA"].iloc[0] == 100000.0
    assert result["REACTIVA"].iloc[2] == 70000.0


def test_parse_dates(sample_df):
    result = parse_dates(sample_df, "FECHA", dayfirst=True)
    assert pd.api.types.is_datetime64_any_dtype(result["FECHA"])
    assert result["FECHA"].iloc[0] == pd.Timestamp("2020-01-01")


def test_filter_by_date():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2025-03-31", "2025-04-01", "2025-05-01"]),
            "val": [1, 2, 3, 4],
        }
    )
    result = filter_by_date(df, "date", "2025-04-01")
    assert len(result) == 2


def test_set_date_index():
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "val": [1, 2],
        }
    )
    result = set_date_index(df)
    assert result.index.name == "FECHA"
    assert len(result) == 2
