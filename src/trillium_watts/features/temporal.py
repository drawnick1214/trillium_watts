"""Temporal feature extraction from a DatetimeIndex."""

from __future__ import annotations

import pandas as pd


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, day, weekday, weekofyear, quarter, dayofyear columns.

    Assumes the DataFrame has a DatetimeIndex.
    """
    df = df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["weekday"] = df.index.weekday  # Monday=0, Sunday=6
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["quarter"] = df.index.quarter
    df["dayofyear"] = df.index.dayofyear
    return df
