"""Outlier detection and removal using the IQR method with linear interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_outliers_iqr(
    series: pd.Series,
    factor: float = 1.5,
) -> pd.Series:
    """Return a boolean mask of outlier positions using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def replace_outliers_with_interpolation(
    df: pd.DataFrame,
    column: str,
    factor: float = 1.5,
) -> tuple[pd.DataFrame, int]:
    """Detect outliers via IQR, replace with NaN, then linearly interpolate.

    Returns the cleaned DataFrame and the number of outliers replaced.
    """
    df = df.copy()
    outliers = detect_outliers_iqr(df[column], factor)
    n_outliers = int(outliers.sum())

    df[column] = df[column].where(~outliers, np.nan)
    df[column] = df[column].interpolate(method="linear")

    return df, n_outliers
