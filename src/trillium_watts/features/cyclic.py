"""Cyclic (sin/cos) encoding for temporal features."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def encode_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encoded versions of month, dayofyear, weekday, weekofyear.

    Expects the DataFrame to already contain month, dayofyear, weekday, weekofyear columns.
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["weekofyear_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["weekofyear_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    return df


def compute_cyclic_for_date(date: datetime) -> dict[str, float]:
    """Compute cyclic features for a single date.

    Used during autoregressive prediction to build the feature vector
    for future dates.
    """
    month = date.month
    doy = date.timetuple().tm_yday
    weekday = date.weekday()
    woy = date.isocalendar().week

    return {
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dayofyear_sin": np.sin(2 * np.pi * doy / 365),
        "dayofyear_cos": np.cos(2 * np.pi * doy / 365),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "weekofyear_sin": np.sin(2 * np.pi * woy / 52),
        "weekofyear_cos": np.cos(2 * np.pi * woy / 52),
    }
