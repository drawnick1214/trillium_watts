"""Imputation strategies for missing values in the energy dataset."""

from __future__ import annotations

import pandas as pd


def impute_activa_by_reference(
    df: pd.DataFrame,
    missing_periods: list[tuple[str, str]],
) -> pd.DataFrame:
    """Impute missing ACTIVA values using year-ago reference adjusted by surrounding-period delta.

    For each missing period, computes the average ACTIVA 2 months before and
    2 months after the gap, calculates a delta, then fills each missing date
    with the value from the same date one year prior plus the delta.
    """
    df = df.copy()
    periods = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in missing_periods]

    for start, end in periods:
        anterior_inicio = (start - pd.DateOffset(months=2)).replace(day=1)
        anterior_fin = start - pd.DateOffset(days=1)

        posterior_inicio = end + pd.DateOffset(days=1)
        posterior_fin = (end + pd.DateOffset(months=2)).replace(day=1) + pd.offsets.MonthEnd(0)

        prom_ant = df.loc[anterior_inicio:anterior_fin, "ACTIVA"].dropna().mean()
        prom_pos = df.loc[posterior_inicio:posterior_fin, "ACTIVA"].dropna().mean()
        delta = prom_pos - prom_ant

        dias_hueco = pd.date_range(start, end, freq="D")
        for fecha in dias_hueco:
            try:
                fecha_referencia = fecha - pd.DateOffset(years=1)
                valor_ref = df.loc[fecha_referencia, "ACTIVA"]
                if pd.notna(valor_ref):
                    df.at[fecha, "ACTIVA"] = valor_ref + delta
            except KeyError:
                continue

    return df


def impute_fp_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Fill FP null values with the column mode."""
    df = df.copy()
    moda_fp = df["FP"].mode()[0]
    df["FP"] = df["FP"].fillna(moda_fp)
    return df


def impute_reactiva_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill then backward fill REACTIVA null values."""
    df = df.copy()
    df["REACTIVA"] = df["REACTIVA"].ffill().bfill()
    return df


def run_imputation_pipeline(
    df: pd.DataFrame,
    missing_periods: list[tuple[str, str]],
) -> pd.DataFrame:
    """Run all imputation steps in sequence."""
    df = impute_activa_by_reference(df, missing_periods)
    df = impute_fp_mode(df)
    df = impute_reactiva_fill(df)
    return df
