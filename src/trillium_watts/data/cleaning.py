"""Data cleaning — type conversion, date parsing, filtering, index setup."""

from __future__ import annotations

import pandas as pd


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove commas from string values and convert columns to float."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].str.replace(",", "").astype(float)
    return df


def parse_dates(
    df: pd.DataFrame,
    date_column: str = "FECHA",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Convert a column to datetime."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=dayfirst)
    return df


def filter_by_date(
    df: pd.DataFrame,
    date_column: str,
    cutoff: str,
) -> pd.DataFrame:
    """Keep only rows where date_column < cutoff."""
    return df[df[date_column] < cutoff].copy()


def set_date_index(df: pd.DataFrame, date_column: str = "FECHA") -> pd.DataFrame:
    """Set the specified column as the DataFrame index."""
    return df.set_index(date_column)


def run_cleaning_pipeline(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    date_column: str = "FECHA",
    date_cutoff: str = "2025-04-01",
) -> pd.DataFrame:
    """Run the full cleaning pipeline: convert types, parse dates, filter, set index."""
    if numeric_columns is None:
        numeric_columns = ["ACTIVA", "REACTIVA"]
    df = convert_numeric_columns(df, numeric_columns)
    df = parse_dates(df, date_column)
    df = filter_by_date(df, date_column, date_cutoff)
    df = set_date_index(df, date_column)
    return df
