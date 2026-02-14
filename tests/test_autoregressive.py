"""Tests for the prediction export module."""

from pathlib import Path

import pandas as pd
import pytest

from trillium_watts.prediction.export import export_predictions_csv


def test_export_predictions_csv(tmp_path):
    # Create sample data
    dates_hist = pd.date_range("2024-01-01", periods=5, freq="D")
    df_hist = pd.DataFrame({"ACTIVA": [100, 200, 300, 400, 500]}, index=dates_hist)
    df_hist.index.name = "FECHA"

    dates_pred = pd.date_range("2024-01-06", periods=3, freq="D")
    predictions = pd.Series([550, 600, 650], index=dates_pred)

    output_path = tmp_path / "test_predictions.csv"
    result_path = export_predictions_csv(df_hist, predictions, output_path)

    assert result_path.exists()

    df = pd.read_csv(result_path)
    assert len(df) == 8
    assert set(df.columns) == {"Fecha", "ACTIVA", "Tipo"}
    assert df[df["Tipo"] == "Historica"].shape[0] == 5
    assert df[df["Tipo"] == "Predicha"].shape[0] == 3
