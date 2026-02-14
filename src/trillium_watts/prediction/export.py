"""Export predictions to CSV for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_predictions_csv(
    historical_df: pd.DataFrame,
    predictions: pd.Series,
    output_path: str | Path,
    target_column: str = "ACTIVA",
    label_historical: str = "Historica",
    label_predicted: str = "Predicha",
) -> Path:
    """Create a unified CSV with columns [Fecha, ACTIVA, Tipo].

    Combines historical data and model predictions into a single file
    that the Streamlit app consumes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Historical portion
    df_hist = pd.DataFrame(
        {
            "Fecha": historical_df.index,
            target_column: historical_df[target_column].values,
            "Tipo": label_historical,
        }
    )

    # Predicted portion
    df_pred = pd.DataFrame(
        {
            "Fecha": predictions.index,
            target_column: predictions.values,
            "Tipo": label_predicted,
        }
    )

    df_combined = pd.concat([df_hist, df_pred], ignore_index=True)
    df_combined = df_combined.sort_values("Fecha").reset_index(drop=True)
    df_combined.to_csv(output_path, index=False)
    return output_path
