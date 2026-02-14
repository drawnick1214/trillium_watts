"""Model persistence — save and load trained models and scalers."""

from __future__ import annotations

from pathlib import Path

import joblib
from tensorflow.keras.models import Sequential, load_model as keras_load_model


def save_model(
    model: Sequential,
    scaler_X,
    scaler_y,
    directory: str | Path,
    model_name: str = "best_model",
) -> None:
    """Save a Keras model and its scalers to disk."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    model.save(directory / f"{model_name}.keras")
    joblib.dump(scaler_X, directory / f"{model_name}_scaler_X.joblib")
    joblib.dump(scaler_y, directory / f"{model_name}_scaler_y.joblib")


def load_model(
    directory: str | Path,
    model_name: str = "best_model",
) -> tuple[Sequential, object, object]:
    """Load a saved Keras model and its scalers.

    Returns (model, scaler_X, scaler_y).
    """
    directory = Path(directory)
    model = keras_load_model(directory / f"{model_name}.keras")
    scaler_X = joblib.load(directory / f"{model_name}_scaler_X.joblib")
    scaler_y = joblib.load(directory / f"{model_name}_scaler_y.joblib")
    return model, scaler_X, scaler_y
