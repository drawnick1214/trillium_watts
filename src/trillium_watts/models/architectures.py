"""Neural network architectures — LSTM and GRU model builders."""

from __future__ import annotations

from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    units: int,
    dropout: float,
    learning_rate: float,
    input_shape: tuple[int, int],
) -> Sequential:
    """Build and compile an LSTM model."""
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gru_model(
    units: int,
    dropout: float,
    learning_rate: float,
    input_shape: tuple[int, int],
) -> Sequential:
    """Build and compile a GRU model."""
    model = Sequential([
        GRU(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_model(
    model_type: str,
    units: int,
    dropout: float,
    learning_rate: float,
    input_shape: tuple[int, int],
) -> Sequential:
    """Factory function — dispatches to the LSTM or GRU builder."""
    builders = {"lstm": build_lstm_model, "gru": build_gru_model}
    if model_type not in builders:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(builders)}")
    return builders[model_type](units, dropout, learning_rate, input_shape)
