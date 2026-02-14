"""Sliding window sequence creation, data splitting, and scaling."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(
    data: np.ndarray,
    window_size: int,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for time series modelling.

    Args:
        data: 2D array of shape (n_timesteps, n_features).
        window_size: Number of past timesteps in each input window.
        target_index: Column index of the target variable in ``data``.

    Returns:
        X: 3D array of shape (n_samples, window_size, n_features).
        y: 1D array of shape (n_samples,).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size, target_index])
    return np.array(X), np.array(y)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temporal train/test split (no shuffle)."""
    split_index = int(len(X) * train_ratio)
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def fit_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[MinMaxScaler, MinMaxScaler]:
    """Fit MinMaxScalers on training data.

    X is reshaped to 2D for fitting, y is reshaped to (n, 1).
    """
    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train.reshape(-1, X_train.shape[2]))

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    return scaler_X, scaler_y


def apply_scalers(
    X: np.ndarray,
    y: np.ndarray,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform data using fitted scalers."""
    n_samples, window, n_features = X.shape
    X_scaled = scaler_X.transform(X.reshape(-1, n_features)).reshape(n_samples, window, n_features)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled
