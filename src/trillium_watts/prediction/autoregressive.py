"""Autoregressive multi-step forecasting."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from trillium_watts.features.cyclic import compute_cyclic_for_date


def predict_future(
    model,
    initial_sequence_scaled: np.ndarray,
    num_steps: int,
    scaler_X: MinMaxScaler,
    features_df: pd.DataFrame,
    target_name: str = "ACTIVA",
    features_list: list[str] | None = None,
) -> pd.Series:
    """Autoregressive multi-step prediction.

    Uses the trained model to predict the next day, then feeds that prediction
    back as input for subsequent days.

    Args:
        model: Trained Keras model.
        initial_sequence_scaled: Shape (1, window_size, n_features), already scaled.
        num_steps: Number of future days to predict.
        scaler_X: Fitted MinMaxScaler for features.
        features_df: DataFrame with historical data (used for last date and feature order).
        target_name: Name of the target column.
        features_list: Ordered list of feature column names.

    Returns:
        pd.Series indexed by future dates with unscaled predicted values.
    """
    if features_list is None:
        raise ValueError("features_list must be provided.")

    idx_map = {feat: i for i, feat in enumerate(features_list)}
    tgt_idx = idx_map[target_name]
    data_min = scaler_X.data_min_
    data_range = scaler_X.data_range_

    current_input = initial_sequence_scaled.copy()
    last_unscaled = scaler_X.inverse_transform(current_input[0, -1].reshape(1, -1))[0]
    last_date = features_df.index[-1]

    future_scaled = []
    for i in range(num_steps):
        pred_scaled = model.predict(current_input, verbose=0)[0, 0]
        future_scaled.append(pred_scaled)

        unscaled_target = pred_scaled * data_range[tgt_idx] + data_min[tgt_idx]

        date = last_date + timedelta(days=i + 1)
        cyc = compute_cyclic_for_date(date)

        new_unscaled = last_unscaled.copy()
        new_unscaled[tgt_idx] = unscaled_target

        # Update temporal and cyclic features
        temporal_values = {
            "year": date.year,
            "month": date.month,
            "dayofyear": date.timetuple().tm_yday,
            "weekday": date.weekday(),
            "weekofyear": date.isocalendar().week,
        }
        for feat, val in {**temporal_values, **cyc}.items():
            if feat in idx_map:
                new_unscaled[idx_map[feat]] = val

        new_scaled = scaler_X.transform(new_unscaled.reshape(1, -1))[0]
        current_input = np.concatenate(
            [current_input[:, 1:, :], new_scaled.reshape(1, 1, -1)],
            axis=1,
        )
        last_unscaled = new_unscaled

    future_unscaled = np.array(future_scaled) * data_range[tgt_idx] + data_min[tgt_idx]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(num_steps)]
    return pd.Series(future_unscaled, index=future_dates)


def prepare_initial_sequence(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    scaler_X: MinMaxScaler,
) -> np.ndarray:
    """Extract and scale the last ``window_size`` rows as the initial input.

    Returns array of shape (1, window_size, n_features).
    """
    last_seq = df[features].iloc[-window_size:].values
    return scaler_X.transform(last_seq).reshape(1, window_size, len(features))
