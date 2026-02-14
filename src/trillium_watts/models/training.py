"""Model training — grid search, best-model selection, retraining, evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

from trillium_watts.models.architectures import build_model


def grid_search(
    model_type: str,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    early_stopping_patience: int = 10,
) -> list[dict]:
    """Run grid search over hyperparameters.

    Returns a list of result dicts, each containing:
        params, val_mae, val_loss, epochs_ran, history
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    results = []

    for params in ParameterGrid(param_grid):
        model = build_model(
            model_type,
            units=params["units"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            input_shape=input_shape,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stop],
            verbose=1,
        )

        val_mae = min(history.history["val_mae"])
        results.append(
            {
                "params": params,
                "val_mae": val_mae,
                "val_loss": min(history.history["val_loss"]),
                "epochs_ran": len(history.history["val_mae"]),
                "history": history.history,
            }
        )

    return results


def select_best_params(results: list[dict], metric: str = "val_mae") -> dict:
    """Select the best hyperparameters from grid search results."""
    best = min(results, key=lambda r: r[metric])
    return best


def retrain_final_model(
    model_type: str,
    best_params: dict,
    X_all: np.ndarray,
    y_all: np.ndarray,
    early_stopping_patience: int = 10,
) -> Sequential:
    """Retrain a model with all data using the best hyperparameters."""
    params = best_params["params"]
    input_shape = (X_all.shape[1], X_all.shape[2])

    model = build_model(
        model_type,
        units=params["units"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        input_shape=input_shape,
    )

    model.fit(
        X_all,
        y_all,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[EarlyStopping(monitor="loss", patience=early_stopping_patience, restore_best_weights=True)],
        verbose=0,
    )

    return model


def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: MinMaxScaler,
) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 on the test set (inverse-scaled)."""
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "y_pred": y_pred,
        "y_true": y_true,
    }
