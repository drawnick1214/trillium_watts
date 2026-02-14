"""Train models via grid search, select best, retrain on all data, save."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trillium_watts.config import load_config, get_project_root
from trillium_watts.models.sequences import create_sequences, split_data, fit_scalers, apply_scalers
from trillium_watts.models.training import grid_search, select_best_params, retrain_final_model, evaluate_model
from trillium_watts.models.persistence import save_model


def main():
    config = load_config()
    root = get_project_root()

    # Load processed data
    processed_path = root / config.data.processed_data_path
    print(f"Loading processed data from {processed_path}...")
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    features = config.features.feature_columns
    target = config.features.target
    data = df[features].values
    target_index = features.index(target)

    # Create sequences
    window_size = config.model.window_size
    X_raw, y_raw = create_sequences(data, window_size, target_index)
    print(f"Created {len(X_raw)} sequences with window_size={window_size}")

    # Split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(X_raw, y_raw, config.model.train_split_ratio)

    # Scale
    scaler_X, scaler_y = fit_scalers(X_train_raw, y_train_raw)
    X_train, y_train = apply_scalers(X_train_raw, y_train_raw, scaler_X, scaler_y)
    X_test, y_test = apply_scalers(X_test_raw, y_test_raw, scaler_X, scaler_y)

    # Grid search for GRU (best performing model)
    model_type = "gru"
    print(f"\nRunning grid search for {model_type.upper()}...")
    results = grid_search(
        model_type,
        config.model.param_grid,
        X_train, y_train,
        X_test, y_test,
        early_stopping_patience=config.model.early_stopping.patience,
    )

    best = select_best_params(results)
    print(f"\nBest params: {best['params']}")
    print(f"Best val_mae: {best['val_mae']:.4f}")

    # Retrain on all data
    print("\nRetraining on all data...")
    X_all, y_all = apply_scalers(X_raw, y_raw, scaler_X, scaler_y)
    model = retrain_final_model(
        model_type, best, X_all, y_all,
        early_stopping_patience=config.model.early_stopping.patience,
    )

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, scaler_y)
    print(f"\nTest metrics:")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R2:   {metrics['r2']:.4f}")

    # Save
    save_dir = root / config.model.model_save_path
    save_model(model, scaler_X, scaler_y, save_dir)
    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()
