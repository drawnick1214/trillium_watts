"""Run autoregressive prediction and export CSV for the Streamlit app."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trillium_watts.config import load_config, get_project_root
from trillium_watts.models.persistence import load_model
from trillium_watts.prediction.autoregressive import predict_future, prepare_initial_sequence
from trillium_watts.prediction.export import export_predictions_csv


def main():
    config = load_config()
    root = get_project_root()

    # Load processed data
    processed_path = root / config.data.processed_data_path
    print(f"Loading processed data from {processed_path}...")
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)

    features = config.features.feature_columns
    target = config.features.target
    window_size = config.model.window_size
    num_steps = config.prediction.default_horizon

    # Load trained model
    model_dir = root / config.model.model_save_path
    print(f"Loading model from {model_dir}...")
    model, scaler_X, scaler_y = load_model(model_dir)

    # Fit a full-data scaler for the autoregressive prediction
    scaler_full = MinMaxScaler()
    scaler_full.fit(df[features].values)

    # Prepare initial sequence
    initial_seq = prepare_initial_sequence(df, features, window_size, scaler_full)

    # Predict
    print(f"Predicting {num_steps} days into the future...")
    predictions = predict_future(
        model=model,
        initial_sequence_scaled=initial_seq,
        num_steps=num_steps,
        scaler_X=scaler_full,
        features_df=df,
        target_name=target,
        features_list=features,
    )
    print(f"Predictions:\n{predictions}")

    # Export
    output_path = root / config.data.predictions_path
    viz_config = config.visualization
    export_predictions_csv(
        df, predictions, output_path,
        target_column=target,
        label_historical=viz_config.tipo_labels["historical"],
        label_predicted=viz_config.tipo_labels["predicted"],
    )
    print(f"\nPredictions exported to {output_path}")


if __name__ == "__main__":
    main()
