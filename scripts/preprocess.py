"""Run the full preprocessing pipeline: load -> clean -> impute -> outliers -> features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trillium_watts.config import load_config, get_project_root
from trillium_watts.data.loader import load_raw_data
from trillium_watts.data.cleaning import run_cleaning_pipeline
from trillium_watts.data.imputation import run_imputation_pipeline
from trillium_watts.data.outliers import replace_outliers_with_interpolation
from trillium_watts.features.pipeline import build_feature_pipeline


def main():
    config = load_config()
    root = get_project_root()

    # 1. Load raw data
    raw_path = root / config.data.raw_data_path
    print(f"Loading raw data from {raw_path}...")
    df = load_raw_data(raw_path, config.data.csv_separator, config.data.csv_encoding)

    # 2. Clean
    print("Cleaning data...")
    df = run_cleaning_pipeline(
        df,
        numeric_columns=["ACTIVA", "REACTIVA"],
        date_column=config.data.date_column,
        date_cutoff=config.data.date_cutoff,
    )

    # 3. Feature engineering (needs to happen before imputation since temporal features are needed)
    print("Engineering features...")
    df = build_feature_pipeline(df)

    # 4. Impute missing values
    print("Imputing missing values...")
    df = run_imputation_pipeline(df, config.data.missing_periods)

    # 5. Remove outliers
    print("Removing outliers...")
    df, n_outliers = replace_outliers_with_interpolation(df, "ACTIVA")
    print(f"  Replaced {n_outliers} outliers in ACTIVA.")

    # 6. Save processed data
    output_path = root / config.data.processed_data_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
