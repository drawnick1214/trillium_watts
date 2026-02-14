"""Configuration loader — reads YAML config and exposes typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class DataConfig:
    google_drive_file_id: str
    raw_data_path: str
    processed_data_path: str
    predictions_path: str
    csv_separator: str
    csv_encoding: str
    date_column: str
    date_cutoff: str
    missing_periods: list[list[str]]


@dataclass
class FeaturesConfig:
    target: str
    feature_columns: list[str]


@dataclass
class EarlyStoppingConfig:
    monitor: str
    patience: int
    restore_best_weights: bool


@dataclass
class ModelConfig:
    window_size: int
    train_split_ratio: float
    model_save_path: str
    param_grid: dict
    early_stopping: EarlyStoppingConfig


@dataclass
class PredictionConfig:
    horizons: list[int]
    default_horizon: int


@dataclass
class SolarConfig:
    default_h_radiation: float
    default_performance_ratio: float
    scenarios: dict[str, int]


@dataclass
class EconomicConfig:
    kwh_per_liter_diesel: float
    co2_per_liter_diesel: float
    diesel_price_cop: float


@dataclass
class VisualizationConfig:
    tipo_labels: dict[str, str]
    colors: dict[str, str]


@dataclass
class Config:
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    prediction: PredictionConfig
    solar: SolarConfig
    economic: EconomicConfig
    visualization: VisualizationConfig


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file and return a Config dataclass."""
    path = Path(path) if path else _DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        model=ModelConfig(
            **{k: v for k, v in raw["model"].items() if k != "early_stopping"},
            early_stopping=EarlyStoppingConfig(**raw["model"]["early_stopping"]),
        ),
        prediction=PredictionConfig(**raw["prediction"]),
        solar=SolarConfig(**raw["solar"]),
        economic=EconomicConfig(**raw["economic"]),
        visualization=VisualizationConfig(**raw["visualization"]),
    )


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
