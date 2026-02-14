# TrilliumWatts — R3newAI

Energy demand forecasting and solar energy simulation for **Leticia, Amazonas, Colombia** (Non-Interconnected Zone).

This project uses Deep Learning (LSTM/GRU) to predict daily electrical energy demand, then simulates the economic and environmental impact of photovoltaic solar installations across different capacity scenarios.

## Project Structure

```
trillium_watts/
├── config/default.yaml          # All configuration (constants, hyperparams, paths)
├── src/trillium_watts/          # Python package
│   ├── config.py                # YAML config loader
│   ├── data/                    # Loading, cleaning, imputation, outlier removal
│   ├── features/                # Temporal + cyclic feature engineering
│   ├── models/                  # Sequences, LSTM/GRU architectures, training, persistence
│   ├── prediction/              # Autoregressive forecasting + CSV export
│   ├── simulation/              # Solar energy, economics, scenario management
│   └── visualization/           # Matplotlib (EDA) + Plotly (Streamlit) plots
├── app/streamlit_app.py         # Interactive web dashboard
├── scripts/                     # CLI pipeline: download, preprocess, train, predict
├── notebooks/                   # EDA and experiment notebooks
├── data/                        # Raw, processed, and prediction data
├── models/                      # Saved model weights and scalers
└── tests/                       # Unit tests
```

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Full Pipeline

```bash
make download     # Download raw data from Google Drive
make preprocess   # Clean, impute, engineer features
make train        # Grid search + train best GRU model
make predict      # Autoregressive forecast -> data/predictions/
```

Or run all at once:

```bash
make all
```

### Launch Dashboard

```bash
make app
# or
streamlit run app/streamlit_app.py
```

### Run Tests

```bash
make test
```

## Data

- **Source**: Daily energy consumption data for Leticia, Amazonas, Colombia
- **Variables**: ACTIVA (kWh), REACTIVA, FP, ALLSKY_SFC_SW_DWN (solar radiation), T2M (temperature)
- **Period**: 2015-2025

## Models

- **GRU** (best, R2 ~0.63) and **LSTM** architectures
- 15-day sliding window with 12 features (target + cyclic temporal + meteorological)
- Autoregressive multi-step forecasting (7, 15, or 30 days)

## Solar Simulation

Three capacity scenarios:
- Small: 100 kW
- Medium: 1 MW
- Large: 5 MW

Calculates: energy generation, diesel displacement, CO2 reduction, economic savings (COP).

## Team — Samsung Innovation Campus 2024

- Edward Nicolas Diaz Cristancho
- Juliana Utrera Florez
- Laura Rico Aldana
