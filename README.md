# TrilliumWatts — R3newAI

Energy demand forecasting and solar energy simulation for **Leticia, Amazonas, Colombia** (Non-Interconnected Zone).

This project uses Deep Learning (LSTM/GRU) to predict daily electrical energy demand, then simulates the economic and environmental impact of photovoltaic solar installations across different capacity scenarios.

## Executive Summary

### Objective

Leticia, Amazonas is a non-interconnected zone in Colombia that relies heavily on diesel generation to meet its electricity needs. This project develops a complete machine learning pipeline to (1) forecast short-term energy demand and (2) evaluate the feasibility of solar photovoltaic installations as a partial replacement for diesel generation.

### Methodology

A GRU (Gated Recurrent Unit) neural network was trained on ~10 years of daily energy consumption data (2015–2025), incorporating 12 features: active energy demand, cyclic-encoded temporal features (month, day of year, weekday, week of year), reactive energy, surface solar radiation, and temperature. The model uses a 15-day sliding window with autoregressive multi-step forecasting to produce 30-day demand predictions.

Data preprocessing included imputation of five known missing periods via year-ago reference values, IQR-based outlier removal with linear interpolation, and MinMaxScaler normalization. Hyperparameter tuning was performed via grid search over network units (32/64), dropout rates (0.2/0.3), and training configurations, with early stopping on validation loss.

### Key Results

| Metric | Value |
|--------|-------|
| Best model | GRU |
| R² score | ~0.63 |
| Sliding window | 15 days |
| Prediction horizon | 30 days |
| Historical daily demand range | 100,000–150,000 kWh/day |
| Predicted demand (Apr 2025) | 108,000–138,000 kWh/day |

### Solar Scenario Analysis

Three photovoltaic capacity scenarios were simulated against predicted demand (assuming 4.5 kWh/m²/day solar radiation and 0.80 performance ratio):

| Scenario | Capacity | Demand Satisfied | Diesel Displaced | CO2 Reduced | Economic Savings (COP) |
|----------|----------|-----------------|-------------------|-------------|----------------------|
| Small | 100 kW | ~0.3–0.5% | Minimal | Minimal | Minimal |
| Medium | 1 MW | ~3–5% | Moderate | Moderate | Moderate |
| Large | 5 MW | ~15–25% | Significant | Significant | Significant |

Even the largest scenario (5 MW) covers only a fraction of total demand, highlighting that solar PV alone cannot replace diesel generation in this region but can meaningfully reduce fuel consumption, CO2 emissions, and operating costs. A hybrid approach combining solar with other renewable sources or grid interconnection would be needed for full diesel displacement.

### Conclusions

1. **GRU outperforms LSTM** for this time series, achieving R² ~0.63 with a compact architecture.
2. **Demand patterns are seasonal and consistent**, with the model capturing weekly and annual cycles effectively.
3. **Solar PV at 5 MW scale** can displace up to ~25% of diesel demand, delivering tangible environmental and economic benefits.
4. **The interactive dashboard** (deployed at [trillium-watts.onrender.com](https://trillium-watts.onrender.com)) enables stakeholders to explore scenarios with adjustable parameters for radiation, performance ratio, and economic assumptions.

---

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

## Link to web app access
https://trillium-watts.onrender.com

## Team — Samsung Innovation Campus 2024

- Edward Nicolas Diaz Cristancho
- Juliana Utrera Florez
- Laura Rico Aldana
