# TrilliumWatts — R3newAI

Energy demand forecasting and solar energy simulation for **Leticia, Amazonas, Colombia** (Non-Interconnected Zone).

This project uses Deep Learning (LSTM/GRU) to predict daily electrical energy demand, then simulates the economic and environmental impact of photovoltaic solar installations across different capacity scenarios.

## Executive Summary

### 1. Context

Leticia is the capital of the Amazonas department in southern Colombia, located at the triple border with Brazil and Peru. As a **Non-Interconnected Zone (ZNI)**, it is not connected to Colombia's national electrical grid (Sistema Interconectado Nacional). The city depends almost entirely on **diesel-powered thermal generation** to supply electricity to its ~50,000 inhabitants, making its energy supply expensive, logistically constrained, and environmentally costly. Diesel fuel must be transported by river or air into one of the most biodiverse regions on Earth, where every liter burned contributes to CO2 emissions and local air pollution in the heart of the Amazon rainforest.

### 2. Problem

The reliance on diesel generation in Leticia creates three compounding challenges:

- **Economic**: Diesel fuel costs are high and volatile, with prices inflated by remote transportation logistics. At COP ~2,554/liter, electricity generation costs far exceed the national average.
- **Environmental**: Each liter of diesel emits ~2.20 kg of CO2. In a region whose ecological value is global, continued fossil fuel dependence contradicts Colombia's climate commitments and the Amazon's conservation priorities.
- **Planning**: Without accurate short-term demand forecasting, energy operators cannot efficiently plan fuel procurement, generation schedules, or evaluate renewable energy alternatives with confidence.

There is a clear need for data-driven tools that can both **predict energy demand** and **quantify the impact of transitioning to renewable sources** — enabling informed decision-making for Leticia's energy future.

### 3. Proposed Solution

TrilliumWatts delivers an end-to-end machine learning pipeline with two core capabilities:

**A. Demand Forecasting** — A GRU (Gated Recurrent Unit) deep learning model trained on ~10 years of daily energy data (2015–2025), using 12 features including active energy consumption, cyclic-encoded temporal patterns (month, day of year, weekday, week of year), reactive energy, surface solar radiation, and temperature. The model employs a 15-day sliding window with autoregressive multi-step forecasting to produce predictions up to 30 days ahead.

**B. Solar PV Simulation** — An economic and environmental impact simulator that evaluates three photovoltaic installation scenarios (100 kW, 1 MW, and 5 MW) against forecasted demand. For each scenario, it calculates solar energy generation, diesel displacement, CO2 emission reductions, and cost savings in COP — using configurable parameters for solar radiation, performance ratio, and fuel economics.

Both capabilities are integrated into an **interactive Streamlit dashboard** ([trillium-watts.onrender.com](https://trillium-watts.onrender.com)) that allows stakeholders to explore scenarios, adjust assumptions, and visualize results in real time.

### 4. Key Indicators

#### Model Performance

| Indicator | Value |
|-----------|-------|
| Best architecture | GRU |
| R² score (test set) | ~0.63 |
| Sliding window size | 15 days |
| Number of features | 12 |
| Prediction horizon | Up to 30 days |
| Historical daily demand | 100,000–150,000 kWh/day |
| Predicted demand (Apr 2025) | 108,000–138,000 kWh/day |

#### Solar Scenario Comparison

Simulated with default parameters: 4.5 kWh/m²/day solar radiation, 0.80 performance ratio, COP 2,553.59/L diesel.

| Scenario | Capacity | Demand Satisfied | Diesel Displaced | CO2 Reduction | Economic Savings |
|----------|----------|-----------------|-------------------|---------------|-----------------|
| Small | 100 kW | ~0.3–0.5% | Minimal | Minimal | Minimal |
| Medium | 1 MW | ~3–5% | Moderate | Moderate | Moderate |
| Large | 5 MW | ~15–25% | Significant | Significant | Significant |

#### Key Takeaways

1. **GRU outperforms LSTM** for this time series, achieving the best R² with a compact architecture and lower computational cost.
2. **Demand follows stable seasonal and weekly patterns**, which the model captures effectively — predicted values for April 2025 are consistent with historical ranges.
3. **A 5 MW solar installation could satisfy up to ~25% of daily demand**, delivering meaningful diesel displacement, CO2 reductions, and economic savings — but solar PV alone cannot fully replace diesel generation.
4. **A hybrid energy strategy** combining solar with other renewables or future grid interconnection would be necessary for full decarbonization of Leticia's electricity supply.

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
