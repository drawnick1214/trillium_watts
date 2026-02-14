"""Streamlit dashboard — Solar energy simulation for Leticia, Colombia."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path for package imports
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from trillium_watts.config import load_config
from trillium_watts.simulation.solar import calculate_solar_energy
from trillium_watts.simulation.scenarios import simulate_all_scenarios
from trillium_watts.visualization.plots_plotly import (
    create_demand_time_series_figure,
    create_scenario_comparison_figure,
)

# --- Load config ---
config = load_config()

# --- Load predictions data ---
@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Fecha"])
    df = df.sort_values("Fecha")
    # Standardize Tipo labels
    tipo_map = {
        "Histórico": "Historica",
        "Historico": "Historica",
        "historico": "Historica",
        "Historica": "Historica",
        "historica": "Historica",
        "Predicción": "Predicha",
        "Prediccion": "Predicha",
        "prediccion": "Predicha",
        "Predicha": "Predicha",
        "predicha": "Predicha",
    }
    df["Tipo"] = df["Tipo"].map(tipo_map).fillna(df["Tipo"])
    return df


predictions_path = _PROJECT_ROOT / config.data.predictions_path
df = load_predictions(str(predictions_path))

# --- Sidebar: simulation parameters ---
st.sidebar.header("Parametros de Simulacion")

H = st.sidebar.slider(
    "Radiacion Solar H (kWh/m2)",
    1.0, 8.0,
    config.solar.default_h_radiation,
    step=0.1,
)
PR = st.sidebar.slider(
    "Performance Ratio (PR)",
    0.60, 0.95,
    config.solar.default_performance_ratio,
    step=0.01,
)

horizonte = st.sidebar.selectbox(
    "Horizonte de Prediccion",
    ["7 dias", "15 dias", "30 dias"],
)
dias = int(horizonte.split()[0])

st.sidebar.subheader("Escenarios de Capacidad")
escenarios_selection = {}
for name in config.solar.scenarios:
    escenarios_selection[name] = st.sidebar.checkbox(name, value=True)

st.sidebar.subheader("Parametros Economicos")
kwh_por_litro = st.sidebar.number_input("kWh/L (Diesel)", value=config.economic.kwh_per_liter_diesel, step=0.1)
co2_por_litro = st.sidebar.number_input("CO2/L (kg)", value=config.economic.co2_per_liter_diesel, step=0.1)
cop_diesel = st.sidebar.number_input("COP/L (Precio diesel)", value=config.economic.diesel_price_cop, step=10.0)

# --- Filter predicted data ---
df_predicha = df[df["Tipo"] == "Predicha"].copy()
df_sim = df_predicha.head(dias)

# --- Title ---
st.title("Simulacion de Energia Solar - Leticia, Colombia")

# --- Demand time series chart ---
st.subheader("Demanda Energetica (Historica + Predicha)")

viz = config.visualization
fig_ts = create_demand_time_series_figure(
    df,
    color_historical=viz.colors["historical"],
    color_predicted=viz.colors["predicted"],
)
st.plotly_chart(fig_ts, use_container_width=True)

# --- Scenario simulation ---
st.subheader("Beneficios por Escenario Solar")
st.markdown(f"Calculo acumulado de beneficios para un horizonte de **{dias} dias**.")

active_scenarios = {
    name: kw
    for name, kw in config.solar.scenarios.items()
    if escenarios_selection.get(name, False)
}

if not active_scenarios:
    st.warning("Selecciona al menos un escenario solar.")
else:
    demand_array = df_sim["ACTIVA"].values

    # Metrics per scenario
    for name, capacity_kw in active_scenarios.items():
        daily_solar = calculate_solar_energy(capacity_kw, H, PR)
        energy_total = daily_solar * dias
        diesel_saved = energy_total / kwh_por_litro
        co2_avoided = diesel_saved * co2_por_litro
        savings = diesel_saved * cop_diesel

        with st.expander(f"Resultados para {name}"):
            st.metric(f"{name} - Energia Generada Total", f"{energy_total:,.0f} kWh")
            st.metric(f"{name} - Diesel Ahorrado", f"{diesel_saved:,.0f} L")
            st.metric(f"{name} - CO2 Evitado", f"{co2_avoided:,.0f} kg")
            st.metric(f"{name} - Ahorro Economico", f"${savings:,.0f} COP")

    # Summary table
    df_summary = simulate_all_scenarios(
        scenarios=active_scenarios,
        demand_array=demand_array,
        num_days=dias,
        h_radiation=H,
        performance_ratio=PR,
        kwh_per_liter=kwh_por_litro,
        co2_per_liter=co2_por_litro,
        diesel_price_cop=cop_diesel,
    )

    # Comparison chart
    st.subheader("Comparativa de Beneficios por Escenario")
    chart_cols = [
        "Escenario",
        "Generacion Solar Total (kWh)",
        "Litros Diesel Ahorrados",
        "Reduccion CO2 (kg)",
        "Ahorro Economico (COP)",
    ]
    fig_bar = create_scenario_comparison_figure(df_summary[chart_cols])
    st.plotly_chart(fig_bar, use_container_width=True)

    # Summary data table
    st.subheader("Tabla Resumen")
    st.dataframe(df_summary, use_container_width=True)

# --- Footer ---
st.caption(
    "Datos de demanda para Leticia, Amazonas. "
    "Parametros personalizables para analisis energetico, economico y ambiental."
)
