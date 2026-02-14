"""Scenario management and simulation loop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trillium_watts.simulation.economics import calculate_diesel_savings
from trillium_watts.simulation.solar import calculate_solar_energy


def get_active_scenarios(
    all_scenarios: dict[str, int],
    selected: dict[str, bool],
) -> dict[str, int]:
    """Filter scenarios by user selection."""
    return {name: kw for name, kw in all_scenarios.items() if selected.get(name, False)}


def simulate_all_scenarios(
    scenarios: dict[str, int],
    demand_array: np.ndarray,
    num_days: int,
    h_radiation: float,
    performance_ratio: float,
    kwh_per_liter: float,
    co2_per_liter: float,
    diesel_price_cop: float,
) -> pd.DataFrame:
    """Run simulation for all active scenarios and return a summary DataFrame."""
    rows = []
    for name, capacity_kw in scenarios.items():
        daily_solar = calculate_solar_energy(capacity_kw, h_radiation, performance_ratio)
        savings = calculate_diesel_savings(
            daily_solar, demand_array, kwh_per_liter, co2_per_liter, diesel_price_cop
        )

        total_demand = float(demand_array.sum())
        satisfaction_pct = (savings["solar_energy_used_kwh"] / total_demand * 100) if total_demand > 0 else 0.0

        rows.append(
            {
                "Escenario": name,
                "Demanda Total Predicha (kWh)": total_demand,
                "Generacion Solar Total (kWh)": daily_solar * num_days,
                "Capacidad Satisfaccion Demanda (%)": satisfaction_pct,
                "Litros Diesel Ahorrados": savings["diesel_saved_liters"],
                "Ahorro Economico (COP)": savings["economic_savings_cop"],
                "Reduccion CO2 (kg)": savings["co2_avoided_kg"],
            }
        )

    return pd.DataFrame(rows)
