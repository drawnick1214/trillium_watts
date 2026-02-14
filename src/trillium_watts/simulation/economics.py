"""Economic and environmental impact calculations."""

from __future__ import annotations

import numpy as np


def calculate_diesel_savings(
    solar_energy_daily_kwh: float,
    demand_kwh_array: np.ndarray,
    kwh_per_liter: float,
    co2_per_liter: float,
    diesel_price_cop: float,
) -> dict[str, float]:
    """Calculate diesel savings, CO2 reduction, and economic savings.

    Solar energy displaces diesel generation up to the daily demand.

    Returns dict with keys:
        diesel_saved_liters, co2_avoided_kg, economic_savings_cop,
        solar_energy_used_kwh
    """
    solar_used = np.minimum(solar_energy_daily_kwh, demand_kwh_array)
    diesel_saved = solar_used / kwh_per_liter
    co2_avoided = diesel_saved * co2_per_liter
    economic_savings = diesel_saved * diesel_price_cop

    return {
        "diesel_saved_liters": float(diesel_saved.sum()),
        "co2_avoided_kg": float(co2_avoided.sum()),
        "economic_savings_cop": float(economic_savings.sum()),
        "solar_energy_used_kwh": float(solar_used.sum()),
    }
