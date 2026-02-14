"""Tests for simulation modules."""

import numpy as np
import pytest

from trillium_watts.simulation.solar import calculate_solar_energy
from trillium_watts.simulation.economics import calculate_diesel_savings
from trillium_watts.simulation.scenarios import simulate_all_scenarios


def test_calculate_solar_energy():
    # 100 kW * 4.5 kWh/m2 * 0.80 = 360 kWh
    result = calculate_solar_energy(100, 4.5, 0.80)
    assert abs(result - 360.0) < 0.01


def test_calculate_solar_energy_zero_capacity():
    assert calculate_solar_energy(0, 4.5, 0.80) == 0.0


def test_calculate_diesel_savings():
    demand = np.array([500.0, 500.0, 500.0])
    result = calculate_diesel_savings(
        solar_energy_daily_kwh=360.0,
        demand_kwh_array=demand,
        kwh_per_liter=3.0,
        co2_per_liter=2.20,
        diesel_price_cop=2553.59,
    )
    # Solar used = min(360, 500) * 3 days = 1080 kWh
    assert abs(result["solar_energy_used_kwh"] - 1080.0) < 0.01
    # Diesel saved = 1080 / 3 = 360 L
    assert abs(result["diesel_saved_liters"] - 360.0) < 0.01
    # CO2 = 360 * 2.20 = 792 kg
    assert abs(result["co2_avoided_kg"] - 792.0) < 0.01


def test_calculate_diesel_savings_demand_lower_than_solar():
    demand = np.array([100.0, 100.0])
    result = calculate_diesel_savings(
        solar_energy_daily_kwh=500.0,
        demand_kwh_array=demand,
        kwh_per_liter=3.0,
        co2_per_liter=2.20,
        diesel_price_cop=2553.59,
    )
    # Solar used = min(500, 100) * 2 = 200 kWh (capped at demand)
    assert abs(result["solar_energy_used_kwh"] - 200.0) < 0.01


def test_simulate_all_scenarios():
    scenarios = {"Small": 100, "Medium": 1000}
    demand = np.array([500.0, 500.0, 500.0])
    result = simulate_all_scenarios(
        scenarios=scenarios,
        demand_array=demand,
        num_days=3,
        h_radiation=4.5,
        performance_ratio=0.80,
        kwh_per_liter=3.0,
        co2_per_liter=2.20,
        diesel_price_cop=2553.59,
    )
    assert len(result) == 2
    assert "Escenario" in result.columns
    assert result.iloc[0]["Escenario"] == "Small"
    assert result.iloc[1]["Escenario"] == "Medium"
