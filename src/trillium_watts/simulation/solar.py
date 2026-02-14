"""Solar energy generation calculation."""

from __future__ import annotations


def calculate_solar_energy(
    capacity_kw: float,
    h_radiation: float,
    performance_ratio: float,
) -> float:
    """Calculate daily solar energy generation in kWh.

    E = Pr * H * PR

    Args:
        capacity_kw: Nominal system power (kW).
        h_radiation: Mean daily solar radiation (kWh/m2/day).
        performance_ratio: System performance ratio (0.6-0.95).

    Returns:
        Daily energy generated in kWh.
    """
    return capacity_kw * h_radiation * performance_ratio
