"""Synthetic warehouse-slotting data for constraint-learning demonstrations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_warehouse_slotting_data(
    n_samples: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic SKU-location observations with hidden feasibility rules.

    The features represent item characteristics and slot-location conditions.
    ``feasible_slotting`` is defined by simultaneous picking-time, congestion,
    ergonomic, and hidden nonlinear stability requirements.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)

    sku_velocity = rng.uniform(5.0, 220.0, n_samples)
    unit_weight = rng.uniform(0.2, 28.0, n_samples)
    cube = rng.uniform(0.002, 0.18, n_samples)
    aisle_distance = rng.uniform(5.0, 140.0, n_samples)
    replenishment_frequency = rng.uniform(0.05, 9.0, n_samples)
    neighboring_pick_density = rng.uniform(0.05, 1.0, n_samples)

    hidden_stability_score = (
        1.40
        - 0.0045 * aisle_distance * np.sqrt(sku_velocity / 100.0)
        - 0.055 * np.maximum(unit_weight - 15.0, 0.0)
        - 2.20 * np.maximum(cube - 0.10, 0.0)
        - 0.065 * replenishment_frequency * neighboring_pick_density
        - 0.55 * np.maximum(neighboring_pick_density - 0.72, 0.0) ** 2
        + 0.18 * np.exp(-((aisle_distance - 35.0) / 35.0) ** 2)
    )
    hidden_stable = hidden_stability_score >= 0.0

    picking_time = (
        28.0
        + 0.24 * aisle_distance
        + 0.055 * sku_velocity
        + 0.90 * np.sqrt(unit_weight)
        + 22.0 * cube
        + 1.40 * replenishment_frequency
        + 18.0 * neighboring_pick_density
        - 0.05 * sku_velocity * np.exp(-aisle_distance / 50.0)
        + rng.normal(0.0, 3.0, n_samples)
    )

    congestion_score = (
        0.12
        + 0.0022 * sku_velocity
        + 0.055 * replenishment_frequency
        + 0.48 * neighboring_pick_density
        + 0.75 * cube
        - 0.0012 * aisle_distance
        + rng.normal(0.0, 0.04, n_samples)
    )
    congestion_score = np.clip(congestion_score, 0.0, 1.5)

    ergonomic_risk = (
        0.08
        + 0.025 * unit_weight
        + 1.60 * cube
        + 0.002 * aisle_distance
        + 0.10 * neighboring_pick_density
        + rng.normal(0.0, 0.03, n_samples)
    )
    ergonomic_risk = np.clip(ergonomic_risk, 0.0, 1.5)

    feasible_slotting = (
        hidden_stable
        & (picking_time <= 78.0)
        & (congestion_score <= 0.85)
        & (ergonomic_risk <= 0.82)
    )

    return pd.DataFrame(
        {
            "sku_velocity": sku_velocity,
            "unit_weight": unit_weight,
            "cube": cube,
            "aisle_distance": aisle_distance,
            "replenishment_frequency": replenishment_frequency,
            "neighboring_pick_density": neighboring_pick_density,
            "picking_time": picking_time,
            "congestion_score": congestion_score,
            "ergonomic_risk": ergonomic_risk,
            "hidden_stability_score": hidden_stability_score,
            "feasible_slotting": feasible_slotting.astype(int),
        }
    )
