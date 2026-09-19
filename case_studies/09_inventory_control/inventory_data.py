"""Synthetic inventory-policy observations for constraint learning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_inventory_policy_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(random_state)

    reorder_point = rng.uniform(40.0, 500.0, n_samples)
    order_quantity = rng.uniform(30.0, 360.0, n_samples)
    lead_time = rng.uniform(0.5, 6.0, n_samples)
    demand_mean = rng.uniform(40.0, 220.0, n_samples)
    demand_cv = rng.uniform(0.10, 0.65, n_samples)
    safety_stock = rng.uniform(0.0, 200.0, n_samples)

    cycle_demand = demand_mean * lead_time
    demand_sigma = demand_mean * demand_cv * np.sqrt(lead_time)
    z_score = (reorder_point + safety_stock - cycle_demand) / (
        demand_sigma + 1e-9
    )
    service_level = 1.0 / (1.0 + np.exp(-1.5 * z_score))
    stockout_frequency = np.clip(
        1.0 - service_level + rng.normal(0.0, 0.015, n_samples),
        0.0,
        1.0,
    )
    holding_cost = 0.08 * (
        reorder_point + safety_stock + 0.5 * order_quantity
    )
    replenishment_instability = np.abs(order_quantity - demand_mean) / (
        demand_mean + 1e-9
    )
    total_cost = (
        holding_cost
        + 220.0 * stockout_frequency
        + 0.03 * order_quantity
    )

    operationally_acceptable = (
        (service_level >= 0.90)
        & (stockout_frequency <= 0.12)
        & (holding_cost <= 50.0)
        & (replenishment_instability <= 1.20)
    )
    hard_policy_compliant = (
        (order_quantity <= 320.0)
        & ((reorder_point + safety_stock) <= 600.0)
    )

    return pd.DataFrame(
        {
            "reorder_point": reorder_point,
            "order_quantity": order_quantity,
            "lead_time": lead_time,
            "demand_mean": demand_mean,
            "demand_cv": demand_cv,
            "safety_stock": safety_stock,
            "service_level": service_level,
            "stockout_frequency": stockout_frequency,
            "holding_cost": holding_cost,
            "replenishment_instability": replenishment_instability,
            "total_cost": total_cost,
            "operationally_acceptable": operationally_acceptable.astype(int),
            "hard_policy_compliant": hard_policy_compliant.astype(int),
        }
    )
