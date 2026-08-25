"""Synthetic supply-chain data for constraint-learning demonstrations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_supply_chain_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic order-cycle observations with hidden feasibility rules.

    Features describe demand uncertainty, replenishment, capacity, and transport.
    The target ``feasible_service`` is derived from simultaneous service and cost
    requirements plus a hidden nonlinear stability envelope.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)

    supplier_lead_time = rng.uniform(2.0, 18.0, n_samples)
    demand_cv = rng.uniform(0.05, 0.65, n_samples)
    order_quantity = rng.uniform(150.0, 1800.0, n_samples)
    safety_stock = rng.uniform(20.0, 600.0, n_samples)
    supplier_utilization = rng.uniform(0.45, 1.05, n_samples)
    transport_time = rng.uniform(0.5, 8.0, n_samples)

    demand_rate = 95.0 + 35.0 * demand_cv + rng.normal(0.0, 4.0, n_samples)
    coverage_days = (order_quantity + safety_stock) / np.maximum(demand_rate, 1.0)

    hidden_stability_score = (
        1.25
        - 0.055 * supplier_lead_time
        - 1.25 * demand_cv
        - 1.55 * np.maximum(supplier_utilization - 0.78, 0.0) ** 2
        - 0.045 * transport_time
        + 0.055 * np.sqrt(np.maximum(safety_stock, 0.0))
        - 0.0025 * np.maximum(order_quantity - 1250.0, 0.0)
    )
    hidden_stable = hidden_stability_score >= 0.0

    service_level = (
        0.995
        - 0.010 * supplier_lead_time
        - 0.18 * demand_cv
        - 0.20 * np.maximum(supplier_utilization - 0.82, 0.0)
        - 0.008 * transport_time
        + 0.00016 * safety_stock
        + 0.003 * np.minimum(coverage_days, 10.0)
        + rng.normal(0.0, 0.012, n_samples)
    )
    service_level = np.clip(service_level, 0.0, 1.0)

    holding_cost = 0.055 * (order_quantity / 2.0 + safety_stock)
    expediting_cost = 40.0 * np.maximum(supplier_utilization - 0.80, 0.0)
    uncertainty_cost = 85.0 * demand_cv**1.35
    lead_time_cost = 2.2 * supplier_lead_time + 1.8 * transport_time
    total_logistics_cost = (
        120.0 + holding_cost + expediting_cost + uncertainty_cost + lead_time_cost
        + rng.normal(0.0, 8.0, n_samples)
    )
    total_logistics_cost = np.maximum(total_logistics_cost, 0.0)

    feasible_service = (
        hidden_stable
        & (service_level >= 0.90)
        & (total_logistics_cost <= 265.0)
    )

    return pd.DataFrame(
        {
            "supplier_lead_time": supplier_lead_time,
            "demand_cv": demand_cv,
            "order_quantity": order_quantity,
            "safety_stock": safety_stock,
            "supplier_utilization": supplier_utilization,
            "transport_time": transport_time,
            "service_level": service_level,
            "total_logistics_cost": total_logistics_cost,
            "hidden_stability_score": hidden_stability_score,
            "feasible_service": feasible_service.astype(int),
        }
    )
