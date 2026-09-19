"""Synthetic multi-product line-balancing observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_line_balancing_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(random_state)

    product_mix_a = rng.uniform(0.0, 1.0, n_samples)
    cycle_time_variability = rng.uniform(0.03, 0.45, n_samples)
    staffing_level = rng.integers(5, 26, n_samples)
    buffer_capacity = rng.uniform(5.0, 70.0, n_samples)
    changeover_frequency = rng.uniform(0.10, 4.0, n_samples)
    target_throughput = rng.uniform(40.0, 180.0, n_samples)

    mix_complexity = 4.0 * product_mix_a * (1.0 - product_mix_a)
    mix_penalty = 1.0 - 0.16 * mix_complexity
    effective_capacity = (
        staffing_level
        * (8.2 - 2.8 * cycle_time_variability)
        * mix_penalty
        * (1.0 - 0.035 * changeover_frequency)
        * (0.94 + 0.06 * np.minimum(buffer_capacity / 35.0, 1.0))
    )
    utilization = target_throughput / (effective_capacity + 1e-9)
    throughput_ratio = (
        np.minimum(1.0, effective_capacity / target_throughput)
        * (
            1.0
            - 0.08 * cycle_time_variability
            - 0.025 * changeover_frequency / (1.0 + buffer_capacity / 20.0)
        )
    )
    wip = (
        np.maximum(utilization - 0.75, 0.0) * target_throughput * 0.55
        + np.maximum(changeover_frequency - 2.0, 0.0) * 3.0
    )
    idle_fraction = np.clip(1.0 - utilization, 0.0, 1.0)
    resource_cost = (
        50.0 * staffing_level
        + 2.0 * buffer_capacity
        + 10.0 * changeover_frequency
    )

    stable_operation = (
        (throughput_ratio >= 0.92)
        & (utilization <= 1.00)
        & (wip <= 35.0)
        & (idle_fraction <= 0.38)
    )
    hard_capacity_compliant = (
        (staffing_level <= 22)
        & (buffer_capacity <= 60.0)
        & (changeover_frequency <= 3.5)
    )

    return pd.DataFrame(
        {
            "product_mix_a": product_mix_a,
            "cycle_time_variability": cycle_time_variability,
            "staffing_level": staffing_level,
            "buffer_capacity": buffer_capacity,
            "changeover_frequency": changeover_frequency,
            "target_throughput": target_throughput,
            "effective_capacity": effective_capacity,
            "utilization": utilization,
            "throughput_ratio": throughput_ratio,
            "wip": wip,
            "idle_fraction": idle_fraction,
            "resource_cost": resource_cost,
            "stable_operation": stable_operation.astype(int),
            "hard_capacity_compliant": hard_capacity_compliant.astype(int),
        }
    )
