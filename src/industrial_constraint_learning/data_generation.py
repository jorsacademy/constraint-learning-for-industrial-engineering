"""Synthetic manufacturing data for constraint-learning demonstrations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_manufacturing_data(
    n_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic manufacturing process data set.

    The process has known hidden feasibility constraints:

    1. 150 <= temperature <= 350
    2. 2 <= pressure <= 8
    3. pressure <= -0.02 * (temperature - 250)^2 + 8

    Yield is generated separately from feasibility. Feasible points tend to have
    high yield, while infeasible points tend to have low yield. Yield is clipped
    to [0, 100] because it is interpreted as a percentage.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)
    temperature = rng.uniform(50.0, 450.0, n_samples)
    pressure = rng.uniform(0.0, 10.0, n_samples)

    physical_feasible = (
        (temperature >= 150.0)
        & (temperature <= 350.0)
        & (pressure >= 2.0)
        & (pressure <= 8.0)
        & (pressure <= -0.02 * (temperature - 250.0) ** 2 + 8.0)
    )

    product_yield = np.empty(n_samples, dtype=float)

    feasible_idx = np.where(physical_feasible)[0]
    infeasible_idx = np.where(~physical_feasible)[0]

    if feasible_idx.size:
        t = temperature[feasible_idx]
        p = pressure[feasible_idx]
        distance = np.sqrt(((t - 250.0) / 100.0) ** 2 + ((p - 6.0) / 3.0) ** 2)
        deterministic_yield = 85.0 + (1.0 - np.minimum(distance, 1.0)) * 15.0
        product_yield[feasible_idx] = deterministic_yield + rng.normal(
            0.0, 2.0, feasible_idx.size
        )

    if infeasible_idx.size:
        product_yield[infeasible_idx] = rng.normal(30.0, 10.0, infeasible_idx.size)

    product_yield = np.clip(product_yield, 0.0, 100.0)

    return pd.DataFrame(
        {
            "temperature": temperature,
            "pressure": pressure,
            "yield": product_yield,
            "physical_feasible": physical_feasible.astype(int),
        }
    )
