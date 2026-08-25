"""Synthetic data generator for the energy-efficient machine settings case study."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_energy_efficiency_data(
    n_samples: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic machine-operation data set.

    The simulated machine is controlled by spindle speed, feed rate, and load.
    Operational outcomes include energy per unit, throughput, and quality.

    The hidden acceptable-operation label requires all of the following:

    - equipment stability constraints are satisfied,
    - energy per unit <= 2.00 kWh/unit,
    - throughput >= 78 units/hour,
    - quality score >= 93.

    The nonlinear stability envelope is intentionally hidden from the learner and
    is retained only for benchmark evaluation.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)

    spindle_speed = rng.uniform(1000.0, 5000.0, n_samples)
    feed_rate = rng.uniform(50.0, 500.0, n_samples)
    load = rng.uniform(0.20, 1.00, n_samples)

    speed_scaled = (spindle_speed - 3000.0) / 1500.0
    feed_scaled = (feed_rate - 275.0) / 150.0
    load_scaled = (load - 0.60) / 0.25

    energy_per_unit = (
        1.75
        + 0.28 * speed_scaled**2
        + 0.18 * load_scaled**2
        + 0.12 * (feed_scaled - 0.25) ** 2
        - 0.18 * feed_scaled
        + 0.06 * speed_scaled * load_scaled
        + rng.normal(0.0, 0.06, n_samples)
    )

    throughput = (
        78.0
        + 12.0 * feed_scaled
        + 6.0 * speed_scaled
        + 7.0 * load_scaled
        - 4.0 * speed_scaled**2
        + rng.normal(0.0, 2.5, n_samples)
    )

    quality_score = (
        96.5
        - 1.8 * speed_scaled**2
        - 1.6 * (feed_scaled - 0.15) ** 2
        - 2.2 * (load_scaled - 0.10) ** 2
        + rng.normal(0.0, 0.8, n_samples)
    )

    stable_operation = (
        (spindle_speed >= 1500.0)
        & (spindle_speed <= 4600.0)
        & (load <= 0.92)
        & (
            (speed_scaled / 1.10) ** 2
            + ((load - 0.55) / 0.42) ** 2
            <= 1.60
        )
    )

    acceptable_operation = (
        stable_operation
        & (energy_per_unit <= 2.00)
        & (throughput >= 78.0)
        & (quality_score >= 93.0)
    )

    return pd.DataFrame(
        {
            "spindle_speed": spindle_speed,
            "feed_rate": feed_rate,
            "load": load,
            "energy_per_unit": energy_per_unit,
            "throughput": throughput,
            "quality_score": quality_score,
            "stable_operation": stable_operation.astype(int),
            "acceptable_operation": acceptable_operation.astype(int),
        }
    )
