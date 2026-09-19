"""Synthetic workforce-shift observations for constraint learning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_workforce_shift_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(random_state)

    staffing_level = rng.integers(6, 31, n_samples)
    shift_length = rng.uniform(6.0, 12.0, n_samples)
    consecutive_shifts = rng.integers(1, 7, n_samples)
    skill_mix = rng.uniform(0.35, 1.0, n_samples)
    demand_rate = rng.uniform(250.0, 1200.0, n_samples)
    break_coverage = rng.uniform(0.50, 1.0, n_samples)

    capacity = (
        staffing_level
        * shift_length
        * (4.2 + 2.8 * skill_mix)
        * break_coverage
    )
    utilization = demand_rate / (capacity + 1e-9)
    service_level = np.clip(
        1.0
        - 0.75 * np.maximum(utilization - 0.78, 0.0)
        - 0.025 * np.maximum(consecutive_shifts - 3, 0)
        + rng.normal(0.0, 0.02, n_samples),
        0.0,
        1.0,
    )
    overtime_hours = np.maximum(
        np.maximum(utilization - 0.90, 0.0) * staffing_level * shift_length * 0.35
        + np.maximum(consecutive_shifts - 4, 0) * 2.5
        + rng.normal(0.0, 1.0, n_samples),
        0.0,
    )
    workload_index = (
        utilization
        * (
            1.0
            + 0.045 * (shift_length - 8.0)
            + 0.055 * (consecutive_shifts - 2)
        )
        / (0.85 + 0.15 * skill_mix)
    )
    staffing_cost = staffing_level * shift_length * (1.0 + 0.35 * skill_mix)

    operationally_acceptable = (
        (service_level >= 0.90)
        & (overtime_hours <= 20.0)
        & (workload_index <= 1.08)
    )
    policy_compliant = (
        (shift_length <= 11.5)
        & (consecutive_shifts <= 5)
        & (break_coverage >= 0.60)
    )

    return pd.DataFrame(
        {
            "staffing_level": staffing_level,
            "shift_length": shift_length,
            "consecutive_shifts": consecutive_shifts,
            "skill_mix": skill_mix,
            "demand_rate": demand_rate,
            "break_coverage": break_coverage,
            "utilization": utilization,
            "service_level": service_level,
            "overtime_hours": overtime_hours,
            "workload_index": workload_index,
            "staffing_cost": staffing_cost,
            "operationally_acceptable": operationally_acceptable.astype(int),
            "policy_compliant": policy_compliant.astype(int),
        }
    )
