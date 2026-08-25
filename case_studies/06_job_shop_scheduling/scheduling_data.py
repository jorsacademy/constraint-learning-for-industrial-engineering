"""Synthetic job-shop scheduling data for constraint-learning demonstrations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_job_shop_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic scheduling episodes with hidden feasibility rules.

    Each row summarizes a scheduling scenario rather than an individual job.
    Feasibility requires acceptable tardiness, overtime, and work-in-process plus
    a hidden nonlinear stability condition that captures workload interactions.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)

    job_count = rng.integers(20, 121, n_samples)
    due_date_tightness = rng.uniform(0.65, 1.45, n_samples)
    machine_utilization = rng.uniform(0.50, 1.08, n_samples)
    processing_time_cv = rng.uniform(0.05, 0.75, n_samples)
    setup_time_ratio = rng.uniform(0.02, 0.35, n_samples)
    machine_availability = rng.uniform(0.78, 1.00, n_samples)

    overload = np.maximum(machine_utilization - machine_availability, 0.0)
    instability_score = (
        1.65
        - 1.8 * overload
        - 1.1 * processing_time_cv
        - 1.5 * setup_time_ratio
        - 1.15 * np.maximum(0.95 - due_date_tightness, 0.0)
        - 0.0028 * np.maximum(job_count - 80, 0) ** 1.2
        + 0.9 * (machine_availability - 0.80)
        - 0.65 * processing_time_cv * setup_time_ratio
    )
    hidden_stable = instability_score >= 0.0

    mean_tardiness = (
        1.5
        + 16.0 * overload
        + 5.5 * processing_time_cv
        + 9.0 * setup_time_ratio
        + 7.0 * np.maximum(0.95 - due_date_tightness, 0.0)
        + 0.035 * np.maximum(job_count - 55, 0)
        - 5.0 * (machine_availability - 0.80)
        + rng.normal(0.0, 0.9, n_samples)
    )
    mean_tardiness = np.maximum(mean_tardiness, 0.0)

    overtime_hours = (
        0.8
        + 18.0 * overload
        + 0.055 * np.maximum(job_count - 45, 0)
        + 5.0 * setup_time_ratio
        + 2.5 * processing_time_cv
        + rng.normal(0.0, 0.8, n_samples)
    )
    overtime_hours = np.maximum(overtime_hours, 0.0)

    wip_level = (
        8.0
        + 0.22 * job_count
        + 28.0 * overload
        + 8.0 * processing_time_cv
        + 10.0 * setup_time_ratio
        + rng.normal(0.0, 2.2, n_samples)
    )
    wip_level = np.maximum(wip_level, 0.0)

    feasible_schedule = (
        hidden_stable
        & (mean_tardiness <= 6.0)
        & (overtime_hours <= 7.5)
        & (wip_level <= 38.0)
    )

    return pd.DataFrame(
        {
            "job_count": job_count.astype(float),
            "due_date_tightness": due_date_tightness,
            "machine_utilization": machine_utilization,
            "processing_time_cv": processing_time_cv,
            "setup_time_ratio": setup_time_ratio,
            "machine_availability": machine_availability,
            "mean_tardiness": mean_tardiness,
            "overtime_hours": overtime_hours,
            "wip_level": wip_level,
            "hidden_stability_score": instability_score,
            "feasible_schedule": feasible_schedule.astype(int),
        }
    )
