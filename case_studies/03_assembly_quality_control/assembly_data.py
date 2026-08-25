"""Synthetic assembly-process data for constraint-learning experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_assembly_quality_data(
    n_samples: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible assembly quality-control data set.

    The synthetic process contains five controllable or observable variables:
    tightening torque, tightening angle, tool speed, insertion force, and
    component temperature. A hidden nonlinear stability surface determines
    whether a setting combination can reliably produce an acceptable joint.

    End-of-line quality metrics are generated from the same latent process but
    include measurement and process noise. The final ``acceptable_operation``
    label requires all engineering requirements to be satisfied.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)

    torque = rng.uniform(20.0, 80.0, n_samples)
    angle = rng.uniform(30.0, 150.0, n_samples)
    tool_speed = rng.uniform(100.0, 500.0, n_samples)
    insertion_force = rng.uniform(100.0, 500.0, n_samples)
    component_temperature = rng.uniform(15.0, 45.0, n_samples)

    z_torque = (torque - 50.0) / 18.0
    z_angle = (angle - 90.0) / 35.0
    z_speed = (tool_speed - 300.0) / 150.0
    z_force = (insertion_force - 300.0) / 140.0
    z_temperature = (component_temperature - 30.0) / 12.0

    hidden_stability_score = (
        z_torque**2
        + 0.80 * z_angle**2
        + 0.45 * z_speed**2
        + 0.35 * z_force**2
        + 0.25 * z_temperature**2
        + 0.35 * z_torque * z_angle
        - 0.20 * z_speed * z_force
    )

    quality_score = np.clip(
        100.0 - 4.0 * hidden_stability_score + rng.normal(0.0, 1.5, n_samples),
        0.0,
        100.0,
    )

    joint_strength = np.clip(
        95.0
        - 8.0 * np.abs(z_torque)
        - 5.0 * np.abs(z_angle)
        - 3.0 * np.abs(z_force)
        + rng.normal(0.0, 2.0, n_samples),
        0.0,
        100.0,
    )

    cycle_time = (
        8.5
        - 0.004 * (tool_speed - 100.0)
        + 0.002 * np.abs(angle - 90.0)
        + rng.normal(0.0, 0.25, n_samples)
    )

    hidden_stable = hidden_stability_score <= 1.35
    acceptable_operation = (
        hidden_stable
        & (quality_score >= 93.0)
        & (joint_strength >= 82.0)
        & (cycle_time <= 8.6)
    )

    return pd.DataFrame(
        {
            "torque": torque,
            "angle": angle,
            "tool_speed": tool_speed,
            "insertion_force": insertion_force,
            "component_temperature": component_temperature,
            "quality_score": quality_score,
            "joint_strength": joint_strength,
            "cycle_time": cycle_time,
            "hidden_stability_score": hidden_stability_score,
            "hidden_stable": hidden_stable.astype(int),
            "acceptable_operation": acceptable_operation.astype(int),
        }
    )
