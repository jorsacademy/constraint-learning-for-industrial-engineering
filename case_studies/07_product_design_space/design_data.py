"""Synthetic product-design experiments for constraint learning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_product_design_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate nonlinear product-design simulation outcomes."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(random_state)

    wall_thickness = rng.uniform(1.2, 8.0, n_samples)
    material_strength = rng.uniform(250.0, 900.0, n_samples)
    rib_ratio = rng.uniform(0.04, 0.35, n_samples)
    tolerance = rng.uniform(0.02, 0.30, n_samples)
    mass = rng.uniform(5.0, 30.0, n_samples)

    stress = (
        320.0
        / (
            wall_thickness
            * np.sqrt(material_strength / 300.0)
            * (1.0 + 1.8 * rib_ratio)
        )
        + rng.normal(0.0, 3.0, n_samples)
    )
    deformation = (
        10.0 / (wall_thickness * (1.0 + 2.5 * rib_ratio))
        + rng.normal(0.0, 0.15, n_samples)
    )
    peak_temperature = (
        55.0
        + 55.0 / (wall_thickness + 0.5)
        + 18.0 * (1.0 - rib_ratio)
        + rng.normal(0.0, 2.0, n_samples)
    )
    design_cost = (
        2.2 * mass
        + 0.045 * material_strength
        + 12.0 / (tolerance + 0.03)
        + 35.0 * rib_ratio
    )

    operational_feasible = (
        (stress <= 85.0)
        & (deformation <= 2.5)
        & (peak_temperature <= 92.0)
        & (design_cost <= 155.0)
    )
    hard_design_compliant = (tolerance >= 0.05) & (mass <= 26.0)

    return pd.DataFrame(
        {
            "wall_thickness": wall_thickness,
            "material_strength": material_strength,
            "rib_ratio": rib_ratio,
            "tolerance": tolerance,
            "mass": mass,
            "stress": stress,
            "deformation": deformation,
            "peak_temperature": peak_temperature,
            "design_cost": design_cost,
            "operational_feasible": operational_feasible.astype(int),
            "hard_design_compliant": hard_design_compliant.astype(int),
        }
    )
