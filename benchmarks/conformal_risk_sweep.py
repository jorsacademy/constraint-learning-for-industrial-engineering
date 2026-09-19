"""Sweep conformal risk levels for the manufacturing benchmark."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning import (  # noqa: E402
    ManufacturingConstraintLearner,
    generate_manufacturing_data,
)


def main() -> None:
    data = generate_manufacturing_data(n_samples=10000, random_state=42)
    learner = ManufacturingConstraintLearner(
        data,
        random_state=42,
        label_mode="physical_feasibility",
    ).fit_feasibility_classifier()

    temperatures = np.linspace(150.0, 350.0, 81)
    pressures = np.linspace(2.0, 8.0, 61)
    tt, pp = np.meshgrid(temperatures, pressures)
    candidates = pd.DataFrame(
        {"temperature": tt.ravel(), "pressure": pp.ravel()}
    )

    rows = []
    for alpha in (0.20, 0.10, 0.05, 0.02):
        if learner.safety_filter is None:
            raise RuntimeError("Safety filter was not fitted")
        if alpha < learner.safety_filter.minimum_attainable_p_value:
            continue

        evaluation = learner.evaluate_safety_filter(alpha=alpha)
        optimizer = learner.risk_controlled_optimizer(alpha=alpha)
        solution = optimizer.optimize(
            candidates,
            objective=lambda frame: (
                0.02 * frame["temperature"] + 0.60 * frame["pressure"]
            ),
            hard_constraint=lambda frame: (
                frame["temperature"].between(150.0, 350.0)
                & frame["pressure"].between(2.0, 8.0)
            ),
            maximize=False,
        )
        true_feasible = learner.true_physical_feasibility(
            np.array([solution.point["temperature"]]),
            np.array([solution.point["pressure"]]),
        )
        rows.append(
            {
                "alpha": alpha,
                "threshold": evaluation.probability_threshold,
                "heldout_false_feasible": evaluation.false_feasible_rate,
                "heldout_false_infeasible": evaluation.false_infeasible_rate,
                "accepted_fraction": evaluation.accepted_fraction,
                "selected_temperature": solution.point["temperature"],
                "selected_pressure": solution.point["pressure"],
                "selected_objective": solution.objective_value,
                "selected_true_feasible": bool(true_feasible[0]),
            }
        )

    result = pd.DataFrame(rows)
    print("Conformal risk / conservatism sweep")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
