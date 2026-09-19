"""Compare candidate search, MILP, and CP-SAT learned-constraint optimization."""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning import (  # noqa: E402
    LinearConstraintSpec,
    ManufacturingConstraintLearner,
    RiskControlledTreeSurrogate,
    SurrogateCPSATOptimizer,
    SurrogateMILPOptimizer,
    generate_manufacturing_data,
    sample_uniform_design,
)


def main() -> None:
    alpha = 0.10
    data = generate_manufacturing_data(n_samples=10000, random_state=42)
    learner = ManufacturingConstraintLearner(
        data,
        random_state=42,
        label_mode="physical_feasibility",
    ).fit_feasibility_classifier()

    bounds = {
        "temperature": (150.0, 350.0),
        "pressure": (2.0, 8.0),
    }
    design = sample_uniform_design(bounds, 15000, random_state=43)
    surrogate = RiskControlledTreeSurrogate(
        learner.feature_columns,
        max_depth=6,
        min_samples_leaf=15,
        random_state=43,
    ).fit_from_teacher(learner, design, alpha=alpha)

    temperatures = np.linspace(150.0, 350.0, 161)
    pressures = np.linspace(2.0, 8.0, 121)
    tt, pp = np.meshgrid(temperatures, pressures)
    fallback = pd.DataFrame(
        {"temperature": tt.ravel(), "pressure": pp.ravel()}
    )

    hard_constraints = [
        LinearConstraintSpec([1.0, 0.0], lower_bound=180.0),
        LinearConstraintSpec([0.0, 1.0], lower_bound=2.5),
    ]
    objective = [0.02, 0.60]

    rows = []

    start = perf_counter()
    candidate_optimizer = learner.risk_controlled_optimizer(alpha=alpha)
    candidate = candidate_optimizer.optimize(
        fallback,
        objective=lambda frame: (
            objective[0] * frame["temperature"]
            + objective[1] * frame["pressure"]
        ),
        hard_constraint=lambda frame: (
            (frame["temperature"] >= 180.0)
            & (frame["pressure"] >= 2.5)
        ),
        maximize=False,
    )
    rows.append(
        {
            "backend": "candidate_search",
            "objective": candidate.objective_value,
            "temperature": candidate.point["temperature"],
            "pressure": candidate.point["pressure"],
            "teacher_probability": candidate.feasibility_probability,
            "conformal_p": float(
                learner.conformal_p_values(
                    pd.DataFrame([candidate.point[["temperature", "pressure"]]])
                )[0]
            ),
            "audited_safe": True,
            "fallback": False,
            "seconds": perf_counter() - start,
        }
    )

    start = perf_counter()
    milp_result = SurrogateMILPOptimizer(
        learner,
        surrogate,
        bounds,
        alpha=alpha,
    ).optimize(
        objective,
        hard_constraints=hard_constraints,
        maximize=False,
        fallback_candidates=fallback,
    )
    rows.append(
        {
            "backend": "milp_highs",
            "objective": milp_result.objective_value,
            "temperature": milp_result.point["temperature"],
            "pressure": milp_result.point["pressure"],
            "teacher_probability": milp_result.teacher_probability,
            "conformal_p": milp_result.conformal_p_value,
            "audited_safe": milp_result.audited_safe,
            "fallback": milp_result.used_fallback,
            "seconds": perf_counter() - start,
        }
    )

    start = perf_counter()
    cpsat_result = SurrogateCPSATOptimizer(
        learner,
        surrogate,
        bounds,
        feature_scales={"temperature": 10, "pressure": 100},
        alpha=alpha,
    ).optimize(
        objective,
        hard_constraints=hard_constraints,
        maximize=False,
        fallback_candidates=fallback,
        num_search_workers=1,
    )
    rows.append(
        {
            "backend": "cp_sat",
            "objective": cpsat_result.objective_value,
            "temperature": cpsat_result.point["temperature"],
            "pressure": cpsat_result.point["pressure"],
            "teacher_probability": cpsat_result.teacher_probability,
            "conformal_p": cpsat_result.conformal_p_value,
            "audited_safe": cpsat_result.audited_safe,
            "fallback": cpsat_result.used_fallback,
            "seconds": perf_counter() - start,
        }
    )

    print("Tree-surrogate fidelity:")
    print(surrogate.fidelity_)
    print("\nSolver comparison:")
    print(
        pd.DataFrame(rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )


if __name__ == "__main__":
    main()
