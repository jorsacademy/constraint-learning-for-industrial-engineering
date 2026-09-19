"""Tests for CP-SAT learned-constraint embedding."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning.cpsat_embedding import (  # noqa: E402
    SurrogateCPSATOptimizer,
)
from industrial_constraint_learning.milp_embedding import LinearConstraintSpec  # noqa: E402
from industrial_constraint_learning.surrogate import (  # noqa: E402
    RiskControlledTreeSurrogate,
    sample_uniform_design,
)


class AxisAlignedTeacher:
    feature_columns = ("x1", "x2")

    @staticmethod
    def _safe(frame: pd.DataFrame) -> np.ndarray:
        return (
            (frame["x1"].to_numpy(dtype=float) >= 0.60)
            & (frame["x2"].to_numpy(dtype=float) >= 0.40)
        )

    def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        safe = self._safe(candidates)
        p = np.where(safe, 0.98, 0.05)
        return np.column_stack([1.0 - p, p])

    def conformal_p_values(self, candidates: pd.DataFrame) -> np.ndarray:
        return np.where(self._safe(candidates), 0.01, 0.50)

    def risk_controlled_threshold(self, alpha: float = 0.05) -> float:
        return 0.90


def test_cpsat_tree_embedding_passes_teacher_and_hard_constraint_audit() -> None:
    teacher = AxisAlignedTeacher()
    bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    design = sample_uniform_design(bounds, 7000, random_state=61)
    surrogate = RiskControlledTreeSurrogate(
        teacher.feature_columns,
        max_depth=3,
        min_samples_leaf=8,
        random_state=61,
    ).fit_from_teacher(teacher, design, alpha=0.10)

    optimizer = SurrogateCPSATOptimizer(
        teacher,
        surrogate,
        bounds,
        feature_scales={"x1": 1000, "x2": 1000},
        alpha=0.10,
    )
    result = optimizer.optimize(
        [1.0, 1.0],
        maximize=True,
        hard_constraints=[
            LinearConstraintSpec([1.0, 1.0], upper_bound=1.60)
        ],
    )

    assert result.audited_safe
    assert not result.used_fallback
    assert result.surrogate_predicted_safe
    assert result.point["x1"] >= 0.60 - 1e-9
    assert result.point["x2"] >= 0.40 - 1e-9
    assert result.point["x1"] + result.point["x2"] <= 1.60 + 1e-6


def test_cpsat_uses_audited_fallback_for_low_fidelity_surrogate() -> None:
    teacher = AxisAlignedTeacher()
    bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    design = sample_uniform_design(bounds, 5000, random_state=62)
    surrogate = RiskControlledTreeSurrogate(
        teacher.feature_columns,
        max_depth=1,
        min_samples_leaf=10,
        random_state=62,
    ).fit_from_teacher(teacher, design, alpha=0.10)

    fallback = pd.DataFrame(
        {
            "x1": [0.60, 0.70, 0.80, 0.90],
            "x2": [0.40, 0.45, 0.50, 0.60],
        }
    )
    result = SurrogateCPSATOptimizer(
        teacher,
        surrogate,
        bounds,
        feature_scales={"x1": 1000, "x2": 1000},
        alpha=0.10,
    ).optimize(
        [1.0, 1.0],
        maximize=False,
        fallback_candidates=fallback,
    )

    assert result.audited_safe
    assert result.used_fallback
    assert result.conformal_p_value <= 0.10
