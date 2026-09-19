"""Tests for solver-embeddable learned constraints."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning.milp_embedding import (  # noqa: E402
    LinearConstraintSpec,
    SurrogateMILPOptimizer,
)
from industrial_constraint_learning.surrogate import (  # noqa: E402
    RiskControlledTreeSurrogate,
    sample_uniform_design,
)
from industrial_constraint_learning.tabular import TabularConstraintLearner  # noqa: E402


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


def test_tree_surrogate_distills_axis_aligned_teacher() -> None:
    teacher = AxisAlignedTeacher()
    bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    design = sample_uniform_design(bounds, 6000, random_state=4)

    surrogate = RiskControlledTreeSurrogate(
        teacher.feature_columns,
        max_depth=3,
        min_samples_leaf=10,
        random_state=4,
    ).fit_from_teacher(teacher, design, alpha=0.10)

    assert surrogate.fidelity_ is not None
    assert surrogate.fidelity_.balanced_accuracy > 0.97
    assert surrogate.fidelity_.false_safe_rate < 0.03
    assert len(surrogate.safe_leaf_paths()) >= 1


def test_milp_embedding_solves_tree_and_passes_teacher_audit() -> None:
    teacher = AxisAlignedTeacher()
    bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    design = sample_uniform_design(bounds, 7000, random_state=5)
    surrogate = RiskControlledTreeSurrogate(
        teacher.feature_columns,
        max_depth=3,
        min_samples_leaf=8,
        random_state=5,
    ).fit_from_teacher(teacher, design, alpha=0.10)

    optimizer = SurrogateMILPOptimizer(
        teacher,
        surrogate,
        bounds,
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
    assert result.point["x1"] >= 0.60 - 1e-6
    assert result.point["x2"] >= 0.40 - 1e-6
    assert result.point["x1"] + result.point["x2"] <= 1.60 + 1e-6


def test_milp_embedding_falls_back_when_shallow_surrogate_is_unsafe() -> None:
    teacher = AxisAlignedTeacher()
    bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    design = sample_uniform_design(bounds, 5000, random_state=6)
    surrogate = RiskControlledTreeSurrogate(
        teacher.feature_columns,
        max_depth=1,
        min_samples_leaf=10,
        random_state=6,
    ).fit_from_teacher(teacher, design, alpha=0.10)

    fallback = pd.DataFrame(
        {
            "x1": [0.60, 0.70, 0.80, 0.90],
            "x2": [0.40, 0.45, 0.50, 0.60],
        }
    )
    optimizer = SurrogateMILPOptimizer(
        teacher,
        surrogate,
        bounds,
        alpha=0.10,
    )
    result = optimizer.optimize(
        [1.0, 1.0],
        maximize=False,
        fallback_candidates=fallback,
    )

    assert result.audited_safe
    assert result.used_fallback
    assert result.conformal_p_value <= 0.10


def test_real_tabular_learner_can_act_as_teacher_for_milp() -> None:
    rng = np.random.default_rng(9)
    x1 = rng.uniform(-2.0, 2.0, 5000)
    x2 = rng.uniform(-2.0, 2.0, 5000)
    feasible = (
        (x1 >= -0.8)
        & (x2 >= -0.6)
        & ((x1 + 0.6 * x2) >= 0.0)
    ).astype(int)
    data = pd.DataFrame({"x1": x1, "x2": x2, "feasible": feasible})

    learner = TabularConstraintLearner(
        data,
        feature_columns=["x1", "x2"],
        label_column="feasible",
        random_state=9,
    ).fit(tune=False)

    bounds = {"x1": (-2.0, 2.0), "x2": (-2.0, 2.0)}
    design = sample_uniform_design(bounds, 8000, random_state=10)
    surrogate = RiskControlledTreeSurrogate(
        learner.feature_columns,
        max_depth=5,
        min_samples_leaf=15,
        random_state=10,
    ).fit_from_teacher(learner, design, alpha=0.10)

    assert surrogate.fidelity_ is not None
    assert surrogate.fidelity_.balanced_accuracy > 0.85

    fallback = sample_uniform_design(bounds, 3000, random_state=11)
    result = SurrogateMILPOptimizer(
        learner,
        surrogate,
        bounds,
        alpha=0.10,
    ).optimize(
        [1.0, 0.25],
        maximize=False,
        fallback_candidates=fallback,
    )

    assert result.audited_safe
    assert result.conformal_p_value <= 0.10
