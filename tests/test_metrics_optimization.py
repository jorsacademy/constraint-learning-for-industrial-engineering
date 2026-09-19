"""Tests for operational region metrics and safe candidate optimization."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning.metrics import evaluate_binary_region  # noqa: E402
from industrial_constraint_learning.optimization import SafeCandidateOptimizer  # noqa: E402


class DummyProbabilityModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(X["x"].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def test_binary_region_metrics_match_expected_rates() -> None:
    truth = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    metrics = evaluate_binary_region(truth, pred)
    assert metrics.intersection_over_union == pytest.approx(1.0 / 3.0)
    assert metrics.false_feasible_rate == pytest.approx(0.5)
    assert metrics.false_infeasible_rate == pytest.approx(0.5)
    assert metrics.feasible_precision == pytest.approx(0.5)
    assert metrics.feasible_recall == pytest.approx(0.5)


def test_safe_candidate_optimizer_respects_probability_and_hard_constraints() -> None:
    candidates = pd.DataFrame(
        {
            "x": [0.6, 0.8, 0.95],
            "cost": [2.0, 5.0, 8.0],
        }
    )
    optimizer = SafeCandidateOptimizer(
        DummyProbabilityModel(),
        feature_columns=["x"],
        min_probability=0.75,
    )
    result = optimizer.optimize(
        candidates,
        objective=lambda frame: frame["cost"],
        hard_constraint=lambda frame: frame["cost"] <= 6.0,
        maximize=True,
    )
    assert result.point["x"] == pytest.approx(0.8)
    assert result.objective_value == pytest.approx(5.0)
    assert result.feasibility_probability == pytest.approx(0.8)
    assert result.safe_candidates == 1


def test_safe_candidate_optimizer_raises_when_no_safe_candidate_exists() -> None:
    candidates = pd.DataFrame({"x": [0.1, 0.2], "cost": [1.0, 2.0]})
    optimizer = SafeCandidateOptimizer(
        DummyProbabilityModel(),
        feature_columns=["x"],
        min_probability=0.9,
    )
    with pytest.raises(RuntimeError, match="No candidate"):
        optimizer.optimize(candidates, objective=lambda frame: frame["cost"])
