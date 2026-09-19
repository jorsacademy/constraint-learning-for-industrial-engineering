"""Tests for class-conditional conformal safety filtering."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning.conformal import ConformalSafetyFilter  # noqa: E402


def test_conformal_p_values_are_rank_based_and_monotone() -> None:
    scores = np.array([0.10, 0.20, 0.30, 0.80, 0.90])
    labels = np.array([0, 0, 0, 1, 1])
    safety = ConformalSafetyFilter().fit(scores, labels)

    candidate_scores = np.array([0.15, 0.25, 0.35, 0.95])
    p = safety.p_values(candidate_scores)

    assert np.all(np.diff(p) <= 0.0)
    assert p[-1] == pytest.approx(0.25)
    assert safety.minimum_attainable_p_value == pytest.approx(0.25)


def test_probability_threshold_matches_conformal_acceptance() -> None:
    infeasible_scores = np.linspace(0.05, 0.60, 99)
    feasible_scores = np.linspace(0.70, 0.99, 30)
    scores = np.concatenate([infeasible_scores, feasible_scores])
    labels = np.concatenate(
        [np.zeros(infeasible_scores.size), np.ones(feasible_scores.size)]
    )
    safety = ConformalSafetyFilter().fit(scores, labels)

    alpha = 0.05
    threshold = safety.probability_threshold(alpha)
    candidates = np.linspace(0.0, 1.0, 1001)

    by_p_value = safety.accept(candidates, alpha=alpha)
    by_threshold = candidates >= threshold
    assert np.array_equal(by_p_value, by_threshold)


def test_too_small_alpha_reports_finite_sample_resolution() -> None:
    safety = ConformalSafetyFilter().fit(
        np.array([0.1, 0.2, 0.3, 0.8]),
        np.array([0, 0, 0, 1]),
    )
    with pytest.raises(RuntimeError, match="finite-sample resolution"):
        safety.probability_threshold(alpha=0.10)
