"""Tests for the reusable tabular constraint learner."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning.tabular import TabularConstraintLearner  # noqa: E402


def test_tabular_constraint_learner_smoke() -> None:
    rng = np.random.default_rng(7)
    x1 = rng.uniform(-2.0, 2.0, 800)
    x2 = rng.uniform(-2.0, 2.0, 800)
    label = ((x1**2 + x2**2) <= 1.5).astype(int)
    data = pd.DataFrame({"x1": x1, "x2": x2, "feasible": label})
    learner = TabularConstraintLearner(
        data,
        feature_columns=["x1", "x2"],
        label_column="feasible",
        random_state=7,
    ).fit(tune=False)
    evaluation = learner.evaluate()
    assert evaluation.balanced_accuracy > 0.8
    assert evaluation.average_precision > 0.8
    assert evaluation.confusion_matrix.shape == (2, 2)


def test_tabular_learner_exposes_conformal_safety_filter() -> None:
    rng = np.random.default_rng(11)
    x1 = rng.uniform(-2.0, 2.0, 2000)
    x2 = rng.uniform(-2.0, 2.0, 2000)
    label = ((x1**2 + x2**2) <= 1.5).astype(int)
    data = pd.DataFrame({"x1": x1, "x2": x2, "feasible": label})

    learner = TabularConstraintLearner(
        data,
        feature_columns=["x1", "x2"],
        label_column="feasible",
        random_state=11,
    ).fit(tune=False)

    threshold = learner.risk_controlled_threshold(alpha=0.10)
    evaluation = learner.evaluate_safety_filter(alpha=0.10)

    assert 0.0 < threshold <= 1.0
    assert evaluation.probability_threshold == threshold
    assert evaluation.calibration_infeasible_count > 50
    assert evaluation.minimum_attainable_p_value < 0.10
