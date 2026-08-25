"""Tests for the manufacturing constraint-learning workflow."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning import (  # noqa: E402
    ManufacturingConstraintLearner,
    generate_manufacturing_data,
)


def test_generated_yield_is_bounded() -> None:
    data = generate_manufacturing_data(n_samples=500, random_state=1)
    assert data["yield"].between(0.0, 100.0).all()


def test_data_generation_is_reproducible() -> None:
    first = generate_manufacturing_data(n_samples=250, random_state=7)
    second = generate_manufacturing_data(n_samples=250, random_state=7)
    assert first.equals(second)


def test_known_optimal_region_is_physically_feasible() -> None:
    temperature = np.array([250.0])
    pressure = np.array([6.0])
    result = ManufacturingConstraintLearner.true_physical_feasibility(
        temperature,
        pressure,
    )
    assert bool(result[0])


def test_obviously_infeasible_points_match_ground_truth() -> None:
    temperature = np.array([100.0, 400.0, 250.0])
    pressure = np.array([9.0, 6.0, 9.0])
    result = ManufacturingConstraintLearner.true_physical_feasibility(
        temperature,
        pressure,
    )
    assert not result.any()


def test_default_classifier_recognizes_reference_points() -> None:
    data = generate_manufacturing_data(n_samples=4000, random_state=2)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()
    assert learner.predict_feasible(250.0, 6.0)
    assert not learner.predict_feasible(100.0, 9.0)


def test_evaluation_returns_expected_metrics() -> None:
    data = generate_manufacturing_data(n_samples=2000, random_state=4)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()
    evaluation = learner.evaluate()
    assert evaluation.confusion_matrix.shape == (2, 2)
    assert 0.0 <= evaluation.balanced_accuracy <= 1.0
    assert 0.0 <= evaluation.f1 <= 1.0
    assert 0.0 <= evaluation.roc_auc <= 1.0
    assert 0.0 <= evaluation.average_precision <= 1.0


def test_high_yield_mode_does_not_require_physical_label_for_fitting() -> None:
    data = generate_manufacturing_data(n_samples=2000, random_state=5).drop(
        columns="physical_feasible"
    )
    learner = ManufacturingConstraintLearner(
        data,
        label_mode="high_yield",
    ).fit_feasibility_classifier()
    evaluation = learner.evaluate()
    assert evaluation.confusion_matrix.shape == (2, 2)


def test_hyperparameter_tuning_records_results() -> None:
    data = generate_manufacturing_data(n_samples=1500, random_state=6)
    learner = ManufacturingConstraintLearner(data).tune_hyperparameters(
        cv_splits=3,
        scoring="average_precision",
    )
    assert learner.best_params_ is not None
    assert learner.cv_best_score_ is not None
    assert 0.0 <= learner.cv_best_score_ <= 1.0


def test_high_yield_bounds_are_ordered() -> None:
    data = generate_manufacturing_data(n_samples=2000, random_state=8)
    learner = ManufacturingConstraintLearner(data)
    bounds = learner.learn_high_yield_bounds(quantile_margin=0.02)
    for values in bounds.values():
        assert values["min"] < values["max"]
