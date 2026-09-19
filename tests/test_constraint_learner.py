"""Tests for the manufacturing constraint-learning workflow."""

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


def test_calibrated_feasibility_probability_separates_reference_points() -> None:
    data = generate_manufacturing_data(n_samples=4000, random_state=12)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()

    good_probability = learner.predict_feasibility_probability(250.0, 6.0)
    bad_probability = learner.predict_feasibility_probability(100.0, 9.0)

    assert 0.0 <= bad_probability <= 1.0
    assert 0.0 <= good_probability <= 1.0
    assert good_probability > bad_probability
    assert learner.predict_safe(250.0, 6.0, min_probability=0.75)
    assert not learner.predict_safe(100.0, 9.0, min_probability=0.75)


def test_boundary_metrics_report_meaningful_region_recovery() -> None:
    data = generate_manufacturing_data(n_samples=5000, random_state=14)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()
    metrics = learner.boundary_metrics(grid_resolution=100)

    assert 0.0 <= metrics.intersection_over_union <= 1.0
    assert 0.0 <= metrics.false_feasible_rate <= 1.0
    assert 0.0 <= metrics.false_infeasible_rate <= 1.0
    assert metrics.intersection_over_union > 0.50


def test_safe_optimizer_finds_high_value_candidate_inside_learned_region() -> None:
    data = generate_manufacturing_data(n_samples=5000, random_state=15)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()

    temperatures = np.linspace(150.0, 350.0, 41)
    pressures = np.linspace(2.0, 8.0, 31)
    tt, pp = np.meshgrid(temperatures, pressures)
    candidates = pd.DataFrame(
        {
            "temperature": tt.ravel(),
            "pressure": pp.ravel(),
        }
    )

    optimizer = learner.safe_optimizer(min_probability=0.75)
    result = optimizer.optimize(
        candidates,
        objective=lambda frame: -(
            ((frame["temperature"] - 250.0) / 100.0) ** 2
            + ((frame["pressure"] - 6.0) / 3.0) ** 2
        ),
        hard_constraint=lambda frame: (
            frame["temperature"].between(150.0, 350.0)
            & frame["pressure"].between(2.0, 8.0)
        ),
        maximize=True,
    )

    assert result.feasibility_probability >= 0.75
    assert abs(result.point["temperature"] - 250.0) <= 5.0
    assert abs(result.point["pressure"] - 6.0) <= 0.5
