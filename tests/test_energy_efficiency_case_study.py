"""Tests for the energy-efficient machine settings case study."""

from pathlib import Path
import sys

CASE_STUDY_DIR = (
    Path(__file__).resolve().parents[1]
    / "case_studies"
    / "02_energy_efficient_machine_settings"
)
if str(CASE_STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_STUDY_DIR))

from energy_constraint_learner import EnergyConstraintLearner
from energy_data import generate_energy_efficiency_data


def test_energy_data_is_reproducible() -> None:
    first = generate_energy_efficiency_data(n_samples=300, random_state=7)
    second = generate_energy_efficiency_data(n_samples=300, random_state=7)
    assert first.equals(second)


def test_energy_data_contains_both_classes() -> None:
    data = generate_energy_efficiency_data(n_samples=1500, random_state=11)
    assert set(data["acceptable_operation"].unique()) == {0, 1}


def test_acceptable_points_meet_observed_performance_requirements() -> None:
    data = generate_energy_efficiency_data(n_samples=1200, random_state=12)
    acceptable = data[data["acceptable_operation"] == 1]

    assert not acceptable.empty
    assert (acceptable["energy_per_unit"] <= 2.00).all()
    assert (acceptable["throughput"] >= 78.0).all()
    assert (acceptable["quality_score"] >= 93.0).all()


def test_energy_constraint_learner_returns_valid_metrics() -> None:
    data = generate_energy_efficiency_data(n_samples=1800, random_state=21)
    learner = EnergyConstraintLearner(data, random_state=21).fit(tune=False)
    evaluation = learner.evaluate()

    assert evaluation.confusion_matrix.shape == (2, 2)
    assert 0.0 <= evaluation.balanced_accuracy <= 1.0
    assert 0.0 <= evaluation.f1 <= 1.0
    assert 0.0 <= evaluation.roc_auc <= 1.0
    assert 0.0 <= evaluation.average_precision <= 1.0


def test_best_observed_point_is_acceptable() -> None:
    data = generate_energy_efficiency_data(n_samples=1800, random_state=22)
    learner = EnergyConstraintLearner(data, random_state=22)
    best = learner.best_observed_energy_efficient_point()

    assert best["acceptable_operation"] == 1
    assert best["energy_per_unit"] <= 2.00
    assert best["throughput"] >= 78.0
    assert best["quality_score"] >= 93.0


def test_estimated_energy_saving_is_positive() -> None:
    data = generate_energy_efficiency_data(n_samples=1800, random_state=23)
    learner = EnergyConstraintLearner(data, random_state=23)
    assert learner.estimated_energy_saving() > 0.0
