"""Tests for the assembly quality-control constraint-learning case study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "case_studies" / "03_assembly_quality_control"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from assembly_constraint_learner import AssemblyQualityConstraintLearner
from assembly_data import generate_assembly_quality_data


def test_assembly_data_is_reproducible() -> None:
    first = generate_assembly_quality_data(n_samples=300, random_state=7)
    second = generate_assembly_quality_data(n_samples=300, random_state=7)
    assert first.equals(second)


def test_acceptable_label_satisfies_engineering_requirements() -> None:
    data = generate_assembly_quality_data(n_samples=1000, random_state=8)
    accepted = data[data["acceptable_operation"] == 1]
    assert not accepted.empty
    assert (accepted["hidden_stable"] == 1).all()
    assert (accepted["quality_score"] >= 93.0).all()
    assert (accepted["joint_strength"] >= 82.0).all()
    assert (accepted["cycle_time"] <= 8.6).all()


def test_class_distribution_is_non_degenerate() -> None:
    data = generate_assembly_quality_data(n_samples=1500, random_state=9)
    rate = data["acceptable_operation"].mean()
    assert 0.05 < rate < 0.50


def test_model_recovers_useful_operating_region() -> None:
    data = generate_assembly_quality_data(n_samples=2500, random_state=10)
    learner = AssemblyQualityConstraintLearner(data, random_state=10).fit(tune=False)
    evaluation = learner.evaluate()
    assert evaluation.balanced_accuracy > 0.85
    assert evaluation.roc_auc > 0.95
    assert evaluation.average_precision > 0.85
    assert evaluation.false_accept_rate < 0.15


def test_nominal_center_is_predicted_acceptable() -> None:
    data = generate_assembly_quality_data(n_samples=3000, random_state=11)
    learner = AssemblyQualityConstraintLearner(data, random_state=11).fit(tune=False)
    assert learner.predict_acceptable(
        torque=50.0,
        angle=90.0,
        tool_speed=300.0,
        insertion_force=300.0,
        component_temperature=30.0,
    )


def test_extreme_setting_is_rejected() -> None:
    data = generate_assembly_quality_data(n_samples=3000, random_state=12)
    learner = AssemblyQualityConstraintLearner(data, random_state=12).fit(tune=False)
    assert not learner.predict_acceptable(
        torque=78.0,
        angle=145.0,
        tool_speed=480.0,
        insertion_force=480.0,
        component_temperature=44.0,
    )
