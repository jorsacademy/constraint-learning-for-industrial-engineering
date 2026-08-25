"""Basic tests for the manufacturing constraint-learning example."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning import (
    ManufacturingConstraintLearner,
    generate_manufacturing_data,
)


def test_generated_yield_is_bounded() -> None:
    data = generate_manufacturing_data(n_samples=500, random_state=1)
    assert data["yield"].between(0.0, 100.0).all()


def test_known_optimal_region_is_physically_feasible() -> None:
    data = generate_manufacturing_data(n_samples=2000, random_state=2)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()
    assert learner.predict_feasible(250.0, 6.0)


def test_obviously_infeasible_point_is_rejected() -> None:
    data = generate_manufacturing_data(n_samples=2000, random_state=3)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()
    assert not learner.predict_feasible(100.0, 9.0)


def test_evaluation_returns_binary_confusion_matrix() -> None:
    data = generate_manufacturing_data(n_samples=1000, random_state=4)
    learner = ManufacturingConstraintLearner(data).fit_feasibility_classifier()
    evaluation = learner.evaluate()
    assert evaluation.confusion_matrix.shape == (2, 2)
