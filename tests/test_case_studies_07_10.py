"""Executable regression tests for case studies 07-10."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CASE_DIRS = [
    ROOT / "case_studies" / "07_product_design_space",
    ROOT / "case_studies" / "08_workforce_shift_scheduling",
    ROOT / "case_studies" / "09_inventory_control",
    ROOT / "case_studies" / "10_multi_product_line_balancing",
]
for path in [SRC, *CASE_DIRS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from design_constraint_learner import ProductDesignConstraintLearner  # noqa: E402
from design_data import generate_product_design_data  # noqa: E402
from inventory_constraint_learner import InventoryConstraintLearner  # noqa: E402
from inventory_data import generate_inventory_policy_data  # noqa: E402
from line_balancing_constraint_learner import (  # noqa: E402
    LineBalancingConstraintLearner,
)
from line_balancing_data import generate_line_balancing_data  # noqa: E402
from workforce_constraint_learner import WorkforceConstraintLearner  # noqa: E402
from workforce_data import generate_workforce_shift_data  # noqa: E402


@pytest.mark.parametrize(
    ("generator", "learner_class", "label_column"),
    [
        (
            generate_product_design_data,
            ProductDesignConstraintLearner,
            "operational_feasible",
        ),
        (
            generate_workforce_shift_data,
            WorkforceConstraintLearner,
            "operationally_acceptable",
        ),
        (
            generate_inventory_policy_data,
            InventoryConstraintLearner,
            "operationally_acceptable",
        ),
        (
            generate_line_balancing_data,
            LineBalancingConstraintLearner,
            "stable_operation",
        ),
    ],
)
def test_remaining_case_studies_learn_nonlinear_feasible_regions(
    generator,
    learner_class,
    label_column,
) -> None:
    first = generator(n_samples=2500, random_state=31)
    second = generator(n_samples=2500, random_state=31)
    assert first.equals(second)
    assert 0.05 < first[label_column].mean() < 0.95

    learner = learner_class(first, random_state=31).fit(tune=False)
    evaluation = learner.evaluate()
    assert evaluation.confusion_matrix.shape == (2, 2)
    assert evaluation.balanced_accuracy > 0.75
    assert evaluation.average_precision > 0.80
    assert 0.0 <= evaluation.false_feasible_rate <= 1.0
    assert 0.0 <= evaluation.false_infeasible_rate <= 1.0


@pytest.mark.parametrize(
    ("generator", "learner_class", "objective", "hard_column"),
    [
        (
            generate_product_design_data,
            ProductDesignConstraintLearner,
            "design_cost",
            "hard_design_compliant",
        ),
        (
            generate_workforce_shift_data,
            WorkforceConstraintLearner,
            "staffing_cost",
            "policy_compliant",
        ),
        (
            generate_inventory_policy_data,
            InventoryConstraintLearner,
            "total_cost",
            "hard_policy_compliant",
        ),
        (
            generate_line_balancing_data,
            LineBalancingConstraintLearner,
            "resource_cost",
            "hard_capacity_compliant",
        ),
    ],
)
def test_remaining_case_studies_screen_candidates_with_hard_constraints(
    generator,
    learner_class,
    objective,
    hard_column,
) -> None:
    data = generator(n_samples=3000, random_state=37)
    learner = learner_class(data, random_state=37).fit(tune=False)
    result = learner.safe_optimizer(min_probability=0.50).optimize(
        data,
        objective=lambda frame: frame[objective],
        hard_constraint=lambda frame: frame[hard_column].astype(bool),
        maximize=False,
    )
    assert result.feasibility_probability >= 0.50
    assert bool(result.point[hard_column])
    assert result.safe_candidates > 0
