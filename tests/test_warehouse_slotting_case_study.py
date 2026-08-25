"""Tests for the warehouse-slotting case study."""

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parents[1] / "case_studies" / "05_warehouse_slotting"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from warehouse_constraint_learner import WarehouseSlottingConstraintLearner  # noqa: E402
from warehouse_data import generate_warehouse_slotting_data  # noqa: E402


def test_warehouse_data_is_reproducible() -> None:
    a = generate_warehouse_slotting_data(n_samples=300, random_state=20)
    b = generate_warehouse_slotting_data(n_samples=300, random_state=20)
    assert a.equals(b)


def test_feasible_label_matches_engineering_rules() -> None:
    data = generate_warehouse_slotting_data(n_samples=1000, random_state=21)
    expected = (
        (data["hidden_stability_score"] >= 0.0)
        & (data["picking_time"] <= 78.0)
        & (data["congestion_score"] <= 0.85)
        & (data["ergonomic_risk"] <= 0.82)
    ).astype(int)
    assert (data["feasible_slotting"] == expected).all()


def test_warehouse_classes_are_non_degenerate() -> None:
    data = generate_warehouse_slotting_data(n_samples=2500, random_state=22)
    share = data["feasible_slotting"].mean()
    assert 0.05 < share < 0.70


def test_warehouse_model_has_useful_held_out_performance() -> None:
    data = generate_warehouse_slotting_data(n_samples=3500, random_state=23)
    learner = WarehouseSlottingConstraintLearner(data, random_state=23).fit(tune=False)
    evaluation = learner.evaluate()
    assert evaluation.balanced_accuracy >= 0.80
    assert evaluation.roc_auc >= 0.90
    assert evaluation.average_precision >= 0.75
    assert evaluation.unsafe_accept_rate <= 0.20


def test_nominal_slot_is_predicted_feasible() -> None:
    data = generate_warehouse_slotting_data(n_samples=4000, random_state=24)
    learner = WarehouseSlottingConstraintLearner(data, random_state=24).fit(tune=False)
    assert learner.predict_feasible(
        sku_velocity=140.0,
        unit_weight=6.0,
        cube=0.04,
        aisle_distance=25.0,
        replenishment_frequency=2.0,
        neighboring_pick_density=0.30,
    )


def test_stressed_slot_is_predicted_infeasible() -> None:
    data = generate_warehouse_slotting_data(n_samples=4000, random_state=25)
    learner = WarehouseSlottingConstraintLearner(data, random_state=25).fit(tune=False)
    assert not learner.predict_feasible(
        sku_velocity=210.0,
        unit_weight=26.0,
        cube=0.16,
        aisle_distance=130.0,
        replenishment_frequency=8.5,
        neighboring_pick_density=0.95,
    )


def test_feasible_region_improves_picking_and_congestion() -> None:
    data = generate_warehouse_slotting_data(n_samples=3000, random_state=26)
    learner = WarehouseSlottingConstraintLearner(data, random_state=26)
    improvement = learner.estimate_operational_improvement()
    assert improvement["picking_time_reduction_pct"] > 0.0
    assert improvement["congestion_reduction_pct"] > 0.0
