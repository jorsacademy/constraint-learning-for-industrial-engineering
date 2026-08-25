"""Tests for the supply-chain feasibility case study."""

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parents[1] / "case_studies" / "04_supply_chain_feasibility"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from supply_chain_constraint_learner import SupplyChainConstraintLearner  # noqa: E402
from supply_chain_data import generate_supply_chain_data  # noqa: E402


def test_supply_chain_data_is_reproducible() -> None:
    a = generate_supply_chain_data(n_samples=300, random_state=10)
    b = generate_supply_chain_data(n_samples=300, random_state=10)
    assert a.equals(b)


def test_feasible_label_matches_engineering_rules() -> None:
    data = generate_supply_chain_data(n_samples=1000, random_state=11)
    expected = (
        (data["hidden_stability_score"] >= 0.0)
        & (data["service_level"] >= 0.90)
        & (data["total_logistics_cost"] <= 265.0)
    ).astype(int)
    assert (data["feasible_service"] == expected).all()


def test_supply_chain_classes_are_non_degenerate() -> None:
    data = generate_supply_chain_data(n_samples=2500, random_state=12)
    share = data["feasible_service"].mean()
    assert 0.03 < share < 0.80


def test_supply_chain_model_has_useful_held_out_performance() -> None:
    data = generate_supply_chain_data(n_samples=3500, random_state=13)
    learner = SupplyChainConstraintLearner(data, random_state=13).fit(tune=False)
    evaluation = learner.evaluate()
    assert evaluation.balanced_accuracy >= 0.75
    assert evaluation.roc_auc >= 0.85
    assert evaluation.average_precision >= 0.55


def test_nominal_point_is_predicted_feasible() -> None:
    data = generate_supply_chain_data(n_samples=4000, random_state=14)
    learner = SupplyChainConstraintLearner(data, random_state=14).fit(tune=False)
    assert learner.predict_feasible(
        supplier_lead_time=5.0,
        demand_cv=0.15,
        order_quantity=700.0,
        safety_stock=300.0,
        supplier_utilization=0.65,
        transport_time=2.0,
    )


def test_stressed_point_is_predicted_infeasible() -> None:
    data = generate_supply_chain_data(n_samples=4000, random_state=15)
    learner = SupplyChainConstraintLearner(data, random_state=15).fit(tune=False)
    assert not learner.predict_feasible(
        supplier_lead_time=16.0,
        demand_cv=0.60,
        order_quantity=1600.0,
        safety_stock=40.0,
        supplier_utilization=1.02,
        transport_time=7.0,
    )
