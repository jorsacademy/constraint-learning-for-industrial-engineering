"""Tests for the job-shop scheduling case study."""

from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parents[1] / "case_studies" / "06_job_shop_scheduling"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from scheduling_constraint_learner import JobShopConstraintLearner  # noqa: E402
from scheduling_data import generate_job_shop_data  # noqa: E402


def test_job_shop_data_is_reproducible() -> None:
    a = generate_job_shop_data(n_samples=300, random_state=21)
    b = generate_job_shop_data(n_samples=300, random_state=21)
    assert a.equals(b)


def test_feasible_schedule_matches_engineering_rules() -> None:
    data = generate_job_shop_data(n_samples=1000, random_state=22)
    expected = (
        (data["hidden_stability_score"] >= 0.0)
        & (data["mean_tardiness"] <= 6.0)
        & (data["overtime_hours"] <= 7.5)
        & (data["wip_level"] <= 38.0)
    ).astype(int)
    assert (data["feasible_schedule"] == expected).all()


def test_job_shop_classes_are_non_degenerate() -> None:
    data = generate_job_shop_data(n_samples=2500, random_state=23)
    share = data["feasible_schedule"].mean()
    assert 0.03 < share < 0.85


def test_job_shop_model_has_useful_held_out_performance() -> None:
    data = generate_job_shop_data(n_samples=3500, random_state=24)
    learner = JobShopConstraintLearner(data, random_state=24).fit(tune=False)
    evaluation = learner.evaluate()
    assert evaluation.balanced_accuracy >= 0.75
    assert evaluation.roc_auc >= 0.85
    assert evaluation.average_precision >= 0.55
    assert evaluation.false_feasible_rate <= 0.30


def test_nominal_schedule_is_predicted_feasible() -> None:
    data = generate_job_shop_data(n_samples=4000, random_state=25)
    learner = JobShopConstraintLearner(data, random_state=25).fit(tune=False)
    assert learner.predict_feasible(
        job_count=45.0,
        due_date_tightness=1.20,
        machine_utilization=0.72,
        processing_time_cv=0.20,
        setup_time_ratio=0.08,
        machine_availability=0.97,
    )


def test_stressed_schedule_is_predicted_infeasible() -> None:
    data = generate_job_shop_data(n_samples=4000, random_state=26)
    learner = JobShopConstraintLearner(data, random_state=26).fit(tune=False)
    assert not learner.predict_feasible(
        job_count=110.0,
        due_date_tightness=0.72,
        machine_utilization=1.04,
        processing_time_cv=0.68,
        setup_time_ratio=0.31,
        machine_availability=0.82,
    )
