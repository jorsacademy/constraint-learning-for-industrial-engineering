"""Tests for distribution shift monitoring and online adaptation."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning.adaptive import (  # noqa: E402
    AdaptiveConstraintController,
    AdaptivePolicy,
)
from industrial_constraint_learning.drift import DistributionShiftMonitor  # noqa: E402
from industrial_constraint_learning.tabular import TabularConstraintLearner  # noqa: E402


FEATURES = ("x1", "x2")


def make_linear_data(
    n: int,
    seed: int,
    *,
    shift: float = 0.0,
    reverse_concept: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(shift, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    margin = x1 + 0.8 * x2
    feasible = (margin < 0.0) if reverse_concept else (margin >= 0.0)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "feasible": feasible.astype(int),
        }
    )


def fit_learner(data: pd.DataFrame, seed: int = 7) -> TabularConstraintLearner:
    return TabularConstraintLearner(
        data,
        feature_columns=FEATURES,
        label_column="feasible",
        random_state=seed,
    ).fit(tune=False)


def test_distribution_shift_monitor_distinguishes_stable_and_shifted_batches() -> None:
    reference = make_linear_data(4000, 1)
    stable = make_linear_data(1200, 2)
    shifted = make_linear_data(1200, 3, shift=1.5)

    learner = fit_learner(reference, seed=1)
    monitor = DistributionShiftMonitor(
        FEATURES,
        feature_psi_threshold=0.20,
        score_psi_threshold=0.20,
        min_batch_size=100,
    ).fit(
        reference,
        learner.predict_proba(reference)[:, 1],
        labels=reference["feasible"],
    )

    stable_report = monitor.detect(
        stable,
        learner.predict_proba(stable)[:, 1],
        labels=stable["feasible"],
    )
    shifted_report = monitor.detect(
        shifted,
        learner.predict_proba(shifted)[:, 1],
        labels=shifted["feasible"],
    )

    assert not stable_report.covariate_drift
    assert stable_report.max_feature_psi < 0.20
    assert shifted_report.covariate_drift
    assert "x1" in shifted_report.shifted_features
    assert shifted_report.max_feature_psi >= 0.20
    assert shifted_report.any_drift


def test_controller_keeps_valid_model_for_stable_batch() -> None:
    reference = make_linear_data(5000, 10)
    batch = make_linear_data(1500, 11)
    learner = fit_learner(reference, seed=10)

    controller = AdaptiveConstraintController(
        learner,
        policy=AdaptivePolicy(
            alpha=0.10,
            max_balanced_accuracy_drop=0.15,
            max_batch_false_feasible_rate=0.25,
        ),
        drift_monitor=DistributionShiftMonitor(
            FEATURES,
            feature_psi_threshold=0.25,
            score_psi_threshold=0.25,
            min_batch_size=100,
        ),
    )
    result = controller.process_batch(batch)

    assert result.action == "none"
    assert result.model_valid
    assert result.version == 1
    assert not result.drift.any_drift


def test_controller_recalibrates_for_moderate_distribution_shift() -> None:
    reference = make_linear_data(5000, 20)
    batch = make_linear_data(1600, 21, shift=0.45)
    learner = fit_learner(reference, seed=20)

    controller = AdaptiveConstraintController(
        learner,
        policy=AdaptivePolicy(
            alpha=0.10,
            max_balanced_accuracy_drop=0.20,
            max_batch_false_feasible_rate=0.30,
            max_safety_false_feasible_rate=0.25,
            severe_feature_psi=5.0,
            severe_score_psi=5.0,
            min_class_count_for_recalibration=30,
        ),
        drift_monitor=DistributionShiftMonitor(
            FEATURES,
            feature_psi_threshold=0.04,
            score_psi_threshold=5.0,
            label_rate_delta_threshold=0.50,
            min_batch_size=100,
        ),
    )
    result = controller.process_batch(batch)

    assert result.drift.covariate_drift
    assert result.action == "recalibrated"
    assert result.model_valid
    assert result.version == 2
    assert result.safety_evaluation is not None


def test_controller_retrains_after_concept_performance_drift() -> None:
    reference = make_linear_data(5000, 30)
    batch = make_linear_data(1600, 31, reverse_concept=True)
    learner = fit_learner(reference, seed=30)

    controller = AdaptiveConstraintController(
        learner,
        policy=AdaptivePolicy(
            alpha=0.10,
            max_balanced_accuracy_drop=0.08,
            min_balanced_accuracy=0.70,
            max_batch_false_feasible_rate=0.25,
            max_safety_false_feasible_rate=0.25,
            recent_window_size=1200,
            retrain_tune=False,
        ),
        drift_monitor=DistributionShiftMonitor(
            FEATURES,
            feature_psi_threshold=0.30,
            score_psi_threshold=0.30,
            label_rate_delta_threshold=0.30,
            min_batch_size=100,
        ),
    )
    result = controller.process_batch(batch)

    assert result.batch_balanced_accuracy is not None
    assert result.batch_balanced_accuracy < 0.30
    assert result.action == "retrained"
    assert result.model_valid
    assert result.version == 2
    assert controller.learner.evaluate().balanced_accuracy >= 0.70


def test_unlabeled_shift_invalidates_until_labels_arrive() -> None:
    reference = make_linear_data(5000, 40)
    shifted = make_linear_data(1200, 41, shift=1.5).drop(columns="feasible")
    learner = fit_learner(reference, seed=40)

    controller = AdaptiveConstraintController(
        learner,
        drift_monitor=DistributionShiftMonitor(
            FEATURES,
            feature_psi_threshold=0.15,
            score_psi_threshold=0.15,
            min_batch_size=100,
        ),
    )
    result = controller.process_batch(shifted)

    assert result.drift.any_drift
    assert result.action == "invalidated_no_labels"
    assert not result.model_valid
    assert not controller.model_valid
