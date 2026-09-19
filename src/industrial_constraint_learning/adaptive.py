"""Online invalidation, recalibration, retraining, and deployment control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .conformal import ConformalSafetyFilter, SafetyFilterEvaluation
from .drift import DistributionShiftMonitor, DriftReport
from .tabular import TabularConstraintLearner


AdaptationAction = Literal[
    "none",
    "recalibrated",
    "retrained",
    "invalidated_no_labels",
    "invalidated_failed_validation",
]


@dataclass(frozen=True)
class AdaptivePolicy:
    """Operational gates for online constraint adaptation."""

    alpha: float = 0.10
    max_balanced_accuracy_drop: float = 0.10
    min_balanced_accuracy: float = 0.65
    max_batch_false_feasible_rate: float = 0.20
    max_safety_false_feasible_rate: float = 0.20
    severe_feature_psi: float = 0.50
    severe_score_psi: float = 0.50
    recent_window_size: int = 5000
    recalibration_fraction: float = 0.60
    min_class_count_for_recalibration: int = 20
    retrain_tune: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= self.max_balanced_accuracy_drop < 1.0:
            raise ValueError("max_balanced_accuracy_drop must be in [0, 1)")
        if not 0.0 < self.min_balanced_accuracy <= 1.0:
            raise ValueError("min_balanced_accuracy must be in (0, 1]")
        if not 0.0 <= self.max_batch_false_feasible_rate <= 1.0:
            raise ValueError("max_batch_false_feasible_rate must be in [0, 1]")
        if not 0.0 <= self.max_safety_false_feasible_rate <= 1.0:
            raise ValueError("max_safety_false_feasible_rate must be in [0, 1]")
        if self.severe_feature_psi <= 0.0 or self.severe_score_psi <= 0.0:
            raise ValueError("severe PSI thresholds must be positive")
        if self.recent_window_size < 200:
            raise ValueError("recent_window_size must be at least 200")
        if not 0.1 <= self.recalibration_fraction <= 0.9:
            raise ValueError("recalibration_fraction must be in [0.1, 0.9]")
        if self.min_class_count_for_recalibration < 2:
            raise ValueError("min_class_count_for_recalibration must be at least 2")


@dataclass(frozen=True)
class AdaptiveUpdateResult:
    """Result of processing one monitored batch."""

    action: AdaptationAction
    model_valid: bool
    version: int
    reason: str
    drift: DriftReport
    batch_balanced_accuracy: float | None
    batch_false_feasible_rate: float | None
    safety_evaluation: SafetyFilterEvaluation | None


LearnerBuilder = Callable[[pd.DataFrame], TabularConstraintLearner]


def _false_feasible_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


class AdaptiveConstraintController:
    """Manage drift detection and safe online model lifecycle.

    A drift event invalidates the active learned safety layer until either:
    1. an independently validated conformal recalibration succeeds, or
    2. a retrained candidate passes deployment gates.

    Unlabeled drift is detected but cannot be safely recalibrated/relearned, so
    the learned safety layer remains invalidated until labels arrive.
    """

    def __init__(
        self,
        learner: TabularConstraintLearner,
        *,
        policy: AdaptivePolicy | None = None,
        drift_monitor: DistributionShiftMonitor | None = None,
        learner_builder: LearnerBuilder | None = None,
        reference_data: pd.DataFrame | None = None,
    ) -> None:
        if learner.model is None or learner.classifier_model is None:
            raise ValueError("learner must be fitted before controller creation")
        if learner.safety_filter is None:
            raise ValueError("learner must have a fitted conformal safety filter")

        self.learner = learner
        self.policy = policy or AdaptivePolicy()
        self.learner_builder = learner_builder
        self.model_valid = True
        self.version = 1

        self.reference_data = (
            reference_data.copy()
            if reference_data is not None
            else learner.data.copy()
        )
        self.history = learner.data.copy()

        self.drift_monitor = drift_monitor or DistributionShiftMonitor(
            learner.feature_columns
        )
        reference_scores = learner.predict_proba(self.reference_data)[:, 1]
        reference_labels = (
            self.reference_data[learner.label_column].astype(int).to_numpy()
            if learner.label_column in self.reference_data.columns
            else None
        )
        self.drift_monitor.fit(
            self.reference_data,
            reference_scores,
            labels=reference_labels,
        )
        self._baseline_balanced_accuracy = learner.evaluate().balanced_accuracy

    def _builder(self, data: pd.DataFrame) -> TabularConstraintLearner:
        if self.learner_builder is not None:
            return self.learner_builder(data)

        if type(self.learner) is not TabularConstraintLearner:
            try:
                return type(self.learner)(
                    data,
                    test_size=self.learner.test_size,
                    random_state=self.learner.random_state,
                )
            except TypeError:
                pass

        return TabularConstraintLearner(
            data,
            feature_columns=self.learner.feature_columns,
            label_column=self.learner.label_column,
            test_size=self.learner.test_size,
            random_state=self.learner.random_state,
            calibration_cv_splits=self.learner.calibration_cv_splits,
            safety_calibration_size=self.learner.safety_calibration_size,
        )

    def _batch_metrics(
        self,
        batch: pd.DataFrame,
    ) -> tuple[float | None, float | None]:
        if self.learner.label_column not in batch.columns:
            return None, None
        y = batch[self.learner.label_column].astype(int).to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise ValueError("batch labels must be binary 0/1")
        if self.learner.classifier_model is None:
            raise RuntimeError("Active classifier model is missing")
        pred = self.learner.classifier_model.predict(
            batch.loc[:, self.learner.feature_columns]
        )
        if np.unique(y).size < 2:
            balanced = None
        else:
            balanced = float(balanced_accuracy_score(y, pred))
        return balanced, _false_feasible_rate(y, pred)

    def _performance_degraded(
        self,
        balanced: float | None,
        false_feasible: float | None,
    ) -> bool:
        accuracy_bad = (
            balanced is not None
            and (
                balanced < self.policy.min_balanced_accuracy
                or balanced
                < self._baseline_balanced_accuracy
                - self.policy.max_balanced_accuracy_drop
            )
        )
        unsafe_accept_bad = (
            false_feasible is not None
            and false_feasible > self.policy.max_batch_false_feasible_rate
        )
        return bool(accuracy_bad or unsafe_accept_bad)

    def _enough_labels_for_recalibration(self, batch: pd.DataFrame) -> bool:
        if self.learner.label_column not in batch.columns:
            return False
        counts = batch[self.learner.label_column].astype(int).value_counts()
        return all(
            int(counts.get(label, 0)) >= self.policy.min_class_count_for_recalibration
            for label in (0, 1)
        )

    def _recalibrate(
        self,
        batch: pd.DataFrame,
    ) -> SafetyFilterEvaluation | None:
        if self.learner.model is None:
            raise RuntimeError("Active risk-score model is missing")
        if not self._enough_labels_for_recalibration(batch):
            return None

        train, validation = train_test_split(
            batch,
            train_size=self.policy.recalibration_fraction,
            random_state=self.learner.random_state + self.version,
            stratify=batch[self.learner.label_column].astype(int),
        )
        train_scores = self.learner.predict_proba(train)[:, 1]
        candidate_filter = ConformalSafetyFilter().fit(
            train_scores,
            train[self.learner.label_column].astype(int).to_numpy(),
        )
        if self.policy.alpha < candidate_filter.minimum_attainable_p_value:
            return None

        validation_scores = self.learner.predict_proba(validation)[:, 1]
        evaluation = candidate_filter.evaluate(
            validation_scores,
            validation[self.learner.label_column].astype(int).to_numpy(),
            alpha=self.policy.alpha,
        )
        if (
            evaluation.false_feasible_rate
            <= self.policy.max_safety_false_feasible_rate
        ):
            self.learner.safety_filter = candidate_filter
            return evaluation
        return None

    def _candidate_passes(
        self,
        candidate: TabularConstraintLearner,
    ) -> tuple[bool, SafetyFilterEvaluation | None]:
        evaluation = candidate.evaluate()
        if evaluation.balanced_accuracy < self.policy.min_balanced_accuracy:
            return False, None
        if (
            evaluation.false_feasible_rate
            > self.policy.max_batch_false_feasible_rate
        ):
            return False, None

        try:
            safety = candidate.evaluate_safety_filter(alpha=self.policy.alpha)
        except RuntimeError:
            return False, None
        if (
            safety.false_feasible_rate
            > self.policy.max_safety_false_feasible_rate
        ):
            return False, safety
        return True, safety

    def _retrain(
        self,
        batch: pd.DataFrame,
    ) -> tuple[TabularConstraintLearner | None, SafetyFilterEvaluation | None]:
        combined = pd.concat([self.history, batch], ignore_index=True)
        recent = combined.tail(self.policy.recent_window_size).copy()
        labels = recent[self.learner.label_column].astype(int)
        if set(labels.unique()) != {0, 1}:
            return None, None

        candidate = self._builder(recent)
        candidate.fit(tune=self.policy.retrain_tune)
        passed, safety = self._candidate_passes(candidate)
        if not passed:
            return None, safety
        return candidate, safety

    def _refresh_reference(self, batch: pd.DataFrame) -> None:
        scores = self.learner.predict_proba(batch)[:, 1]
        labels = (
            batch[self.learner.label_column].astype(int).to_numpy()
            if self.learner.label_column in batch.columns
            else None
        )
        self.drift_monitor.fit(batch, scores, labels=labels)
        self.reference_data = batch.copy()

    def process_batch(self, batch: pd.DataFrame) -> AdaptiveUpdateResult:
        """Detect drift and adapt the active learned constraint when justified."""
        missing = set(self.learner.feature_columns).difference(batch.columns)
        if missing:
            raise ValueError(f"Missing batch feature columns: {sorted(missing)}")
        if batch.empty:
            raise ValueError("batch must not be empty")

        scores = self.learner.predict_proba(batch)[:, 1]
        labels = (
            batch[self.learner.label_column].astype(int).to_numpy()
            if self.learner.label_column in batch.columns
            else None
        )
        drift = self.drift_monitor.detect(batch, scores, labels=labels)
        balanced, false_feasible = self._batch_metrics(batch)
        performance_degraded = self._performance_degraded(
            balanced,
            false_feasible,
        )

        if not drift.any_drift and not performance_degraded:
            self.history = pd.concat([self.history, batch], ignore_index=True).tail(
                self.policy.recent_window_size
            )
            return AdaptiveUpdateResult(
                action="none",
                model_valid=self.model_valid,
                version=self.version,
                reason="No material distribution or performance drift detected.",
                drift=drift,
                batch_balanced_accuracy=balanced,
                batch_false_feasible_rate=false_feasible,
                safety_evaluation=None,
            )

        self.model_valid = False
        if labels is None:
            return AdaptiveUpdateResult(
                action="invalidated_no_labels",
                model_valid=False,
                version=self.version,
                reason=(
                    "Distribution shift detected without labels; the learned "
                    "safety layer cannot be recalibrated or revalidated."
                ),
                drift=drift,
                batch_balanced_accuracy=None,
                batch_false_feasible_rate=None,
                safety_evaluation=None,
            )

        severe_drift = (
            drift.max_feature_psi >= self.policy.severe_feature_psi
            or drift.score_psi >= self.policy.severe_score_psi
        )
        if not performance_degraded and not severe_drift:
            safety = self._recalibrate(batch)
            if safety is not None:
                self.history = pd.concat(
                    [self.history, batch], ignore_index=True
                ).tail(self.policy.recent_window_size)
                self._refresh_reference(batch)
                self.model_valid = True
                self.version += 1
                return AdaptiveUpdateResult(
                    action="recalibrated",
                    model_valid=True,
                    version=self.version,
                    reason=(
                        "Drift detected but predictive performance remained "
                        "acceptable; conformal safety calibration was refreshed."
                    ),
                    drift=drift,
                    batch_balanced_accuracy=balanced,
                    batch_false_feasible_rate=false_feasible,
                    safety_evaluation=safety,
                )

        candidate, safety = self._retrain(batch)
        if candidate is not None:
            self.learner = candidate
            self.history = candidate.data.copy()
            self._baseline_balanced_accuracy = candidate.evaluate().balanced_accuracy
            self._refresh_reference(batch)
            self.model_valid = True
            self.version += 1
            return AdaptiveUpdateResult(
                action="retrained",
                model_valid=True,
                version=self.version,
                reason=(
                    "Performance degradation or severe drift required full "
                    "relearning; the candidate passed deployment gates."
                ),
                drift=drift,
                batch_balanced_accuracy=balanced,
                batch_false_feasible_rate=false_feasible,
                safety_evaluation=safety,
            )

        return AdaptiveUpdateResult(
            action="invalidated_failed_validation",
            model_valid=False,
            version=self.version,
            reason=(
                "Drift invalidated the active learned safety layer and no "
                "recalibrated/retrained candidate passed deployment gates."
            ),
            drift=drift,
            batch_balanced_accuracy=balanced,
            batch_false_feasible_rate=false_feasible,
            safety_evaluation=safety,
        )
