"""Reusable tabular constraint learner for industrial case studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .conformal import ConformalSafetyFilter, SafetyFilterEvaluation
from .optimization import SafeCandidateOptimizer


@dataclass(frozen=True)
class TabularConstraintEvaluation:
    """Held-out metrics for a generic tabular learned constraint."""

    confusion_matrix: np.ndarray
    balanced_accuracy: float
    f1: float
    roc_auc: float
    average_precision: float
    false_feasible_rate: float
    false_infeasible_rate: float


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


class TabularConstraintLearner:
    """Learn nonlinear binary operational constraints from tabular observations."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        label_column: str,
        *,
        test_size: float = 0.25,
        random_state: int = 42,
        calibration_cv_splits: int = 3,
        safety_calibration_size: float = 0.20,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if calibration_cv_splits < 2:
            raise ValueError("calibration_cv_splits must be at least 2")
        if not 0.0 < safety_calibration_size < 0.5:
            raise ValueError("safety_calibration_size must be between 0 and 0.5")
        required = set(feature_columns) | {label_column}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        labels = data[label_column].astype(int)
        if set(labels.unique()) != {0, 1}:
            raise ValueError("label_column must contain both binary classes 0 and 1")

        self.data = data.copy()
        self.feature_columns = tuple(feature_columns)
        self.label_column = label_column
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.calibration_cv_splits = int(calibration_cv_splits)
        self.safety_calibration_size = float(safety_calibration_size)
        self.model: CalibratedClassifierCV | None = None
        self.safety_filter: ConformalSafetyFilter | None = None
        self.best_params_: Dict[str, object] | None = None
        self.cv_best_score_: float | None = None
        self._split = None

    @staticmethod
    def _base_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def _calibrated(self, estimator: Pipeline) -> CalibratedClassifierCV:
        return CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=StratifiedKFold(
                n_splits=self.calibration_cv_splits,
                shuffle=True,
                random_state=self.random_state,
            ),
            n_jobs=-1,
        )

    def fit(
        self,
        *,
        tune: bool = True,
        cv_splits: int = 3,
        scoring: str = "average_precision",
    ) -> "TabularConstraintLearner":
        if cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")
        X = self.data.loc[:, self.feature_columns]
        y = self.data[self.label_column].astype(int)
        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        X_fit, X_safety, y_fit, y_safety = train_test_split(
            X_train_pool,
            y_train_pool,
            test_size=self.safety_calibration_size,
            random_state=self.random_state + 1,
            stratify=y_train_pool,
        )

        estimator = self._base_pipeline()
        self.best_params_ = None
        self.cv_best_score_ = None
        if tune:
            search = GridSearchCV(
                estimator,
                param_grid={
                    "classifier__C": [1.0, 5.0, 10.0, 25.0],
                    "classifier__gamma": ["scale", 0.05, 0.1, 0.25],
                },
                scoring=scoring,
                cv=StratifiedKFold(
                    n_splits=cv_splits,
                    shuffle=True,
                    random_state=self.random_state,
                ),
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_fit, y_fit)
            estimator = clone(search.best_estimator_)
            self.best_params_ = dict(search.best_params_)
            self.cv_best_score_ = float(search.best_score_)

        self.model = self._calibrated(estimator)
        self.model.fit(X_fit, y_fit)
        safety_scores = np.asarray(
            self.model.predict_proba(X_safety)[:, 1],
            dtype=float,
        )
        self.safety_filter = ConformalSafetyFilter().fit(
            safety_scores,
            y_safety.to_numpy(),
        )
        self._split = (X_fit, X_test, y_fit, y_test)
        return self

    def evaluate(self) -> TabularConstraintEvaluation:
        if self.model is None or self._split is None:
            raise RuntimeError("Fit the learner before evaluation")
        _, X_test, _, y_test = self._split
        pred = self.model.predict(X_test)
        prob = self.model.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        return TabularConstraintEvaluation(
            confusion_matrix=matrix,
            balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
            f1=float(f1_score(y_test, pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, prob)),
            average_precision=float(average_precision_score(y_test, prob)),
            false_feasible_rate=_ratio(int(fp), int(fp + tn)),
            false_infeasible_rate=_ratio(int(fn), int(fn + tp)),
        )

    def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit the learner before prediction")
        missing = set(self.feature_columns).difference(candidates.columns)
        if missing:
            raise ValueError(f"Missing candidate columns: {sorted(missing)}")
        return self.model.predict_proba(candidates.loc[:, self.feature_columns])

    def risk_controlled_threshold(self, alpha: float = 0.05) -> float:
        """Return the probability threshold implied by conformal calibration."""
        if self.safety_filter is None:
            raise RuntimeError("Fit the learner before safety calibration")
        return self.safety_filter.probability_threshold(alpha=alpha)

    def evaluate_safety_filter(
        self,
        alpha: float = 0.05,
    ) -> SafetyFilterEvaluation:
        """Evaluate conformal screening on the untouched held-out test set."""
        if self.model is None or self._split is None or self.safety_filter is None:
            raise RuntimeError("Fit the learner before safety evaluation")
        _, X_test, _, y_test = self._split
        scores = np.asarray(self.model.predict_proba(X_test)[:, 1], dtype=float)
        return self.safety_filter.evaluate(
            scores,
            y_test.to_numpy(),
            alpha=alpha,
        )

    def conformal_p_values(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return infeasible-class conformal p-values for candidate rows."""
        if self.safety_filter is None:
            raise RuntimeError("Fit the learner before conformal safety scoring")
        probabilities = self.predict_proba(candidates)[:, 1]
        return self.safety_filter.p_values(probabilities)

    def risk_controlled_optimizer(
        self,
        alpha: float = 0.05,
    ) -> SafeCandidateOptimizer:
        """Build an optimizer from the conformal probability threshold.

        The class-conditional conformal guarantee is marginal for a future
        candidate. Selecting the best point from many screened candidates is an
        additional selection problem and requires separate validation.
        """
        threshold = self.risk_controlled_threshold(alpha=alpha)
        if not 0.0 < threshold <= 1.0:
            raise RuntimeError(
                "The conformal threshold is outside the probability range; "
                "the requested alpha cannot produce a usable safe set."
            )
        if self.model is None:
            raise RuntimeError("Fit the learner before building an optimizer")
        return SafeCandidateOptimizer(
            self.model,
            feature_columns=self.feature_columns,
            min_probability=threshold,
        )

    def safe_optimizer(self, min_probability: float = 0.50) -> SafeCandidateOptimizer:
        if self.model is None:
            raise RuntimeError("Fit the learner before building an optimizer")
        return SafeCandidateOptimizer(
            self.model,
            feature_columns=self.feature_columns,
            min_probability=min_probability,
        )

    def descriptive_feasible_bounds(
        self,
        quantile_margin: float = 0.02,
    ) -> Dict[str, Dict[str, float]]:
        if not 0.0 <= quantile_margin < 0.5:
            raise ValueError("quantile_margin must be in [0, 0.5)")
        feasible = self.data[self.data[self.label_column].astype(bool)]
        if feasible.empty:
            raise RuntimeError("No feasible observations are available")
        return {
            feature: {
                "min": float(feasible[feature].quantile(quantile_margin)),
                "max": float(feasible[feature].quantile(1.0 - quantile_margin)),
            }
            for feature in self.feature_columns
        }

    def best_observed_feasible(
        self,
        objective_column: str,
        *,
        maximize: bool = False,
    ) -> pd.Series:
        if objective_column not in self.data.columns:
            raise ValueError(f"Unknown objective column: {objective_column}")
        feasible = self.data[self.data[self.label_column].astype(bool)]
        if feasible.empty:
            raise RuntimeError("No feasible observations are available")
        index = (
            feasible[objective_column].idxmax()
            if maximize
            else feasible[objective_column].idxmin()
        )
        return feasible.loc[index]
