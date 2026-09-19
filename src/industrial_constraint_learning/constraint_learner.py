"""Constraint-learning models for the manufacturing case study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .conformal import ConformalSafetyFilter, SafetyFilterEvaluation
from .metrics import BinaryRegionMetrics, evaluate_binary_region
from .optimization import SafeCandidateOptimizer

LabelMode = Literal["physical_feasibility", "high_yield"]


@dataclass
class ConstraintEvaluation:
    """Held-out evaluation results for a learned constraint classifier."""

    confusion_matrix: np.ndarray
    classification_report: Dict[str, dict]
    balanced_accuracy: float
    f1: float
    roc_auc: float
    average_precision: float


class ManufacturingConstraintLearner:
    """Learn and evaluate nonlinear operating constraints from manufacturing data.

    Two learning modes are supported:

    ``physical_feasibility``
        Uses the synthetic ground-truth physical feasibility label. This mode is
        useful as a benchmark because the learned boundary can be compared with
        the known hidden constraint.

    ``high_yield``
        Builds labels only from observed yield and does not use
        ``physical_feasible`` during fitting. This more closely represents an
        industrial setting where the true physical constraint is unknown.
    """

    feature_columns: Tuple[str, str] = ("temperature", "pressure")

    def __init__(
        self,
        data: pd.DataFrame,
        high_yield_threshold: float = 85.0,
        test_size: float = 0.25,
        random_state: int = 42,
        label_mode: LabelMode = "physical_feasibility",
        calibration_cv_splits: int = 3,
        safety_calibration_size: float = 0.10,
    ) -> None:
        required = {"temperature", "pressure", "yield"}
        if label_mode == "physical_feasibility":
            required.add("physical_feasible")
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if label_mode not in ("physical_feasibility", "high_yield"):
            raise ValueError(f"Unsupported label_mode: {label_mode}")
        if calibration_cv_splits < 2:
            raise ValueError("calibration_cv_splits must be at least 2")
        if not 0.0 < safety_calibration_size < 0.5:
            raise ValueError("safety_calibration_size must be between 0 and 0.5")

        self.data = data.copy()
        self.high_yield_threshold = high_yield_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.label_mode = label_mode
        self.calibration_cv_splits = calibration_cv_splits
        self.safety_calibration_size = float(safety_calibration_size)
        self.classifier_model: CalibratedClassifierCV | None = None
        self.model: CalibratedClassifierCV | None = None
        self.safety_filter: ConformalSafetyFilter | None = None
        self.simple_bounds: Dict[str, Dict[str, float]] = {}
        self.best_params_: Dict[str, object] | None = None
        self.cv_best_score_: float | None = None
        self._split = None

    def _labels(self) -> pd.Series:
        if self.label_mode == "physical_feasibility":
            return self.data["physical_feasible"].astype(int)
        return (self.data["yield"] >= self.high_yield_threshold).astype(int)

    @staticmethod
    def _base_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
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

    def _calibrated_model(self, estimator: Pipeline) -> CalibratedClassifierCV:
        """Wrap an estimator with cross-validated sigmoid probability calibration."""
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

    def _train_test_safety_split(self, X, y):
        """Create independent fit, safety-calibration, and held-out test sets."""
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
        return X_fit, X_safety, X_test, y_fit, y_safety, y_test

    def _fit_safety_filter(self, X_safety, y_safety) -> None:
        if self.model is None:
            raise RuntimeError("Fit the classifier before safety calibration")
        scores = np.asarray(self.model.predict_proba(X_safety)[:, 1], dtype=float)
        self.safety_filter = ConformalSafetyFilter().fit(scores, y_safety.to_numpy())

    def fit_feasibility_classifier(self) -> "ManufacturingConstraintLearner":
        """Fit the nonlinear SVM and an independent conformal safety filter."""
        X = self.data[list(self.feature_columns)]
        y = self._labels()

        X_fit, X_safety, X_test, y_fit, y_safety, y_test = (
            self._train_test_safety_split(X, y)
        )
        base_estimator = self._base_pipeline()
        X_classifier = pd.concat([X_fit, X_safety]).sort_index()
        y_classifier = pd.concat([y_fit, y_safety]).sort_index()
        self.classifier_model = self._calibrated_model(clone(base_estimator))
        self.classifier_model.fit(X_classifier, y_classifier)
        self.model = self._calibrated_model(base_estimator)
        self.model.fit(X_fit, y_fit)
        self._fit_safety_filter(X_safety, y_safety)
        self._split = (X_fit, X_test, y_fit, y_test)
        self.best_params_ = None
        self.cv_best_score_ = None
        return self

    def tune_hyperparameters(
        self,
        cv_splits: int = 5,
        scoring: str = "average_precision",
    ) -> "ManufacturingConstraintLearner":
        """Tune RBF-SVM hyperparameters using stratified cross-validation."""
        if cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")

        X = self.data[list(self.feature_columns)]
        y = self._labels()
        X_fit, X_safety, X_test, y_fit, y_safety, y_test = (
            self._train_test_safety_split(X, y)
        )

        search = GridSearchCV(
            estimator=self._base_pipeline(),
            param_grid={
                "classifier__C": [0.5, 1.0, 5.0, 10.0, 25.0, 50.0],
                "classifier__gamma": ["scale", 0.05, 0.1, 0.25, 0.5, 1.0],
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

        best_estimator = clone(search.best_estimator_)
        X_classifier = pd.concat([X_fit, X_safety]).sort_index()
        y_classifier = pd.concat([y_fit, y_safety]).sort_index()
        self.classifier_model = self._calibrated_model(clone(best_estimator))
        self.classifier_model.fit(X_classifier, y_classifier)
        self.model = self._calibrated_model(best_estimator)
        self.model.fit(X_fit, y_fit)
        self._fit_safety_filter(X_safety, y_safety)
        self._split = (X_fit, X_test, y_fit, y_test)
        self.best_params_ = dict(search.best_params_)
        self.cv_best_score_ = float(search.best_score_)
        return self

    def _continuous_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return continuous feasibility scores for ranking and curve metrics."""
        if self.model is None:
            raise RuntimeError("Fit the classifier before requesting scores")
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        return np.asarray(self.model.decision_function(X), dtype=float)

    def evaluate(self) -> ConstraintEvaluation:
        """Evaluate the learned constraint on held-out data."""
        if self.model is None or self.classifier_model is None or self._split is None:
            raise RuntimeError("Fit the classifier before evaluation")

        _, X_test, _, y_test = self._split
        predictions = self.classifier_model.predict(X_test)
        scores = self._continuous_scores(X_test)
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        return ConstraintEvaluation(
            confusion_matrix=confusion_matrix(y_test, predictions),
            classification_report=report,
            balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
            f1=float(f1_score(y_test, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, scores)),
            average_precision=float(average_precision_score(y_test, scores)),
        )

    def evaluate_against_physical_truth(self) -> ConstraintEvaluation:
        """Evaluate any fitted model against physical truth on the held-out rows.

        This is especially useful in ``high_yield`` mode: physical feasibility is
        not used for training, but synthetic experiments can still quantify how
        well outcome-derived constraints recover the hidden physical region.
        """
        if "physical_feasible" not in self.data.columns:
            raise RuntimeError("physical_feasible is required for benchmark evaluation")
        if self.model is None or self.classifier_model is None or self._split is None:
            raise RuntimeError("Fit the classifier before evaluation")

        _, X_test, _, _ = self._split
        y_true = self.data.loc[X_test.index, "physical_feasible"].astype(int)
        predictions = self.classifier_model.predict(X_test)
        scores = self._continuous_scores(X_test)
        report = classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        return ConstraintEvaluation(
            confusion_matrix=confusion_matrix(y_true, predictions),
            classification_report=report,
            balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
            f1=float(f1_score(y_true, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, scores)),
            average_precision=float(average_precision_score(y_true, scores)),
        )

    def learn_high_yield_bounds(
        self,
        quantile_margin: float = 0.01,
    ) -> Dict[str, Dict[str, float]]:
        """Learn robust descriptive bounds for high-yield observations.

        These are descriptive operating bounds, not exact physical constraints.
        Quantiles reduce sensitivity to individual noisy observations.
        """
        if not 0.0 <= quantile_margin < 0.5:
            raise ValueError("quantile_margin must be in [0, 0.5)")

        subset = self.data[self.data["yield"] >= self.high_yield_threshold]
        if "physical_feasible" in subset.columns:
            subset = subset[subset["physical_feasible"] == 1]
        if subset.empty:
            raise RuntimeError("No high-yield observations are available")

        lower_q = quantile_margin
        upper_q = 1.0 - quantile_margin
        self.simple_bounds = {
            column: {
                "min": float(subset[column].quantile(lower_q)),
                "max": float(subset[column].quantile(upper_q)),
            }
            for column in self.feature_columns
        }
        return self.simple_bounds

    def best_observed_feasible_point(self) -> pd.Series:
        """Return the highest-yield physically feasible observation."""
        if "physical_feasible" not in self.data.columns:
            raise RuntimeError("physical_feasible is required for this benchmark method")
        feasible = self.data[self.data["physical_feasible"] == 1]
        if feasible.empty:
            raise RuntimeError("No physically feasible observations are available")
        return feasible.loc[feasible["yield"].idxmax()]

    def predict_feasible(self, temperature: float, pressure: float) -> bool:
        """Predict whether an operating point belongs to the learned region."""
        if self.classifier_model is None:
            raise RuntimeError("Fit the classifier before prediction")
        X = pd.DataFrame({"temperature": [temperature], "pressure": [pressure]})
        return bool(self.classifier_model.predict(X)[0])

    def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return calibrated feasibility probabilities for candidate rows."""
        if self.model is None:
            raise RuntimeError("Fit the classifier before prediction")
        missing = set(self.feature_columns).difference(candidates.columns)
        if missing:
            raise ValueError(f"Missing candidate columns: {sorted(missing)}")
        return self.model.predict_proba(candidates.loc[:, self.feature_columns])

    def conformal_p_values(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return infeasible-class conformal p-values for candidate rows."""
        if self.safety_filter is None:
            raise RuntimeError("Fit the classifier before conformal safety scoring")
        probabilities = self.predict_proba(candidates)[:, 1]
        return self.safety_filter.p_values(probabilities)

    def predict_feasibility_probability(
        self,
        temperature: float,
        pressure: float,
    ) -> float:
        """Return the calibrated model probability of feasibility."""
        if self.model is None:
            raise RuntimeError("Fit the classifier before prediction")
        X = pd.DataFrame({"temperature": [temperature], "pressure": [pressure]})
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("The fitted model does not expose probabilities")
        return float(self.model.predict_proba(X)[0, 1])

    def predict_safe(
        self,
        temperature: float,
        pressure: float,
        min_probability: float = 0.50,
    ) -> bool:
        """Accept an operating point only above a configured probability threshold."""
        if not 0.0 < min_probability <= 1.0:
            raise ValueError("min_probability must be in (0, 1]")
        return self.predict_feasibility_probability(temperature, pressure) >= min_probability

    def conformal_p_value(
        self,
        temperature: float,
        pressure: float,
    ) -> float:
        """Return the infeasible-class conformal p-value for an operating point."""
        if self.safety_filter is None:
            raise RuntimeError("Fit the classifier before conformal safety scoring")
        score = self.predict_feasibility_probability(temperature, pressure)
        return float(self.safety_filter.p_values(np.array([score]))[0])

    def risk_controlled_threshold(self, alpha: float = 0.05) -> float:
        """Return the probability threshold implied by the conformal safety filter."""
        if self.safety_filter is None:
            raise RuntimeError("Fit the classifier before safety calibration")
        return self.safety_filter.probability_threshold(alpha=alpha)

    def evaluate_safety_filter(
        self,
        alpha: float = 0.05,
    ) -> SafetyFilterEvaluation:
        """Evaluate conformal screening on the untouched held-out test set."""
        if self.model is None or self._split is None or self.safety_filter is None:
            raise RuntimeError("Fit the classifier before safety evaluation")
        _, X_test, _, y_test = self._split
        scores = np.asarray(self.model.predict_proba(X_test)[:, 1], dtype=float)
        return self.safety_filter.evaluate(
            scores,
            y_test.to_numpy(),
            alpha=alpha,
        )

    def risk_controlled_optimizer(
        self,
        alpha: float = 0.05,
    ) -> SafeCandidateOptimizer:
        """Build an optimizer using the conformal score threshold.

        The conformal guarantee is marginal for a future candidate under
        exchangeability. Optimizing over many screened candidates introduces
        selection effects that require separate validation.
        """
        threshold = self.risk_controlled_threshold(alpha=alpha)
        if not 0.0 < threshold <= 1.0:
            raise RuntimeError(
                "The conformal threshold is outside the probability range; "
                "the requested alpha cannot produce a usable safe set."
            )
        if self.model is None:
            raise RuntimeError("Fit the classifier before building an optimizer")
        return SafeCandidateOptimizer(
            model=self.model,
            feature_columns=self.feature_columns,
            min_probability=threshold,
        )

    def boundary_metrics(self, grid_resolution: int = 200) -> BinaryRegionMetrics:
        """Evaluate learned-vs-true feasible regions on a dense synthetic grid."""
        if self.classifier_model is None:
            raise RuntimeError("Fit the classifier before boundary evaluation")
        if grid_resolution < 10:
            raise ValueError("grid_resolution must be at least 10")

        t_values = np.linspace(50.0, 450.0, grid_resolution)
        p_values = np.linspace(0.0, 10.0, grid_resolution)
        tt, pp = np.meshgrid(t_values, p_values)
        grid = pd.DataFrame(
            {"temperature": tt.ravel(), "pressure": pp.ravel()}
        )
        learned = self.classifier_model.predict(grid).reshape(tt.shape)
        truth = self.true_physical_feasibility(tt, pp)
        return evaluate_binary_region(truth, learned)

    def safe_optimizer(
        self,
        min_probability: float = 0.50,
    ) -> SafeCandidateOptimizer:
        """Build a candidate optimizer using the fitted learned constraint."""
        if self.model is None:
            raise RuntimeError("Fit the classifier before building an optimizer")
        return SafeCandidateOptimizer(
            model=self.model,
            feature_columns=self.feature_columns,
            min_probability=min_probability,
        )

    @staticmethod
    def true_physical_feasibility(
        temperature: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the known synthetic physical constraint for benchmarking."""
        return (
            (temperature >= 150.0)
            & (temperature <= 350.0)
            & (pressure >= 2.0)
            & (pressure <= 8.0)
            & (pressure <= -0.02 * (temperature - 250.0) ** 2 + 8.0)
        )

    def _save_or_show(
        self,
        fig: plt.Figure,
        output_path: str | Path | None,
        show: bool,
    ) -> None:
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_boundary_comparison(
        self,
        output_path: str | Path | None = None,
        show: bool = True,
        grid_resolution: int = 350,
    ) -> None:
        """Compare the learned decision region with the known synthetic truth."""
        if self.classifier_model is None:
            raise RuntimeError("Fit the classifier before plotting")

        t_values = np.linspace(50.0, 450.0, grid_resolution)
        p_values = np.linspace(0.0, 10.0, grid_resolution)
        tt, pp = np.meshgrid(t_values, p_values)
        grid = pd.DataFrame(
            {"temperature": tt.ravel(), "pressure": pp.ravel()}
        )
        learned = self.classifier_model.predict(grid).reshape(tt.shape)
        truth = self.true_physical_feasibility(tt, pp).astype(int)

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            self.data["temperature"],
            self.data["pressure"],
            c=self.data["yield"],
            cmap="viridis",
            alpha=0.55,
            s=22,
        )
        fig.colorbar(scatter, ax=ax, label="yield")
        ax.contour(
            tt,
            pp,
            truth,
            levels=[0.5],
            linewidths=2.2,
            linestyles="--",
        )
        ax.contour(
            tt,
            pp,
            learned,
            levels=[0.5],
            linewidths=2.2,
        )
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Pressure (MPa)")
        ax.set_title("True vs Learned Constraint Boundary")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        self._save_or_show(fig, output_path, show)

    def plot_roc_pr_curves(
        self,
        output_directory: str | Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot ROC and precision-recall curves for the held-out test set."""
        if self.model is None or self._split is None:
            raise RuntimeError("Fit the classifier before plotting")
        _, X_test, _, y_test = self._split
        scores = self._continuous_scores(X_test)

        fpr, tpr, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        roc_auc = roc_auc_score(y_test, scores)
        avg_precision = average_precision_score(y_test, scores)

        output_dir = Path(output_directory) if output_directory is not None else None

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        ax_roc.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Held-out ROC Curve")
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.25)
        fig_roc.tight_layout()
        roc_path = output_dir / "roc_curve.png" if output_dir is not None else None
        self._save_or_show(fig_roc, roc_path, show)

        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        ax_pr.plot(recall, precision, label=f"Average precision = {avg_precision:.3f}")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Held-out Precision-Recall Curve")
        ax_pr.legend()
        ax_pr.grid(True, alpha=0.25)
        fig_pr.tight_layout()
        pr_path = output_dir / "precision_recall_curve.png" if output_dir is not None else None
        self._save_or_show(fig_pr, pr_path, show)
