"""Constraint learner for energy-efficient machine operation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


@dataclass
class EnergyConstraintEvaluation:
    """Held-out evaluation metrics for acceptable-operation classification."""

    confusion_matrix: np.ndarray
    classification_report: Dict[str, dict]
    balanced_accuracy: float
    f1: float
    roc_auc: float
    average_precision: float


class EnergyConstraintLearner:
    """Learn nonlinear acceptable machine-setting regions from observed data."""

    feature_columns: Tuple[str, str, str] = (
        "spindle_speed",
        "feed_rate",
        "load",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        required = {
            *self.feature_columns,
            "energy_per_unit",
            "throughput",
            "quality_score",
            "acceptable_operation",
        }
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")

        self.data = data.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.model: Pipeline | None = None
        self.best_params_: Dict[str, object] | None = None
        self._split = None

    def fit(self, tune: bool = True, cv_folds: int = 5) -> "EnergyConstraintLearner":
        """Fit an RBF-SVM classifier for the acceptable operating region."""
        X = self.data[list(self.feature_columns)]
        y = self.data["acceptable_operation"].astype(int)

        if y.nunique() < 2:
            raise RuntimeError("Both acceptable and unacceptable observations are required")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        if tune:
            if cv_folds < 2:
                raise ValueError("cv_folds must be at least 2")
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid={
                    "classifier__C": [1.0, 5.0, 10.0, 25.0],
                    "classifier__gamma": ["scale", 0.1, 0.3, 0.7],
                },
                scoring="average_precision",
                cv=cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            self.best_params_ = dict(search.best_params_)
        else:
            pipeline.set_params(classifier__C=10.0, classifier__gamma="scale")
            pipeline.fit(X_train, y_train)
            self.model = pipeline
            self.best_params_ = {
                "classifier__C": 10.0,
                "classifier__gamma": "scale",
            }

        self._split = (X_train, X_test, y_train, y_test)
        return self

    def evaluate(self) -> EnergyConstraintEvaluation:
        """Evaluate acceptable-operation detection on held-out observations."""
        if self.model is None or self._split is None:
            raise RuntimeError("Call fit before evaluate")

        _, X_test, _, y_test = self._split
        predictions = self.model.predict(X_test)
        scores = self.model.predict_proba(X_test)[:, 1]

        return EnergyConstraintEvaluation(
            confusion_matrix=confusion_matrix(y_test, predictions),
            classification_report=classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
            balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
            f1=float(f1_score(y_test, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, scores)),
            average_precision=float(average_precision_score(y_test, scores)),
        )

    def predict_acceptable(
        self,
        spindle_speed: float,
        feed_rate: float,
        load: float,
    ) -> bool:
        """Predict whether a proposed machine setting is acceptable."""
        if self.model is None:
            raise RuntimeError("Call fit before prediction")

        point = pd.DataFrame(
            {
                "spindle_speed": [spindle_speed],
                "feed_rate": [feed_rate],
                "load": [load],
            }
        )
        return bool(self.model.predict(point)[0])

    def best_observed_energy_efficient_point(self) -> pd.Series:
        """Return the lowest-energy observed point satisfying all requirements."""
        acceptable = self.data[self.data["acceptable_operation"] == 1]
        if acceptable.empty:
            raise RuntimeError("No acceptable observations are available")
        return acceptable.loc[acceptable["energy_per_unit"].idxmin()]

    def summarize_acceptable_region(self, quantile_margin: float = 0.02) -> Dict[str, Dict[str, float]]:
        """Return descriptive quantile bounds for observed acceptable settings."""
        if not 0.0 <= quantile_margin < 0.5:
            raise ValueError("quantile_margin must be in [0, 0.5)")

        acceptable = self.data[self.data["acceptable_operation"] == 1]
        if acceptable.empty:
            raise RuntimeError("No acceptable observations are available")

        lower = quantile_margin
        upper = 1.0 - quantile_margin
        return {
            column: {
                "min": float(acceptable[column].quantile(lower)),
                "max": float(acceptable[column].quantile(upper)),
            }
            for column in self.feature_columns
        }

    def estimated_energy_saving(self) -> float:
        """Estimate mean energy reduction versus all observed machine settings."""
        acceptable = self.data[self.data["acceptable_operation"] == 1]
        if acceptable.empty:
            raise RuntimeError("No acceptable observations are available")

        baseline = float(self.data["energy_per_unit"].mean())
        efficient = float(acceptable["energy_per_unit"].mean())
        return 100.0 * (baseline - efficient) / baseline

    def save_evaluation_curves(self, output_dir: str | Path) -> None:
        """Save ROC and precision-recall curves for the held-out test data."""
        if self.model is None or self._split is None:
            raise RuntimeError("Call fit before plotting")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        _, X_test, _, y_test = self._split
        scores = self.model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label="RBF-SVM")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Energy-Efficiency Constraint Learning: ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path / "energy_efficiency_roc_curve.png", dpi=160)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_test, scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Energy-Efficiency Constraint Learning: Precision-Recall Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path / "energy_efficiency_precision_recall_curve.png", dpi=160)
        plt.close(fig)

    def save_operating_region_slice(
        self,
        output_dir: str | Path,
        load_value: float = 0.60,
        grid_resolution: int = 180,
    ) -> None:
        """Save a 2D speed-feed slice of the learned 3D acceptable region."""
        if self.model is None:
            raise RuntimeError("Call fit before plotting")
        if grid_resolution < 20:
            raise ValueError("grid_resolution must be at least 20")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        speed = np.linspace(
            self.data["spindle_speed"].min(),
            self.data["spindle_speed"].max(),
            grid_resolution,
        )
        feed = np.linspace(
            self.data["feed_rate"].min(),
            self.data["feed_rate"].max(),
            grid_resolution,
        )
        speed_grid, feed_grid = np.meshgrid(speed, feed)
        grid = pd.DataFrame(
            {
                "spindle_speed": speed_grid.ravel(),
                "feed_rate": feed_grid.ravel(),
                "load": np.full(speed_grid.size, load_value),
            }
        )
        probability = self.model.predict_proba(grid)[:, 1].reshape(speed_grid.shape)

        fig, ax = plt.subplots(figsize=(9, 6))
        contour = ax.contourf(
            speed_grid,
            feed_grid,
            probability,
            levels=np.linspace(0.0, 1.0, 11),
            alpha=0.75,
        )
        fig.colorbar(contour, ax=ax, label="Predicted acceptable-operation probability")
        ax.contour(
            speed_grid,
            feed_grid,
            probability,
            levels=[0.5],
            linewidths=2.0,
        )
        ax.set_xlabel("Spindle Speed (rpm)")
        ax.set_ylabel("Feed Rate (mm/min)")
        ax.set_title(f"Learned Acceptable Region at Load = {load_value:.2f}")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path / "energy_efficiency_operating_region.png", dpi=160)
        plt.close(fig)
