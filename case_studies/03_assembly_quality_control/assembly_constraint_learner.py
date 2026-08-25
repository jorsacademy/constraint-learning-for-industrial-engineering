"""Constraint-learning model for the assembly quality-control case study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class AssemblyEvaluation:
    """Held-out metrics for the learned acceptable-operation region."""

    confusion_matrix: np.ndarray
    balanced_accuracy: float
    f1: float
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    false_accept_rate: float
    false_reject_rate: float


class AssemblyQualityConstraintLearner:
    """Learn nonlinear assembly settings associated with acceptable quality."""

    feature_columns = (
        "torque",
        "angle",
        "tool_speed",
        "insertion_force",
        "component_temperature",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        required = set(self.feature_columns) | {"acceptable_operation"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")

        self.data = data.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.model: Pipeline | GridSearchCV | None = None
        self._split = None

    def fit(self, tune: bool = True) -> "AssemblyQualityConstraintLearner":
        """Fit an RBF-SVM classifier to the observed acceptable-operation label."""
        X = self.data[list(self.feature_columns)]
        y = self.data["acceptable_operation"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        pipeline = Pipeline(
            [
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
            cv = StratifiedKFold(
                n_splits=4,
                shuffle=True,
                random_state=self.random_state,
            )
            search = GridSearchCV(
                pipeline,
                param_grid={
                    "classifier__C": [1.0, 5.0, 10.0],
                    "classifier__gamma": ["scale", 0.1, 0.3],
                },
                scoring="average_precision",
                cv=cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            self.model = search
        else:
            pipeline.set_params(classifier__C=10.0, classifier__gamma=0.1)
            pipeline.fit(X_train, y_train)
            self.model = pipeline

        self._split = (X_train, X_test, y_train, y_test)
        return self

    def evaluate(self) -> AssemblyEvaluation:
        """Evaluate the learned constraint region on held-out observations."""
        if self.model is None or self._split is None:
            raise RuntimeError("Call fit before evaluate")

        _, X_test, _, y_test = self._split
        prediction = self.model.predict(X_test)
        probability = self.model.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, prediction)
        tn, fp, fn, tp = matrix.ravel()

        false_accept_rate = fp / (fp + tn) if fp + tn else 0.0
        false_reject_rate = fn / (fn + tp) if fn + tp else 0.0

        return AssemblyEvaluation(
            confusion_matrix=matrix,
            balanced_accuracy=float(balanced_accuracy_score(y_test, prediction)),
            f1=float(f1_score(y_test, prediction, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, probability)),
            average_precision=float(average_precision_score(y_test, probability)),
            precision=float(precision_score(y_test, prediction, zero_division=0)),
            recall=float(recall_score(y_test, prediction, zero_division=0)),
            false_accept_rate=float(false_accept_rate),
            false_reject_rate=float(false_reject_rate),
        )

    def acceptable_operating_bounds(
        self,
        quantile_margin: float = 0.02,
    ) -> Dict[str, Dict[str, float]]:
        """Return descriptive quantile bounds for observed acceptable operation."""
        if not 0.0 <= quantile_margin < 0.5:
            raise ValueError("quantile_margin must be in [0, 0.5)")

        subset = self.data[self.data["acceptable_operation"] == 1]
        if subset.empty:
            raise RuntimeError("No acceptable observations are available")

        return {
            column: {
                "min": float(subset[column].quantile(quantile_margin)),
                "max": float(subset[column].quantile(1.0 - quantile_margin)),
            }
            for column in self.feature_columns
        }

    def best_observed_joint(self) -> pd.Series:
        """Return the acceptable observation with the highest joint strength."""
        subset = self.data[self.data["acceptable_operation"] == 1]
        if subset.empty:
            raise RuntimeError("No acceptable observations are available")
        return subset.loc[subset["joint_strength"].idxmax()]

    def predict_acceptable(self, **settings: float) -> bool:
        """Predict whether one complete setting vector is acceptable."""
        if self.model is None:
            raise RuntimeError("Call fit before prediction")
        missing = [name for name in self.feature_columns if name not in settings]
        if missing:
            raise ValueError(f"Missing settings: {missing}")
        frame = pd.DataFrame([{name: settings[name] for name in self.feature_columns}])
        return bool(self.model.predict(frame)[0])

    def save_evaluation_plots(self, output_dir: str | Path) -> None:
        """Save ROC, precision-recall, and operating-region slice figures."""
        if self.model is None or self._split is None:
            raise RuntimeError("Call fit before plotting")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _, X_test, _, y_test = self._split
        probability = self.model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probability)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, probability):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Assembly Quality Constraint Learning: ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path / "assembly_quality_roc_curve.png", dpi=160)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_test, probability)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Assembly Quality Constraint Learning: Precision-Recall Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path / "assembly_quality_precision_recall_curve.png", dpi=160)
        plt.close(fig)

        torque_grid = np.linspace(20.0, 80.0, 220)
        angle_grid = np.linspace(30.0, 150.0, 220)
        torque_mesh, angle_mesh = np.meshgrid(torque_grid, angle_grid)
        grid = pd.DataFrame(
            {
                "torque": torque_mesh.ravel(),
                "angle": angle_mesh.ravel(),
                "tool_speed": 300.0,
                "insertion_force": 300.0,
                "component_temperature": 30.0,
            }
        )
        region = self.model.predict(grid).reshape(torque_mesh.shape)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.contourf(torque_mesh, angle_mesh, region, levels=[-0.5, 0.5, 1.5], alpha=0.25)
        observed = self.data.sample(min(len(self.data), 1500), random_state=self.random_state)
        scatter = ax.scatter(
            observed["torque"],
            observed["angle"],
            c=observed["acceptable_operation"],
            alpha=0.45,
            s=18,
        )
        fig.colorbar(scatter, ax=ax, label="acceptable_operation")
        ax.set_xlabel("Tightening Torque")
        ax.set_ylabel("Tightening Angle")
        ax.set_title(
            "Learned Assembly Operating Region\n"
            "Slice: tool speed=300, insertion force=300, temperature=30"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path / "assembly_quality_operating_region.png", dpi=160)
        plt.close(fig)
