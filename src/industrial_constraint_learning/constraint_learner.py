"""Constraint-learning models for the manufacturing case study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ConstraintEvaluation:
    """Held-out evaluation results for the feasibility classifier."""

    confusion_matrix: np.ndarray
    classification_report: Dict[str, dict]


class ManufacturingConstraintLearner:
    """Learn and evaluate feasible operating regions from manufacturing data."""

    feature_columns: Tuple[str, str] = ("temperature", "pressure")

    def __init__(
        self,
        data: pd.DataFrame,
        high_yield_threshold: float = 85.0,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        required = {"temperature", "pressure", "yield", "physical_feasible"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")

        self.data = data.copy()
        self.high_yield_threshold = high_yield_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.model: Pipeline | None = None
        self.simple_bounds: Dict[str, Dict[str, float]] = {}
        self._split = None

    def fit_feasibility_classifier(self) -> "ManufacturingConstraintLearner":
        """Fit a nonlinear classifier for the physical feasible region."""
        X = self.data[list(self.feature_columns)]
        y = self.data["physical_feasible"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        probability=False,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.model.fit(X_train, y_train)
        self._split = (X_train, X_test, y_train, y_test)
        return self

    def evaluate(self) -> ConstraintEvaluation:
        """Evaluate the learned physical feasibility constraint on held-out data."""
        if self.model is None or self._split is None:
            raise RuntimeError("Call fit_feasibility_classifier before evaluate")

        _, X_test, _, y_test = self._split
        predictions = self.model.predict(X_test)
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions)
        return ConstraintEvaluation(matrix, report)

    def learn_high_yield_bounds(self, quantile_margin: float = 0.01) -> Dict[str, Dict[str, float]]:
        """Learn interpretable bounds that summarize high-yield feasible operation.

        These are descriptive operating bounds, not exact physical constraints.
        Quantiles reduce sensitivity to single noisy observations.
        """
        if not 0.0 <= quantile_margin < 0.5:
            raise ValueError("quantile_margin must be in [0, 0.5)")

        subset = self.data[
            (self.data["physical_feasible"] == 1)
            & (self.data["yield"] >= self.high_yield_threshold)
        ]
        if subset.empty:
            raise RuntimeError("No high-yield feasible observations are available")

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
        """Return the highest-yield feasible observation in the data set."""
        feasible = self.data[self.data["physical_feasible"] == 1]
        if feasible.empty:
            raise RuntimeError("No physically feasible observations are available")
        return feasible.loc[feasible["yield"].idxmax()]

    def predict_feasible(self, temperature: float, pressure: float) -> bool:
        """Predict whether an operating point belongs to the learned feasible region."""
        if self.model is None:
            raise RuntimeError("Call fit_feasibility_classifier before prediction")
        X = pd.DataFrame(
            {"temperature": [temperature], "pressure": [pressure]}
        )
        return bool(self.model.predict(X)[0])

    def plot_learned_region(self) -> None:
        """Visualize data, the learned feasible region, and descriptive bounds."""
        if self.model is None:
            raise RuntimeError("Call fit_feasibility_classifier before plotting")

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            self.data["temperature"],
            self.data["pressure"],
            c=self.data["yield"],
            cmap="viridis",
            alpha=0.65,
            s=28,
        )
        fig.colorbar(scatter, ax=ax, label="yield")

        DecisionBoundaryDisplay.from_estimator(
            self.model,
            self.data[list(self.feature_columns)],
            response_method="predict",
            alpha=0.18,
            ax=ax,
            grid_resolution=300,
        )

        if self.simple_bounds:
            t_bounds = self.simple_bounds["temperature"]
            p_bounds = self.simple_bounds["pressure"]
            ax.axvline(t_bounds["min"], linestyle="--")
            ax.axvline(t_bounds["max"], linestyle="--")
            ax.axhline(p_bounds["min"], linestyle="--")
            ax.axhline(p_bounds["max"], linestyle="--")

        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Pressure (MPa)")
        ax.set_title("Learned Feasible Region for the Manufacturing Process")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
