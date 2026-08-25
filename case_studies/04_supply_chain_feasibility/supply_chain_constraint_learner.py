"""Constraint learner for the supply-chain feasibility case study."""

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
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class SupplyChainEvaluation:
    confusion_matrix: np.ndarray
    balanced_accuracy: float
    f1: float
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    missed_failure_rate: float


class SupplyChainConstraintLearner:
    """Learn combinations of supply-chain settings that sustain service targets."""

    feature_columns: Tuple[str, ...] = (
        "supplier_lead_time",
        "demand_cv",
        "order_quantity",
        "safety_stock",
        "supplier_utilization",
        "transport_time",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        required = set(self.feature_columns) | {
            "service_level",
            "total_logistics_cost",
            "feasible_service",
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

    def fit(self, tune: bool = True) -> "SupplyChainConstraintLearner":
        X = self.data[list(self.feature_columns)]
        y = self.data["feasible_service"].astype(int)
        if y.nunique() < 2:
            raise RuntimeError("Both feasible and infeasible observations are required")

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
            search = GridSearchCV(
                pipeline,
                param_grid={
                    "classifier__C": [1.0, 3.0, 10.0],
                    "classifier__gamma": ["scale", 0.1, 0.3],
                },
                scoring="average_precision",
                cv=4,
                n_jobs=None,
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            self.best_params_ = dict(search.best_params_)
        else:
            pipeline.set_params(classifier__C=3.0, classifier__gamma="scale")
            pipeline.fit(X_train, y_train)
            self.model = pipeline
            self.best_params_ = {
                "classifier__C": 3.0,
                "classifier__gamma": "scale",
            }

        self._split = (X_train, X_test, y_train, y_test)
        return self

    def evaluate(self) -> SupplyChainEvaluation:
        if self.model is None or self._split is None:
            raise RuntimeError("Call fit before evaluate")

        _, X_test, _, y_test = self._split
        pred = self.model.predict(X_test)
        score = self.model.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        missed_failure_rate = fp / (tn + fp) if (tn + fp) else 0.0

        return SupplyChainEvaluation(
            confusion_matrix=matrix,
            balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
            f1=float(f1_score(y_test, pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, score)),
            average_precision=float(average_precision_score(y_test, score)),
            precision=float(precision_score(y_test, pred, zero_division=0)),
            recall=float(recall_score(y_test, pred, zero_division=0)),
            missed_failure_rate=float(missed_failure_rate),
        )

    def descriptive_bounds(self, quantile_margin: float = 0.02) -> Dict[str, Dict[str, float]]:
        if not 0.0 <= quantile_margin < 0.5:
            raise ValueError("quantile_margin must be in [0, 0.5)")
        feasible = self.data[self.data["feasible_service"] == 1]
        if feasible.empty:
            raise RuntimeError("No feasible observations are available")
        return {
            col: {
                "min": float(feasible[col].quantile(quantile_margin)),
                "max": float(feasible[col].quantile(1.0 - quantile_margin)),
            }
            for col in self.feature_columns
        }

    def best_observed_feasible_point(self) -> pd.Series:
        feasible = self.data[self.data["feasible_service"] == 1]
        if feasible.empty:
            raise RuntimeError("No feasible observations are available")
        score = feasible["total_logistics_cost"] + 300.0 * (1.0 - feasible["service_level"])
        return feasible.loc[score.idxmin()]

    def predict_feasible(self, **kwargs: float) -> bool:
        if self.model is None:
            raise RuntimeError("Call fit before prediction")
        missing = [col for col in self.feature_columns if col not in kwargs]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X = pd.DataFrame([{col: kwargs[col] for col in self.feature_columns}])
        return bool(self.model.predict(X)[0])

    def plot_curves(self, output_dir: str | Path) -> None:
        if self.model is None or self._split is None:
            raise RuntimeError("Call fit before plotting")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _, X_test, _, y_test = self._split
        score = self.model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, score)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, score):.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Supply-Chain Feasibility ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "supply_chain_roc_curve.png", dpi=160)
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, score)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, label=f"AP = {average_precision_score(y_test, score):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Supply-Chain Feasibility Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "supply_chain_precision_recall_curve.png", dpi=160)
        plt.close()

    def plot_operating_slice(
        self,
        output_dir: str | Path,
        supplier_utilization: float = 0.75,
        order_quantity: float = 800.0,
        safety_stock: float = 250.0,
        transport_time: float = 3.0,
    ) -> None:
        """Plot a lead-time/demand-volatility slice of the six-dimensional region."""
        if self.model is None:
            raise RuntimeError("Call fit before plotting")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        lead = np.linspace(2.0, 18.0, 180)
        cv = np.linspace(0.05, 0.65, 180)
        xx, yy = np.meshgrid(lead, cv)
        grid = pd.DataFrame(
            {
                "supplier_lead_time": xx.ravel(),
                "demand_cv": yy.ravel(),
                "order_quantity": order_quantity,
                "safety_stock": safety_stock,
                "supplier_utilization": supplier_utilization,
                "transport_time": transport_time,
            }
        )
        zz = self.model.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.30)
        sample = self.data.sample(min(1000, len(self.data)), random_state=self.random_state)
        plt.scatter(
            sample["supplier_lead_time"],
            sample["demand_cv"],
            c=sample["feasible_service"],
            s=16,
            alpha=0.45,
        )
        plt.xlabel("Supplier Lead Time (days)")
        plt.ylabel("Demand Coefficient of Variation")
        plt.title("Learned Supply-Chain Feasibility Region: 2D Slice")
        plt.tight_layout()
        plt.savefig(output / "supply_chain_feasibility_region.png", dpi=160)
        plt.close()
