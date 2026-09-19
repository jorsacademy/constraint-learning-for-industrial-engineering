"""Decision-tree surrogates for solver-embeddable learned constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree


class RiskControlledTeacher(Protocol):
    """Minimal teacher interface required for surrogate distillation."""

    feature_columns: Sequence[str]

    def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray: ...

    def conformal_p_values(self, candidates: pd.DataFrame) -> np.ndarray: ...

    def risk_controlled_threshold(self, alpha: float = 0.05) -> float: ...


@dataclass(frozen=True)
class SurrogateFidelity:
    """Held-out agreement between the solver surrogate and learned teacher."""

    accuracy: float
    balanced_accuracy: float
    false_safe_rate: float
    false_unsafe_rate: float
    safe_precision: float
    teacher_safe_fraction: float
    surrogate_safe_fraction: float
    test_size: int


@dataclass(frozen=True)
class TreePathCondition:
    """One axis-aligned split condition on a root-to-leaf path."""

    feature_index: int
    feature_name: str
    sense: str
    threshold: float


@dataclass(frozen=True)
class SafeLeafPath:
    """A tree leaf predicted as safe together with its path constraints."""

    leaf_id: int
    conditions: tuple[TreePathCondition, ...]


def sample_uniform_design(
    bounds: Mapping[str, tuple[float, float]],
    n_samples: int,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample a reproducible uniform design inside finite variable bounds."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not bounds:
        raise ValueError("bounds must not be empty")

    rng = np.random.default_rng(random_state)
    columns: dict[str, np.ndarray] = {}
    for feature, (lower, upper) in bounds.items():
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("surrogate-sampling bounds must be finite")
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {feature}: {lower}, {upper}")
        columns[feature] = rng.uniform(lower, upper, n_samples)
    return pd.DataFrame(columns)


class RiskControlledTreeSurrogate:
    """Distill a conformal-safe teacher region into a decision tree.

    The decision tree is intentionally interpretable and solver-embeddable.
    Fidelity is measured against the original teacher on a held-out design set.
    The surrogate does not replace the teacher: every solver solution must still
    be audited against the original calibrated/conformal model.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        *,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        split_test_size: float = 0.25,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if not 0.0 < split_test_size < 0.5:
            raise ValueError("split_test_size must be between 0 and 0.5")

        self.feature_columns = tuple(feature_columns)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.split_test_size = float(split_test_size)
        self.model: DecisionTreeClassifier | None = None
        self.alpha: float | None = None
        self.teacher_probability_threshold: float | None = None
        self.fidelity_: SurrogateFidelity | None = None

    def _validate_design(self, design_points: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(design_points.columns)
        if missing:
            raise ValueError(f"Missing surrogate features: {sorted(missing)}")
        if design_points.empty:
            raise ValueError("design_points must not be empty")
        X = design_points.loc[:, self.feature_columns].astype(float)
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("design_points must contain finite feature values")
        return X

    @staticmethod
    def _fidelity(y_true: np.ndarray, y_pred: np.ndarray) -> SurrogateFidelity:
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        def ratio(num: int, den: int) -> float:
            return float(num / den) if den else 0.0

        return SurrogateFidelity(
            accuracy=float(np.mean(y_true == y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            false_safe_rate=ratio(int(fp), int(fp + tn)),
            false_unsafe_rate=ratio(int(fn), int(fn + tp)),
            safe_precision=ratio(int(tp), int(tp + fp)),
            teacher_safe_fraction=float(np.mean(y_true)),
            surrogate_safe_fraction=float(np.mean(y_pred)),
            test_size=int(y_true.size),
        )

    def fit_from_teacher(
        self,
        teacher: RiskControlledTeacher,
        design_points: pd.DataFrame,
        *,
        alpha: float = 0.10,
    ) -> "RiskControlledTreeSurrogate":
        """Fit a tree to the teacher's conformal-safe region."""
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if tuple(teacher.feature_columns) != self.feature_columns:
            raise ValueError("teacher feature order must match surrogate feature order")

        X = self._validate_design(design_points)
        p_values = np.asarray(teacher.conformal_p_values(X), dtype=float).reshape(-1)
        if p_values.size != len(X):
            raise ValueError("teacher conformal_p_values returned the wrong length")
        y = (p_values <= alpha).astype(int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                "surrogate design must contain both teacher-safe and teacher-unsafe points"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.split_test_size,
            random_state=self.random_state,
            stratify=y,
        )
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
        )
        tree.fit(X_train, y_train)
        self.model = tree
        self.alpha = float(alpha)
        self.teacher_probability_threshold = float(
            teacher.risk_controlled_threshold(alpha=alpha)
        )
        self.fidelity_ = self._fidelity(y_test, tree.predict(X_test).astype(int))
        return self

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit the surrogate before prediction")
        X = self._validate_design(candidates)
        return self.model.predict(X).astype(int)

    def safe_leaf_paths(self) -> tuple[SafeLeafPath, ...]:
        """Extract every root-to-leaf path whose predicted class is safe."""
        if self.model is None:
            raise RuntimeError("Fit the surrogate before extracting tree paths")

        tree = self.model.tree_
        classes = self.model.classes_
        paths: list[SafeLeafPath] = []

        def walk(node: int, conditions: list[TreePathCondition]) -> None:
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                predicted_class = int(classes[int(np.argmax(tree.value[node][0]))])
                if predicted_class == 1:
                    paths.append(
                        SafeLeafPath(
                            leaf_id=int(node),
                            conditions=tuple(conditions),
                        )
                    )
                return

            feature_index = int(tree.feature[node])
            feature_name = self.feature_columns[feature_index]
            threshold = float(tree.threshold[node])
            walk(
                int(tree.children_left[node]),
                conditions
                + [
                    TreePathCondition(
                        feature_index=feature_index,
                        feature_name=feature_name,
                        sense="<=",
                        threshold=threshold,
                    )
                ],
            )
            walk(
                int(tree.children_right[node]),
                conditions
                + [
                    TreePathCondition(
                        feature_index=feature_index,
                        feature_name=feature_name,
                        sense=">",
                        threshold=threshold,
                    )
                ],
            )

        walk(0, [])
        if not paths:
            raise RuntimeError("The fitted surrogate contains no safe leaves")
        return tuple(paths)
