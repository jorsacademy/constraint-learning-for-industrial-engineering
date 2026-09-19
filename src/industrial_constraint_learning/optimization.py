"""Downstream optimization utilities for learned probabilistic constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd


class ProbabilisticConstraintModel(Protocol):
    """Minimal interface required by :class:`SafeCandidateOptimizer`."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


ObjectiveFunction = Callable[
    [pd.DataFrame],
    pd.Series | np.ndarray | Sequence[float],
]
HardConstraint = Callable[
    [pd.DataFrame],
    pd.Series | np.ndarray | Sequence[bool],
]


@dataclass(frozen=True)
class CandidateOptimizationResult:
    """Best candidate satisfying hard and learned probabilistic constraints."""

    point: pd.Series
    objective_value: float
    feasibility_probability: float
    candidates_evaluated: int
    safe_candidates: int


class SafeCandidateOptimizer:
    """Optimize over an explicit candidate set under a learned safe constraint.

    The class deliberately does not claim that a probabilistic classifier is a
    deterministic engineering constraint. A candidate is accepted only when it
    satisfies all user-supplied hard constraints and its predicted feasibility
    probability exceeds ``min_probability``.
    """

    def __init__(
        self,
        model: ProbabilisticConstraintModel,
        feature_columns: Sequence[str],
        min_probability: float = 0.95,
    ) -> None:
        if not 0.0 < min_probability <= 1.0:
            raise ValueError("min_probability must be in (0, 1]")
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        self.model = model
        self.feature_columns = tuple(feature_columns)
        self.min_probability = float(min_probability)

    @staticmethod
    def _coerce_vector(values, n_rows: int, *, name: str, dtype=None) -> np.ndarray:
        array = np.asarray(values, dtype=dtype).reshape(-1)
        if array.size != n_rows:
            raise ValueError(f"{name} must return one value per candidate")
        return array

    def optimize(
        self,
        candidates: pd.DataFrame,
        objective: ObjectiveFunction,
        *,
        hard_constraint: HardConstraint | None = None,
        maximize: bool = True,
    ) -> CandidateOptimizationResult:
        """Return the best safe candidate according to ``objective``."""
        if candidates.empty:
            raise ValueError("candidates must not be empty")
        missing = set(self.feature_columns).difference(candidates.columns)
        if missing:
            raise ValueError(f"Missing candidate columns: {sorted(missing)}")

        features = candidates.loc[:, self.feature_columns]
        probabilities = np.asarray(self.model.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape != (len(candidates), 2):
            raise ValueError("model.predict_proba must return shape (n_candidates, 2)")
        feasible_probability = probabilities[:, 1]
        if not np.isfinite(feasible_probability).all():
            raise ValueError("model returned non-finite feasibility probabilities")

        safe_mask = feasible_probability >= self.min_probability
        if hard_constraint is not None:
            hard_mask = self._coerce_vector(
                hard_constraint(candidates),
                len(candidates),
                name="hard_constraint",
                dtype=bool,
            )
            safe_mask &= hard_mask

        safe_indices = np.flatnonzero(safe_mask)
        if safe_indices.size == 0:
            raise RuntimeError("No candidate satisfies the configured safe constraints")

        objective_values = self._coerce_vector(
            objective(candidates),
            len(candidates),
            name="objective",
            dtype=float,
        )
        if not np.isfinite(objective_values[safe_indices]).all():
            raise ValueError("objective returned non-finite values for safe candidates")

        safe_objectives = objective_values[safe_indices]
        relative_best = int(
            np.argmax(safe_objectives) if maximize else np.argmin(safe_objectives)
        )
        best_index = int(safe_indices[relative_best])

        return CandidateOptimizationResult(
            point=candidates.iloc[best_index].copy(),
            objective_value=float(objective_values[best_index]),
            feasibility_probability=float(feasible_probability[best_index]),
            candidates_evaluated=int(len(candidates)),
            safe_candidates=int(safe_indices.size),
        )
