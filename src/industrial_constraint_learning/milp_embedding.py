"""MILP embedding for risk-controlled decision-tree surrogates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .surrogate import RiskControlledTeacher, RiskControlledTreeSurrogate


@dataclass(frozen=True)
class LinearConstraintSpec:
    """Linear hard constraint over the ordered decision features."""

    coefficients: tuple[float, ...]
    lower_bound: float = -np.inf
    upper_bound: float = np.inf

    def __init__(
        self,
        coefficients: Sequence[float],
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
    ) -> None:
        object.__setattr__(
            self,
            "coefficients",
            tuple(float(value) for value in coefficients),
        )
        object.__setattr__(self, "lower_bound", float(lower_bound))
        object.__setattr__(self, "upper_bound", float(upper_bound))


@dataclass(frozen=True)
class MILPEmbeddingResult:
    """Audited result from surrogate-embedded mixed-integer optimization."""

    point: pd.Series
    objective_value: float
    solver_objective_value: float
    solver_status: int
    solver_message: str
    selected_leaf_id: int | None
    surrogate_predicted_safe: bool
    teacher_probability: float
    conformal_p_value: float
    audited_safe: bool
    used_fallback: bool
    surrogate_fidelity_balanced_accuracy: float | None


class SurrogateMILPOptimizer:
    """Embed safe decision-tree leaves as a disjunctive MILP constraint.

    The tree formulation is exact for the fitted tree surrogate. Because the tree
    only approximates the original learned/conformal constraint, the returned
    solver point is always audited against the teacher. If the audit fails, the
    optimizer either uses an explicitly supplied candidate fallback set or fails
    closed.
    """

    def __init__(
        self,
        teacher: RiskControlledTeacher,
        surrogate: RiskControlledTreeSurrogate,
        bounds: Mapping[str, tuple[float, float]],
        *,
        alpha: float = 0.10,
        split_epsilon: float = 1e-7,
    ) -> None:
        if surrogate.model is None:
            raise ValueError("surrogate must be fitted before MILP construction")
        if tuple(teacher.feature_columns) != tuple(surrogate.feature_columns):
            raise ValueError("teacher and surrogate feature order must match")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if split_epsilon <= 0.0:
            raise ValueError("split_epsilon must be positive")

        self.teacher = teacher
        self.surrogate = surrogate
        self.feature_columns = tuple(surrogate.feature_columns)
        self.alpha = float(alpha)
        self.split_epsilon = float(split_epsilon)

        if set(bounds) != set(self.feature_columns):
            raise ValueError("bounds must be supplied for every feature and no others")
        self.bounds = {
            feature: (float(bounds[feature][0]), float(bounds[feature][1]))
            for feature in self.feature_columns
        }
        for feature, (lower, upper) in self.bounds.items():
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"Invalid finite bounds for {feature}")

    def _validate_objective(self, coefficients: Sequence[float]) -> np.ndarray:
        objective = np.asarray(coefficients, dtype=float).reshape(-1)
        if objective.size != len(self.feature_columns):
            raise ValueError("objective must contain one coefficient per feature")
        if not np.isfinite(objective).all():
            raise ValueError("objective coefficients must be finite")
        return objective

    def _validate_hard_constraints(
        self,
        hard_constraints: Sequence[LinearConstraintSpec],
    ) -> None:
        n = len(self.feature_columns)
        for spec in hard_constraints:
            if len(spec.coefficients) != n:
                raise ValueError(
                    "every hard constraint must contain one coefficient per feature"
                )
            if spec.lower_bound > spec.upper_bound:
                raise ValueError("hard constraint lower_bound exceeds upper_bound")

    def _tree_constraints(
        self,
        hard_constraints: Sequence[LinearConstraintSpec],
    ) -> tuple[LinearConstraint, tuple[int, ...]]:
        safe_leaves = self.surrogate.safe_leaf_paths()
        n_x = len(self.feature_columns)
        n_z = len(safe_leaves)
        n_vars = n_x + n_z

        rows: list[np.ndarray] = []
        lowers: list[float] = []
        uppers: list[float] = []

        # Exactly one safe leaf must be active.
        row = np.zeros(n_vars, dtype=float)
        row[n_x:] = 1.0
        rows.append(row)
        lowers.append(1.0)
        uppers.append(1.0)

        for leaf_position, leaf in enumerate(safe_leaves):
            z_index = n_x + leaf_position
            for condition in leaf.conditions:
                feature_index = condition.feature_index
                lower, upper = self.bounds[condition.feature_name]
                threshold = float(condition.threshold)
                row = np.zeros(n_vars, dtype=float)

                if condition.sense == "<=":
                    # x <= t + M(1-z)
                    M = max(upper - threshold, 0.0)
                    row[feature_index] = 1.0
                    row[z_index] = M
                    rows.append(row)
                    lowers.append(-np.inf)
                    uppers.append(threshold + M)
                elif condition.sense == ">":
                    epsilon = self.split_epsilon * max(1.0, upper - lower)
                    target = threshold + epsilon
                    # x >= target - M(1-z)
                    M = max(target - lower, 0.0)
                    row[feature_index] = 1.0
                    row[z_index] = -M
                    rows.append(row)
                    lowers.append(target - M)
                    uppers.append(np.inf)
                else:
                    raise RuntimeError(f"Unsupported tree condition: {condition.sense}")

        for spec in hard_constraints:
            row = np.zeros(n_vars, dtype=float)
            row[:n_x] = np.asarray(spec.coefficients, dtype=float)
            rows.append(row)
            lowers.append(spec.lower_bound)
            uppers.append(spec.upper_bound)

        matrix = np.vstack(rows)
        return (
            LinearConstraint(
                matrix,
                np.asarray(lowers, dtype=float),
                np.asarray(uppers, dtype=float),
            ),
            tuple(leaf.leaf_id for leaf in safe_leaves),
        )

    def _audit(self, point: pd.Series) -> tuple[float, float, bool, bool]:
        frame = pd.DataFrame(
            [{feature: float(point[feature]) for feature in self.feature_columns}]
        )
        surrogate_safe = bool(self.surrogate.predict(frame)[0])
        probabilities = np.asarray(self.teacher.predict_proba(frame), dtype=float)
        if probabilities.shape != (1, 2):
            raise ValueError("teacher predict_proba must return shape (n, 2)")
        probability = float(probabilities[0, 1])
        p_values = np.asarray(self.teacher.conformal_p_values(frame), dtype=float)
        if p_values.size != 1:
            raise ValueError("teacher conformal_p_values returned the wrong length")
        p_value = float(p_values[0])
        audited_safe = bool(p_value <= self.alpha)
        return probability, p_value, audited_safe, surrogate_safe

    def _hard_mask(
        self,
        candidates: pd.DataFrame,
        hard_constraints: Sequence[LinearConstraintSpec],
    ) -> np.ndarray:
        matrix = candidates.loc[:, self.feature_columns].to_numpy(dtype=float)
        mask = np.ones(len(candidates), dtype=bool)
        for j, feature in enumerate(self.feature_columns):
            lower, upper = self.bounds[feature]
            mask &= matrix[:, j] >= lower - 1e-9
            mask &= matrix[:, j] <= upper + 1e-9
        for spec in hard_constraints:
            value = matrix @ np.asarray(spec.coefficients, dtype=float)
            mask &= value >= spec.lower_bound - 1e-9
            mask &= value <= spec.upper_bound + 1e-9
        return mask

    def _fallback(
        self,
        candidates: pd.DataFrame,
        objective: np.ndarray,
        hard_constraints: Sequence[LinearConstraintSpec],
        *,
        maximize: bool,
    ) -> MILPEmbeddingResult:
        missing = set(self.feature_columns).difference(candidates.columns)
        if missing:
            raise ValueError(f"Missing fallback columns: {sorted(missing)}")
        if candidates.empty:
            raise RuntimeError("Fallback candidate set is empty")

        hard_mask = self._hard_mask(candidates, hard_constraints)
        p_values = np.asarray(
            self.teacher.conformal_p_values(
                candidates.loc[:, self.feature_columns]
            ),
            dtype=float,
        ).reshape(-1)
        safe_mask = p_values <= self.alpha
        eligible = np.flatnonzero(hard_mask & safe_mask)
        if eligible.size == 0:
            raise RuntimeError(
                "MILP solution failed teacher audit and no audited fallback candidate exists"
            )

        matrix = candidates.loc[:, self.feature_columns].to_numpy(dtype=float)
        values = matrix @ objective
        local = int(
            np.argmax(values[eligible]) if maximize else np.argmin(values[eligible])
        )
        index = int(eligible[local])
        point = candidates.iloc[index].loc[list(self.feature_columns)].copy()
        probability, p_value, audited_safe, surrogate_safe = self._audit(point)
        return MILPEmbeddingResult(
            point=point,
            objective_value=float(values[index]),
            solver_objective_value=float("nan"),
            solver_status=-1,
            solver_message="Fallback candidate search after failed teacher audit",
            selected_leaf_id=None,
            surrogate_predicted_safe=surrogate_safe,
            teacher_probability=probability,
            conformal_p_value=p_value,
            audited_safe=audited_safe,
            used_fallback=True,
            surrogate_fidelity_balanced_accuracy=(
                self.surrogate.fidelity_.balanced_accuracy
                if self.surrogate.fidelity_ is not None
                else None
            ),
        )

    def optimize(
        self,
        objective_coefficients: Sequence[float],
        *,
        hard_constraints: Sequence[LinearConstraintSpec] = (),
        maximize: bool = False,
        integer_features: Sequence[str] = (),
        fallback_candidates: pd.DataFrame | None = None,
        solver_options: Mapping[str, object] | None = None,
    ) -> MILPEmbeddingResult:
        """Solve the tree-embedded MILP and audit the result against the teacher."""
        objective = self._validate_objective(objective_coefficients)
        self._validate_hard_constraints(hard_constraints)

        integer_set = set(integer_features)
        unknown_integer = integer_set.difference(self.feature_columns)
        if unknown_integer:
            raise ValueError(f"Unknown integer features: {sorted(unknown_integer)}")

        safe_leaves = self.surrogate.safe_leaf_paths()
        n_x = len(self.feature_columns)
        n_z = len(safe_leaves)
        c = np.zeros(n_x + n_z, dtype=float)
        c[:n_x] = -objective if maximize else objective

        lower_bounds = np.array(
            [self.bounds[feature][0] for feature in self.feature_columns]
            + [0.0] * n_z,
            dtype=float,
        )
        upper_bounds = np.array(
            [self.bounds[feature][1] for feature in self.feature_columns]
            + [1.0] * n_z,
            dtype=float,
        )
        integrality = np.zeros(n_x + n_z, dtype=int)
        for index, feature in enumerate(self.feature_columns):
            if feature in integer_set:
                integrality[index] = 1
        integrality[n_x:] = 1

        constraints, leaf_ids = self._tree_constraints(hard_constraints)
        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=constraints,
            options=dict(solver_options or {}),
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                f"MILP solver failed with status={result.status}: {result.message}"
            )

        point = pd.Series(
            {
                feature: float(result.x[index])
                for index, feature in enumerate(self.feature_columns)
            }
        )
        z_values = np.asarray(result.x[n_x:], dtype=float)
        selected_leaf = (
            int(leaf_ids[int(np.argmax(z_values))])
            if z_values.size
            else None
        )
        probability, p_value, audited_safe, surrogate_safe = self._audit(point)
        objective_value = float(np.dot(objective, result.x[:n_x]))

        if not audited_safe:
            if fallback_candidates is None:
                raise RuntimeError(
                    "MILP surrogate solution failed the original conformal teacher "
                    "audit and no fallback candidate set was supplied"
                )
            return self._fallback(
                fallback_candidates,
                objective,
                hard_constraints,
                maximize=maximize,
            )

        return MILPEmbeddingResult(
            point=point,
            objective_value=objective_value,
            solver_objective_value=float(result.fun),
            solver_status=int(result.status),
            solver_message=str(result.message),
            selected_leaf_id=selected_leaf,
            surrogate_predicted_safe=surrogate_safe,
            teacher_probability=probability,
            conformal_p_value=p_value,
            audited_safe=True,
            used_fallback=False,
            surrogate_fidelity_balanced_accuracy=(
                self.surrogate.fidelity_.balanced_accuracy
                if self.surrogate.fidelity_ is not None
                else None
            ),
        )
