"""CP-SAT embedding for risk-controlled decision-tree surrogates."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from .milp_embedding import LinearConstraintSpec
from .surrogate import RiskControlledTeacher, RiskControlledTreeSurrogate


@dataclass(frozen=True)
class CPSATEmbeddingResult:
    """Audited result from discretized tree-embedded CP-SAT optimization."""

    point: pd.Series
    objective_value: float
    solver_status: str
    selected_leaf_id: int | None
    surrogate_predicted_safe: bool
    teacher_probability: float
    conformal_p_value: float
    audited_safe: bool
    used_fallback: bool
    surrogate_fidelity_balanced_accuracy: float | None


class SurrogateCPSATOptimizer:
    """Embed tree-safe leaves with CP-SAT reified constraints.

    Each continuous feature is represented on an integer lattice using the
    supplied feature scale. Tree split thresholds are encoded directly on that
    lattice with reified constraints, avoiding big-M constants. The CP-SAT
    solution is always audited against the original conformal teacher.
    """

    def __init__(
        self,
        teacher: RiskControlledTeacher,
        surrogate: RiskControlledTreeSurrogate,
        bounds: Mapping[str, tuple[float, float]],
        *,
        feature_scales: Mapping[str, int] | None = None,
        alpha: float = 0.10,
        linear_scale: int = 1_000_000,
    ) -> None:
        if surrogate.model is None:
            raise ValueError("surrogate must be fitted before CP-SAT construction")
        if tuple(teacher.feature_columns) != tuple(surrogate.feature_columns):
            raise ValueError("teacher and surrogate feature order must match")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if linear_scale < 1:
            raise ValueError("linear_scale must be positive")

        self.teacher = teacher
        self.surrogate = surrogate
        self.feature_columns = tuple(surrogate.feature_columns)
        self.alpha = float(alpha)
        self.linear_scale = int(linear_scale)

        if set(bounds) != set(self.feature_columns):
            raise ValueError("bounds must be supplied for every feature and no others")
        self.bounds = {
            feature: (float(bounds[feature][0]), float(bounds[feature][1]))
            for feature in self.feature_columns
        }
        for feature, (lower, upper) in self.bounds.items():
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"Invalid finite bounds for {feature}")

        scales = feature_scales or {
            feature: 1000 for feature in self.feature_columns
        }
        if set(scales) != set(self.feature_columns):
            raise ValueError("feature_scales must be supplied for every feature")
        self.feature_scales = {feature: int(scales[feature]) for feature in self.feature_columns}
        if any(scale <= 0 for scale in self.feature_scales.values()):
            raise ValueError("feature scales must be positive integers")

    def _scaled_bounds(self, feature: str) -> tuple[int, int]:
        lower, upper = self.bounds[feature]
        scale = self.feature_scales[feature]
        lower_i = int(ceil(lower * scale - 1e-12))
        upper_i = int(floor(upper * scale + 1e-12))
        if lower_i > upper_i:
            raise ValueError(f"Scaled bounds are empty for feature {feature}")
        return lower_i, upper_i

    def _audit(self, point: pd.Series) -> tuple[float, float, bool, bool]:
        frame = pd.DataFrame(
            [{feature: float(point[feature]) for feature in self.feature_columns}]
        )
        surrogate_safe = bool(self.surrogate.predict(frame)[0])
        probabilities = np.asarray(self.teacher.predict_proba(frame), dtype=float)
        probability = float(probabilities[0, 1])
        p_values = np.asarray(self.teacher.conformal_p_values(frame), dtype=float)
        p_value = float(p_values[0])
        return probability, p_value, bool(p_value <= self.alpha), surrogate_safe

    def _hard_satisfied(
        self,
        point: pd.Series,
        hard_constraints: Sequence[LinearConstraintSpec],
    ) -> bool:
        vector = point.loc[list(self.feature_columns)].to_numpy(dtype=float)
        for spec in hard_constraints:
            if len(spec.coefficients) != len(self.feature_columns):
                raise ValueError(
                    "every hard constraint must contain one coefficient per feature"
                )
            value = float(np.dot(np.asarray(spec.coefficients, dtype=float), vector))
            if value < spec.lower_bound - 1e-7 or value > spec.upper_bound + 1e-7:
                return False
        return True

    def _fallback(
        self,
        candidates: pd.DataFrame,
        objective: np.ndarray,
        hard_constraints: Sequence[LinearConstraintSpec],
        *,
        maximize: bool,
    ) -> CPSATEmbeddingResult:
        missing = set(self.feature_columns).difference(candidates.columns)
        if missing:
            raise ValueError(f"Missing fallback columns: {sorted(missing)}")
        eligible: list[int] = []
        p_values = np.asarray(
            self.teacher.conformal_p_values(candidates.loc[:, self.feature_columns]),
            dtype=float,
        ).reshape(-1)
        for i in range(len(candidates)):
            point = candidates.iloc[i]
            in_bounds = all(
                self.bounds[feature][0] - 1e-9
                <= float(point[feature])
                <= self.bounds[feature][1] + 1e-9
                for feature in self.feature_columns
            )
            if (
                in_bounds
                and p_values[i] <= self.alpha
                and self._hard_satisfied(point, hard_constraints)
            ):
                eligible.append(i)

        if not eligible:
            raise RuntimeError(
                "CP-SAT solution failed teacher audit and no audited fallback candidate exists"
            )
        matrix = candidates.loc[:, self.feature_columns].to_numpy(dtype=float)
        values = matrix @ objective
        eligible_array = np.asarray(eligible, dtype=int)
        local = int(
            np.argmax(values[eligible_array])
            if maximize
            else np.argmin(values[eligible_array])
        )
        index = int(eligible_array[local])
        point = candidates.iloc[index].loc[list(self.feature_columns)].copy()
        probability, p_value, audited_safe, surrogate_safe = self._audit(point)
        return CPSATEmbeddingResult(
            point=point,
            objective_value=float(values[index]),
            solver_status="FALLBACK",
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
        fallback_candidates: pd.DataFrame | None = None,
        time_limit_seconds: float | None = None,
        num_search_workers: int = 1,
    ) -> CPSATEmbeddingResult:
        """Solve the discretized tree embedding and audit against the teacher."""
        objective = np.asarray(objective_coefficients, dtype=float).reshape(-1)
        if objective.size != len(self.feature_columns):
            raise ValueError("objective must contain one coefficient per feature")
        if not np.isfinite(objective).all():
            raise ValueError("objective coefficients must be finite")
        if num_search_workers < 1:
            raise ValueError("num_search_workers must be positive")

        model = cp_model.CpModel()
        x_vars: list[cp_model.IntVar] = []
        for feature in self.feature_columns:
            lower_i, upper_i = self._scaled_bounds(feature)
            x_vars.append(model.NewIntVar(lower_i, upper_i, feature))

        safe_leaves = self.surrogate.safe_leaf_paths()
        leaf_vars = [
            model.NewBoolVar(f"leaf_{leaf.leaf_id}")
            for leaf in safe_leaves
        ]
        model.Add(sum(leaf_vars) == 1)

        for leaf_var, leaf in zip(leaf_vars, safe_leaves):
            for condition in leaf.conditions:
                feature = condition.feature_name
                scale = self.feature_scales[feature]
                x_var = x_vars[condition.feature_index]
                if condition.sense == "<=":
                    threshold_i = int(floor(condition.threshold * scale + 1e-12))
                    model.Add(x_var <= threshold_i).OnlyEnforceIf(leaf_var)
                elif condition.sense == ">":
                    threshold_i = int(floor(condition.threshold * scale + 1e-12)) + 1
                    model.Add(x_var >= threshold_i).OnlyEnforceIf(leaf_var)
                else:
                    raise RuntimeError(f"Unsupported tree condition: {condition.sense}")

        for spec in hard_constraints:
            if len(spec.coefficients) != len(self.feature_columns):
                raise ValueError(
                    "every hard constraint must contain one coefficient per feature"
                )
            integer_coefficients = [
                int(
                    round(
                        coefficient
                        * self.linear_scale
                        / self.feature_scales[feature]
                    )
                )
                for coefficient, feature in zip(
                    spec.coefficients,
                    self.feature_columns,
                )
            ]
            for raw, scaled in zip(spec.coefficients, integer_coefficients):
                if raw != 0.0 and scaled == 0:
                    raise ValueError(
                        "linear_scale is too small to preserve a nonzero "
                        "hard-constraint coefficient"
                    )
            expression = sum(
                coefficient * variable
                for coefficient, variable in zip(integer_coefficients, x_vars)
            )
            if np.isfinite(spec.lower_bound):
                model.Add(
                    expression >= int(ceil(spec.lower_bound * self.linear_scale - 1e-9))
                )
            if np.isfinite(spec.upper_bound):
                model.Add(
                    expression <= int(floor(spec.upper_bound * self.linear_scale + 1e-9))
                )

        objective_coefficients_int = [
            int(
                round(
                    coefficient
                    * self.linear_scale
                    / self.feature_scales[feature]
                )
            )
            for coefficient, feature in zip(objective, self.feature_columns)
        ]
        for raw, scaled in zip(objective, objective_coefficients_int):
            if raw != 0.0 and scaled == 0:
                raise ValueError(
                    "linear_scale is too small to preserve a nonzero "
                    "objective coefficient"
                )
        objective_expression = sum(
            coefficient * variable
            for coefficient, variable in zip(objective_coefficients_int, x_vars)
        )
        if maximize:
            model.Maximize(objective_expression)
        else:
            model.Minimize(objective_expression)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = int(num_search_workers)
        if time_limit_seconds is not None:
            if time_limit_seconds <= 0.0:
                raise ValueError("time_limit_seconds must be positive")
            solver.parameters.max_time_in_seconds = float(time_limit_seconds)

        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"CP-SAT solver failed with status={status_name}")

        point = pd.Series(
            {
                feature: solver.Value(variable) / self.feature_scales[feature]
                for feature, variable in zip(self.feature_columns, x_vars)
            }
        )
        selected_leaf = None
        for leaf_var, leaf in zip(leaf_vars, safe_leaves):
            if solver.Value(leaf_var):
                selected_leaf = int(leaf.leaf_id)
                break

        probability, p_value, audited_safe, surrogate_safe = self._audit(point)
        hard_ok = self._hard_satisfied(point, hard_constraints)
        objective_value = float(np.dot(objective, point.to_numpy(dtype=float)))

        if not audited_safe or not hard_ok:
            if fallback_candidates is None:
                raise RuntimeError(
                    "CP-SAT surrogate solution failed original teacher/hard-constraint "
                    "audit and no fallback candidate set was supplied"
                )
            return self._fallback(
                fallback_candidates,
                objective,
                hard_constraints,
                maximize=maximize,
            )

        return CPSATEmbeddingResult(
            point=point,
            objective_value=objective_value,
            solver_status=status_name,
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
