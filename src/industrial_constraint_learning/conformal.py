"""Class-conditional conformal filters for learned feasibility scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyFilterEvaluation:
    """Held-out operating characteristics of a conformal safety filter."""

    alpha: float
    probability_threshold: float
    false_feasible_rate: float
    false_infeasible_rate: float
    accepted_fraction: float
    accepted_feasible_fraction: float
    calibration_infeasible_count: int
    minimum_attainable_p_value: float


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


class ConformalSafetyFilter:
    """Class-conditional conformal filter using infeasible calibration scores.

    The fitted model score must increase with evidence for feasibility. For a
    candidate score s, the null hypothesis is that the candidate belongs to
    the infeasible class. The conformal p-value is

        (1 + count(calibration infeasible scores >= s)) / (n_infeasible + 1).

    Under exchangeability between the infeasible calibration observations and a
    future infeasible observation, accepting only when p <= alpha controls the
    marginal false-feasible probability at level alpha.

    This is a single-candidate / marginal statement. Selecting the best point
    from many screened candidates is an additional selection problem and is not
    automatically covered by the same guarantee.
    """

    def __init__(self) -> None:
        self._infeasible_scores: np.ndarray | None = None

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("scores must contain at least one value")
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite")
        return values

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ConformalSafetyFilter":
        """Fit the reference distribution from independently calibrated data."""
        values = self._validate_scores(scores)
        y = np.asarray(labels).astype(int).reshape(-1)
        if y.size != values.size:
            raise ValueError("scores and labels must contain the same number of rows")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("labels must be binary 0/1")

        infeasible_scores = values[y == 0]
        if infeasible_scores.size == 0:
            raise ValueError("calibration data must contain infeasible observations")

        self._infeasible_scores = np.sort(infeasible_scores)
        return self

    @property
    def calibration_infeasible_count(self) -> int:
        if self._infeasible_scores is None:
            raise RuntimeError("Fit the safety filter before use")
        return int(self._infeasible_scores.size)

    @property
    def minimum_attainable_p_value(self) -> float:
        return 1.0 / (self.calibration_infeasible_count + 1.0)

    def p_values(self, scores: np.ndarray) -> np.ndarray:
        """Return conservative class-conditional conformal p-values."""
        if self._infeasible_scores is None:
            raise RuntimeError("Fit the safety filter before use")
        values = self._validate_scores(scores)

        left = np.searchsorted(self._infeasible_scores, values, side="left")
        greater_equal = self._infeasible_scores.size - left
        return (1.0 + greater_equal) / (self._infeasible_scores.size + 1.0)

    def accept(self, scores: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Return a mask of candidates passing the conformal safety filter."""
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        return self.p_values(scores) <= alpha

    def probability_threshold(self, alpha: float = 0.05) -> float:
        """Return the score threshold equivalent to p_value <= alpha."""
        if self._infeasible_scores is None:
            raise RuntimeError("Fit the safety filter before use")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if alpha < self.minimum_attainable_p_value:
            raise RuntimeError(
                "alpha is below the finite-sample resolution of the safety "
                f"calibration set; minimum attainable p-value is "
                f"{self.minimum_attainable_p_value:.6f}"
            )

        n = self._infeasible_scores.size
        allowed_ge = int(np.floor(alpha * (n + 1) - 1.0 + 1e-12))
        if allowed_ge >= n:
            return float("-inf")

        descending = self._infeasible_scores[::-1]
        boundary_score = float(descending[allowed_ge])
        return float(np.nextafter(boundary_score, np.inf))

    def evaluate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        alpha: float = 0.05,
    ) -> SafetyFilterEvaluation:
        """Evaluate the safety filter on labeled held-out observations."""
        values = self._validate_scores(scores)
        y = np.asarray(labels).astype(int).reshape(-1)
        if y.size != values.size:
            raise ValueError("scores and labels must contain the same number of rows")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("labels must be binary 0/1")

        accepted = self.accept(values, alpha=alpha)
        infeasible = y == 0
        feasible = y == 1
        false_feasible = int(np.sum(accepted & infeasible))
        false_infeasible = int(np.sum(~accepted & feasible))
        accepted_count = int(np.sum(accepted))
        accepted_feasible = int(np.sum(accepted & feasible))

        return SafetyFilterEvaluation(
            alpha=float(alpha),
            probability_threshold=self.probability_threshold(alpha),
            false_feasible_rate=_ratio(false_feasible, int(np.sum(infeasible))),
            false_infeasible_rate=_ratio(false_infeasible, int(np.sum(feasible))),
            accepted_fraction=float(np.mean(accepted)),
            accepted_feasible_fraction=_ratio(accepted_feasible, accepted_count),
            calibration_infeasible_count=self.calibration_infeasible_count,
            minimum_attainable_p_value=self.minimum_attainable_p_value,
        )
