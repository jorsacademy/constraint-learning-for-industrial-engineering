"""Distribution-shift diagnostics for learned industrial constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftReport:
    """Distribution-shift diagnostics for one incoming batch."""

    feature_psi: Dict[str, float]
    max_feature_psi: float
    shifted_features: tuple[str, ...]
    score_psi: float
    label_rate_delta: float | None
    covariate_drift: bool
    score_drift: bool
    target_rate_drift: bool
    any_drift: bool
    sample_size: int


class DistributionShiftMonitor:
    """Monitor feature and model-score distributions with PSI.

    Reference histogram bins are defined by reference quantiles. Population
    Stability Index (PSI) is used as an engineering drift diagnostic rather than
    as a formal hypothesis test. Thresholds therefore remain configurable and
    should be validated for the application.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        *,
        n_bins: int = 10,
        feature_psi_threshold: float = 0.20,
        score_psi_threshold: float = 0.20,
        label_rate_delta_threshold: float = 0.10,
        min_batch_size: int = 50,
        epsilon: float = 1e-6,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if n_bins < 3:
            raise ValueError("n_bins must be at least 3")
        if feature_psi_threshold <= 0.0 or score_psi_threshold <= 0.0:
            raise ValueError("PSI thresholds must be positive")
        if not 0.0 < label_rate_delta_threshold < 1.0:
            raise ValueError("label_rate_delta_threshold must be in (0, 1)")
        if min_batch_size < 10:
            raise ValueError("min_batch_size must be at least 10")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.feature_columns = tuple(feature_columns)
        self.n_bins = int(n_bins)
        self.feature_psi_threshold = float(feature_psi_threshold)
        self.score_psi_threshold = float(score_psi_threshold)
        self.label_rate_delta_threshold = float(label_rate_delta_threshold)
        self.min_batch_size = int(min_batch_size)
        self.epsilon = float(epsilon)

        self._feature_edges: Dict[str, np.ndarray] = {}
        self._feature_reference: Dict[str, np.ndarray] = {}
        self._score_edges: np.ndarray | None = None
        self._score_reference: np.ndarray | None = None
        self._reference_label_rate: float | None = None

    def _validate_frame(self, data: pd.DataFrame) -> None:
        missing = set(self.feature_columns).difference(data.columns)
        if missing:
            raise ValueError(f"Missing drift-monitor columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("data must not be empty")

    def _edges(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Drift-monitor values must be finite")
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        internal = np.unique(np.quantile(values, quantiles))
        return np.concatenate(([-np.inf], internal, [np.inf]))

    def _proportions(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
        proportions = counts.astype(float) / max(int(np.sum(counts)), 1)
        proportions = np.clip(proportions, self.epsilon, None)
        return proportions / proportions.sum()

    def _psi(
        self,
        values: np.ndarray,
        edges: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        current = self._proportions(values, edges)
        return float(np.sum((current - reference) * np.log(current / reference)))

    def fit(
        self,
        data: pd.DataFrame,
        scores: np.ndarray,
        *,
        labels: np.ndarray | pd.Series | None = None,
    ) -> "DistributionShiftMonitor":
        """Establish a new deployment reference distribution."""
        self._validate_frame(data)
        score_values = np.asarray(scores, dtype=float).reshape(-1)
        if score_values.size != len(data):
            raise ValueError("scores must contain one value per reference row")
        if not np.isfinite(score_values).all():
            raise ValueError("scores must be finite")

        self._feature_edges = {}
        self._feature_reference = {}
        for feature in self.feature_columns:
            values = data[feature].to_numpy(dtype=float)
            edges = self._edges(values)
            self._feature_edges[feature] = edges
            self._feature_reference[feature] = self._proportions(values, edges)

        self._score_edges = self._edges(score_values)
        self._score_reference = self._proportions(score_values, self._score_edges)

        self._reference_label_rate = None
        if labels is not None:
            y = np.asarray(labels).astype(int).reshape(-1)
            if y.size != len(data):
                raise ValueError("labels must contain one value per reference row")
            if not np.isin(y, [0, 1]).all():
                raise ValueError("labels must be binary 0/1")
            self._reference_label_rate = float(np.mean(y))
        return self

    def detect(
        self,
        data: pd.DataFrame,
        scores: np.ndarray,
        *,
        labels: np.ndarray | pd.Series | None = None,
    ) -> DriftReport:
        """Compare a batch with the active deployment reference."""
        self._validate_frame(data)
        if not self._feature_reference or self._score_reference is None:
            raise RuntimeError("Fit the drift monitor before detection")
        if len(data) < self.min_batch_size:
            raise ValueError(
                f"Batch size {len(data)} is below min_batch_size={self.min_batch_size}"
            )

        score_values = np.asarray(scores, dtype=float).reshape(-1)
        if score_values.size != len(data):
            raise ValueError("scores must contain one value per batch row")
        if not np.isfinite(score_values).all():
            raise ValueError("scores must be finite")

        feature_psi = {
            feature: self._psi(
                data[feature].to_numpy(dtype=float),
                self._feature_edges[feature],
                self._feature_reference[feature],
            )
            for feature in self.feature_columns
        }
        shifted = tuple(
            feature
            for feature, value in feature_psi.items()
            if value >= self.feature_psi_threshold
        )
        max_feature_psi = max(feature_psi.values(), default=0.0)
        score_psi = self._psi(
            score_values,
            self._score_edges,
            self._score_reference,
        )

        label_rate_delta = None
        target_rate_drift = False
        if labels is not None:
            y = np.asarray(labels).astype(int).reshape(-1)
            if y.size != len(data):
                raise ValueError("labels must contain one value per batch row")
            if not np.isin(y, [0, 1]).all():
                raise ValueError("labels must be binary 0/1")
            if self._reference_label_rate is not None:
                label_rate_delta = abs(float(np.mean(y)) - self._reference_label_rate)
                target_rate_drift = (
                    label_rate_delta >= self.label_rate_delta_threshold
                )

        covariate_drift = bool(shifted)
        score_drift = score_psi >= self.score_psi_threshold
        return DriftReport(
            feature_psi=feature_psi,
            max_feature_psi=float(max_feature_psi),
            shifted_features=shifted,
            score_psi=float(score_psi),
            label_rate_delta=label_rate_delta,
            covariate_drift=covariate_drift,
            score_drift=score_drift,
            target_rate_drift=target_rate_drift,
            any_drift=bool(covariate_drift or score_drift or target_rate_drift),
            sample_size=int(len(data)),
        )
