"""Operational metrics for learned feasible regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryRegionMetrics:
    """Metrics for a binary learned feasible region against known truth.

    ``false_feasible_rate`` is the fraction of truly infeasible points that are
    incorrectly accepted by the learned region. ``false_infeasible_rate`` is
    the fraction of truly feasible points incorrectly rejected.
    """

    intersection_over_union: float
    false_feasible_rate: float
    false_infeasible_rate: float
    feasible_precision: float
    feasible_recall: float
    predicted_feasible_fraction: float
    true_feasible_fraction: float


def _safe_ratio(numerator: float, denominator: float, *, empty: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def evaluate_binary_region(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> BinaryRegionMetrics:
    """Evaluate a learned binary feasible region."""
    truth = np.asarray(y_true).astype(bool).ravel()
    pred = np.asarray(y_pred).astype(bool).ravel()
    if truth.size == 0:
        raise ValueError("y_true must contain at least one observation")
    if truth.size != pred.size:
        raise ValueError("y_true and y_pred must contain the same number of elements")

    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))

    union = tp + fp + fn
    return BinaryRegionMetrics(
        intersection_over_union=_safe_ratio(tp, union, empty=1.0),
        false_feasible_rate=_safe_ratio(fp, fp + tn),
        false_infeasible_rate=_safe_ratio(fn, fn + tp),
        feasible_precision=_safe_ratio(tp, tp + fp),
        feasible_recall=_safe_ratio(tp, tp + fn),
        predicted_feasible_fraction=float(np.mean(pred)),
        true_feasible_fraction=float(np.mean(truth)),
    )
