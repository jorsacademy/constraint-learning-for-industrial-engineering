"""Sequential drift-detection and online-adaptation benchmark."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning import (  # noqa: E402
    AdaptiveConstraintController,
    AdaptivePolicy,
    DistributionShiftMonitor,
    TabularConstraintLearner,
)


FEATURES = ("x1", "x2")


def make_batch(
    n: int,
    seed: int,
    *,
    shift: float = 0.0,
    reverse_concept: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(shift, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    margin = x1 + 0.8 * x2
    feasible = (margin < 0.0) if reverse_concept else (margin >= 0.0)
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "feasible": feasible.astype(int)}
    )


def build_learner(data: pd.DataFrame) -> TabularConstraintLearner:
    return TabularConstraintLearner(
        data,
        feature_columns=FEATURES,
        label_column="feasible",
        random_state=42,
    ).fit(tune=False)


def main() -> None:
    baseline = make_batch(6000, 42)
    learner = build_learner(baseline)
    controller = AdaptiveConstraintController(
        learner,
        policy=AdaptivePolicy(
            alpha=0.10,
            max_balanced_accuracy_drop=0.10,
            min_balanced_accuracy=0.70,
            max_batch_false_feasible_rate=0.25,
            max_safety_false_feasible_rate=0.25,
            severe_feature_psi=0.60,
            severe_score_psi=0.60,
            recent_window_size=1600,
            retrain_tune=False,
        ),
        drift_monitor=DistributionShiftMonitor(
            FEATURES,
            feature_psi_threshold=0.08,
            score_psi_threshold=0.20,
            label_rate_delta_threshold=0.20,
            min_batch_size=100,
        ),
    )

    stream = [
        ("stable", make_batch(1500, 100)),
        ("covariate_shift", make_batch(1500, 101, shift=0.45)),
        (
            "concept_shift",
            make_batch(1600, 102, shift=0.45, reverse_concept=True),
        ),
        (
            "post_retrain_stable",
            make_batch(1500, 103, shift=0.45, reverse_concept=True),
        ),
    ]

    rows = []
    for name, batch in stream:
        result = controller.process_batch(batch)
        rows.append(
            {
                "batch": name,
                "action": result.action,
                "version": result.version,
                "valid": result.model_valid,
                "max_feature_psi": result.drift.max_feature_psi,
                "score_psi": result.drift.score_psi,
                "balanced_accuracy": result.batch_balanced_accuracy,
                "false_feasible_rate": result.batch_false_feasible_rate,
            }
        )

    report = pd.DataFrame(rows)
    print("Sequential online adaptation benchmark")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
