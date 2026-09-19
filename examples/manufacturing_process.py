"""Run the end-to-end manufacturing constraint-learning benchmark."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_constraint_learning import (  # noqa: E402
    ManufacturingConstraintLearner,
    generate_manufacturing_data,
)


def print_evaluation(title: str, evaluation) -> None:
    print(f"\n{title}")
    print("Confusion matrix:")
    print(evaluation.confusion_matrix)
    print(
        "Metrics: "
        f"balanced_accuracy={evaluation.balanced_accuracy:.3f}, "
        f"f1={evaluation.f1:.3f}, "
        f"roc_auc={evaluation.roc_auc:.3f}, "
        f"average_precision={evaluation.average_precision:.3f}"
    )


def main() -> None:
    figures = ROOT / "figures"
    data = generate_manufacturing_data(n_samples=5000, random_state=42)

    print("Synthetic manufacturing process data:")
    print(data.describe(include="all"))
    print(f"Physical feasibility rate: {data['physical_feasible'].mean():.3f}")
    print(f"High-yield rate: {(data['yield'] >= 85.0).mean():.3f}")

    benchmark = ManufacturingConstraintLearner(
        data,
        high_yield_threshold=85.0,
        test_size=0.25,
        random_state=42,
        label_mode="physical_feasibility",
    )
    benchmark.tune_hyperparameters(cv_splits=5, scoring="average_precision")
    print("\nBenchmark model tuned using physical feasibility labels")
    print(f"Best parameters: {benchmark.best_params_}")
    print(f"Best cross-validation score: {benchmark.cv_best_score_:.3f}")
    print_evaluation("Held-out benchmark evaluation", benchmark.evaluate())

    bounds = benchmark.learn_high_yield_bounds(quantile_margin=0.01)
    print("\nDescriptive high-yield operating bounds:")
    for parameter, values in bounds.items():
        print(f"  {parameter}: {values['min']:.2f} to {values['max']:.2f}")

    best = benchmark.best_observed_feasible_point()
    print("\nBest observed physically feasible point:")
    print(f"  temperature: {best['temperature']:.2f} °C")
    print(f"  pressure: {best['pressure']:.2f} MPa")
    print(f"  observed yield: {best['yield']:.2f}%")

    benchmark.plot_boundary_comparison(
        figures / "true_vs_learned_boundary.png",
        show=False,
    )
    benchmark.plot_roc_pr_curves(figures, show=False)

    region_metrics = benchmark.boundary_metrics(grid_resolution=200)
    print("\nLearned feasible-region diagnostics:")
    print(f"  IoU: {region_metrics.intersection_over_union:.3f}")
    print(f"  false-feasible rate: {region_metrics.false_feasible_rate:.3f}")
    print(f"  false-infeasible rate: {region_metrics.false_infeasible_rate:.3f}")

    temperatures = np.linspace(150.0, 350.0, 81)
    pressures = np.linspace(2.0, 8.0, 61)
    tt, pp = np.meshgrid(temperatures, pressures)
    candidates = pd.DataFrame(
        {"temperature": tt.ravel(), "pressure": pp.ravel()}
    )
    risk_alpha = 0.10
    safety_evaluation = benchmark.evaluate_safety_filter(alpha=risk_alpha)
    risk_threshold = benchmark.risk_controlled_threshold(alpha=risk_alpha)
    print("\nConformal safety-filter diagnostics:")
    print(f"  alpha: {risk_alpha:.3f}")
    print(f"  probability threshold: {risk_threshold:.3f}")
    print(
        "  held-out false-feasible rate: "
        f"{safety_evaluation.false_feasible_rate:.3f}"
    )
    print(
        "  held-out false-infeasible rate: "
        f"{safety_evaluation.false_infeasible_rate:.3f}"
    )
    print(f"  accepted fraction: {safety_evaluation.accepted_fraction:.3f}")

    safe_optimizer = benchmark.risk_controlled_optimizer(alpha=risk_alpha)
    safe_solution = safe_optimizer.optimize(
        candidates,
        objective=lambda frame: (
            0.02 * frame["temperature"] + 0.60 * frame["pressure"]
        ),
        hard_constraint=lambda frame: (
            frame["temperature"].between(150.0, 350.0)
            & frame["pressure"].between(2.0, 8.0)
        ),
        maximize=False,
    )
    print("\nIllustrative safe operating-point optimization:")
    print(f"  temperature: {safe_solution.point['temperature']:.2f} °C")
    print(f"  pressure: {safe_solution.point['pressure']:.2f} MPa")
    print(
        "  learned feasibility probability: "
        f"{safe_solution.feasibility_probability:.3f}"
    )
    print(f"  process-effort proxy: {safe_solution.objective_value:.3f}")

    outcome_only = ManufacturingConstraintLearner(
        data,
        high_yield_threshold=85.0,
        test_size=0.25,
        random_state=42,
        label_mode="high_yield",
    )
    outcome_only.tune_hyperparameters(cv_splits=5, scoring="average_precision")
    print("\nOutcome-only model tuned without physical feasibility labels")
    print(f"Best parameters: {outcome_only.best_params_}")
    print(f"Best cross-validation score: {outcome_only.cv_best_score_:.3f}")
    print_evaluation("Held-out high-yield classification", outcome_only.evaluate())
    print_evaluation(
        "Recovery of hidden physical feasibility",
        outcome_only.evaluate_against_physical_truth(),
    )
    outcome_only.plot_boundary_comparison(
        figures / "outcome_only_vs_true_boundary.png",
        show=False,
    )

    print(f"\nFigures saved to: {figures}")


if __name__ == "__main__":
    main()
