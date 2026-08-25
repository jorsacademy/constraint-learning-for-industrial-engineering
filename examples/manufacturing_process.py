"""Run the end-to-end manufacturing constraint-learning benchmark."""

from pathlib import Path
import sys

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
