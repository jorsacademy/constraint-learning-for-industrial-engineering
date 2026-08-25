"""Run the manufacturing constraint-learning example."""

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


def main() -> None:
    data = generate_manufacturing_data(n_samples=2000, random_state=42)
    print("Synthetic manufacturing process data:")
    print(data.describe(include="all"))

    learner = ManufacturingConstraintLearner(
        data,
        high_yield_threshold=85.0,
        test_size=0.25,
        random_state=42,
    )
    learner.fit_feasibility_classifier()

    evaluation = learner.evaluate()
    print("\nHeld-out confusion matrix:")
    print(evaluation.confusion_matrix)
    print("\nHeld-out classification metrics:")
    for label in ("0", "1"):
        metrics = evaluation.classification_report[label]
        print(
            f"class {label}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}"
        )

    bounds = learner.learn_high_yield_bounds(quantile_margin=0.01)
    print("\nDescriptive high-yield operating bounds:")
    for parameter, values in bounds.items():
        print(f"  {parameter}: {values['min']:.2f} to {values['max']:.2f}")

    best = learner.best_observed_feasible_point()
    print("\nBest observed physically feasible point:")
    print(f"  temperature: {best['temperature']:.2f} °C")
    print(f"  pressure: {best['pressure']:.2f} MPa")
    print(f"  observed yield: {best['yield']:.2f}%")

    learner.plot_learned_region()


if __name__ == "__main__":
    main()
