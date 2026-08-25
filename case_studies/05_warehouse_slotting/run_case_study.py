"""Run the warehouse-slotting constraint-learning case study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

from warehouse_constraint_learner import WarehouseSlottingConstraintLearner  # noqa: E402
from warehouse_data import generate_warehouse_slotting_data  # noqa: E402


def main() -> None:
    data = generate_warehouse_slotting_data(n_samples=6000, random_state=42)
    learner = WarehouseSlottingConstraintLearner(data, random_state=42).fit(tune=True)
    evaluation = learner.evaluate()

    print("Warehouse slotting constraint-learning case study")
    print(f"Feasible share: {data['feasible_slotting'].mean():.3f}")
    print(f"Best parameters: {learner.best_params_}")
    print("Confusion matrix:")
    print(evaluation.confusion_matrix)
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"F1 score: {evaluation.f1:.3f}")
    print(f"ROC AUC: {evaluation.roc_auc:.3f}")
    print(f"Average precision: {evaluation.average_precision:.3f}")
    print(f"Precision: {evaluation.precision:.3f}")
    print(f"Recall: {evaluation.recall:.3f}")
    print(f"Unsafe accept rate: {evaluation.unsafe_accept_rate:.3f}")

    print("\nDescriptive feasible-slotting bounds:")
    for name, bounds in learner.descriptive_bounds().items():
        print(f"  {name}: {bounds['min']:.3f} to {bounds['max']:.3f}")

    best = learner.best_observed_feasible_slot()
    print("\nBest observed feasible slotting point:")
    for name in learner.feature_columns:
        print(f"  {name}: {best[name]:.3f}")
    print(f"  picking_time: {best['picking_time']:.3f} s")
    print(f"  congestion_score: {best['congestion_score']:.3f}")
    print(f"  ergonomic_risk: {best['ergonomic_risk']:.3f}")

    improvement = learner.estimate_operational_improvement()
    print("\nObserved operational improvement inside feasible region:")
    print(f"  picking time reduction: {improvement['picking_time_reduction_pct']:.2f}%")
    print(f"  congestion reduction: {improvement['congestion_reduction_pct']:.2f}%")

    figures = ROOT / "figures"
    learner.plot_curves(figures)
    learner.plot_operating_slice(figures)
    print(f"\nFigures written to: {figures}")


if __name__ == "__main__":
    main()
