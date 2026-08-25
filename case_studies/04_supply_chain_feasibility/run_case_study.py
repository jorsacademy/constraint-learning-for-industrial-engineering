"""Run the supply-chain feasibility case study."""

from pathlib import Path

from supply_chain_constraint_learner import SupplyChainConstraintLearner
from supply_chain_data import generate_supply_chain_data


def main() -> None:
    data = generate_supply_chain_data(n_samples=5000, random_state=42)
    learner = SupplyChainConstraintLearner(data, test_size=0.25, random_state=42)
    learner.fit(tune=True)
    evaluation = learner.evaluate()

    print("Supply-chain feasibility benchmark")
    print(f"Observations: {len(data)}")
    print(f"Feasible share: {data['feasible_service'].mean():.3f}")
    print(f"Best parameters: {learner.best_params_}")
    print("\nHeld-out metrics:")
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"F1 score: {evaluation.f1:.3f}")
    print(f"ROC AUC: {evaluation.roc_auc:.3f}")
    print(f"Average precision: {evaluation.average_precision:.3f}")
    print(f"Precision: {evaluation.precision:.3f}")
    print(f"Recall: {evaluation.recall:.3f}")
    print(f"Missed failure rate: {evaluation.missed_failure_rate:.3f}")
    print("Confusion matrix:")
    print(evaluation.confusion_matrix)

    bounds = learner.descriptive_bounds(quantile_margin=0.02)
    print("\nDescriptive feasible operating bounds:")
    for name, values in bounds.items():
        print(f"  {name}: {values['min']:.3f} to {values['max']:.3f}")

    best = learner.best_observed_feasible_point()
    print("\nBest observed feasible service-cost point:")
    for feature in learner.feature_columns:
        print(f"  {feature}: {best[feature]:.3f}")
    print(f"  service_level: {best['service_level']:.3f}")
    print(f"  total_logistics_cost: {best['total_logistics_cost']:.2f}")

    figures = Path(__file__).resolve().parents[2] / "figures"
    learner.plot_curves(figures)
    learner.plot_operating_slice(figures)
    print(f"\nFigures written to: {figures}")


if __name__ == "__main__":
    main()
