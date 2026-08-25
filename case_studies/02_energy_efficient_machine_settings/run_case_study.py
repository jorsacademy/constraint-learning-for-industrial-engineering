"""Run the energy-efficient machine settings case study."""

from __future__ import annotations

from pathlib import Path

from energy_constraint_learner import EnergyConstraintLearner
from energy_data import generate_energy_efficiency_data


def main() -> None:
    data = generate_energy_efficiency_data(n_samples=4000, random_state=42)

    print("Synthetic energy-efficiency data summary:")
    print(data.describe())
    print(
        "\nAcceptable-operation share: "
        f"{100.0 * data['acceptable_operation'].mean():.2f}%"
    )

    learner = EnergyConstraintLearner(
        data,
        test_size=0.25,
        random_state=42,
    ).fit(tune=True, cv_folds=5)

    evaluation = learner.evaluate()

    print("\nBest cross-validated hyperparameters:")
    print(learner.best_params_)

    print("\nHeld-out confusion matrix:")
    print(evaluation.confusion_matrix)

    print("\nHeld-out metrics:")
    print(f"  balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"  F1: {evaluation.f1:.3f}")
    print(f"  ROC AUC: {evaluation.roc_auc:.3f}")
    print(f"  average precision: {evaluation.average_precision:.3f}")

    class_one = evaluation.classification_report["1"]
    print(
        "  acceptable-operation precision/recall: "
        f"{class_one['precision']:.3f} / {class_one['recall']:.3f}"
    )

    print("\nDescriptive acceptable-region bounds:")
    bounds = learner.summarize_acceptable_region(quantile_margin=0.02)
    for parameter, values in bounds.items():
        print(f"  {parameter}: {values['min']:.2f} to {values['max']:.2f}")

    best = learner.best_observed_energy_efficient_point()
    print("\nBest observed acceptable machine setting:")
    print(f"  spindle speed: {best['spindle_speed']:.2f} rpm")
    print(f"  feed rate: {best['feed_rate']:.2f} mm/min")
    print(f"  load: {best['load']:.3f}")
    print(f"  energy per unit: {best['energy_per_unit']:.3f} kWh/unit")
    print(f"  throughput: {best['throughput']:.2f} units/hour")
    print(f"  quality score: {best['quality_score']:.2f}")

    print(
        "\nEstimated mean energy reduction inside the observed acceptable region: "
        f"{learner.estimated_energy_saving():.2f}%"
    )

    figure_dir = Path(__file__).resolve().parents[2] / "figures"
    learner.save_evaluation_curves(figure_dir)
    learner.save_operating_region_slice(figure_dir, load_value=0.60)

    print(f"\nFigures written to: {figure_dir}")


if __name__ == "__main__":
    main()
