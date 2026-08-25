"""Run the assembly quality-control constraint-learning case study."""

from pathlib import Path

from assembly_constraint_learner import AssemblyQualityConstraintLearner
from assembly_data import generate_assembly_quality_data


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"


def main() -> None:
    data = generate_assembly_quality_data(n_samples=4000, random_state=42)
    print("Synthetic assembly quality-control data:")
    print(data.describe())
    print(f"\nAcceptable-operation rate: {data['acceptable_operation'].mean():.3f}")

    learner = AssemblyQualityConstraintLearner(
        data,
        test_size=0.25,
        random_state=42,
    ).fit(tune=True)

    evaluation = learner.evaluate()
    print("\nHeld-out confusion matrix:")
    print(evaluation.confusion_matrix)
    print("\nHeld-out metrics:")
    print(f"  balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"  F1 score: {evaluation.f1:.3f}")
    print(f"  ROC AUC: {evaluation.roc_auc:.3f}")
    print(f"  average precision: {evaluation.average_precision:.3f}")
    print(f"  precision: {evaluation.precision:.3f}")
    print(f"  recall: {evaluation.recall:.3f}")
    print(f"  false accept rate: {evaluation.false_accept_rate:.3f}")
    print(f"  false reject rate: {evaluation.false_reject_rate:.3f}")

    bounds = learner.acceptable_operating_bounds(quantile_margin=0.02)
    print("\nDescriptive acceptable-operation bounds:")
    for parameter, values in bounds.items():
        print(f"  {parameter}: {values['min']:.2f} to {values['max']:.2f}")

    best = learner.best_observed_joint()
    print("\nBest observed acceptable joint by joint strength:")
    for parameter in learner.feature_columns:
        print(f"  {parameter}: {best[parameter]:.2f}")
    print(f"  quality score: {best['quality_score']:.2f}")
    print(f"  joint strength: {best['joint_strength']:.2f}")
    print(f"  cycle time: {best['cycle_time']:.2f}")

    learner.save_evaluation_plots(FIGURES)
    print(f"\nFigures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
