"""Run the job-shop scheduling constraint-learning case study."""

from pathlib import Path

from scheduling_constraint_learner import JobShopConstraintLearner
from scheduling_data import generate_job_shop_data


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"


def main() -> None:
    data = generate_job_shop_data(n_samples=5000, random_state=42)
    learner = JobShopConstraintLearner(data, random_state=42).fit(tune=True)
    evaluation = learner.evaluate()

    print("Job-shop scheduling constraint-learning benchmark")
    print(f"Feasible share: {data['feasible_schedule'].mean():.3f}")
    print(f"Best parameters: {learner.best_params_}")
    print("Confusion matrix:")
    print(evaluation.confusion_matrix)
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"F1: {evaluation.f1:.3f}")
    print(f"ROC AUC: {evaluation.roc_auc:.3f}")
    print(f"Average precision: {evaluation.average_precision:.3f}")
    print(f"Precision: {evaluation.precision:.3f}")
    print(f"Recall: {evaluation.recall:.3f}")
    print(f"False feasible rate: {evaluation.false_feasible_rate:.3f}")

    print("\nDescriptive bounds for feasible scheduling episodes:")
    for feature, bounds in learner.descriptive_bounds().items():
        print(f"  {feature}: {bounds['min']:.3f} to {bounds['max']:.3f}")

    best = learner.best_observed_feasible_episode()
    print("\nBest observed feasible scheduling episode:")
    print(f"  job count: {best['job_count']:.0f}")
    print(f"  due-date tightness: {best['due_date_tightness']:.3f}")
    print(f"  machine utilization: {best['machine_utilization']:.3f}")
    print(f"  mean tardiness: {best['mean_tardiness']:.3f}")
    print(f"  overtime hours: {best['overtime_hours']:.3f}")
    print(f"  WIP level: {best['wip_level']:.3f}")

    learner.plot_curves(FIGURES)
    learner.plot_operating_slice(FIGURES)
    print(f"\nFigures written to: {FIGURES}")


if __name__ == "__main__":
    main()
