"""Run the workforce-shift constraint-learning case study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CASE = Path(__file__).resolve().parent
for path in (SRC, CASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from workforce_constraint_learner import WorkforceConstraintLearner  # noqa: E402
from workforce_data import generate_workforce_shift_data  # noqa: E402


def main() -> None:
    data = generate_workforce_shift_data(n_samples=5000, random_state=42)
    learner = WorkforceConstraintLearner(data, random_state=42).fit(tune=True)
    evaluation = learner.evaluate()

    print("Workforce shift constraint learning")
    print(f"Operational acceptance rate: {data['operationally_acceptable'].mean():.3f}")
    print(f"Policy-compliance rate: {data['policy_compliant'].mean():.3f}")
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"Average precision: {evaluation.average_precision:.3f}")
    print(f"False-feasible rate: {evaluation.false_feasible_rate:.3f}")

    risk_alpha = 0.10
    safety = learner.evaluate_safety_filter(alpha=risk_alpha)
    print(f"Conformal probability threshold: {safety.probability_threshold:.3f}")
    print(f"Held-out conformal false-feasible rate: {safety.false_feasible_rate:.3f}")
    print(f"Held-out accepted fraction: {safety.accepted_fraction:.3f}")

    optimizer = learner.risk_controlled_optimizer(alpha=risk_alpha)
    result = optimizer.optimize(
        data,
        objective=lambda frame: frame["staffing_cost"],
        hard_constraint=lambda frame: frame["policy_compliant"].astype(bool),
        maximize=False,
    )
    print("\nBest screened observed shift:")
    print(result.point[list(learner.feature_columns)].to_string())
    print(f"Staffing cost proxy: {result.objective_value:.3f}")
    print(f"Learned feasibility probability: {result.feasibility_probability:.3f}")


if __name__ == "__main__":
    main()
