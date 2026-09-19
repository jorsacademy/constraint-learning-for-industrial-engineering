"""Run the multi-product line-balancing constraint-learning case study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CASE = Path(__file__).resolve().parent
for path in (SRC, CASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from line_balancing_constraint_learner import (  # noqa: E402
    LineBalancingConstraintLearner,
)
from line_balancing_data import generate_line_balancing_data  # noqa: E402


def main() -> None:
    data = generate_line_balancing_data(n_samples=5000, random_state=42)
    learner = LineBalancingConstraintLearner(data, random_state=42).fit(tune=True)
    evaluation = learner.evaluate()

    print("Multi-product line-balancing constraint learning")
    print(f"Stable-operation rate: {data['stable_operation'].mean():.3f}")
    print(f"Hard-capacity compliance rate: {data['hard_capacity_compliant'].mean():.3f}")
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"Average precision: {evaluation.average_precision:.3f}")
    print(f"False-feasible rate: {evaluation.false_feasible_rate:.3f}")

    optimizer = learner.safe_optimizer(min_probability=0.50)
    result = optimizer.optimize(
        data,
        objective=lambda frame: frame["resource_cost"],
        hard_constraint=lambda frame: frame["hard_capacity_compliant"].astype(bool),
        maximize=False,
    )
    print("\nBest screened observed line configuration:")
    print(result.point[list(learner.feature_columns)].to_string())
    print(f"Resource-cost proxy: {result.objective_value:.3f}")
    print(f"Learned feasibility probability: {result.feasibility_probability:.3f}")


if __name__ == "__main__":
    main()
