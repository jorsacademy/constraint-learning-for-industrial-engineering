"""Run the product-design-space constraint-learning case study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CASE = Path(__file__).resolve().parent
for path in (SRC, CASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from design_constraint_learner import ProductDesignConstraintLearner  # noqa: E402
from design_data import generate_product_design_data  # noqa: E402


def main() -> None:
    data = generate_product_design_data(n_samples=5000, random_state=42)
    learner = ProductDesignConstraintLearner(data, random_state=42).fit(tune=True)
    evaluation = learner.evaluate()

    print("Product design constraint learning")
    print(f"Operational feasibility rate: {data['operational_feasible'].mean():.3f}")
    print(f"Hard-design compliance rate: {data['hard_design_compliant'].mean():.3f}")
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"Average precision: {evaluation.average_precision:.3f}")
    print(f"False-feasible rate: {evaluation.false_feasible_rate:.3f}")

    optimizer = learner.safe_optimizer(min_probability=0.50)
    result = optimizer.optimize(
        data,
        objective=lambda frame: frame["design_cost"],
        hard_constraint=lambda frame: frame["hard_design_compliant"].astype(bool),
        maximize=False,
    )
    print("\nBest screened observed design:")
    print(result.point[list(learner.feature_columns)].to_string())
    print(f"Design cost: {result.objective_value:.3f}")
    print(f"Learned feasibility probability: {result.feasibility_probability:.3f}")


if __name__ == "__main__":
    main()
