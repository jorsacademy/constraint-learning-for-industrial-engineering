# Case Study 04: Supply Chain Feasibility

## Industrial problem

Learn combinations of demand uncertainty, replenishment settings, supplier loading, and transportation conditions under which required service levels can be maintained without exceeding a logistics-cost ceiling.

This executable synthetic benchmark treats supply-chain feasibility as a nonlinear classification problem. The classifier does not receive the hidden stability equation directly; it learns from observed operating variables and the final feasible-service label.

## Input variables

- supplier lead time,
- demand coefficient of variation,
- order quantity,
- safety stock,
- supplier utilization,
- transport time.

## Observed outcomes

The simulator produces:

- service level,
- total logistics cost,
- a hidden nonlinear stability score used only to construct and audit the benchmark,
- `feasible_service`.

An observation is feasible only when all of the following hold simultaneously:

```text
hidden stability score >= 0
service level >= 0.90
total logistics cost <= 265
```

The hidden stability rule combines lead time, demand variability, supplier utilization, transport time, safety stock, and excessive order quantity. It is intentionally nonlinear so a rectangular min/max rule is not sufficient.

## Constraint-learning formulation

The learner uses:

```text
StandardScaler
    -> RBF Support Vector Classifier
    -> class balancing
    -> stratified train/test split
    -> GridSearchCV
    -> average-precision model selection
```

Average precision is used for model selection because feasible observations can represent a minority of historical order cycles.

## Evaluation

The executable workflow reports:

- confusion matrix,
- balanced accuracy,
- F1 score,
- ROC AUC,
- average precision,
- precision,
- recall,
- missed-failure rate.

In this case study, the missed-failure rate is the share of truly infeasible order cycles that are incorrectly accepted by the learned feasibility model. Operationally, this is more important than raw accuracy because an accepted but unstable supply-chain configuration can lead to stockouts, expediting, or service failure.

The workflow also extracts robust quantile-based descriptive bounds and identifies the best observed feasible point using a combined service-cost score. These summaries are descriptive and are not presented as exact constraints.

## Visualizations

Running the case study writes:

```text
figures/supply_chain_roc_curve.png
figures/supply_chain_precision_recall_curve.png
figures/supply_chain_feasibility_region.png
```

The feasibility-region plot is a two-dimensional lead-time/demand-volatility slice through the six-dimensional learned region while the other variables are held at representative values.

## Run

From the repository root:

```bash
python case_studies/04_supply_chain_feasibility/run_case_study.py
```

## Tests

```bash
pytest -q tests/test_supply_chain_case_study.py
```

The tests cover reproducibility, engineering-rule consistency, class balance, held-out predictive performance, and nominal versus stressed operating points.

## Industrial extension

A production implementation should use time-based validation, supplier- and lane-specific drift monitoring, calibrated probabilities, and explicit safety or contractual constraints. The learned feasibility model can then be embedded inside sourcing, order-quantity, or inventory optimization as a data-driven admissibility constraint rather than used as an unconstrained predictor.
