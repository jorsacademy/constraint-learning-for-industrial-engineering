# Case Study 05: Warehouse Slotting

## Industrial problem

Learn SKU-location combinations that keep picking time, congestion, and ergonomic risk within acceptable limits while preserving warehouse operating stability.

This case study treats slotting feasibility as a nonlinear classification problem. The classifier does not receive the hidden synthetic engineering equation; it only observes item/location features and the resulting feasible/infeasible label.

## Input variables

- `sku_velocity`: expected picks per day,
- `unit_weight`: item weight in kilograms,
- `cube`: item volume in cubic meters,
- `aisle_distance`: distance from dispatch in meters,
- `replenishment_frequency`: replenishments per day,
- `neighboring_pick_density`: normalized local traffic intensity.

## Synthetic operational outputs

The benchmark generates:

- `picking_time`,
- `congestion_score`,
- `ergonomic_risk`,
- `hidden_stability_score`,
- `feasible_slotting`.

A slotting observation is feasible only when all of the following hold:

```text
hidden_stability_score >= 0
picking_time <= 78 seconds
congestion_score <= 0.85
ergonomic_risk <= 0.82
```

The hidden stability score contains nonlinear interactions among travel distance, SKU velocity, weight, cube, replenishment frequency, and local traffic.

## Constraint-learning model

The workflow uses:

```text
StandardScaler
    -> RBF-SVM
    -> class balancing
    -> stratified train/test split
    -> GridSearchCV
    -> average-precision model selection
```

Average precision is emphasized because feasible slotting observations may form a minority class.

## Evaluation

The executable case reports:

- confusion matrix,
- balanced accuracy,
- F1 score,
- ROC AUC,
- average precision,
- precision,
- recall,
- unsafe accept rate.

`unsafe_accept_rate` is the fraction of truly infeasible slotting observations that the classifier incorrectly accepts as feasible. In a warehouse application, these false accepts can create excessive travel, congestion, ergonomic exposure, or unstable replenishment behavior.

## Operational summaries

The learner also provides:

- quantile-based descriptive bounds for feasible slotting settings,
- the best observed feasible slot using a composite picking/congestion/ergonomic score,
- estimated picking-time reduction inside the observed feasible region,
- estimated congestion reduction inside the observed feasible region.

These summaries are descriptive. They are not substitutes for explicit storage-capacity, fire-code, hazardous-material, weight, or rack-engineering constraints.

## Visualization

Running the case study writes:

```text
figures/warehouse_slotting_roc_curve.png
figures/warehouse_slotting_precision_recall_curve.png
figures/warehouse_slotting_feasibility_region.png
```

The feasibility-region figure shows an SKU-velocity/aisle-distance slice of the six-dimensional learned region while the remaining variables are fixed at representative values.

## Run

From the repository root:

```bash
python case_studies/05_warehouse_slotting/run_case_study.py
```

## Tests

```bash
pytest -q tests/test_warehouse_slotting_case_study.py
```

The tests cover reproducibility, engineering-rule consistency, class balance, held-out model quality, nominal/stressed operating points, unsafe-accept behavior, and operational improvement inside the feasible region.

## Extension

A practical next step is to embed the learned feasibility classifier inside a deterministic slotting optimizer. The optimization model would still enforce hard storage and safety rules explicitly, while the learned constraint would screen historically poor SKU-location combinations that are difficult to express analytically.
