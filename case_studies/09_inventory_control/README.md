# Case Study 09: Inventory Control

## Industrial problem

Learn reorder-policy regions that deliver acceptable service, stockout frequency, holding cost, and replenishment stability under varying demand and lead-time conditions.

## Features

- reorder point
- order quantity
- lead time
- demand mean
- demand coefficient of variation
- safety stock

The generator reports service level, stockout frequency, holding cost, replenishment instability, and total cost. The learned target is `operationally_acceptable`. Explicit order/storage policy limits are represented separately by `hard_policy_compliant`.

## Workflow

```text
synthetic replenishment policies
-> calibrated nonlinear feasibility learner
-> held-out classification and unsafe-accept diagnostics
-> hard policy screening
-> learned-probability screening
-> minimum-cost screened policy
```

## Run

```bash
python case_studies/09_inventory_control/run_case_study.py
```

This case is a screening benchmark rather than a stochastic-inventory optimality claim. Learned feasibility complements, rather than replaces, explicit planning constraints.
