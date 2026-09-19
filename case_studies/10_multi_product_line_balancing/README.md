# Case Study 10: Multi-Product Line Balancing

## Industrial problem

Learn product-mix and capacity regions where a shared production line remains stable while meeting throughput requirements without excessive utilization, WIP growth, or idle capacity.

## Features

- product A mix
- cycle-time variability
- staffing level
- buffer capacity
- changeover frequency
- target throughput

The synthetic line model generates effective capacity, utilization, throughput ratio, WIP, idle fraction, and a resource-cost proxy. The learned target is `stable_operation`. Staffing, buffer, and changeover policy limits remain explicit in `hard_capacity_compliant`.

## Workflow

```text
synthetic line configurations
-> calibrated nonlinear stability learner
-> held-out stability evaluation
-> explicit capacity-policy filter
-> learned-probability filter
-> minimum-resource screened configuration
```

## Run

```bash
python case_studies/10_multi_product_line_balancing/run_case_study.py
```

The learned stability surface is intended as a screening layer that can later be embedded in sequencing, line-balancing, or production-planning optimization.
