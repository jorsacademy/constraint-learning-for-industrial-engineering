# Constraint Learning for Industrial Engineering

This repository demonstrates how data-driven constraint learning can be applied to industrial engineering problems. Five case studies are currently fully executable: manufacturing process constraint recovery, energy-efficient machine settings, assembly quality control, supply-chain feasibility, and warehouse slotting.

The project is educational and research-oriented. It separates hard feasibility from high-performance operation, evaluates learned regions on held-out data, uses cross-validation for model selection, and distinguishes descriptive operating bounds from exact constraints.

## Implemented case studies

### 01. Manufacturing process optimization

The manufacturing workflow includes reproducible synthetic process data, known hidden physical constraints for benchmark evaluation, an outcome-only learning mode, stratified train/test separation, RBF-SVM nonlinear decision boundaries, cross-validated hyperparameter tuning, classification metrics, ROC and precision-recall curves, true-vs-learned boundary comparison, descriptive operating bounds, automated tests, and a reproducible notebook.

### 02. Energy-efficient machine settings

The energy-efficiency workflow models spindle speed, feed rate, and machine load while simultaneously enforcing energy, throughput, quality, and hidden equipment-stability requirements. It includes nonlinear constraint learning, class balancing, model selection with average precision, held-out evaluation, energy-saving analysis, interpretable operating summaries, region visualization, and automated tests.

### 03. Assembly quality control

The assembly-quality workflow models tightening torque, tightening angle, tool speed, insertion force, and component temperature. Acceptable operation requires a hidden nonlinear stability condition, adequate end-of-line quality, adequate joint strength, and acceptable cycle time. It reports false accept and false reject rates in addition to standard classification metrics.

### 04. Supply-chain feasibility

The supply-chain workflow models supplier lead time, demand volatility, order quantity, safety stock, supplier utilization, and transport time. Feasibility requires adequate service level, acceptable logistics cost, and a hidden nonlinear stability condition. It includes reproducible synthetic order-cycle data, nonlinear interactions, class-balanced RBF-SVM constraint learning, cross-validated model selection, held-out evaluation, descriptive bounds, service-cost analysis, region visualization, and automated tests.

### 05. Warehouse slotting

The warehouse-slotting workflow models SKU velocity, unit weight, item cube, aisle distance, replenishment frequency, and neighboring pick density. Feasibility requires acceptable picking time, congestion, ergonomic risk, and a hidden nonlinear warehouse-stability condition. It includes:

- reproducible synthetic SKU-location observations,
- a nonlinear hidden stability score,
- a composite feasible-slotting target,
- RBF-SVM constraint learning,
- class balancing and cross-validated hyperparameter tuning,
- balanced accuracy, F1, ROC AUC, average precision, precision, and recall,
- an unsafe-accept rate for infeasible slotting assignments incorrectly accepted by the model,
- quantile-based descriptive bounds,
- identification of the best observed feasible slot,
- estimated picking-time and congestion reduction inside the feasible region,
- ROC and precision-recall curves,
- a two-dimensional SKU-velocity/aisle-distance slice of the learned six-dimensional feasibility region,
- automated tests.

## Repository structure

```text
constraint-learning-for-industrial-engineering/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   └── industrial_constraint_learning/
│       ├── __init__.py
│       ├── data_generation.py
│       └── constraint_learner.py
├── examples/
│   └── manufacturing_process.py
├── notebooks/
│   └── manufacturing_constraint_learning.ipynb
├── figures/
├── case_studies/
│   ├── 01_manufacturing_process_optimization/
│   ├── 02_energy_efficient_machine_settings/
│   ├── 03_assembly_quality_control/
│   ├── 04_supply_chain_feasibility/
│   ├── 05_warehouse_slotting/
│   │   ├── README.md
│   │   ├── warehouse_data.py
│   │   ├── warehouse_constraint_learner.py
│   │   └── run_case_study.py
│   ├── 06_job_shop_scheduling/
│   ├── 07_product_design_space/
│   ├── 08_workforce_shift_scheduling/
│   ├── 09_inventory_control/
│   └── 10_multi_product_line_balancing/
└── tests/
    ├── test_constraint_learner.py
    ├── test_energy_efficiency_case_study.py
    ├── test_assembly_quality_case_study.py
    ├── test_supply_chain_case_study.py
    └── test_warehouse_slotting_case_study.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run the executable benchmarks

```bash
python examples/manufacturing_process.py
python case_studies/02_energy_efficient_machine_settings/run_case_study.py
python case_studies/03_assembly_quality_control/run_case_study.py
python case_studies/04_supply_chain_feasibility/run_case_study.py
python case_studies/05_warehouse_slotting/run_case_study.py
```

The executable case studies write evaluation figures into `figures/`.

## Run the notebook

```bash
jupyter lab notebooks/manufacturing_constraint_learning.ipynb
```

## Run tests

```bash
pytest -q
```

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12.

## Methodological notes

The original manufacturing prototype used DBSCAN to keep the largest high-yield cluster and fitted a quadratic curve directly to feasible points. That approach can be misleading because a polynomial regression through interior feasible observations does not estimate an upper or lower constraint boundary.

The implemented examples treat constraint recovery as classification. In benchmark settings, hidden synthetic rules are available only to evaluate whether the learned region resembles the intended feasible or acceptable region. They are not supplied to the classifier as mathematical constraints.

Descriptive operating bounds are quantile-based summaries. They are intentionally not presented as exact physical constraints.

ROC AUC is reported, but average precision and the precision-recall curve are emphasized because feasible or acceptable observations may be relatively rare. Balanced accuracy is also reported to reduce the risk of interpreting majority-class accuracy as good constraint recovery.

For quality-control applications, false accept and false reject rates are reported because the operational costs of passing a defective item and rejecting a conforming item are asymmetric. For supply-chain and warehouse applications, analogous unsafe-accept metrics highlight configurations that the model would incorrectly treat as feasible.

In real industrial applications, learned constraints should complement rather than replace explicit OEM limits, engineering safety rules, regulatory constraints, contractual constraints, rack/storage constraints, fire-code requirements, and validated process specifications.

## Case studies

Ten industrial engineering applications are organized under `case_studies/`. Cases 01-05 are executable. Cases 06-10 currently contain project specifications that can be expanded into independent computational experiments.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
