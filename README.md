# Constraint Learning for Industrial Engineering

This repository demonstrates how data-driven constraint learning can be applied to industrial engineering problems. Four case studies are currently fully executable: manufacturing process constraint recovery, energy-efficient machine settings, assembly quality control, and supply-chain feasibility.

The project is educational and research-oriented. It separates hard feasibility from high-performance operation, evaluates learned regions on held-out data, uses cross-validation for model selection, and distinguishes descriptive operating bounds from exact constraints.

## Implemented case studies

### 01. Manufacturing process optimization

The manufacturing workflow includes reproducible synthetic process data, known hidden physical constraints for benchmark evaluation, an outcome-only learning mode, stratified train/test separation, RBF-SVM nonlinear decision boundaries, cross-validated hyperparameter tuning, classification metrics, ROC and precision-recall curves, true-vs-learned boundary comparison, descriptive operating bounds, automated tests, and a reproducible notebook.

### 02. Energy-efficient machine settings

The energy-efficiency workflow models spindle speed, feed rate, and machine load while simultaneously enforcing energy, throughput, quality, and hidden equipment-stability requirements. It includes nonlinear constraint learning, class balancing, model selection with average precision, held-out evaluation, energy-saving analysis, interpretable operating summaries, region visualization, and automated tests.

### 03. Assembly quality control

The assembly-quality workflow models tightening torque, tightening angle, tool speed, insertion force, and component temperature. Acceptable operation requires a hidden nonlinear stability condition, adequate end-of-line quality, adequate joint strength, and acceptable cycle time. It reports false accept and false reject rates in addition to standard classification metrics.

### 04. Supply-chain feasibility

The supply-chain workflow models supplier lead time, demand volatility, order quantity, safety stock, supplier utilization, and transport time. Feasibility requires adequate service level, acceptable logistics cost, and a hidden nonlinear stability condition. It includes:

- reproducible synthetic order-cycle data,
- nonlinear supply-chain stability interactions,
- a composite feasible-service target,
- RBF-SVM constraint learning,
- class balancing and cross-validated hyperparameter tuning,
- balanced accuracy, F1, ROC AUC, average precision, precision, and recall,
- a missed-failure rate for infeasible cycles incorrectly accepted by the model,
- robust descriptive bounds for feasible operating settings,
- identification of the best observed service-cost point,
- ROC and precision-recall curves,
- a two-dimensional lead-time/demand-volatility slice of the learned six-dimensional feasible region,
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
│   │   ├── README.md
│   │   ├── supply_chain_data.py
│   │   ├── supply_chain_constraint_learner.py
│   │   └── run_case_study.py
│   ├── 05_warehouse_slotting/
│   ├── 06_job_shop_scheduling/
│   ├── 07_product_design_space/
│   ├── 08_workforce_shift_scheduling/
│   ├── 09_inventory_control/
│   └── 10_multi_product_line_balancing/
└── tests/
    ├── test_constraint_learner.py
    ├── test_energy_efficiency_case_study.py
    ├── test_assembly_quality_case_study.py
    └── test_supply_chain_case_study.py
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

For quality-control applications, false accept and false reject rates are reported because the operational costs of passing a defective item and rejecting a conforming item are asymmetric. For supply-chain applications, the missed-failure rate similarly highlights configurations that the model would incorrectly treat as feasible.

In real industrial applications, learned constraints should complement rather than replace explicit OEM limits, engineering safety rules, regulatory constraints, contractual constraints, and validated process specifications.

## Case studies

Ten industrial engineering applications are organized under `case_studies/`. Cases 01-04 are executable. Cases 05-10 currently contain project specifications that can be expanded into independent computational experiments.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
