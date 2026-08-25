# Constraint Learning for Industrial Engineering

This repository demonstrates how data-driven constraint learning can be applied to industrial engineering problems. Three case studies are currently fully executable: manufacturing process constraint recovery, energy-efficient machine settings, and assembly quality control.

The project is educational and research-oriented. It separates hard feasibility from high-performance operation, evaluates learned regions on held-out data, uses cross-validation for model selection, and distinguishes descriptive operating bounds from exact constraints.

## Implemented case studies

### 01. Manufacturing process optimization

The manufacturing workflow includes:

- reproducible synthetic process data generation,
- known hidden physical constraints for benchmark evaluation,
- an outcome-only learning mode that does not use the hidden feasibility label during training,
- stratified train/test separation,
- RBF-SVM nonlinear decision boundaries,
- cross-validated hyperparameter tuning,
- confusion matrix, balanced accuracy, F1, ROC AUC, and average precision,
- ROC and precision-recall curves,
- true-vs-learned constraint boundary comparison,
- robust quantile summaries for high-yield operation,
- identification of the best observed physically feasible point,
- automated tests,
- a reproducible Jupyter notebook.

The outcome-only mode is particularly important. In a real industrial system, engineers often observe process inputs and outcomes but do not have a perfect label describing the true physical operating envelope. The synthetic benchmark lets us hide that label during fitting and use it only afterward to evaluate constraint recovery.

### 02. Energy-efficient machine settings

The energy-efficiency workflow models spindle speed, feed rate, and machine load while simultaneously enforcing energy, throughput, quality, and hidden equipment-stability requirements. It includes:

- reproducible synthetic machine-operation data,
- a nonlinear hidden stability envelope,
- an acceptable-operation label based on multiple simultaneous performance constraints,
- RBF-SVM constraint learning,
- cross-validated hyperparameter tuning using average precision,
- held-out balanced accuracy, F1, ROC AUC, and average precision,
- descriptive quantile bounds for acceptable settings,
- identification of the lowest-energy observed acceptable point,
- estimated energy reduction relative to the full observed operating population,
- ROC and precision-recall curves,
- a two-dimensional spindle-speed/feed-rate slice of the learned three-dimensional acceptable region,
- automated tests.

### 03. Assembly quality control

The assembly-quality workflow models tightening torque, tightening angle, tool speed, insertion force, and component temperature. Acceptable operation requires a hidden nonlinear stability condition, adequate end-of-line quality, adequate joint strength, and acceptable cycle time. It includes:

- reproducible synthetic assembly-process data,
- a nonlinear interaction-based hidden stability surface,
- a composite acceptable-operation label,
- RBF-SVM constraint learning,
- class balancing and cross-validated hyperparameter tuning,
- balanced accuracy, F1, ROC AUC, average precision, precision, and recall,
- false accept and false reject rates,
- descriptive quantile bounds for acceptable assembly settings,
- identification of the highest-strength observed acceptable joint,
- ROC and precision-recall curves,
- a two-dimensional torque/angle slice of the learned five-dimensional acceptable region,
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
│   │   ├── README.md
│   │   ├── energy_data.py
│   │   ├── energy_constraint_learner.py
│   │   └── run_case_study.py
│   ├── 03_assembly_quality_control/
│   │   ├── README.md
│   │   ├── assembly_data.py
│   │   ├── assembly_constraint_learner.py
│   │   └── run_case_study.py
│   ├── 04_supply_chain_feasibility/
│   ├── 05_warehouse_slotting/
│   ├── 06_job_shop_scheduling/
│   ├── 07_product_design_space/
│   ├── 08_workforce_shift_scheduling/
│   ├── 09_inventory_control/
│   └── 10_multi_product_line_balancing/
└── tests/
    ├── test_constraint_learner.py
    ├── test_energy_efficiency_case_study.py
    └── test_assembly_quality_case_study.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run the manufacturing benchmark

```bash
python examples/manufacturing_process.py
```

## Run the energy-efficiency benchmark

```bash
python case_studies/02_energy_efficient_machine_settings/run_case_study.py
```

## Run the assembly-quality benchmark

```bash
python case_studies/03_assembly_quality_control/run_case_study.py
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

For quality-control applications, false accept and false reject rates are also reported because the operational costs of passing a defective item and rejecting a conforming item are asymmetric.

In real industrial applications, learned constraints should complement rather than replace explicit OEM limits, engineering safety rules, regulatory constraints, and validated process specifications.

## Case studies

Ten industrial engineering applications are organized under `case_studies/`. Cases 01, 02, and 03 are executable. Cases 04-10 currently contain project specifications that can be expanded into independent computational experiments.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
