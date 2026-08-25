# Constraint Learning for Industrial Engineering

This repository demonstrates how data-driven constraint learning can be applied to industrial engineering problems. The primary implemented case study is a synthetic manufacturing process with temperature, pressure, yield, and a known hidden physical operating envelope.

The project is educational and research-oriented. It separates physical feasibility from high-performance operation, evaluates learned regions on held-out data, uses cross-validation for model selection, and distinguishes descriptive operating bounds from exact constraints.

## Manufacturing benchmark

The implemented manufacturing workflow includes:

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
│   ├── 06_job_shop_scheduling/
│   ├── 07_product_design_space/
│   ├── 08_workforce_shift_scheduling/
│   ├── 09_inventory_control/
│   └── 10_multi_product_line_balancing/
└── tests/
    └── test_constraint_learner.py
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

The script writes evaluation figures into `figures/`.

## Run the notebook

```bash
jupyter lab notebooks/manufacturing_constraint_learning.ipynb
```

## Run tests

```bash
pytest -q
```

## Methodological notes

The original prototype used DBSCAN to keep the largest high-yield cluster and fitted a quadratic curve directly to feasible points. That approach can be misleading because a polynomial regression through interior feasible observations does not estimate an upper or lower constraint boundary.

This implementation treats constraint recovery as a classification problem. In the benchmark mode, the model learns the known synthetic feasibility labels. In the outcome-only mode, labels are derived only from observed yield. The hidden physical constraint is then used strictly as an external benchmark.

The simple operating bounds are quantile-based descriptive summaries. They are intentionally not presented as exact physical constraints.

ROC AUC is reported, but average precision and the precision-recall curve are emphasized because feasible or high-yield observations may be relatively rare. Balanced accuracy is also reported to reduce the risk of interpreting majority-class accuracy as good constraint recovery.

## Case studies

Ten industrial engineering applications are organized under `case_studies/`. The manufacturing case is fully implemented in the reusable Python package. The remaining case-study folders define the industrial problem, candidate variables, learning target, constraint-learning formulation, validation strategy, and extension path for future executable implementations.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
