# Constraint Learning for Industrial Engineering

This repository demonstrates how data-driven constraint learning can be applied to industrial engineering problems. The initial case study focuses on a synthetic manufacturing process with temperature, pressure, and yield.

The project is designed as an educational and research-oriented example. It separates physical feasibility from high-performance operation, evaluates learned feasible regions on held-out data, and distinguishes between observed best points and model-based recommendations.

## Initial case study

The manufacturing example includes:

- synthetic process data generation with known hidden constraints,
- explicit feasible/infeasible labels derived from the process rules,
- train/test separation,
- a classifier-based learned feasible region,
- simple interpretable bounds for high-yield operation,
- held-out evaluation metrics,
- visualization of the learned decision region,
- identification of the best observed feasible operating point.

## Repository structure

```text
constraint-learning-for-industrial-engineering/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   └── industrial_constraint_learning/
│       ├── __init__.py
│       ├── data_generation.py
│       └── constraint_learner.py
├── examples/
│   └── manufacturing_process.py
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

## Run the manufacturing example

```bash
python examples/manufacturing_process.py
```

## Methodological notes

The original prototype used DBSCAN to keep the largest high-yield cluster and fitted a quadratic curve directly to feasible points. That approach can be misleading because a polynomial fit to interior feasible observations does not necessarily estimate a constraint boundary. This version instead learns feasibility as a supervised classification problem and evaluates it on unseen data.

The simple min/max bounds are retained only as an interpretable summary of high-yield operation. They should not be treated as exact physical constraints.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
