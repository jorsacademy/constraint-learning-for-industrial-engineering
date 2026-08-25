# Case Study 02: Energy-Efficient Machine Settings

## Industrial problem

Learn machine operating regions that reduce energy consumption without violating throughput, quality, or equipment-stability requirements.

This case study is fully executable. It uses a synthetic machining process controlled by spindle speed, feed rate, and machine load. The learner does not receive the mathematical form of the hidden stability envelope. It learns an acceptable-operation region from observed settings and outcomes.

## Decision variables

- spindle speed (rpm)
- feed rate (mm/min)
- machine load (normalized 0-1)

## Observed outcomes

- energy per unit (kWh/unit)
- throughput (units/hour)
- quality score

## Acceptable-operation definition

A synthetic observation is labeled acceptable only when all of the following hold:

- the hidden equipment-stability envelope is satisfied,
- energy per unit is at most 2.00 kWh/unit,
- throughput is at least 78 units/hour,
- quality score is at least 93.

The hidden stability rule creates a nonlinear feasible region. The classification model is trained on observed machine settings and the acceptable-operation label rather than on the explicit mathematical stability equation.

## Constraint-learning formulation

The acceptable operating region is learned with an RBF-kernel support vector classifier. The workflow includes:

- stratified train/test separation,
- feature standardization,
- class balancing,
- cross-validated hyperparameter tuning,
- average precision as the tuning objective,
- held-out balanced accuracy, F1, ROC AUC, and average precision,
- descriptive quantile bounds for acceptable settings,
- identification of the best observed acceptable point,
- estimated energy reduction relative to the full observed operating population,
- ROC and precision-recall curves,
- a two-dimensional speed-feed slice through the learned three-dimensional region.

## Files

```text
02_energy_efficient_machine_settings/
├── README.md
├── energy_data.py
├── energy_constraint_learner.py
└── run_case_study.py
```

Repository-level tests are located in:

```text
tests/test_energy_efficiency_case_study.py
```

## Run

From the repository root:

```bash
python case_studies/02_energy_efficient_machine_settings/run_case_study.py
```

The script writes the following figures into the repository-level `figures/` directory:

```text
energy_efficiency_roc_curve.png
energy_efficiency_precision_recall_curve.png
energy_efficiency_operating_region.png
```

## Interpretation

The learned region is a decision-support approximation, not a substitute for OEM operating limits, engineering safety constraints, or regulatory requirements. In a real plant, hard equipment and safety constraints should remain explicit. Constraint learning is most useful for discovering additional data-driven operating envelopes that arise from interacting energy, throughput, quality, and process-stability requirements.

## Extension opportunities

Useful next steps include machine-specific models, transfer learning across similar equipment, shift-level drift detection, regression models for energy and throughput, and constrained optimization over the learned acceptable region.
