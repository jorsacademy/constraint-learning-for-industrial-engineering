# Case Study 03: Assembly Quality Control

## Industrial problem

Learn combinations of assembly settings that avoid defects, rework, or joint failure while maintaining cycle-time requirements.

This executable case study models a tightening and insertion operation using five process variables:

- tightening torque,
- tightening angle,
- tool speed,
- insertion force,
- component temperature.

The synthetic process also generates end-of-line quality score, joint strength, and cycle time. A hidden nonlinear stability surface represents interaction effects that are not supplied directly to the learning algorithm.

## Acceptance requirements

An observation is labeled `acceptable_operation = 1` only when all of the following hold:

- the hidden nonlinear stability condition is satisfied,
- quality score is at least 93,
- joint strength is at least 82,
- cycle time is at most 8.6.

The classifier learns from the observed settings and final acceptable/not-acceptable label. The hidden stability equation is retained only to construct and audit the synthetic benchmark.

## Constraint-learning method

The workflow uses:

- reproducible synthetic data generation with NumPy's `default_rng`,
- stratified train/test separation,
- standardization,
- an RBF support-vector classifier,
- class balancing,
- cross-validated hyperparameter tuning,
- average precision as the model-selection score,
- held-out evaluation.

The learned decision surface should be interpreted as a data-driven acceptable-operation region, not as a replacement for validated engineering specifications.

## Evaluation

The executable model reports:

- confusion matrix,
- balanced accuracy,
- F1 score,
- ROC AUC,
- average precision,
- precision,
- recall,
- false accept rate,
- false reject rate.

False acceptance is particularly important in quality-control applications because it represents a defective or unacceptable observation predicted as acceptable. False rejection represents acceptable production that the model rejects.

## Interpretable summaries

The workflow also extracts quantile-based descriptive bounds for each process setting and identifies the highest-strength observed acceptable joint. These summaries are intentionally kept separate from the nonlinear classifier boundary.

## Figures

Running the case study writes:

- `figures/assembly_quality_roc_curve.png`
- `figures/assembly_quality_precision_recall_curve.png`
- `figures/assembly_quality_operating_region.png`

The operating-region figure is a two-dimensional torque/angle slice with tool speed, insertion force, and component temperature fixed at nominal values.

## Run

From the repository root:

```bash
python case_studies/03_assembly_quality_control/run_case_study.py
```

## Tests

The automated tests verify reproducibility, consistency of the acceptance label, non-degenerate class balance, held-out predictive performance, nominal-center acceptance, and rejection of an extreme setting combination.

```bash
pytest -q tests/test_assembly_quality_case_study.py
```

## Extension

A production-oriented extension would add station identifiers, time ordering, calibration monitoring, process-drift detection, and conformal prediction or another uncertainty method for observations close to the learned constraint boundary.
