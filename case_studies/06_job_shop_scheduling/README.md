# Case Study 06: Job Shop Scheduling

## Industrial problem

Learn operating conditions under which a scheduling policy is likely to meet due dates without excessive overtime, queue growth, or work-in-process.

## Implemented variables

- job count
- due-date tightness
- machine utilization
- processing-time variability
- setup-time ratio
- machine availability

## Synthetic benchmark

Each observation represents a complete scheduling episode. The generator produces three operational outcomes:

- mean tardiness,
- overtime hours,
- work-in-process level.

It also evaluates a hidden nonlinear stability score that captures interactions among workload, variability, setups, due-date pressure, and machine availability. The mathematical hidden rule is used only to construct the benchmark target; it is not supplied to the classifier.

A scheduling episode is labeled feasible only when all of the following hold:

```text
hidden stability score >= 0
mean tardiness <= 6.0
 overtime hours <= 7.5
WIP level <= 38.0
```

## Constraint-learning method

The case study uses a standardized RBF support-vector classifier with class balancing. Hyperparameters are selected by cross-validated average precision. The learned model therefore acts as a surrogate feasibility constraint over six scheduling descriptors.

Reported held-out metrics include:

- confusion matrix,
- balanced accuracy,
- F1 score,
- ROC AUC,
- average precision,
- precision,
- recall,
- false feasible rate.

The false feasible rate is the fraction of truly infeasible episodes that the learned constraint incorrectly accepts. This is important when the classifier is used to screen scenarios before a more expensive scheduling or simulation stage.

## Interpretability and visualization

The implementation also provides:

- quantile-based descriptive bounds for feasible episodes,
- identification of the best observed feasible episode using a tardiness/overtime/WIP composite score,
- ROC and precision-recall curves,
- a two-dimensional machine-utilization/due-date-tightness slice of the learned six-dimensional region.

The quantile summaries are descriptive and must not be interpreted as exact scheduling constraints.

## Run

From the repository root:

```bash
python case_studies/06_job_shop_scheduling/run_case_study.py
```

Generated figures are written to `figures/`:

```text
job_shop_roc_curve.png
job_shop_precision_recall_curve.png
job_shop_feasibility_region.png
```

## Test

```bash
pytest -q tests/test_job_shop_scheduling_case_study.py
```

## Extension

A natural next step is to couple the learned feasibility model with a dispatching-rule search, simulation-optimization loop, or mixed-integer scheduling model. The classifier can screen clearly poor workload regimes before solving a detailed scheduling problem, but should not replace hard precedence, machine-capacity, or safety constraints.