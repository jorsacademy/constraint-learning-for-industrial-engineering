# Case Study 07: Product Design Space

## Industrial problem

Learn nonlinear regions of a product-design space associated with acceptable structural, deformation, thermal, and cost outcomes while keeping explicit design rules separate from the learned model.

## Features

- wall thickness
- material strength
- rib ratio
- manufacturing tolerance
- component mass

The synthetic simulator produces stress, deformation, peak temperature, and design cost. The learned label is `operational_feasible`. Explicit tolerance and mass rules are stored separately as `hard_design_compliant` and are enforced by the downstream optimizer rather than learned from data.

## Workflow

```text
synthetic design experiments
-> calibrated RBF-SVM constraint learner
-> held-out feasibility evaluation
-> false-feasible / false-infeasible diagnostics
-> safe candidate screening
-> minimum-cost hard-compliant observed design
```

## Run

```bash
python case_studies/07_product_design_space/run_case_study.py
```

The implementation uses the shared `TabularConstraintLearner` so calibration, evaluation, and optimization semantics are consistent with the other tabular case studies.
