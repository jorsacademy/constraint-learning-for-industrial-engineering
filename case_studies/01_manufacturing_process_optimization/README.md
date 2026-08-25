# Case Study 01: Manufacturing Process Optimization

## Industrial problem

Learn the operating envelope that produces acceptable yield while respecting nonlinear interactions between process settings.

## Example variables

- temperature
- pressure
- feed rate
- cycle time
- material grade

## Learning target

Binary feasibility or high-yield operation derived from process outcomes.

## Constraint-learning formulation

Estimate a nonlinear classification boundary separating feasible and infeasible settings. Use a benchmark with known synthetic constraints first, then repeat the analysis using outcome-only labels.

## Validation

- stratified holdout testing
- balanced accuracy
- F1 score
- ROC AUC
- average precision
- true-vs-learned boundary comparison when ground truth is available

## Status

Implemented in `src/industrial_constraint_learning/`, `examples/manufacturing_process.py`, and `notebooks/manufacturing_constraint_learning.ipynb`.
