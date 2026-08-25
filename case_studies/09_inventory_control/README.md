# Case Study 09: Inventory Control

## Industrial problem

Learn reorder-policy regions that maintain service targets without excessive holding cost, shortages, or unstable replenishment behavior.

## Example variables

- reorder point
- order quantity
- lead time
- demand mean
- demand variability
- safety stock

## Learning target

A policy-feasibility label based on service level, stockout frequency, holding cost, and replenishment constraints.

## Constraint-learning formulation

Generate observations from historical replenishment cycles or simulation experiments and learn combinations of policy and demand parameters associated with acceptable performance.

## Validation

Use service-level recall, stockout-risk precision, total-cost distribution, and validation under demand regimes not used for fitting.

## Extension

Use the learned feasibility model to restrict candidate policies in simulation optimization, Bayesian optimization, or reinforcement-learning inventory control.
