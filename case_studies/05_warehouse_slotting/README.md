# Case Study 05: Warehouse Slotting

## Industrial problem

Learn item-location combinations that keep picking time and congestion within acceptable limits while respecting physical storage restrictions.

## Example variables

- item velocity
- item weight
- item dimensions
- storage level
- distance from dispatch
- aisle traffic

## Learning target

A feasible-slotting label based on picking-time, congestion, ergonomic, and storage-capacity thresholds.

## Constraint-learning formulation

Model feasible slot assignments from historical warehouse operations. The learned model can complement deterministic rules such as weight limits and hazardous-material separation constraints.

## Validation

Evaluate feasible-assignment precision and recall, average picking-time reduction, congestion impact, and performance by product family.

## Extension

Integrate the learned feasibility classifier into a mixed-integer slotting model to prevent the optimizer from proposing historically poor item-location combinations.
