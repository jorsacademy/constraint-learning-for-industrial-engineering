# Case Study 04: Supply Chain Feasibility

## Industrial problem

Learn combinations of demand, lead time, supplier capacity, transportation conditions, and inventory buffers under which service and cost targets remain achievable.

## Example variables

- supplier lead time
- demand volatility
- order quantity
- safety stock
- supplier utilization
- transport time

## Learning target

A feasible-service label such as on-time fulfillment above a required service level while total logistics cost remains below a specified limit.

## Constraint-learning formulation

Use historical order cycles as observations and learn the boundary between stable and unstable operating conditions. Models should account for class imbalance because severe failures may be relatively rare.

## Validation

Use time-based holdout periods, average precision, missed-failure rate, service-level retention, and sensitivity to supplier or lane changes.

## Extension

Use the learned feasibility model as a constraint inside inventory or sourcing optimization rather than as a stand-alone predictor.
