# Case Study 10: Multi-Product Line Balancing

## Industrial problem

Learn workload and product-mix regions in which a shared production line can meet throughput targets without bottleneck overload, excessive idle time, or work-in-process growth.

## Example variables

- product mix
- station cycle times
- staffing level
- buffer capacity
- changeover frequency
- target throughput

## Learning target

A feasible-production label based on throughput attainment, station utilization, queue growth, and work-in-process limits.

## Constraint-learning formulation

Use historical shifts or discrete-event simulation experiments to learn combinations of product mix and capacity conditions that produce stable line performance.

## Validation

Measure feasible-region precision and recall, bottleneck detection accuracy, throughput retention, and robustness under changes in demand mix.

## Extension

Use the learned feasibility surface as a screening constraint inside line-balancing, sequencing, or production-planning optimization.
