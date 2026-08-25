# Case Study 08: Workforce Shift Scheduling

## Industrial problem

Learn staffing and shift-pattern regions associated with acceptable service, workload, and operational reliability.

## Example variables

- staffing level
- shift length
- consecutive shifts
- skill mix
- workload per employee
- break coverage

## Learning target

A feasible-shift label based on service level, overtime, queueing, quality, and policy-compliance thresholds.

## Constraint-learning formulation

Use historical shift-level operational outcomes to learn combinations of staffing and demand conditions under which performance remains acceptable. Hard legal, contractual, and safety rules should remain explicit deterministic constraints rather than being replaced by a learned model.

## Validation

Use time-based validation, service-level recall, overtime reduction, workload balance, and robustness across weekdays, seasons, and demand regimes.

## Extension

Use the learned operational-feasibility model alongside explicit labor rules in workforce optimization or simulation-based staffing analysis.
