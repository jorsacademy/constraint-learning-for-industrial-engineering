# Case Study 06: Job Shop Scheduling

## Industrial problem

Learn operating conditions under which a scheduling policy is likely to meet due dates without excessive overtime, queue growth, or work-in-process.

## Example variables

- job count
- due-date tightness
- machine utilization
- processing-time variability
- setup-time ratio
- machine availability

## Learning target

A schedule-feasibility label such as all critical due dates met while overtime and work-in-process remain within limits.

## Constraint-learning formulation

Generate or collect scheduling episodes, summarize each episode by workload descriptors, and learn a feasibility boundary. The learned constraint can screen dispatching rules or optimization scenarios before expensive schedule generation.

## Validation

Evaluate F1 for feasible schedules, false-feasible rate, tardiness distribution, and generalization across job mixes and machine-failure scenarios.

## Extension

Use learned feasibility as a surrogate constraint inside simulation optimization, reinforcement learning, or multi-objective scheduling.
