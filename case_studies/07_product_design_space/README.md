# Case Study 07: Product Design Space

## Industrial problem

Learn regions of the design space that satisfy structural, thermal, cost, and manufacturability requirements using historical simulations or experiments.

## Example variables

- wall thickness
- material properties
- geometric ratios
- reinforcement dimensions
- manufacturing tolerance
- component mass

## Learning target

A design-feasibility label derived from simulation or test criteria such as stress, deformation, temperature, mass, and cost limits.

## Constraint-learning formulation

Treat expensive simulation results as labeled design points and learn a surrogate feasibility boundary. The model can reduce the number of infeasible designs sent to high-cost FEA or physical testing.

## Validation

Use held-out simulation cases, false-feasible rate, boundary accuracy near critical limits, and validation on a separate design family when possible.

## Extension

Combine the learned constraint with Bayesian optimization or evolutionary search to focus exploration on promising feasible regions.
