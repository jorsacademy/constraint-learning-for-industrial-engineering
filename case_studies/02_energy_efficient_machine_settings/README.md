# Case Study 02: Energy-Efficient Machine Settings

## Industrial problem

Learn machine operating regions that reduce energy consumption without violating throughput, quality, or equipment constraints.

## Example variables

- spindle or motor speed
- feed rate
- load
- idle time
- coolant flow
- cycle time

## Learning target

An acceptable-operation label such as energy per unit below a threshold while throughput and quality remain above required levels.

## Constraint-learning formulation

Treat acceptable production as a constrained classification problem. Learn nonlinear combinations of settings that satisfy multiple performance conditions rather than optimizing energy in isolation.

## Validation

Evaluate precision and recall for acceptable-operation detection, energy savings inside the learned region, throughput retention, and stability across machines or production shifts.

## Extension

Compare a global model with machine-specific models and evaluate whether learned constraints transfer across equipment of the same type.
