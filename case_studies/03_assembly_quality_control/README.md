# Case Study 03: Assembly Quality Control

## Industrial problem

Learn combinations of assembly settings that avoid defects, rework, or joint failure.

## Example variables

- tightening torque
- tightening angle
- tool speed
- insertion force
- component temperature
- station identifier

## Learning target

Pass/fail quality, rework requirement, or a composite quality label derived from dimensional and functional tests.

## Constraint-learning formulation

Learn the feasible quality region from process sensor data and end-of-line inspection outcomes. Interpretable summaries can be extracted from a nonlinear classifier, but should be reported separately from the exact learned decision surface.

## Validation

Use defect-class precision/recall, false-accept rate, false-reject rate, calibration across stations, and time-based validation to detect process drift.

## Extension

Add conformal prediction or uncertainty estimates so the system can flag operating points close to the learned constraint boundary for engineering review.
