# Conformal Safety Filtering

## Objective

A calibrated feasibility probability is useful for ranking candidates, but a fixed
probability threshold such as 0.90 does not by itself define an operational risk
guarantee.

This repository therefore adds a separate class-conditional conformal safety
filter. The filter uses an independent safety-calibration split containing
observed feasible and infeasible examples. Only the infeasible calibration scores
are used to construct the safety reference distribution.

For a candidate with feasibility score s, the null hypothesis is that the point
is infeasible. The conservative conformal p-value is

```text
p_infeasible(s)
    = (1 + number of infeasible calibration scores >= s)
      / (n_infeasible + 1)
```

A candidate passes the filter when:

```text
p_infeasible(s) <= alpha
```

## Data separation

The learning workflow uses three statistically distinct roles:

```text
full data
├── held-out test set
└── training pool
    ├── model-fit subset
    └── safety-calibration subset
```

The probability/scoring SVM and its sigmoid calibration are fitted only on the
model-fit subset. The conformal safety filter is then constructed on the
independent safety-calibration subset. A separate calibrated decision model may use the full pre-test training pool
because it is not used to generate conformal scores. The held-out test set is used only for final evaluation.

## Interpretation

Under exchangeability between the infeasible safety-calibration observations and
a future infeasible observation, the class-conditional conformal p-value is valid.
Using the rule p <= alpha therefore controls the marginal probability that a
future infeasible candidate is incorrectly accepted at level alpha.

This is not the same as a physical or regulatory safety certification.

It also does not automatically provide a family-wise guarantee after screening
many candidates and selecting the best one. Candidate selection can amplify
risk. The current `risk_controlled_optimizer()` therefore uses the conformal
threshold as a screening rule but deliberately documents the distinction between
single-candidate marginal control and selection-aware optimization.

## Finite-sample resolution

With n infeasible calibration observations, the smallest attainable conformal
p-value is:

```text
1 / (n + 1)
```

If alpha is smaller than this value, the requested risk level cannot be resolved
from the available calibration data and the API raises an error instead of
inventing a threshold.

## Diagnostics

`evaluate_safety_filter(alpha)` reports:

- conformal probability threshold;
- held-out false-feasible rate;
- held-out false-infeasible rate;
- accepted fraction;
- feasible share among accepted observations;
- number of infeasible safety-calibration observations;
- minimum attainable conformal p-value.

These quantities expose the operational trade-off between risk control and
conservatism.

## Example

```python
learner.fit(tune=True)

evaluation = learner.evaluate_safety_filter(alpha=0.10)
threshold = learner.risk_controlled_threshold(alpha=0.10)

optimizer = learner.risk_controlled_optimizer(alpha=0.10)
result = optimizer.optimize(
    candidates,
    objective=objective,
    hard_constraint=hard_constraint,
    maximize=False,
)
```

The deterministic `hard_constraint` remains separate from the learned filter.
