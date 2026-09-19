# Distribution Shift and Online Adaptation

## Objective

The learned feasibility boundary and conformal safety filter are valid only while
the deployment environment remains sufficiently similar to the data used for
training and safety calibration. This module adds an explicit lifecycle for
distribution shift:

```text
observe batch
-> detect distribution / score / target-rate shift
-> evaluate current predictive performance when labels are available
-> invalidate learned safety layer if drift is material
-> recalibrate conformal filter when the model is still adequate
-> retrain when predictive performance or severe distribution drift requires it
-> revalidate candidate
-> deploy only after gates pass
```

## Drift signals

`DistributionShiftMonitor` tracks:

- feature-level Population Stability Index (PSI);
- maximum feature PSI and shifted-feature list;
- model-score PSI;
- feasible-label-rate change when labels are available.

PSI is used as an engineering drift diagnostic, not as a formal statistical
hypothesis test. Its thresholds are configurable and should be validated for the
application.

## Fail-closed behavior

A detected shift immediately invalidates the active learned safety layer while an
adaptation decision is being made.

If labels are unavailable, the controller returns
`invalidated_no_labels`. It does not claim that an unlabeled shifted batch can
be safely recalibrated.

## Recalibration versus relearning

The controller distinguishes two cases.

### Conformal recalibration

If distribution drift is detected but the active decision model still meets
performance gates and the drift is not severe, the incoming labeled batch is
split again into:

```text
adaptation calibration
validation
```

A candidate `ConformalSafetyFilter` is fitted on the adaptation partition and
evaluated on the validation partition. It is deployed only if the false-feasible
gate passes.

### Full relearning

If balanced accuracy degrades, false-feasible error rises beyond the configured
limit, or drift severity exceeds the policy threshold, the controller trains a
new learner on the recent rolling window.

The new learner must pass:

- minimum balanced accuracy;
- maximum classification false-feasible rate;
- maximum conformal safety-filter false-feasible rate.

If the candidate fails, the old learned safety layer remains invalidated.

## Deployment state

`AdaptiveConstraintController` maintains:

- `model_valid`;
- monotonically increasing deployment `version`;
- recent rolling history;
- current drift reference distribution;
- current baseline predictive performance.

Successful recalibration or retraining increments the deployment version and
refreshes the monitoring reference.

## Main actions

```text
none
recalibrated
retrained
invalidated_no_labels
invalidated_failed_validation
```

## Example

```python
controller = AdaptiveConstraintController(
    learner,
    policy=AdaptivePolicy(alpha=0.10),
)

result = controller.process_batch(new_labeled_batch)

print(result.action)
print(result.model_valid)
print(result.drift.max_feature_psi)
```

## Limitations

The controller is a research/engineering lifecycle demonstration. Drift thresholds
are not universal constants. PSI does not establish causal concept drift, and
online adaptation does not eliminate the need for process-specific validation,
change control, auditability, or regulatory approval.

Conformal guarantees also rely on the relevance/exchangeability of the calibration
sample for the future observations being screened. Drift detection and
recalibration are therefore safeguards around that assumption, not proof that the
assumption always holds.
