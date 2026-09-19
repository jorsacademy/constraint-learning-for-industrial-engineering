# Solver Embedding of Learned Constraints

## Objective

Candidate screening is useful, but it does not allow a mathematical optimizer to
reason directly over a learned feasible region. This layer distills the active
risk-controlled teacher into an interpretable decision tree and embeds that tree
inside MILP or CP-SAT.

The pipeline is:

```text
calibrated + conformal teacher
-> sample bounded design space
-> decision-tree surrogate
-> held-out fidelity diagnostics
-> exact tree-path solver encoding
-> optimize objective with hard constraints
-> audit solver point against original conformal teacher
-> accept or use audited fallback
```

## What is exact and what is approximate

The distinction is important.

- The decision tree is an approximation of the original learned/conformal safe
  set.
- The MILP encoding is exact for the fitted decision tree.
- The CP-SAT encoding is exact for the fitted decision tree on the chosen
  discretized integer lattice.
- Neither solver backend is allowed to treat tree feasibility as proof of
  teacher feasibility.
- Every returned solver point is re-scored by the original teacher and must
  satisfy the conformal p-value rule.

If the teacher audit fails, the optimizer fails closed unless an explicit
fallback candidate set is supplied.

## Tree distillation

`RiskControlledTreeSurrogate.fit_from_teacher(...)` labels design points using:

```text
p_infeasible(x) <= alpha
```

and fits a bounded-depth `DecisionTreeClassifier`.

Held-out fidelity reports:

- accuracy;
- balanced accuracy;
- false-safe rate;
- false-unsafe rate;
- safe precision;
- teacher-safe fraction;
- surrogate-safe fraction.

For optimization, false-safe fidelity is especially important because a
surrogate that expands the safe set can propose teacher-unsafe points.

## MILP encoding

`SurrogateMILPOptimizer` extracts each safe root-to-leaf path. Every safe leaf
gets a binary activation variable. Exactly one safe leaf must be selected.

For a left tree branch:

```text
x_j <= threshold
```

the MILP uses a bounded big-M implication. For a right branch:

```text
x_j > threshold
```

the formulation uses a small scale-aware epsilon.

Finite variable bounds are mandatory because they define tight big-M constants.

The backend uses `scipy.optimize.milp`, which calls HiGHS.

## CP-SAT encoding

`SurrogateCPSATOptimizer` uses the same extracted safe leaf paths, but each path
condition is added with `OnlyEnforceIf(leaf)`. This avoids big-M constants.

CP-SAT operates on an integer lattice. `feature_scales` map original variables
onto that lattice. For example:

```python
feature_scales = {
    "temperature": 10,  # 0.1 degree resolution
    "pressure": 100,    # 0.01 MPa resolution
}
```

Tree thresholds, hard linear constraints, and the objective are scaled to the
integer model. The final point is evaluated again in original units.

## Hard constraints

Engineering, legal, OEM, capacity, and policy constraints remain explicit solver
constraints. They are not learned by the tree.

```python
hard = [
    LinearConstraintSpec([1.0, 0.0], lower_bound=180.0),
    LinearConstraintSpec([0.0, 1.0], lower_bound=2.5),
]
```

The learned tree describes only the data-driven operational region.

## Mandatory teacher audit and fallback

After the solver returns a point:

```text
solver tree feasibility
-> original probability model
-> original conformal p-value
-> hard-constraint audit
```

If the original teacher rejects the point, the solver result is not returned as
safe. With `fallback_candidates`, the implementation searches that explicit set
for the best teacher-audited point. Without a fallback set, it raises.

This audit/fallback layer is intentionally conservative because optimizing over a
surrogate can exploit approximation errors near the learned boundary.

## Interaction with online adaptation

A solver embedding is a snapshot of one deployment state. If
`AdaptiveConstraintController.version` changes because of recalibration or
retraining, the previous tree surrogate and solver encoding are stale.

Operationally:

```text
deployment version changes
-> discard old tree encoding
-> resample current bounded design space
-> refit surrogate against current teacher
-> recheck fidelity
-> rebuild MILP / CP-SAT model
```

An invalidated controller should not be used to create a new embedding until the
learned safety layer has passed adaptation and revalidation.

## Benchmark

Run:

```bash
python benchmarks/solver_embedding_comparison.py
```

The benchmark compares:

- explicit candidate search;
- HiGHS MILP tree embedding;
- OR-Tools CP-SAT tree embedding.

It reports objective value, operating point, teacher probability, conformal
p-value, audit status, fallback usage, runtime, and surrogate fidelity.

## Limitations

Decision-tree distillation can be conservative or unsafe depending on tree depth,
sampling density, and geometry of the teacher region. Held-out fidelity is
diagnostic, not a mathematical guarantee over the full continuous domain.

MILP big-M formulations require finite, credible feature bounds. CP-SAT requires
a meaningful discretization scale. The final teacher audit is therefore a
required part of the method, not an optional verification step.
