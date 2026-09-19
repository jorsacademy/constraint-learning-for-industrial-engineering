# Constraint Learning for Industrial Engineering

This repository demonstrates how data-driven constraint learning can be applied to industrial engineering problems. All ten case studies are executable: manufacturing process constraint recovery, energy-efficient machine settings, assembly quality control, supply-chain feasibility, warehouse slotting, job-shop scheduling, product design space, workforce shift scheduling, inventory control, and multi-product line balancing.

The project is educational and research-oriented. It separates hard feasibility from high-performance operation, evaluates learned regions on held-out data, uses cross-validation for model selection, and distinguishes descriptive operating bounds from exact constraints. The manufacturing benchmark now also exposes cross-validated probability calibration, operational feasible-region diagnostics, and a downstream candidate optimizer that combines learned probabilistic constraints with explicit hard constraints.

## Implemented case studies

### 01. Manufacturing process optimization

The manufacturing workflow includes reproducible synthetic process data, known hidden physical constraints for benchmark evaluation, an outcome-only learning mode, stratified train/test separation, RBF-SVM nonlinear decision boundaries, cross-validated hyperparameter tuning, classification metrics, ROC and precision-recall curves, true-vs-learned boundary comparison, descriptive operating bounds, automated tests, and a reproducible notebook.

### 02. Energy-efficient machine settings

The energy-efficiency workflow models spindle speed, feed rate, and machine load while simultaneously enforcing energy, throughput, quality, and hidden equipment-stability requirements. It includes nonlinear constraint learning, class balancing, model selection with average precision, held-out evaluation, energy-saving analysis, interpretable operating summaries, region visualization, and automated tests.

### 03. Assembly quality control

The assembly-quality workflow models tightening torque, tightening angle, tool speed, insertion force, and component temperature. Acceptable operation requires a hidden nonlinear stability condition, adequate end-of-line quality, adequate joint strength, and acceptable cycle time. It reports false accept and false reject rates in addition to standard classification metrics.

### 04. Supply-chain feasibility

The supply-chain workflow models supplier lead time, demand volatility, order quantity, safety stock, supplier utilization, and transport time. Feasibility requires adequate service level, acceptable logistics cost, and a hidden nonlinear stability condition. It includes reproducible synthetic order-cycle data, nonlinear interactions, class-balanced RBF-SVM constraint learning, cross-validated model selection, held-out evaluation, descriptive bounds, service-cost analysis, region visualization, and automated tests.

### 05. Warehouse slotting

The warehouse-slotting workflow models SKU velocity, unit weight, item cube, aisle distance, replenishment frequency, and neighboring pick density. Feasibility requires acceptable picking time, congestion, ergonomic risk, and a hidden nonlinear warehouse-stability condition. It includes class-balanced nonlinear constraint learning, operational unsafe-accept metrics, descriptive feasible-region summaries, performance-improvement analysis, region visualization, and automated tests.

### 06. Job-shop scheduling

The job-shop workflow models job count, due-date tightness, machine utilization, processing-time variability, setup-time ratio, and machine availability. A scheduling episode is feasible only when a hidden nonlinear stability condition is satisfied and tardiness, overtime, and work-in-process stay within limits. The implementation includes:

- reproducible synthetic scheduling episodes,
- a nonlinear hidden workload-stability score,
- a composite feasible-schedule target,
- RBF-SVM constraint learning,
- class balancing and cross-validated hyperparameter tuning,
- balanced accuracy, F1, ROC AUC, average precision, precision, and recall,
- a false-feasible rate for infeasible scheduling episodes incorrectly accepted by the model,
- quantile-based descriptive bounds,
- identification of the best observed feasible scheduling episode,
- ROC and precision-recall curves,
- a two-dimensional utilization/due-date-tightness slice of the learned six-dimensional region,
- automated tests.

### 07. Product design space

The product-design workflow learns nonlinear combinations of wall thickness, material strength, rib ratio, tolerance, and mass associated with acceptable structural, deformation, thermal, and cost outcomes. Explicit mass and tolerance rules remain separate hard constraints during downstream screening.

### 08. Workforce shift scheduling

The workforce workflow learns staffing and demand conditions associated with acceptable service level, overtime, and workload. Shift-length, consecutive-shift, and break-coverage policy rules are kept deterministic and are applied before learned-probability screening.

### 09. Inventory control

The inventory workflow learns reorder-policy regions associated with service, stockout, holding-cost, and replenishment-stability targets. Explicit order/storage policy limits remain hard constraints, and a safe candidate optimizer screens minimum-cost observed policies.

### 10. Multi-product line balancing

The line-balancing workflow learns product-mix and capacity conditions associated with stable throughput, utilization, WIP, and idle-capacity behavior. Explicit staffing, buffer, and changeover policies remain separate from the learned stability model.

Cases 07-10 share a reusable calibrated `TabularConstraintLearner` and the same hard-constraint-plus-learned-probability optimization semantics.

## Repository structure

```text
constraint-learning-for-industrial-engineering/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   └── industrial_constraint_learning/
│       ├── __init__.py
│       ├── data_generation.py
│       ├── adaptive.py
│       ├── constraint_learner.py
│       ├── conformal.py
│       ├── cpsat_embedding.py
│       ├── drift.py
│       ├── metrics.py
│       ├── milp_embedding.py
│       ├── optimization.py
│       ├── surrogate.py
│       └── tabular.py
├── examples/
│   └── manufacturing_process.py
├── notebooks/
│   └── manufacturing_constraint_learning.ipynb
├── figures/
├── docs/
│   ├── conformal_safety.md
│   ├── online_adaptation.md
│   └── solver_embedding.md
├── case_studies/
│   ├── 01_manufacturing_process_optimization/
│   ├── 02_energy_efficient_machine_settings/
│   ├── 03_assembly_quality_control/
│   ├── 04_supply_chain_feasibility/
│   ├── 05_warehouse_slotting/
│   ├── 06_job_shop_scheduling/
│   │   ├── README.md
│   │   ├── scheduling_data.py
│   │   ├── scheduling_constraint_learner.py
│   │   └── run_case_study.py
│   ├── 07_product_design_space/
│   ├── 08_workforce_shift_scheduling/
│   ├── 09_inventory_control/
│   └── 10_multi_product_line_balancing/
└── tests/
    ├── test_constraint_learner.py
    ├── test_energy_efficiency_case_study.py
    ├── test_assembly_quality_case_study.py
    ├── test_supply_chain_case_study.py
    ├── test_warehouse_slotting_case_study.py
    └── test_job_shop_scheduling_case_study.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run the executable benchmarks

```bash
python examples/manufacturing_process.py
python case_studies/02_energy_efficient_machine_settings/run_case_study.py
python case_studies/03_assembly_quality_control/run_case_study.py
python case_studies/04_supply_chain_feasibility/run_case_study.py
python case_studies/05_warehouse_slotting/run_case_study.py
python case_studies/06_job_shop_scheduling/run_case_study.py
python case_studies/07_product_design_space/run_case_study.py
python case_studies/08_workforce_shift_scheduling/run_case_study.py
python case_studies/09_inventory_control/run_case_study.py
python case_studies/10_multi_product_line_balancing/run_case_study.py
```

The executable case studies write evaluation figures into `figures/`.

## Run the notebook

```bash
jupyter lab notebooks/manufacturing_constraint_learning.ipynb
```

## Run the conformal risk sweep

```bash
python benchmarks/conformal_risk_sweep.py
```

This benchmark reports the trade-off between requested false-feasible risk,
conformal score threshold, held-out false-feasible/false-infeasible rates,
accepted-set size, and the selected manufacturing operating point.

## Run the online drift-adaptation benchmark

```bash
python benchmarks/online_drift_adaptation.py
```

The benchmark streams stable, covariate-shifted, concept-shifted, and
post-adaptation batches through the full lifecycle:

```text
detect -> invalidate -> recalibrate/retrain -> revalidate -> deploy
```

## Run the solver-embedding benchmark

```bash
python benchmarks/solver_embedding_comparison.py
```

This compares explicit candidate search, HiGHS MILP embedding, and OR-Tools
CP-SAT embedding of the same risk-controlled tree surrogate. Every solver result
is audited against the original calibrated/conformal teacher before acceptance.

## Run tests

```bash
pytest -q
```

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12.

## Risk-controlled conformal safety sets

The learned probability is now followed by an independent class-conditional
conformal safety layer. Training uses separate model-fit, safety-calibration,
and held-out test subsets. The safety filter constructs an infeasible-class
conformal p-value and accepts a future candidate only when that p-value is below
the configured `alpha`.

```python
evaluation = learner.evaluate_safety_filter(alpha=0.10)
threshold = learner.risk_controlled_threshold(alpha=0.10)
optimizer = learner.risk_controlled_optimizer(alpha=0.10)
```

Under exchangeability, this controls the marginal false-feasible probability for
a future infeasible candidate at the chosen alpha level. It is not a physical
safety certificate, and selecting the best point from many screened candidates
is a separate multiple-selection problem that is not automatically covered by
the single-candidate conformal guarantee.

The API exposes the finite-sample resolution `1 / (n_infeasible + 1)`; if an
alpha smaller than that cannot be supported by the calibration sample, the
method raises instead of fabricating a threshold. See
[`docs/conformal_safety.md`](docs/conformal_safety.md) for the statistical
contract and limitations.

## Safe learned constraints and downstream optimization

The manufacturing learner exposes calibrated feasibility probabilities using a sigmoid calibration layer fitted by cross-validation on the training partition. The held-out test partition remains untouched for evaluation.

A learned feasibility probability is not treated as a deterministic engineering guarantee. Downstream decisions can be screened with a configurable threshold:

```python
optimizer = learner.safe_optimizer(min_probability=0.50)
result = optimizer.optimize(
    candidates,
    objective=objective_function,
    hard_constraint=hard_constraint_function,
    maximize=False,
)
```

The optimizer first enforces explicit hard constraints, then requires the learned feasibility probability to exceed the configured threshold, and only then compares objective values. The default threshold of 0.50 matches the calibrated classifier decision rule; it is not a certified safety level. Higher thresholds should be chosen only after validating false-feasible behavior and the resulting feasible-set coverage for the application. This keeps OEM limits, legal rules, capacity limits, precedence relations, and other validated deterministic requirements separate from data-driven constraints.

For synthetic benchmarks with known ground truth, `boundary_metrics()` reports intersection-over-union, false-feasible rate, false-infeasible rate, feasible precision/recall, and learned-vs-true feasible-region size. The false-feasible rate is particularly important because it measures the share of truly infeasible operating points incorrectly accepted by the learned model.

## Direct MILP and CP-SAT embedding of learned constraints

The active conformal-safe teacher can now be distilled into a bounded-depth
decision tree and embedded directly in mathematical optimization.

```text
risk-controlled teacher
-> sampled bounded design space
-> tree surrogate
-> held-out fidelity diagnostics
-> MILP / CP-SAT encoding
-> solver optimization with explicit hard constraints
-> original-teacher audit
-> accept or audited fallback
```

`SurrogateMILPOptimizer` encodes safe tree leaves with binary variables and
tight bound-derived big-M implications using SciPy/HiGHS.
`SurrogateCPSATOptimizer` encodes the same leaf paths with OR-Tools reified
constraints on a user-defined integer lattice.

The tree is only a surrogate of the learned safe set. Solver feasibility is
therefore never treated as sufficient: the returned point must pass the original
conformal p-value rule. If that audit fails, the optimizer fails closed unless
an explicit fallback candidate set has been supplied.

Hard engineering/policy constraints remain direct solver constraints; they are
not distilled into the learned tree.

Solver embeddings are deployment snapshots. After online recalibration or
retraining increments the adaptive controller version, old tree encodings should
be discarded and rebuilt against the current teacher.

See [`docs/solver_embedding.md`](docs/solver_embedding.md) for the formulations,
audit contract, discretization rules, and limitations.

## Distribution shift and online adaptation

The repository now treats model validity as a deployment state rather than a
permanent property. `DistributionShiftMonitor` tracks feature PSI, model-score
PSI, and target-rate changes. `AdaptiveConstraintController` combines those
signals with labeled batch performance to decide among:

```text
none
recalibrated
retrained
invalidated_no_labels
invalidated_failed_validation
```

Material drift invalidates the learned safety layer before adaptation. Moderate
labeled shift can trigger conformal recalibration when the decision model remains
adequate. Performance degradation or severe shift triggers full relearning on a
recent rolling window. A candidate is deployed only after classification and
conformal false-feasible gates pass.

Unlabeled shift is fail-closed: the controller detects it but does not claim that
the conformal layer can be safely refreshed without outcome labels.

See [`docs/online_adaptation.md`](docs/online_adaptation.md) for the lifecycle,
validation gates, monitoring assumptions, and limitations.

## Methodological notes

The original manufacturing prototype used DBSCAN to keep the largest high-yield cluster and fitted a quadratic curve directly to feasible points. That approach can be misleading because a polynomial regression through interior feasible observations does not estimate an upper or lower constraint boundary.

The implemented examples treat constraint recovery as classification. In benchmark settings, hidden synthetic rules are available only to evaluate whether the learned region resembles the intended feasible or acceptable region. They are not supplied to the classifier as mathematical constraints.

Descriptive operating bounds are quantile-based summaries. They are intentionally not presented as exact physical constraints.

ROC AUC is reported, but average precision and the precision-recall curve are emphasized because feasible or acceptable observations may be relatively rare. Balanced accuracy is also reported to reduce the risk of interpreting majority-class accuracy as good constraint recovery.

Probability calibration improves the interpretation of model scores but does not convert empirical probabilities into physical, regulatory, or contractual guarantees. The probability threshold used by the safe candidate optimizer is therefore an operational risk parameter rather than a certified safety level.

For quality-control applications, false accept and false reject rates are reported because the operational costs of passing a defective item and rejecting a conforming item are asymmetric. For supply-chain, warehouse, and scheduling applications, analogous false-feasible or unsafe-accept metrics highlight configurations that the model would incorrectly treat as feasible.

In real industrial applications, learned constraints should complement rather than replace explicit OEM limits, engineering safety rules, regulatory constraints, contractual constraints, precedence relations, machine-capacity limits, storage constraints, fire-code requirements, and validated process specifications.

## Case studies

Ten industrial engineering applications are organized under `case_studies/`, and all ten are executable computational experiments with automated test coverage. Cases 07-10 use the shared calibrated tabular learner and explicitly separate learned operational feasibility from deterministic policy or engineering constraints.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
