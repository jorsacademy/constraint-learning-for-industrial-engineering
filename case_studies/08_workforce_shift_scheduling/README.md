# Case Study 08: Workforce Shift Scheduling

## Industrial problem

Learn staffing and workload regions associated with acceptable service, overtime, and workload outcomes without replacing explicit workforce-policy constraints.

## Features

- staffing level
- shift length
- consecutive shifts
- skill mix
- demand rate
- break coverage

The synthetic operational model generates utilization, service level, overtime hours, workload index, and staffing cost. The learned target is `operationally_acceptable`. Shift-length, consecutive-shift, and break-coverage policy rules remain explicit in `policy_compliant`.

## Workflow

```text
shift-level operational observations
-> calibrated nonlinear constraint learner
-> held-out operational-feasibility evaluation
-> explicit policy filter
-> learned-probability filter
-> minimum-cost screened shift
```

## Run

```bash
python case_studies/08_workforce_shift_scheduling/run_case_study.py
```

The case study intentionally separates descriptive operational learning from legal, contractual, or safety rules that should remain deterministic.
