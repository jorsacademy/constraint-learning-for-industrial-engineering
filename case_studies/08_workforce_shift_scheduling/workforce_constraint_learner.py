"""Constraint learner for workforce shift scheduling."""

from __future__ import annotations

import pandas as pd

from industrial_constraint_learning import TabularConstraintLearner


class WorkforceConstraintLearner(TabularConstraintLearner):
    feature_columns = (
        "staffing_level",
        "shift_length",
        "consecutive_shifts",
        "skill_mix",
        "demand_rate",
        "break_coverage",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            data,
            feature_columns=self.feature_columns,
            label_column="operationally_acceptable",
            test_size=test_size,
            random_state=random_state,
        )
