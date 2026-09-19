"""Constraint learner for multi-product line balancing."""

from __future__ import annotations

import pandas as pd

from industrial_constraint_learning import TabularConstraintLearner


class LineBalancingConstraintLearner(TabularConstraintLearner):
    feature_columns = (
        "product_mix_a",
        "cycle_time_variability",
        "staffing_level",
        "buffer_capacity",
        "changeover_frequency",
        "target_throughput",
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
            label_column="stable_operation",
            test_size=test_size,
            random_state=random_state,
        )
