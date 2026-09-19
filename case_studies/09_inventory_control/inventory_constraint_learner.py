"""Constraint learner for inventory-control policies."""

from __future__ import annotations

import pandas as pd

from industrial_constraint_learning import TabularConstraintLearner


class InventoryConstraintLearner(TabularConstraintLearner):
    feature_columns = (
        "reorder_point",
        "order_quantity",
        "lead_time",
        "demand_mean",
        "demand_cv",
        "safety_stock",
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
