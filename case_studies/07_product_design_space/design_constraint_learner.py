"""Constraint learner for the product-design-space case study."""

from __future__ import annotations

import pandas as pd

from industrial_constraint_learning import TabularConstraintLearner


class ProductDesignConstraintLearner(TabularConstraintLearner):
    feature_columns = (
        "wall_thickness",
        "material_strength",
        "rib_ratio",
        "tolerance",
        "mass",
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
            label_column="operational_feasible",
            test_size=test_size,
            random_state=random_state,
        )
