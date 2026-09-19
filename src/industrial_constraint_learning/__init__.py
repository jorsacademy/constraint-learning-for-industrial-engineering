"""Constraint learning tools for industrial engineering examples."""

from .adaptive import AdaptiveConstraintController, AdaptivePolicy, AdaptiveUpdateResult
from .conformal import ConformalSafetyFilter, SafetyFilterEvaluation
from .constraint_learner import ManufacturingConstraintLearner
from .data_generation import generate_manufacturing_data
from .drift import DistributionShiftMonitor, DriftReport
from .metrics import BinaryRegionMetrics, evaluate_binary_region
from .milp_embedding import LinearConstraintSpec, MILPEmbeddingResult, SurrogateMILPOptimizer
from .optimization import CandidateOptimizationResult, SafeCandidateOptimizer
from .surrogate import RiskControlledTreeSurrogate, SurrogateFidelity, sample_uniform_design
from .tabular import TabularConstraintEvaluation, TabularConstraintLearner

__all__ = [
    "AdaptiveConstraintController",
    "AdaptivePolicy",
    "AdaptiveUpdateResult",
    "BinaryRegionMetrics",
    "CandidateOptimizationResult",
    "ConformalSafetyFilter",
    "DistributionShiftMonitor",
    "DriftReport",
    "LinearConstraintSpec",
    "MILPEmbeddingResult",
    "ManufacturingConstraintLearner",
    "SafeCandidateOptimizer",
    "RiskControlledTreeSurrogate",
    "SafetyFilterEvaluation",
    "SurrogateFidelity",
    "SurrogateMILPOptimizer",
    "TabularConstraintEvaluation",
    "TabularConstraintLearner",
    "evaluate_binary_region",
    "generate_manufacturing_data",
    "sample_uniform_design",
]
