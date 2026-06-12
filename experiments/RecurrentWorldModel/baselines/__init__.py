"""Baselines for the Stage 0 gate.

The decisive Stage 0 test (implementation_plan.md §2.3): does the settling core
beat a *fixed-depth* transformer of **equal active parameters** at an equal data
budget? If a matched fixed-depth model ties it, recurrent depth buys nothing.
"""

from .fixed_depth import (
    FixedDepthConfig,
    FixedDepthTransformer,
    matched_baseline,
)
from .field_model import UnifiedFieldModel, SeparateHeadsModel, FunctionalFieldModel
from .lewm import LeWorldModel
from .sigreg import SIGReg, MultiSubspaceSIGReg
from .bottleneck import ActivationFFN, TBAFPerToken, TBAFVerbatim, CommonMode, make_activation

__all__ = ["FixedDepthConfig", "FixedDepthTransformer", "matched_baseline",
           "UnifiedFieldModel", "SeparateHeadsModel", "FunctionalFieldModel",
           "LeWorldModel", "SIGReg", "MultiSubspaceSIGReg",
           "ActivationFFN", "TBAFPerToken", "TBAFVerbatim", "CommonMode", "make_activation"]
