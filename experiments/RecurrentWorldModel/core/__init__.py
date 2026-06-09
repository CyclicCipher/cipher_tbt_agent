"""Stage 0 core of the Clamped Settling Core.

This package contains the *settling operator* and the machinery to run it to a
fixed point and instrument whether it actually converges. No training loops live
here (repo rule: never run training on the dev machine — Mistake #36). These are
the building blocks Stage 0 of ``Docs/implementation_plan.md`` validates:

    block.py   -- the weight-shared relational settling block f_theta
    deq.py     -- the Deep-Equilibrium wrapper: iterate f_theta to a fixed point,
                  with O(1)-memory gradient options and convergence diagnostics
    halting.py -- convergence-based halting (per-step; per-problem stub)

The four modes (Represent / Perceive / Reason / Learn) are NOT here yet -- they
are Stage 1+ and attach on top of a *validated* settling core. Build settling
first; a mode on a non-converging loop is undebuggable.
"""

from .block import SettlingBlock, SettlingBlockConfig
from .deq import DEQFixedPoint, DEQConfig, FixedPointInfo
from .halting import converged, ChainHalt

__all__ = [
    "SettlingBlock",
    "SettlingBlockConfig",
    "DEQFixedPoint",
    "DEQConfig",
    "FixedPointInfo",
    "converged",
    "ChainHalt",
]
