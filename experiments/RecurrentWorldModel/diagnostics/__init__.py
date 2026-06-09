"""Diagnostics for the Clamped Settling Core.

Stage 0 provides the convergence instrumentation that Risk 1 (does the block
settle reliably and adaptively?) is measured with. Later risks add their own
probes here (consistency-vs-correctness, credit assignment, interference).
"""

from .convergence import ConvergenceMonitor, basin_consistency

__all__ = ["ConvergenceMonitor", "basin_consistency"]
