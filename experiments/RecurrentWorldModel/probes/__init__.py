"""Diagnostic *scripts* (the probes) for the Clamped Settling Core.

Convention for this experiment: probe SOURCE lives here in ``probes/`` (tracked);
probe OUTPUT (metrics JSON, logs, plots) is written to ``diagnostics/``, which is
git-ignored repo-wide (``.gitignore``: ``experiments/*/diagnostics/``). Keep the
two apart — code in ``probes/``, artifacts in ``diagnostics/``.

Stage 0 provides the convergence instrumentation that Risk 1 (does the block
settle reliably and adaptively?) is measured with. Later risks add their own
probes here (consistency-vs-correctness, credit assignment, interference).
"""

from .convergence import ConvergenceMonitor, basin_consistency

__all__ = ["ConvergenceMonitor", "basin_consistency"]
