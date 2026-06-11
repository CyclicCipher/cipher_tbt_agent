"""Stage 0 synthetic reasoning tasks.

Stage 0 is text/static-data testable; it needs a task with a *controllable
difficulty knob* so we can test the two Risk-1 claims (does it converge; does
iterations-to-converge scale with difficulty) and the gate (does iteration beat
fixed depth as difficulty grows). The ARC-style interactive environment (env/)
is Stage 2+; this is the cheap algorithmic stand-in.
"""

from .algorithmic import ModularChain, Batch
from .eventstream import EventStream, TemporalBatch
from .shiftseq import ShiftSeq, ShiftBatch
from .driftfield import DriftField, DriftBatch

__all__ = ["ModularChain", "Batch", "EventStream", "TemporalBatch", "ShiftSeq", "ShiftBatch",
           "DriftField", "DriftBatch"]
