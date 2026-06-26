"""The unified TBT agent — one cortical-column model plays any ARC-AGI-3-style game.

`UnifiedAgent` (full observability) and `PartialObsAgent` (egocentric / partial observability) are the same
model: a spatial cortical column (SR-frame map + the recurrence), a task column joined by the thalamus, the
basal ganglia gating RL/MuZero-valued subgoals (reward.py), over learned mechanics (dynamics + perception).
"""

from .unified_agent import PartialObsAgent, UnifiedAgent

__all__ = ["UnifiedAgent", "PartialObsAgent"]
