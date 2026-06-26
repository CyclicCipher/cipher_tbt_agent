"""The PREDECESSOR symbolic world-model agent — kept as a reference (see the repo README).

A hand-written perceive→induce→infer-goal→plan agent that solves the LockPath ARC replica from frame + score
alone (4/4 levels). The TBT cortical-column agent (`agent/column/`) grew out of this; `score.py` here is also
the shared scorer the column agent is evaluated with. Design notes: `docs/phase1/AGENT_DESIGN.md`.
"""

from .agent import WorldModelAgent
from .world_model import WorldModel

__all__ = ["WorldModelAgent", "WorldModel"]
