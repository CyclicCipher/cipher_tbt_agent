"""hippocampus/ — the full four-part hippocampus (EC-map / DG / CA3 / CA1 + replay); see DESIGN.md.

Allocortex, NOT a neocortical column: it needs none of the column's L5 motor/displacement machinery. It adds what
"reuse a column" cannot: a forkable allocentric world-STATE (the rollout's substrate), multi-chart REMAPPING, the CA3
recurrent ATTRACTOR + one-shot EPISODIC store, and the CA1 novelty comparator.

Built slice by slice, wired as each lands (RULES.md #3). The `Hippocampus` ORCHESTRATOR that composes the subfields lands
with the final slice (DESIGN §3.6); for now this package re-exports the pieces that exist.
"""

from __future__ import annotations

from .ca1 import CA1, Remapper
from .ca3 import CA3
from .dg import DG
from .map import WorldMap
from .replay import Rollout, WorldModel

__all__ = ["WorldMap", "Rollout", "WorldModel", "CA3", "DG", "CA1", "Remapper"]
