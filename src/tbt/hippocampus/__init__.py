"""hippocampus/ — the full four-part hippocampus (EC-map / DG / CA3 / CA1 + replay); see DESIGN.md.

Allocortex, NOT a neocortical column: it needs none of the column's L5 motor/displacement machinery. It adds what
"reuse a column" cannot: a forkable allocentric world-STATE (the rollout's substrate), multi-chart REMAPPING, the CA3
recurrent ATTRACTOR + one-shot EPISODIC store, and the CA1 novelty comparator.

`Hippocampus` (below) is the slice-6 ORCHESTRATOR that composes the subfields into ONE region behind a single agent handle.
"""

from __future__ import annotations

from .ca1 import CA1, Remapper
from .ca3 import CA3
from .dg import DG
from .featurize import WorldFeaturizer
from .map import WorldMap
from .replay import Rollout, WorldModel

__all__ = ["WorldMap", "Rollout", "WorldModel", "CA3", "DG", "CA1", "Remapper", "WorldFeaturizer", "Hippocampus"]


class Hippocampus:
    """The composed hippocampal region (DESIGN §2/§3, slice 6) — allocentric MAP + rollout REPLAY + one-shot EPISODIC CA3 +
    DG separation + CA1 REMAPPING, behind one handle.

    It holds the stateful memory (episodic CA3, DG separation, the chart Remapper) and provides PLANNING over a world-STATE
    the cortex assembles: the hippocampus supplies the state, the memory, and the forward SWEEP; the cortex supplies the
    learned forward MODEL (RULES #5 — compose, never reimplement). The world-state assembly (reading the nav pose + scene
    objects) is the cortex→hippocampus bridge and stays with the agent, which owns both regions. Allocortex — no L5 — so it
    reuses none of the column's deep machinery.

    Three roles, one handle:
      • EPISODIC — `remember`/`recall`: one-shot store of a scene, completion from a partial glimpse (the maze-wall case).
      • REMAPPING — `chart_key`/`visit`: DG-separate an environment signature; recall a known chart or mint a new one, a
        partial view still recalling it (absence ≠ novelty), a contradicted view remapping.
      • PLANNING — `plan`: fork the assembled world-state and roll the cortex's forward model forward to a goal (the rollout;
        the imagined-future widget's substrate)."""

    def __init__(self, n_inputs: int = 512, dims: int = 2, bounds=None, seed: int = 0) -> None:
        self.episodic = CA3()                            # one-shot scene/episode memory (store + partial-cue completion)
        self.dg = DG(n_inputs=n_inputs, seed=seed)       # environment signature → separated chart key
        self.remapper = Remapper()                       # multi-chart remapping (its own CA3 + CA1 comparator)
        self.featurizer = WorldFeaturizer(dims=dims, bounds=bounds)   # world-state → SDR for the value critic (rollout leaf)

    # ── EPISODIC memory (CA3) ──────────────────────────────────────────────────────────────────────────────────────
    def remember(self, episode) -> None:
        """Store an episode (a set of tokens) one-shot."""
        self.episodic.store(episode)

    def recall(self, glimpse) -> set:
        """Complete the whole episode from a partial glimpse (some of its tokens)."""
        return self.episodic.complete(set(glimpse))

    # ── REMAPPING (DG + CA3 + CA1) ─────────────────────────────────────────────────────────────────────────────────
    def chart_key(self, signature) -> frozenset:
        """A DG-separated chart key of an environment signature (distinct environments → well-separated keys)."""
        return self.dg.separate(signature)

    def visit(self, observed):
        """Recall the chart of a known environment (even from a partial view) or mint a new one — returns `(chart_id, CA1Result)`."""
        return self.remapper.visit(observed)

    # ── PLANNING by REPLAY (over a cortex-assembled world + model) ─────────────────────────────────────────────────
    def plan(self, world, model, reward, actions, horizon: int = 12, value=None) -> list:
        """Plan by ROLLOUT: fork `world` and search the cortex's forward `model` forward for the shortest action sequence
        reaching the goal (`reward(world) > 0`), the value critic scoring leaves beyond the horizon. Returns the action
        sequence (empty = already satisfied). `world`/`model` come from the cortex (`Agent.world_state`/`world_model`)."""
        return Rollout(model, reward, actions, horizon, value).plan(world)

    def featurize(self, world) -> frozenset:
        """A world-state → the overlap-bearing SDR the value critic scores at the rollout leaf (`featurize.py`)."""
        return self.featurizer.encode(world)
