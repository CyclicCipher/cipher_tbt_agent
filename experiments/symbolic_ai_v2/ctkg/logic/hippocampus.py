"""
Hippocampus — episodic memory.

Stores two things per timestep:

1. **Activation snapshot** — {NodeId: activation_level} for ALL active nodes,
   including decay residuals. Used by replay (consolidation slow path) to
   re-strengthen transition edges.

2. **Observation record** — the list of NodeIds that were DIRECTLY observed
   in a single observe() call. No decay residuals, no spread activations.
   Used by natural transformation discovery to find which tokens genuinely
   co-occur in the same observation (not just happen to be co-active due
   to decay from previous observations).

The distinction matters: a snapshot says "these nodes were active at this
moment" (includes ghosts from previous steps). An observation record says
"these specific tokens were presented together" (pure signal, no ghosts).
"""
from __future__ import annotations

from dataclasses import dataclass, field


NodeId = int


@dataclass
class Snapshot:
    """One timestep's activation pattern."""
    step: int
    activations: dict[NodeId, float]   # {node_id: activation_level}


@dataclass
class Observation:
    """One observe() call's token list — the tokens actually presented."""
    step: int
    token_nids: list[NodeId]   # the specific nodes from this observation


class Hippocampus:
    """Episodic buffer of activation snapshots and observation records."""

    def __init__(self, max_episodes: int = 2000) -> None:
        self._snapshots: list[Snapshot] = []
        self._observations: list[Observation] = []
        self._max_episodes = max_episodes
        self._step_counter: int = 0

    def store(self, active: dict[NodeId, float],
              observed_nids: list[NodeId] | None = None) -> int:
        """Store the current activation pattern and observation.

        Parameters
        ----------
        active : {node_id: activation_level} for all active nodes (snapshot).
        observed_nids : the specific NodeIds from this observe() call.
            If None, only the snapshot is stored (backward compatibility).

        Returns the snapshot index.
        """
        step = self._step_counter
        self._step_counter += 1

        self._snapshots.append(Snapshot(step=step, activations=dict(active)))
        if observed_nids is not None:
            self._observations.append(Observation(step=step, token_nids=list(observed_nids)))

        # Evict oldest if over limit.
        if len(self._snapshots) > self._max_episodes:
            self._snapshots.pop(0)
        if len(self._observations) > self._max_episodes:
            self._observations.pop(0)

        return step

    def replay(self, index: int | None = None) -> dict[NodeId, float]:
        """Return a stored activation pattern for replay."""
        if not self._snapshots:
            return {}
        if index is None:
            return dict(self._snapshots[-1].activations)
        if 0 <= index < len(self._snapshots):
            return dict(self._snapshots[index].activations)
        return {}

    def episode_count(self) -> int:
        return len(self._snapshots)

    def observation_count(self) -> int:
        return len(self._observations)

    def all_snapshots(self) -> list[Snapshot]:
        """Return all stored snapshots (for replay in consolidation)."""
        return list(self._snapshots)

    def all_observations(self) -> list[Observation]:
        """Return all stored observation records (for NT discovery)."""
        return list(self._observations)

    def clear_before(self, step: int) -> int:
        """Remove snapshots and observations older than the given step."""
        before_snap = len(self._snapshots)
        before_obs = len(self._observations)
        self._snapshots = [s for s in self._snapshots if s.step >= step]
        self._observations = [o for o in self._observations if o.step >= step]
        return (before_snap - len(self._snapshots)) + (before_obs - len(self._observations))
