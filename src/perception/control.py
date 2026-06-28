"""The control-loop planner — drive the tbt Neocortex from a perceived scene.

This is the perception-side ADAPTER that satisfies the agent's `planner` contract (act / reset / new_level /
on_death) by feeding a `StateEncoder`'s scene decode into the domain-general `tbt.Neocortex`. The PLANNING
CAPABILITY lives in the model (`tbt/neocortex.py` — the §5/§6 multi-column control loop: task ⊕ space columns
talking by CMP through the thalamus, SR prioritized-replay navigation, the factored relational push); this file
is the thin glue that hands the Neocortex its OPAQUE inputs:

  - the agent cell + the mover cells              (StateEncoder.movers — pushable objects)
  - the cell graph T                              (StateEncoder.graph  — walkable minus walls/shut doors)
  - the ordered sub-goals (content, node, mover?) (StateEncoder.factors — F's emergent cover/reach terms)
  - the spatial column + cid to bind them on      (StateEncoder.column — the SR-frame map)

No colour / grid / mechanic appears here — it all comes through the StateEncoder, so the same adapter drives any
game whose perception emits a Scene. Lives in `perception/` (not `tbt/`) because it depends on the StateEncoder,
exactly as `perception/scene.py` already depends on `tbt.column`: perception is the task layer above the model.
"""

from __future__ import annotations

import random

from tbt.neocortex import Neocortex

from .scene import StateEncoder


class NeocortexPlanner:
    """A `planner` for `tbt.agent.Agent`: scene → move index, planning with the multi-column Neocortex."""

    def __init__(self, world, gamma: float = 0.9, seed: int = 0):
        self.enc = StateEncoder(world)
        self.neo = Neocortex(gamma=gamma, seed=seed)
        self.deltas = self.enc.deltas
        self.seed = seed
        self.reset()

    def reset(self):                                           # new game
        self.rng = random.Random(self.seed)
        self.enc.reset()
        self.neo.reset()
        self._bound = None

    def new_level(self):                                       # the layout changed → rebuild map, rebind fresh
        self.enc.reset()
        self.neo.reset()
        self._bound = None

    def on_death(self):                                        # GAME_OVER: the env reloads THIS level → fresh progress
        self._bound = None                                     # forces a rebind (clears the Neocortex's _done) next act

    def act(self, scene, explore: float = 0.0):
        """One scene → one move index. Decode the scene into the Neocortex's opaque inputs, (re)bind the sub-goals
        when the map or factor set changes, and let the control loop pick the move (sequence sub-goals, route a
        goal-state by CMP, SR-navigate the agent — or push the focus mover, factored egocentric ⊗ absolute)."""
        if scene.body_pos is None:                             # body not visible → can't plan; wander
            return self.rng.randrange(len(self.deltas))
        if explore and self.rng.random() < explore:           # keep discovery alive (the play loop's epsilon)
            return self.rng.randrange(len(self.deltas))
        col, cid = self.enc.column(scene)                      # the SR-frame map over the walkable cells
        T = self.enc.graph(scene)
        factors = self.enc.factors(scene)                      # F's conjunctive terms: [(cell, "cover"|"reach"), …]
        subgoals = [(i, cell, kind == "cover") for i, (cell, kind) in enumerate(factors)]   # cover ⇒ a mover sub-goal
        key = (id(col), tuple(factors))                        # rebind only when the column or the factors change
        if key != self._bound:
            self.neo.bind(subgoals, col, cid)
            self._bound = key
        agent = scene.body_pos
        if agent not in T:                                     # body off the navigable graph (shouldn't happen)
            return self.rng.randrange(len(self.deltas))
        return self.neo.act(agent, self.enc.movers(scene), T, self.enc.openers(scene))
