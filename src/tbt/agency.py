"""Agency -- discover the controllable SELF with no assumptions, from controllability alone.

The self/body is the part of the frame whose change is caused by the agent's OWN actions. The hand-coded efference
copy got this wrong: it assumed the body is the colour that moves by the issued delta (ACTION1=up, one cell), which
the live games break. Here the self is just what the agent's actions CONTROL -- discovered, not assumed. Over
(frame, action, next_frame) experience the self emerges as the persistent, coherent region that reliably changes
when the agent acts; the static layout is the rest (the recurring map the column navigates). Built on the retina's
exogenous-attention residual (`salient_cells` / `dominant_region`). No assumption about action->direction, a
single-cell body, colour, scrolling, or a goal -- only "what do my actions move".

This is the first factoring the retina enables: split each frame into the static LAYOUT and the dynamic SELF/objects.
Reading each action's effect on the self (the per-action operator) is stage 2; here we recover the self and confirm
it is controllable, coherent, and local -- the thing the old body-id could not find.
"""

from __future__ import annotations

from .retina import dominant_region, salient_cells


class Agency:
    def __init__(self):
        self.steps = 0
        self.caused = 0                                       # steps whose action caused ANY change
        self.change_count: dict = {}                          # cell -> number of steps it changed
        self.per_action: dict = {}                            # action -> [dominant-region centroid] (the operator seed)

    def observe(self, frame, action, nxt) -> None:
        """One step of experience: did the agent's action change the world, where, and how coherently."""
        self.steps += 1
        cells = salient_cells(frame, nxt)
        if not cells:
            return
        self.caused += 1
        for c in cells:
            self.change_count[c] = self.change_count.get(c, 0) + 1
        _, centroid = dominant_region(cells)
        if centroid is not None:
            self.per_action.setdefault(action, []).append(centroid)

    def controllability(self) -> float:
        """Fraction of actions that caused a change -- the agent reliably affects the world, so a self exists."""
        return self.caused / self.steps if self.steps else 0.0

    def dynamic_cells(self, min_count: int = 1) -> set:
        """Cells that changed >= `min_count` times -- the self + other moving objects (vs the static layout)."""
        return {c for c, n in self.change_count.items() if n >= min_count}

    def self_region(self, min_count: int = 1):
        """The dominant coherent dynamic region and its centroid -- the controllable self (the primary moving
        object). Returns (cells, (cx, cy)), or (set(), None) if nothing moved."""
        return dominant_region(self.dynamic_cells(min_count))

    def coherence(self, min_count: int = 1) -> float:
        """How concentrated the dynamics are in ONE object: |dominant region| / |all dynamic cells| (1.0 = a single
        coherent self; low = scattered or multi-object)."""
        dyn = self.dynamic_cells(min_count)
        if not dyn:
            return 0.0
        comp, _ = dominant_region(dyn)
        return len(comp) / len(dyn)
