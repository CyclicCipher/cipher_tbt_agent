"""Layer 2/3 — the OUTPUT / object layer: the Numenta column pooler (MICROCIRCUIT.md §7b, Stage 3).

Objects are STABLE, location-INVARIANT identities — each a fixed sparse SDR of L2/3 cells, ASSIGNED once and HELD active
while the object is explored. PROXIMAL synapses (reusing the shared HTM segment machinery, `htm.HTMLayer` — the SAME
`_learn_segment`/`_connected_match` as L4 and the temporal memory, rule 1) grow from an object's cells to the L4
feature-at-location cells sensed while exploring it, FAST (one-shot: `init_perm ≥ connected`), so ANY of the object's
feature-locations drives (part of) its SDR — that held-stable activity IS the location-invariance.

RECOGNITION (inference) is UNION-then-NARROW: the FIRST sensation activates every object with enough proximal support (a
UNION of candidates); each further sensation INTERSECTS the active set with the still-supported cells, so it NARROWS to the
single object consistent with ALL sensed features (Hawkins, Ahmad & Cui 2017, *A Theory of How Columns…*). Readout = which
learned object SDR the active cells most overlap (associative recall — the M4 primitive, now over the pooled L4
representation). The within-session narrowing IS one column's temporal stability; lateral inter-column VOTING is the sheet
(§3b, deferred). POSE lives in L6 (the anchoring that makes L4 match) — a Stage-4 wiring concern, not here. Pure stdlib.
"""

from __future__ import annotations

import random

from .htm import HTMLayer


class L23Pooler(HTMLayer):
    """The column pooler. `learn(name, l4_cells)` assigns an object a stable SDR and grows its cells' proximal segments
    onto the feature-at-location's L4 cells; `sense(l4_cells)` runs union-then-narrow inference; `best`/`confident` read
    the recognition. Built on `htm.HTMLayer` (like `L4Layer`/`SequenceMemory`): the proximal segments live in `self.seg`
    keyed by L2/3 cell id, and a cell is PROXIMALLY SUPPORTED when its segment has ≥ `activation_threshold` connected
    synapses onto the active L4 cells (the same connected-match the basal channel uses, one mechanism)."""

    def __init__(self, n_cells: int = 1024, sdr_size: int = 24, activation_threshold: int = 3, min_threshold: int = 1,
                 connected: float = 0.5, init_perm: float = 0.55, perm_inc: float = 0.1, perm_dec: float = 0.02,
                 max_syn: int = 512, seed: int = 0) -> None:
        super().__init__(cells_per_column=1, activation_threshold=activation_threshold, min_threshold=min_threshold,
                         connected=connected, init_perm=init_perm, perm_inc=perm_inc, perm_dec=perm_dec, max_syn=max_syn)
        self.n_cells, self.sdr_size = int(n_cells), int(sdr_size)
        self.objects: dict = {}          # name -> the object's ASSIGNED stable SDR (a frozenset of L2/3 cell ids)
        self.active: set = set()         # the currently active object cells (the settled / still-narrowing identity)
        self._rng = random.Random(seed)

    def _proximal(self, cell) -> dict:
        """The one PROXIMAL segment of an L2/3 cell = `{L4 cell: permanence}` (grown from the object's feature-locations)."""
        return self.seg.setdefault(cell, [{}])[0]

    def learn(self, name, l4_cells) -> None:
        """Assign `name` a stable SDR on first sight, then GROW each of its cells' proximal segment onto this feature-at-
        location's L4 cells (called once per sensed location while exploring the object). Over the object's locations each
        of its cells connects to the object's L4 union → any location drives the object (location-invariance). Holds the
        object active (learning stability)."""
        if name not in self.objects:
            self.objects[name] = frozenset(self._rng.sample(range(self.n_cells), self.sdr_size))
        l4 = set(l4_cells)
        for cell in self.objects[name]:
            self._learn_segment(self._proximal(cell), l4, l4, punish=False)   # GROW-ONLY: accumulate the object's L4 union
        self.active = set(self.objects[name])

    def reset(self) -> None:
        """A new recognition session (a new object / a boundary): drop the active set. The library persists."""
        self.active = set()

    def _supported(self, l4_cells) -> set:
        """The L2/3 cells PROXIMALLY DRIVEN by the current L4 = those with ≥ `activation_threshold` CONNECTED proximal
        synapses onto the active L4 cells (the feed-forward / spatial-pooling overlap over L4, §7a)."""
        l4 = set(l4_cells)
        return {cell for cell, ss in self.seg.items() if self._connected_match(ss[0], l4) >= self.activation_threshold}

    def sense(self, l4_cells):
        """One sensation: UNION on the first (every object consistent with it), else NARROW (intersect the active set with
        the still-supported cells; an empty intersection — a contradiction / new object — RE-UNIONS). Returns the best."""
        supported = self._supported(l4_cells)
        if not self.active:
            self.active = supported
        else:
            narrowed = self.active & supported
            self.active = narrowed if narrowed else supported
        return self.best()

    def scores(self) -> dict:
        """Per-object overlap of the active cells with its assigned SDR — the evidence for each library object."""
        return {name: len(self.active & sdr) for name, sdr in self.objects.items()}

    def best(self):
        """The recognised object = the library object whose assigned SDR the active cells most overlap (associative
        recall); None if nothing is active / has overlapped yet."""
        if not self.active or not self.objects:
            return None
        name, ov = max(self.scores().items(), key=lambda kv: kv[1])
        return name if ov > 0 else None

    def confident(self, margin: int = 2) -> bool:
        """Has recognition NARROWED to one object? — the top overlap leads the runner-up by ≥ `margin` (and clears it)."""
        s = sorted(self.scores().values(), reverse=True)
        if not s or s[0] < margin:
            return False
        return len(s) == 1 or s[0] - s[1] >= margin
