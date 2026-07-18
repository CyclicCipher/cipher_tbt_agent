"""hippocampus/featurize.py — the world-state → SDR featuriser for the value critic (closes replay.py's leaf-value seam).

`replay.py` left an honest seam: the `ValueCritic` scores an SDR, but the rollout leaf passed a stand-in `value`. This encodes
a `WorldMap` into the SDR the critic reads, so the rollout gets a LEARNED leaf heuristic (and the critic can be trained on
world-state transitions).

WHY OVERLAP-BEARING, AND WHY A *METRIC* CODE. The critic is a LINEAR read-off over bits (`V = Σ w[bit]`), so SDR OVERLAP =
value SIMILARITY: the featuriser must map NEARBY world-states to OVERLAPPING bit sets, or value cannot generalise. Crucially
"nearby" here is EUCLIDEAN distance, and value must vary SMOOTHLY with it — so position uses a per-axis `ScalarEncoder` (a
sliding bump whose overlap DECREASES MONOTONICALLY with distance), NOT the `GridEncoder`. The grid code is MODULAR — perfect
for location IDENTITY (path integration, where L6a uses it) but it ALIASES over distance, so a far state can spuriously share
bits with a near one and the value gradient goes non-monotonic (`reference_sdr_regime_and_phase_codes`: SDR is great for
IDENTITY, breaks for METRIC — pick the encoder for the job). Each entity is tagged by identity, so a bit is `(entity, axis-bit)`:
the agent's self-location under `"self"`, each object under its id — content bound to place (the thalamus's CMP vote, degenerate
to one content symbol per entity), keeping "object A at X" distinct from "B at X" and from the agent at X.
"""

from __future__ import annotations

from ..encoders import ScalarEncoder


class WorldFeaturizer:
    """`WorldMap` → an overlap-bearing SDR (a frozenset of `(entity, axis-bit)` bits) the `ValueCritic` can score. Position is
    a per-axis metric scalar code, so nearby world-states share bits and value generalises smoothly over space."""

    def __init__(self, dims: int = 2, bounds=None, n: int = 64, w: int = 9) -> None:
        bounds = bounds or [(0, 63)] * dims
        self.axes = [ScalarEncoder(lo, hi, n=n, w=w) for lo, hi in bounds]

    def _place(self, pos) -> set:
        """The metric place code: each axis's scalar SDR, offset into its own bit-field, so x and y do not collide."""
        bits, off = set(), 0
        for enc, x in zip(self.axes, pos):
            bits |= {off + b for b in enc.encode(x).active}
            off += enc.n
        return bits

    def encode(self, world) -> frozenset:
        """The agent self-location (`"self"`) + every object at its place (`oid`). An empty / unlocated world yields the empty
        SDR (value 0 — the honest prior)."""
        bits = set()
        if world.agent is not None:
            bits |= {("self", b) for b in self._place(world.agent[0])}
        for oid, (pos, _R) in world.objects.items():
            bits |= {(oid, b) for b in self._place(pos)}
        return frozenset(bits)
