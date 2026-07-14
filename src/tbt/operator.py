"""operator.py — the TRANSFORM primitive (ARCHITECTURE.md §8). The SECOND of the column's two primitives, distinct from the
ASSOCIATE primitive (`htm.HTMLayer`) and composing with it: the operator moves the LOCATION code (L6a path integration),
then the HTMLayer reads the feature at that new location (L4). This is what makes the forward model order-invariant —
"predict the next feature from the next MOVEMENT, not from the previous feature" (`htm.py`).

WHY THIS IS NOT AN HTMLayer. An HTMLayer MEMORISES (context → next) associations in dendrite segments — per-instance,
tabular, with NO weight-sharing across positions. So a transition learned at one place is a different segment from the SAME
transition at another place: a plain sequence memory scores 0% on a place-value it was never trained on
(`project_place_invariance_needs_factored_state`, ARCHITECTURE §7). Path integration needs the OPPOSITE: one action's
effect, learned once, applied EVERYWHERE. That is a group-structured TRANSFORM of the location code, not an association —
hence a second primitive (do NOT try to unify them; that was a refuted design).

THE MECHANISM (no matrix, no gradient, no ANN — `reference_operator_as_group_representation`, Gao et al. 2021). The
location is a `GridEncoder` SDR: a bump per (scale, axis) MODULE, each module a ring of `scale` cells (`encoders.modules()`
are the contiguous bit-blocks). A translation shifts each module's bump by `δ mod scale`, so the operator for one action is
a **cyclic shift per module** — a block-structured PERMUTATION, fully described by one integer shift per module. It is:
  * LEARNED, not coded — `learn(before, action, after)` reads each module's phase delta (the best cyclic shift by overlap)
    and VOTES; a genuine translation gives a CONSTANT shift across every position, so the modal vote converges immediately
    and GENERALISES to positions never visited (the point of the primitive). We do NOT assume "ACTION1 = north" (that would
    be the bitter-lesson trap, `feedback_bitter_lesson`); the agent discovers what each action does to its own code.
  * a GROUP REPRESENTATION by construction — cyclic shifts compose additively mod scale (abelian), so `apply` chained over a
    sequence dead-reckons correctly; the operator is invertible (shift by −δ).
  * the one-step TRANSITION GENERATOR the successor representation later accumulates (SR = Σ γ^k (policy-avg M)^k, its
    discounted resolvent — ARCHITECTURE §8, ROADMAP Phase 3b); logically prior to the SR.

SCOPE (honest, per RULES — noted, not silently dropped):
  * ABELIAN core; the NON-ABELIAN SE(2) case is BUILT by CONDITIONING (not deferred): reuse THIS operator for the LOCATION
    keyed by (action, heading) — so its shift is a FUNCTION of heading (the semidirect product R²⋊SO(2)) — plus a second
    `ModularOperator` on the HEADING ring for TURN (see `column.py` `learn_pose_move`/`path_integrate_pose`). Non-commutative
    by construction. The `ConjunctiveEncoder` TENSOR (`module_grids()`) remains the route to a CONTINUOUS-heading fully-linear
    form — DEFERRED; keying is the discrete, minimal form that reuses this operator with zero changes.
  * The REGULAR free kernel only. Irregularity (a wall blocks the move, a box gets pushed) is NOT in the operator — it is a
    context-gated OVERRIDE read from the local relational context (which warps the reachability graph / SR), emergent, never
    coded (ARCHITECTURE §8; `reference_obstacle_as_transition_cost`, `reference_l5_operator_kinds`). DEFERRED.

Owned by L6a (path integration) + L5 (the displacement content); see `column.py` §12. Pure stdlib + `encoders`.
Sources: Gao et al. 2021 (path integration as a group representation); Burak & Fiete 2009 (grid continuous attractor);
Hawkins et al. 2019 (displacement cells). Legacy `operator.py`/`pose_operator` (matrix SE(2)) are REFERENCE only, not copied.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable

from tbt.encoders import SDR


class ModularOperator:
    """The abelian path-integration operator over a modular ring code (a `GridEncoder`). Learns, per ACTION, a cyclic shift
    per module by phase-delta voting; applies it as a block-structured permutation of the location SDR. An unlearned action
    acts as the IDENTITY (predicts staying put — the correct prior, and a large prediction error until it is learned).

    `grid` need only expose `modules()` (a partition of `0..n` into contiguous per-module bit-ranges) and `n`."""

    def __init__(self, grid) -> None:
        self.grid = grid
        self.n = int(grid.n)
        mods = grid.modules()
        self.bases = [int(m[0]) for m in mods]                     # first bit of each module (ranges are contiguous)
        self.sizes = [len(m) for m in mods]                        # ring size (= scale) of each module
        self.nmod = len(mods)
        self._bit_module = [0] * self.n                            # bit index -> module index (for apply)
        for mi, m in enumerate(mods):
            for b in m:
                self._bit_module[int(b)] = mi
        self.votes: dict = {}                                      # action -> [Counter(shift -> count) per module]
        self.shifts: dict = {}                                     # action -> [learned modal shift per module]

    # ---- read a module's active cells (phases within the ring) ------------------------------------------------
    def _module_cells(self, sdr: SDR, mi: int):
        base, size = self.bases[mi], self.sizes[mi]
        return {b - base for b in sdr.active if base <= b < base + size}

    @staticmethod
    def _best_shift(before_cells, after_cells, size: int) -> int:
        """The cyclic shift s that maps the before-bump onto the after-bump (max overlap) — the observed per-module phase
        delta. Exact for a clean bump: shifting the window by the true δ gives full overlap, any other shift less."""
        best_s, best_ov = 0, -1
        for s in range(size):
            ov = len({(c + s) % size for c in before_cells} & after_cells)
            if ov > best_ov:
                best_ov, best_s = ov, s
        return best_s

    # ---- LEARN: vote each module's phase delta for this action ------------------------------------------------
    def learn(self, before: SDR, action: Hashable, after: SDR) -> None:
        """Observe one transition `(before, action, after)` on the location code and update `action`'s per-module shift by
        VOTING (the modal observed delta). Position-invariant by construction: the same shift is read at every position."""
        votes = self.votes.setdefault(action, [Counter() for _ in range(self.nmod)])
        shifts = self.shifts.setdefault(action, [0] * self.nmod)
        for mi in range(self.nmod):
            bc, ac = self._module_cells(before, mi), self._module_cells(after, mi)
            if not bc or not ac:
                continue
            s = self._best_shift(bc, ac, self.sizes[mi])
            votes[mi][s] += 1
            shifts[mi] = votes[mi].most_common(1)[0][0]

    # ---- APPLY: path-integrate the location code by the learned operator --------------------------------------
    def apply(self, loc: SDR, action: Hashable) -> SDR:
        """Predict the location AFTER `action` = shift each module's bump by the learned per-module shift. Unlearned action
        → IDENTITY (`loc` unchanged). Composes: `apply(apply(loc, a), b)` shifts by shift[a]+shift[b] mod size (abelian)."""
        shifts = self.shifts.get(action)
        if shifts is None:
            return loc
        new = set()
        for b in loc.active:
            mi = self._bit_module[b]
            base, size = self.bases[mi], self.sizes[mi]
            new.add(base + ((b - base) + shifts[mi]) % size)
        return SDR(loc.n, new)

    def known(self, action: Hashable) -> bool:
        """Has this action been learned (any observation seen)?"""
        return action in self.shifts

    def shift_of(self, action: Hashable):
        """The learned per-module shift vector for an action (for inspection/tests); None if unlearned."""
        return self.shifts.get(action)
