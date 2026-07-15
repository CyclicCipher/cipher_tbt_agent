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


class RotationOperator:
    """ROTATION as a CIRCULAR-BUFFER shift of the orientation-module index (Numenta 2021; rotation plan R2). On a
    multi-orientation `GridEncoder` (modules spread over 360°, ORDERED by orientation), rotating a location by ω = k·(360/N)
    moves each module's phase k steps around its scale's orientation ring, CELL UNCHANGED — the equivariance R2 pins:

        apply(encode(loc), k) == encode(R_ω · loc)          for ω = k·(360/N)

    So rotation is a PERMUTATION of the location code: the SAME TRANSFORM primitive as translation (`ModularOperator`), but
    shifting ACROSS modules (the orientation buffer) instead of WITHIN one (the phase). It is exact, invertible, and composes
    (k1 then k2 == k1+k2) — a group action, not a search.

    CONSTRUCTED, not learned — deliberately. `ModularOperator` LEARNS what an ACTION does (the agent cannot know what ACTION1
    means). A rotation here is not an action to discover but a HYPOTHESIS to test (R3 scans k and ranks by fit), and the shift
    follows from the grid's SUPPLIED orientation geometry (ARCHITECTURE §10 P3: the location code's structure is given, not
    learned — as the grid's scales are). If rotation ever becomes an action the body takes, `ModularOperator` learns it keyed
    on that action; nothing here needs to change.
    """

    def __init__(self, grid) -> None:
        assert not grid._axis_aligned, (
            "RotationOperator needs an ORIENTED grid — GridEncoder(orientations=N) spread over 360°. An axis-aligned "
            "(0°,90°) set is NOT closed under rotation (rotating by 90° needs the 270° direction), so the shift would not "
            "equal the rotation.")
        self.grid = grid
        self.n_orient = int(grid.n_orient)
        mods = grid.modules()
        self.bases = [int(m[0]) for m in mods]                 # first bit of each module (module index = scale*N + orient)
        self._bit_module = [0] * int(grid.n)                   # bit -> module index
        for mi, m in enumerate(mods):
            for b in m:
                self._bit_module[int(b)] = mi

    @property
    def steps(self) -> int:
        """The number of distinct rotation steps = the orientation-buffer size N (ω = k·360/N). R3 scans `range(steps)`."""
        return self.n_orient

    def angle(self, k: int) -> float:
        """The rotation angle in degrees of `k` buffer steps — how an inferred pose is reported."""
        return (int(k) % self.n_orient) * 360.0 / self.n_orient

    def apply(self, loc: SDR, k: int) -> SDR:
        """Rotate the location code by `k` steps: move every active bit from its orientation-module j to module j+k within
        the SAME scale's ring, keeping the cell. k=0 is the identity."""
        k = int(k) % self.n_orient
        if k == 0:
            return loc
        out = set()
        for b in loc.active:
            mi = self._bit_module[b]
            scale, i = divmod(mi, self.n_orient)               # modules are ordered scale-major, orientation-minor
            j = scale * self.n_orient + (i + k) % self.n_orient
            out.add(self.bases[j] + (b - self.bases[mi]))      # same cell, rotated module (same scale ⇒ same ring size)
        return SDR(loc.n, out)
