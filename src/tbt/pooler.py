"""pooler.py — L2/3 TEMPORAL POOLING: the STABLE object-IDENTITY layer (ARCHITECTURE.md §8; column.py §2/§12).

As the sensor moves over one object, L4 emits a STREAM of feature-at-location codes — a DIFFERENT one every fixation. L2/3
pools that stream into a SINGLE, sparse, STABLE identity SDR that persists across the fixations (L4 changes; L2/3 stays the
same) and is RE-POOLED only when L4 says "this doesn't fit" (a prediction error → a different object). Recognition is
INCREMENTAL associative recall — overlap against the learned object library — never recomputed from scratch (the fix for the
"a new object almost every frame" duplication bug, `reference_htm_pooling_recall_heterarchy`).

WHAT L2/3 IS AND IS NOT (`reference_tbt_layers_4_23`). L2/3 holds the object's IDENTITY only — a label. The object's
STRUCTURE (which feature at which location, and the displacements between them) stays DISTRIBUTED across L4 (features),
L6a (locations), L5 (displacements). So this pooler does NOT store a graph of features-at-locations (that was the legacy
`L23_Object`, explicitly retired); it stores L4-stream → identity associations and a small object library.

THE MECHANISM:
  * an OBJECT = a fixed sparse identity SDR (`w` of `n` cells), minted once on first encounter.
  * `support(l4_code, identity)`: how strongly one L4 feature-at-location code backs ONE named identity (connected
    feedforward overlap). This is L2/3's associative recall, and it is the WEIGHT the column's (object, pose) hypothesis
    population is scored with — the column decides WHICH object, this layer says how well each fits.
  * `settle(identity)`: hold the identity the column concluded — L2/3's stable output.
  * LEARN (`mint` + `bind`): reinforce FEEDFORWARD synapses from the active L4 cells → an identity's cells (Hebbian).
    Both are called by `Column.commit` — an EPISODE-level act, never per fixation. See below.

WHY LEARNING IS NOT PER-FIXATION (the 2026-07-15 fix; `notes/rotation_invariance_plan.md` R5). This layer used to commit an
identity at the FIRST fixation of a learning sweep and persist it unconditionally to the end. Measured consequence: two
objects sharing their first feature-at-location MERGED into one chimeric identity holding both their features. At fixation 1
the two objects are genuinely INDISTINGUISHABLE, so recognising the first is correct INFERENCE — the bug was the absence of
REVISION once a later fixation contradicted it, and a per-fixation loop cannot revise, because by the time the contradiction
arrives the earlier fixations have already been bound to the wrong identity. Refutation needs the object's EXTENT, which
arrives only with the whole sweep. So the commitment moved to the episode end (`Column.commit`: buffer → recognise → bind),
which is Monty's structure (Buffer → the end-of-episode learning step) and leaves this layer with a clean split:
**`pool` INFERS (and never mints); `mint`/`bind` LEARN, at the caller's episode boundary.**

WHY IT IS NOT AN HTMLayer (and not a third primitive). Pooling is the ASSOCIATE primitive (Hebbian feedforward binding,
like L4's) in a POOLING regime: the OUTPUT (the identity SDR) is DECOUPLED from the instantaneous input and PERSISTS by
recurrent self-support — which an `HTMLayer` cannot do (its active cells ARE its proximal input). So L2/3 gets this small
dedicated engine, exactly as L6a gets `operator.MotionOperator` for the TRANSFORM it cannot do as sequence memory.

SCOPE (honest, per RULES): object BOUNDARIES during learning are given by the caller's episode (`Column.start_object`); the
fully-unsupervised boundary is the refinement (TBT leaves it open too). Cross-column VOTING over identities is the THALAMUS's
job (a multi-column slice), not this single column's. Pure stdlib. Sources: Hawkins/Ahmad/Cui 2017 (columns); TBP/Monty 2024
(arXiv:2412.18354); Numenta temporal pooler.
"""

from __future__ import annotations

import random
from typing import Hashable


class ColumnPooler:
    """L2/3: pool the L4 feature-at-location stream into a STABLE object-identity SDR (persist + recognise). `n_cells` = the
    identity space; `w` = its sparsity; thresholds are FRACTIONS of `w`. A grown feedforward synapse is CONNECTED at once
    (`init_perm ≥ connected`) so one clean pass suffices to learn an object."""

    def __init__(self, n_cells: int = 2048, w: int = 40, connected: float = 0.5, init_perm: float = 0.55,
                 perm_inc: float = 0.1, perm_dec: float = 0.02, recognize_frac: float = 0.5, seed: int = 0) -> None:
        self.n, self.w = int(n_cells), int(w)
        self.connected, self.init_perm = float(connected), float(init_perm)
        self.perm_inc, self.perm_dec = float(perm_inc), float(perm_dec)
        self.recognize_frac = float(recognize_frac)     # the support a fixation needs to COUNT as matching an identity
        self.rng = random.Random(seed)
        self.ff: dict = {}                 # L4 cell -> {l23 identity cell: permanence}  (feedforward)
        self.objects: list = []            # the library: each object = a frozenset of `w` identity cells
        self.active: frozenset = frozenset()   # the CURRENT identity (persists across fixations)

    def reset(self) -> None:
        """An object BOUNDARY: drop the current identity so the next `pool` starts a fresh object (recognise or mint)."""
        self.active = frozenset()

    # ---- the L2/3 cells SUPPORTED (connected feedforward) by the current L4 code ------------------------------
    def _supported(self, l4_active) -> set:
        sup = set()
        for c in l4_active:
            for l23, p in self.ff.get(c, {}).items():
                if p >= self.connected:
                    sup.add(l23)
        return sup

    def _match(self, obj: frozenset, sup: set) -> float:
        """Fraction of an object's identity cells that the current L4 code supports (the recall score)."""
        return len(obj & sup) / self.w if self.w else 0.0

    def support(self, l4_active, identity: frozenset) -> float:
        """How strongly the current L4 code supports ONE NAMED identity, in [0, 1] — the graded evidence a caller needs when
        it is testing a specific object HYPOTHESIS rather than asking "which object is this?" (`pool`'s job). L2/3 owns
        identity matching, so recognition-by-evidence reads it from here instead of re-deriving it (RULES #5)."""
        return self._match(identity, self._supported(frozenset(l4_active)))

    # ---- LEARN: the two acts, both driven by `Column.commit` at an EPISODE boundary ---------------------------
    def mint(self) -> frozenset:
        """A NEW object: a fresh random sparse identity SDR (sparse → negligible overlap with existing objects), added to the
        library. Minting is a LEARNING act, taken once a whole swept episode is not explained by any known object — never
        per fixation, and never at inference (see the module docstring)."""
        obj = frozenset(self.rng.sample(range(self.n), self.w))
        self.objects.append(obj)
        return obj

    def bind(self, l4_active, identity: frozenset) -> None:
        """LEARN one fixation onto an identity — Hebbian: strengthen feedforward from the active L4 cells onto the identity's
        cells (grown CONNECTED, so one clean pass suffices). The caller MUST pass a PREDICTED (non-burst) L4 code: a burst
        code is location-agnostic, so binding it would teach "feature → object" (feature-only recognition) rather than
        "feature-at-LOCATION → object", and the arrangement would stop being load-bearing."""
        for c in l4_active:
            syn = self.ff.setdefault(c, {})
            for l23 in identity:
                syn[l23] = min(1.0, syn.get(l23, self.init_perm) + self.perm_inc)

    # ---- INFER: hold the identity the column concluded --------------------------------------------------------
    def settle(self, identity: frozenset) -> frozenset:
        """L2/3's OUTPUT: hold the identity the column has concluded (empty = nothing recognised). The conclusion is the
        COLUMN's — a narrowed population of (object, pose) hypotheses (`Column.perceive`/`recognize`), which subsumes the
        support-only recall this layer used to do alone: that could only answer "which object does this code support?",
        never "…and where am I on it?", so it silently required the sensor to already be at the object's learned origin
        (measured: an object shifted by (7,3) read as UNRECOGNISED). `support` below is still this layer's — the population
        weighs each hypothesis with it."""
        self.active = identity
        return identity

