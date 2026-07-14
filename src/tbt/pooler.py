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

WHY IT IS NOT AN HTMLayer (and not a third primitive). Pooling is the ASSOCIATE primitive (Hebbian feedforward binding,
like L4's) in a POOLING regime: the OUTPUT (the identity SDR) is DECOUPLED from the instantaneous input and PERSISTS by
recurrent self-support — which an `HTMLayer` cannot do (its active cells ARE its proximal input). So L2/3 gets this small
dedicated engine, exactly as L6a gets `operator.ModularOperator` for the TRANSFORM it cannot do as sequence memory. The two
column primitives (ASSOCIATE + TRANSFORM, ARCHITECTURE §8) are unchanged; this is a specialised associative recognition
layer, not a new primitive.

THE MECHANISM:
  * an OBJECT = a fixed sparse identity SDR (`w` of `n` cells), minted once on first encounter.
  * LEARN: for each fixation, reinforce FEEDFORWARD synapses from the active L4 cells → the active identity's cells (Hebbian).
    The identity is minted at an object's ONSET (a `reset` boundary) and KEPT across the object's fixations (persistence) —
    so one object → one identity, not one-per-fixation. Re-encountering a known object RECOGNISES + reinforces it (no
    duplicate).
  * RECOGNISE (infer): the identity whose cells are best SUPPORTED (connected feedforward) by the current L4 code; once
    settled it PERSISTS while still supported, and re-pools on a mismatch. Overlap-recall, O(library), not a scan.

SCOPE (honest, per RULES): object BOUNDARIES during learning are given by `reset` (one object per learning episode); the
unsupervised boundary (mint only on an L4 prediction ERROR / burst) is the refinement. Recognition here is ~one-shot for
objects with DISTINCT feature-at-location codes; genuine INCREMENTAL disambiguation shows up only when objects SHARE codes
(ambiguity) — a harder test, deferred. Cross-column VOTING over identities is the THALAMUS's job (a multi-column slice),
not this single column's. Pure stdlib. Sources: Hawkins/Ahmad/Cui 2017 (columns); TBP/Monty 2024; Numenta temporal pooler.
"""

from __future__ import annotations

import random
from typing import Hashable


class ColumnPooler:
    """L2/3: pool the L4 feature-at-location stream into a STABLE object-identity SDR (persist + recognise). `n_cells` = the
    identity space; `w` = its sparsity; thresholds are FRACTIONS of `w`. A grown feedforward synapse is CONNECTED at once
    (`init_perm ≥ connected`) so one clean pass suffices to learn an object."""

    def __init__(self, n_cells: int = 2048, w: int = 40, connected: float = 0.5, init_perm: float = 0.55,
                 perm_inc: float = 0.1, perm_dec: float = 0.02, recognize_frac: float = 0.5, persist_frac: float = 0.5,
                 seed: int = 0) -> None:
        self.n, self.w = int(n_cells), int(w)
        self.connected, self.init_perm = float(connected), float(init_perm)
        self.perm_inc, self.perm_dec = float(perm_inc), float(perm_dec)
        self.recognize_frac, self.persist_frac = float(recognize_frac), float(persist_frac)
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

    def _mint(self) -> frozenset:
        """A NEW object: a fresh random sparse identity SDR (sparse → negligible overlap with existing objects)."""
        obj = frozenset(self.rng.sample(range(self.n), self.w))
        self.objects.append(obj)
        return obj

    def _reinforce(self, l4_active, identity: frozenset) -> None:
        """Hebbian: strengthen feedforward from the active L4 cells onto the active identity's cells (grown CONNECTED)."""
        for c in l4_active:
            syn = self.ff.setdefault(c, {})
            for l23 in identity:
                syn[l23] = min(1.0, syn.get(l23, self.init_perm) + self.perm_inc)

    # ---- pool one fixation: persist / recognise / mint, then bind ---------------------------------------------
    def pool(self, l4_active, learn: bool = True, bursting: bool = False) -> frozenset:
        """Pool one L4 feature-at-location code into the identity. If an identity is already active and still consistent
        (or we are LEARNING one object across its fixations), PERSIST it. Otherwise: a `bursting` fixation (L4 predicted
        NOTHING here — a novel feature-at-location, a location-agnostic code) must NOT be used to RECOGNISE a different known
        object (that is the feature-only-recognition trap); it is the NOVELTY signal → mint (learning) / nothing (inference).
        A SETTLED (predicted, non-burst) code is reliable → recognise the best-supported known object, else mint / nothing.
        Pooling the PREDICTED stream, treating the burst as novelty, is the theory (`reference_htm_pooling_recall_heterarchy`)."""
        if bursting:
            # L4 predicted NOTHING here — this feature-at-location is not (yet) learned, so the code is a location-agnostic
            # BURST, unreliable for recognition. LEARNING: still learning the current object → PERSIST, pool nothing (let L4
            # train first). INFERENCE: the current object does not predict here → recognition FAILURE (the boundary signal).
            return self.active if learn else frozenset()
        l4_active = frozenset(l4_active)                            # a SETTLED (predicted, location-specific) code — reliable
        sup = self._supported(l4_active)
        if self.active and (learn or self._match(self.active, sup) >= self.persist_frac):
            identity = self.active                                   # PERSIST (one object per episode / still supported)
        else:
            best, best_m = None, 0.0
            for obj in self.objects:                                # RECOGNISE by overlap-recall (O(library), not a scan)
                m = self._match(obj, sup)
                if m > best_m:
                    best, best_m = obj, m
            identity = best if (best is not None and best_m >= self.recognize_frac) else (self._mint() if learn else frozenset())
        self.active = identity
        if learn and identity:
            self._reinforce(l4_active, identity)
        return identity

    def which(self) -> int:
        """The index of the current identity in the library (a stable integer label), or -1 if none/unrecognised."""
        return self.objects.index(self.active) if self.active in self.objects else -1
