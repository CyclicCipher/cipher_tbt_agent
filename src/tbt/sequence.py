"""The ONE temporal sequence memory (ARCHITECTURE.md §5) — context-conditioned next-element prediction.

HTM temporal memory (Hawkins & Ahmad 2016): a cell's CONTEXT (its distal/basal input = the recently-active cells) puts
it in a PREDICTIVE state; when input arrives, the predicted cells fire and INHIBIT their siblings, making the
representation CONTEXT-SPECIFIC. So the SAME element in a different context becomes a different (predictive) state, and
HIGH-ORDER sequences (ABCD vs XBCY) collapse to FIRST-ORDER transitions over the context-specific states — the predictive
cells ARE the next-element prediction.

This is ONE mechanism (§5), reused wherever a layer predicts the next element of a sequence — L4 (next feature / content
dynamics), L2/3 (next DISPLACEMENT / an object's BEHAVIOR, indexed by its PHASE), L5 (next action / a motor skill) —
differing ONLY in the element type and the context that drives it. L6 is the exception (its temporal structure is the
SR). Never reimplement it per layer (rule 1); the same difference the neuroscience notes — "the primary difference being
the contextual input sent to the active dendrites."
"""

from __future__ import annotations

import numpy as np

from .htm import HTMLayer


class SequenceMemory(HTMLayer):
    """HTM TEMPORAL MEMORY (Hawkins & Ahmad 2016) — predictive cells in an SDR, learned online. Each element activates a
    set of COLUMNS (a hashable symbol → a disjoint column block, grown online; a real content SDR's active bits ARE the
    columns). Each column holds `M` CELLS; a cell fires ALONE when a distal SEGMENT (synapses onto prior-active cells)
    put it in the PREDICTIVE state, so the SAME column fires a DIFFERENT cell in a different context → HIGH-ORDER
    sequences emerge (no fixed `order` tuple). An unpredicted column BURSTS (all cells fire) — the LITERAL surprise /
    novelty. The active-CELL SDR is the PHASE (§5's one recurrence); the predictive cells' columns ARE the next-element
    prediction. `order` (back-compat) selects the high-order depth: `order<=1` → 1 cell/column (first-order), else `M`.

    A THIRD dendrite channel — **APICAL** (top-down feedback, `observe(element, apical=...)`) — is a pure TIEBREAK on the
    basal prediction (Hawkins & Ahmad 2016; Numenta apical_tiebreak_temporal_memory): among the BASALLY-predicted cells it
    NARROWS to the apically-supported subset IF ANY exist, else falls back to ALL of them. An active BASAL segment predicts
    a cell; an active APICAL segment ALONE does NOT — so apical never fires a cell, never causes/prevents a burst, and its
    learning is DOWNSTREAM of the basal winner. Its role is to CONFIRM / bias toward an expected sequence and DETECT
    mismatch; it CANNOT split a cell that basal made shared, so it does not disambiguate identical-prefix sequences (that
    is the L2/3 pooler's job — MICROCIRCUIT.md Stage 3). Apical is OPTIONAL; `apical=None` reduces EXACTLY to the basal TM
    above (basal = lateral, predicts NEXT; apical = feedback, refines NOW)."""

    def __init__(self, order: int = 2, cells_per_column: int = 8, w: int = 8, activation_threshold: int = 4,
                 min_threshold: int = 2, connected: float = 0.5, init_perm: float = 0.40, perm_inc: float = 0.15,
                 perm_dec: float = 0.05, predicted_dec: float = 0.01, max_syn: int = 24):
        super().__init__(cells_per_column=(1 if order <= 1 else cells_per_column),   # order<=1 → first-order (1 cell/column)
                         activation_threshold=activation_threshold, min_threshold=min_threshold, connected=connected,
                         init_perm=init_perm, perm_inc=perm_inc, perm_dec=perm_dec, max_syn=max_syn)
        self.w = int(w)                                          # active columns per symbolic element
        self.predicted_dec = float(predicted_dec)               # punishment on a basal segment that predicted a column that did not activate
        self._enc: dict = {}                                     # element → its column block (grown online; the CategoryEncoder)
        self._active: set = set()                              # active cells this step (the PHASE)
        self._winners: set = set()                            # winner cells this step (presynaptic candidates for growth)
        self._predictive: set = set()                         # cells in the PREDICTIVE state (for next step)
        self._active_segs: dict = {}                          # cell → the segment(s) that made it predictive (for reinforce/punish)
        self._burst = False

    # ---- the symbol → columns encoder (disjoint block per element; a real SDR passes its active bits as columns) ----
    def _cols(self, element):
        block = self._enc.get(element)
        if block is None:
            base = len(self._enc) * self.w
            block = self._enc[element] = set(range(base, base + self.w))
        return block

    def observe(self, element, apical=None) -> None:
        """One element arrived (§5). ACTIVATE cells: a column with basally-predicted cells fires them — and if the top-down
        feedback SUPPORTS a subset (apical tiebreak), fires ONLY those; but if NONE of the basal predictions is apically
        supported it falls back to ALL of them (an active basal segment predicts a cell; an active apical segment ALONE
        does NOT — Hawkins & Ahmad 2016; Numenta apical_tiebreak_temporal_memory). A column with NO basal prediction
        BURSTS (all cells) and its WINNER is BASAL-determined (best-matching segment, else the deterministic least-used
        cell) — apical never causes or prevents a burst. Firing / winner cells reinforce their basal segment and, DOWNSTREAM
        of that basal choice, learn their APICAL segment onto the current feedback. PUNISH basal segments that predicted a
        column that did not activate; recompute the PREDICTIVE cells for next step. Learned segments (basal + apical) persist
        across `reset`. `apical` = the active top-down feedback bits (an iterable, distinct from the (col, cell) key space);
        None = no feedback (EXACTLY the basal TM). NB apical CONFIRMS / biases / detects-mismatch — it cannot split a cell
        basal made shared, so it does NOT disambiguate identical-prefix sequences; that is the L2/3 pooler's job (Stage 3)."""
        cols = self._cols(element)
        prev_active, prev_winners = self._active, self._winners
        apical = set(apical) if apical else None
        apic_pred = self._apical_predicted(apical) if apical else set()   # cells CONNECTED-supported by the feedback (the tiebreak set)
        active, winners = set(), set()
        self._burst = False
        for col in cols:
            basal = [(col, c) for c in range(self.M) if (col, c) in self._predictive]
            if basal:                                           # PREDICTED (basal): the depolarised cells fire
                supported = [cell for cell in basal if cell in apic_pred]
                for cell in (supported or basal):               # APICAL TIEBREAK: narrow to the apically-supported cells if ANY; else keep ALL basal
                    active.add(cell)
                    winners.add(cell)
                    for s in self._active_segs.get(cell, []):   # reinforce the basal segment(s) that correctly predicted this cell
                        self._learn_segment(s, prev_active, prev_winners)
                    if apical is not None:
                        self._learn_apical(cell, apical)         # learn apical DOWNSTREAM, on the basally-chosen cells
            else:                                               # BURST: no basal prediction (surprise) → all cells fire, BASAL-determined winner
                self._burst = True
                for c in range(self.M):
                    active.add((col, c))
                wcell, best = self._winner(col, prev_active)
                winners.add(wcell)
                if prev_winners:                                # (a sequence start has no context to learn)
                    if best is None:
                        best = {}
                        self.seg.setdefault(wcell, []).append(best)
                    self._learn_segment(best, prev_active, prev_winners)
                if apical is not None:
                    self._learn_apical(wcell, apical)
        for cell in self._predictive:                           # PUNISH: predicted a column that did NOT activate → weaken it
            if cell[0] not in cols:
                for s in self._active_segs.get(cell, []):
                    for c in list(s):
                        if c in prev_active:
                            s[c] -= self.predicted_dec
                            if s[c] <= 0.0:
                                del s[c]
        self._active, self._winners = active, winners
        self._recompute_predictive()

    def _recompute_predictive(self) -> None:
        """A cell is PREDICTIVE (depolarised) for next step iff a basal segment connected-matches the now-active cells (the
        base's `_predicted`, driven by the recurrent PHASE = `self._active`); track those segments for reinforce/punish."""
        self._predictive, self._active_segs = self._predicted(self._active)

    # ---- read-outs: decode the predictive cells' columns back to an element -----------------------------------
    def predicted_columns(self) -> set:
        return {col for (col, _c) in self._predictive}

    def candidates(self) -> set:
        """The elements whose column block overlaps the predicted columns — the competing next-element hypotheses (one =
        confident, several = an ambiguous/branching context, none = a burst)."""
        pc = self.predicted_columns()
        return {e for e, block in self._enc.items() if block & pc}

    def predict(self):
        """The predicted next element = the one whose block the predictive cells most cover, or None (a burst: nothing
        predicted). The predictive cells ARE the prediction (HTM); this decodes them back to an element."""
        pc = self.predicted_columns()
        best, best_ov = None, 0
        for e, block in self._enc.items():
            ov = len(block & pc)
            if ov > best_ov:
                best, best_ov = e, ov
        return best

    def confident(self) -> bool:
        """Is the prediction UNAMBIGUOUS? — exactly one element is predicted and its FULL block is covered (mastered vs a
        still-branching context; the learning signal that was the symbolic `len(Counter)==1`)."""
        cands = self.candidates()
        return len(cands) == 1 and self._enc[next(iter(cands))] <= self.predicted_columns()

    def reset(self) -> None:
        """A sequence boundary: clear the active/winner/predictive state (the PHASE) so context is not carried across it.
        The learned segments + the encoder PERSIST."""
        self._active, self._winners, self._predictive, self._active_segs = set(), set(), set(), {}
        self._burst = False


def inverse(displacement):
    """BACKWARD MODELLING (§5): the INVERSE of an SE(2) displacement. Because operators are invertible group elements,
    running a behavior BACKWARD is just applying the inverse displacements in REVERSE order (the stapler: closing IS
    opening reversed) — it is NOT a separate mechanism, only the forward sequence memory with inverse operators. Uses:
    RETRODICTION (infer the pose that PRECEDED a state: `prior = pose @ inverse(d)`) and reverse-replay credit assignment
    (walk a trajectory backward to propagate reward to the earlier states/actions that led to it)."""
    return np.linalg.inv(np.asarray(displacement, dtype=float))


class Behavior:
    """An OTHER object's DYNAMICS as a learned temporal sequence of DISPLACEMENTS (§5) — the self/other unification: your
    EFFERENCE drives self-motion (`column.forward`), and this learned behavior drives an OTHER object's next displacement,
    both by the SAME apply-operator-to-pose. `observe(pose)` reads the object's SE(2) pose each step, takes the body-frame
    displacement `pose_before⁻¹·pose_after`, quantizes it to a hashable KEY, and a `SequenceMemory` over the keys learns
    the behavior (a patrol, a toggle-cycle, an opening/closing) indexed by its PHASE. `predict()` returns the next
    displacement (a 3×3 SE(2) matrix) to apply — the OTHER-object driver of the forward model. Because operators are
    invertible, the behavior also runs BACKWARD via `predict().inverse` (P3c)."""

    def __init__(self, order: int = 2, pos_tol: float = 0.5, ang_bins: int = 16):
        self.mem = SequenceMemory(order=order)
        self.disps: dict = {}                                    # displacement KEY -> the representative displacement (3×3)
        self.pos_tol, self.ang_bins = pos_tol, ang_bins
        self._prev = None                                        # the previous observed pose (3×3)

    def _key(self, d):
        """A hashable, tolerance-quantized key for a displacement (so a repeating behavior recurs as the SAME symbol)."""
        return (round(float(d[0, 2]) / self.pos_tol), round(float(d[1, 2]) / self.pos_tol),
                round(float(np.arctan2(d[1, 0], d[0, 0]) / (2 * np.pi) * self.ang_bins)) % self.ang_bins)

    def observe(self, pose) -> None:
        """The object's SE(2) pose this step: learn the DISPLACEMENT since the last pose into the sequence memory."""
        pose = np.asarray(pose, dtype=float)
        if self._prev is not None:
            d = np.linalg.inv(self._prev) @ pose                 # the body-frame displacement (the operator)
            k = self._key(d)
            self.disps.setdefault(k, d)
            self.mem.observe(k)
        self._prev = pose

    def predict(self):
        """The predicted next DISPLACEMENT (3×3), from the behavior's phase — or None (no learned continuation yet)."""
        k = self.mem.predict()
        return self.disps.get(k) if k is not None else None

    def reset(self) -> None:
        """A boundary (the object left / a new episode): drop the phase + the last pose; keep the learned behavior."""
        self.mem.reset()
        self._prev = None
