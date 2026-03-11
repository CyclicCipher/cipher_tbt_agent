"""MorphismGraph — the single core data structure and algorithm.

Implements Graph-SEQUITUR: a generalization of Nevill-Manning & Witten (1997)
from 1D sequences to arbitrary labeled directed graphs.  All operations are
O(1) amortised per observation; total O(n) for n observations.

Two invariants, both entailed by MDL (zero free parameters):
  1. Edge-pair uniqueness: no triple (A →[e1]→ B →[e2]→ C) recurs in the
     current parse without being replaced by a Composition.
  2. Rule utility: every Composition is used at least twice.

Segment boundaries arise automatically when a triple is seen for the first
time (count == 1).  Compositions arise when a triple is seen a second time
(count == 2).  No surprise threshold, no chunk-size cap.

Data representation:
  symbols   : list[Symbol]             — indexed directly by integer ID, O(1)
  atoms     : dict[str, int]           — observation value → atom ID, O(1)
  edges     : dict[(int,int,int), int] — (src, etype, tgt) → count
  pairs     : dict[(int,int,int,int,int), int] — (Q,e1,P,e2,S) → count
  rules     : dict[int, (int,int,int)] — comp_id → (left, etype, right)
  rules_inv : dict[(int,int,int), int] — (left, etype, right) → comp_id
  _out      : dict[int, dict[int, dict[int,int]]] — output index for predict()
  _buf      : list[(int, int|None)]    — current chunk buffer

All dict keys are plain Python tuples (no packed-integer encoding).  This is
correct for all corpus sizes.  See DATA_FORMATS.md for the Rust struct layout
that replaces these tuples in the production rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .topology import Topology


# ── Symbol types ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Symbol:
    sid:   int   # unique integer ID; also the list index in MorphismGraph.symbols
    level: int   # 0 = atom, >0 = composition


@dataclass(slots=True)
class Atom(Symbol):
    value:       str               # the raw observation string
    predicted:   bool = False      # True if created by a rule, not observed in the corpus
    concept_ids: frozenset = frozenset()  # FCA type IDs (Phase 19 L1): which
                                          # distributional types this atom belongs to.
                                          # Populated by LiveCTKG.on_segment() after
                                          # each FCA pass.  Empty until first pass.
                                          # frozenset is immutable: update via
                                          # atom.concept_ids = atom.concept_ids | {id}


@dataclass(slots=True)
class Composition(Symbol):
    left:  int   # left constituent symbol ID
    etype: int   # edge type integer connecting left → right
    right: int   # right constituent symbol ID


# ── MorphismGraph ─────────────────────────────────────────────────────────────

class MorphismGraph:
    """Graph-SEQUITUR morphism graph.

    Core API:
      observe(value, edge_type_int)         — process one token, O(1) amortised
      observe_sequence(data, topology)      — process a full sequence
      observe_edge(src_val, etype, tgt_val) — record one explicit edge (2D+)
      flush()                               — emit remaining buffer at end of seq
      on_segment(callback)                  — register segment-boundary callback
      predict_dist(src_id, etype_int)       — P(next | src, etype) dict
      decompose(sid)                        — (left, etype, right) for compositions
    """

    def __init__(self, topology=None) -> None:
        """Create a MorphismGraph.

        topology : optional Topology instance.  If provided, LiveCTKG is
                   automatically wired as a segment-boundary callback, making
                   FCA adjunction back-off available in all prediction functions
                   without any further setup.

                   ActiveInferenceTracker is NOT auto-wired here: its
                   O(|edges|) hook is prohibitively slow for large corpora.
                   Attach it explicitly when needed:
                       ait = ActiveInferenceTracker(mg)
        """
        # Symbol table — list-indexed by ID for O(1) access
        self.symbols:   list[Symbol]            = []
        self.atoms:     dict[str, int]          = {}   # value → atom ID

        # Edge and pair counts — tuple keys per DATA_FORMATS.md
        self.edges:     dict[tuple, int]        = {}   # (src, etype, tgt) → count
        self.pairs:     dict[tuple, int]        = {}   # (Q,e1,P,e2,S) → count

        # Pruning support (GOALS.md §8, BLUEPRINT.md §"Pruning and Forgetting")
        # _pairs_rdigram: right-digram → set of pair_keys containing it.
        #   Used by composition-triggered pruning to find dead pairs in O(degree).
        # _pair_born: pair_key → boundary-count at insertion, for singleton pruning.
        # _n_pruned: cumulative count of entries removed from pairs.
        self._pairs_rdigram: dict[tuple, set]   = {}   # (P,e,S) → set[pair_key]
        self._pair_born:     dict[tuple, int]   = {}   # pair_key → boundary at birth
        self._n_pruned:      int                = 0

        # Composition rules
        self.rules:     dict[int, tuple]        = {}   # comp_id → (l, etype, r)
        self.rules_inv: dict[tuple, int]        = {}   # (l, etype, r) → comp_id

        # Output index for O(degree) prediction: _out[src][etype][tgt] = count
        self._out: dict[int, dict[int, dict[int, int]]] = {}

        # Current chunk buffer: list of (symbol_id, incoming_edge_type_int | None)
        # _buf[i][1] = the edge from symbol i-1 to symbol i (None for i==0)
        self._buf: list[tuple[int, Optional[int]]] = []

        # Segment-boundary callbacks: called with (chunk_buf, graph) on each boundary
        self._callbacks: list[Callable] = []

        # Pre-observation hooks: called with (ctx_id, etype, tgt_id) BEFORE
        # any model update.  ctx_id is None on the first token (no context).
        # Used by ActiveInferenceTracker to compute prediction_error from the
        # prior; does NOT modify graph state.
        self._observe_hooks: list[Callable] = []

        # Running counters
        self._n_obs:        int = 0   # total observations
        self._n_boundaries: int = 0   # total segment boundaries emitted

        # Digram counts for composition trigger.
        # Separate from triple pair counts: triggers composition when a
        # (left, etype, right) digram is seen twice regardless of context Q.
        self.digrams: dict[tuple, int] = {}

        # Composition-level edge recording.
        # After observe() detects that the right sub-pair (P →[e2]→ S) matches a
        # known composition C, this is set to C's symbol ID.  The NEXT observe()
        # call records edge (C →[incoming_etype]→ next_token) into edges / _out,
        # enabling predict_dist(C_id, etype) for multi-level (trigram+) prediction.
        self._pending_comp_ctx: Optional[int] = None

        # FCA type registry (Phase 19 L1 — Marcus type-token distinction).
        # _fca_type_ids: maps a frozenset of edge-type ints (the FCA attribute set,
        #   i.e. the distributional signature of a type) to a stable integer type ID.
        # _fca_types: inverse list — type_id → frozenset[int] of edge types.
        # These are updated by LiveCTKG.on_segment() after each FCA pass.
        # Atom.concept_ids stores the set of type IDs each atom belongs to.
        self._fca_type_ids: dict[frozenset, int] = {}
        self._fca_types:    list[frozenset]      = []

        # Optional full-capability sub-systems (set by _wire when topology is given)
        self._ctkg = None   # LiveCTKG: FCA + CTKG sheaf, enables type back-off
        self._ait  = None   # ActiveInferenceTracker: not auto-wired (see docstring)

        if topology is not None:
            self._wire(topology)

    # ── Full-capability wiring ────────────────────────────────────────────────

    def _wire(self, topology) -> None:
        """Attach full capabilities for *topology*.

        Wires LiveCTKG as a segment-boundary callback.  After each chunk,
        FCA discovers formal concepts, which are sheaf-merged into a global
        CTKG.  The resulting atom_type_map() is then used automatically by
        predict() / perplexity_multilevel() for FCA adjunction back-off:
        when a context has no direct edges, the back-off pools edges from
        all atoms that share the same structural type.

        Uses lazy imports to avoid the circular-import chain
            morphism → ctkg_live → morphism  (safe at call time).

        ActiveInferenceTracker is intentionally not wired here.  Its hook
        runs O(|edges|) per observation and is prohibitively slow during
        large-corpus training.  Attach it explicitly if needed:
            from reasoning.active_inference import ActiveInferenceTracker
            ait = ActiveInferenceTracker(mg)
        """
        from ..reasoning.ctkg_live import LiveCTKG  # lazy — safe after module init
        self._ctkg = LiveCTKG(topology)
        self.on_segment(self._ctkg.on_segment)

    # ── Atom management ───────────────────────────────────────────────────────

    def _get_or_create_atom(self, value: str) -> int:
        """Return the atom ID for value, creating it if unseen.  O(1)."""
        sid = self.atoms.get(value)
        if sid is None:
            sid = len(self.symbols)
            atom = Atom(sid=sid, level=0, value=value)
            self.symbols.append(atom)
            self.atoms[value] = sid
        return sid

    def get_or_create_atom(self, value: str, coarse_type: str = '') -> int:
        """Return atom ID for value, creating a *predicted* atom if unseen.

        Unlike the private _get_or_create_atom (used during corpus ingestion),
        this marks newly created atoms as predicted=True so they can be
        distinguished from observed atoms.  Predicted atoms have zero edges
        initially; they participate in prediction but not endofunctor building.

        coarse_type is stored for Phase 19 type-membership tracking but not
        yet attached to the Atom dataclass (kept in mg._predicted_coarse_types).
        """
        sid = self.atoms.get(value)
        if sid is not None:
            return sid
        sid = len(self.symbols)
        atom = Atom(sid=sid, level=0, value=value, predicted=True)
        self.symbols.append(atom)
        self.atoms[value] = sid
        if coarse_type:
            if not hasattr(self, '_predicted_coarse_types'):
                self._predicted_coarse_types: dict[int, str] = {}
            self._predicted_coarse_types[sid] = coarse_type
        return sid

    # ── FCA type registry (Phase 19 L1 — Marcus type-token distinction) ────────

    def fca_type_id(self, attr_set: frozenset) -> int:
        """Return the stable integer ID for the FCA type identified by attr_set.

        attr_set is a frozenset of edge-type integers — the distributional
        signature shared by all atoms in a formal concept's extent.  Two atoms
        in different chunks with identical edge-type sets get the same type ID.

        Creates a new type ID the first time attr_set is seen.  Subsequent calls
        with the same attr_set return the same integer (stable across chunks).
        """
        tid = self._fca_type_ids.get(attr_set)
        if tid is None:
            tid = len(self._fca_types)
            self._fca_type_ids[attr_set] = tid
            self._fca_types.append(attr_set)
        return tid

    def atoms_of_type(self, type_id: int) -> list[int]:
        """Return all atom IDs currently registered as members of type_id.

        Linear scan — O(|atoms|).  Intended for offline analysis and rule
        indexing, not for inner-loop prediction.  For prediction, consult
        atom.concept_ids directly.
        """
        result = []
        for sid, sym in enumerate(self.symbols):
            if isinstance(sym, Atom) and type_id in sym.concept_ids:
                result.append(sid)
        return result

    # ── Output index maintenance ──────────────────────────────────────────────

    def _inc_out(self, src: int, etype: int, tgt: int, count: int) -> None:
        """Update the output index to reflect a new edge count."""
        if src not in self._out:
            self._out[src] = {}
        emap = self._out[src]
        if etype not in emap:
            emap[etype] = {}
        emap[etype][tgt] = count

    # ── Core observation ──────────────────────────────────────────────────────

    def observe(self, value: str, edge_type: Optional[int]) -> bool:
        """Process one observation.

        value     : the raw observation (character, token, pixel colour, etc.)
        edge_type : integer code of the edge that leads from the previous
                    observation to this one.  Pass None for the first observation
                    in each sequence (no incoming edge).

        Returns True if a segment boundary was detected (this triple has never
        been seen before, meaning the current chunk has reached a natural end).

        The buffer is flushed on a boundary: everything up to and including the
        previous symbol is emitted to registered callbacks, and the current
        symbol starts a new chunk.
        """
        self._n_obs += 1
        S = self._get_or_create_atom(value)

        # ── Fire pre-observation hooks (BEFORE any model update) ─────────────
        # Hooks see the prior state: edges not yet incremented, no new compositions.
        # ctx_id is None on the first token (nothing in buffer yet).
        if self._observe_hooks:
            ctx_id = self._buf[-1][0] if self._buf else None
            for hook in self._observe_hooks:
                hook(ctx_id, edge_type, S, self)

        # ── First observation or start of a new chunk after a boundary ──────
        if not self._buf:
            self._pending_comp_ctx = None  # no incoming edge; can't record comp edge
            self._buf.append((S, edge_type))
            return False

        # ── Record the edge P → S ────────────────────────────────────────────
        P, _e_incoming_P = self._buf[-1]
        # edge_type here is the edge FROM P TO S (the incoming edge of S)
        edge_key = (P, edge_type, S)
        new_edge_count = self.edges.get(edge_key, 0) + 1
        self.edges[edge_key] = new_edge_count
        self._inc_out(P, edge_type, S, new_edge_count)

        # ── Record composition-level edge if a composition was pending ────────
        # _pending_comp_ctx holds the ID of the composition that ended at P.
        # Now that we know P was followed by S via edge_type, record that edge
        # from the composition into the same edges / _out table.
        if self._pending_comp_ctx is not None and edge_type is not None:
            cid      = self._pending_comp_ctx
            self._pending_comp_ctx = None
            comp_cnt = self.edges.get((cid, edge_type, S), 0) + 1
            self.edges[(cid, edge_type, S)] = comp_cnt
            self._inc_out(cid, edge_type, S, comp_cnt)

        # ── Digram-based composition trigger ─────────────────────────────────
        # Create a Composition whenever (P →[edge_type]→ S) has been seen twice,
        # regardless of which context Q preceded it.  More aggressive than the
        # old triple-based trigger, so depth grows faster.
        if edge_type is not None:
            dgram_key = (P, edge_type, S)
            if dgram_key not in self.rules_inv:
                old_dgram = self.digrams.get(dgram_key, 0)
                self.digrams[dgram_key] = old_dgram + 1
                if old_dgram == 1:
                    # Second occurrence → create composition + rewrite earlier buf
                    self._create_composition(P, edge_type, S)

        # ── Check pair (Q →[e1]→ P →[e2]→ S) for boundary detection ─────────
        #   e1 = the edge from Q to P = the incoming edge stored for P in _buf
        #   e2 = edge_type = the incoming edge of S (edge from P to S)
        #   Pairs are only checked when both edges are valid (non-None).
        is_boundary = False
        if (len(self._buf) >= 2
                and self._buf[-1][1] is not None  # e1 must be valid
                and edge_type is not None):        # e2 must be valid
            Q, _ = self._buf[-2]
            e1   = self._buf[-1][1]   # edge Q → P
            e2   = edge_type          # edge P → S
            pair_key = (Q, e1, P, e2, S)
            old_count = self.pairs.get(pair_key, 0)
            self.pairs[pair_key] = old_count + 1

            if old_count == 0:
                # First time this triple is seen → segment boundary.
                # Register in reverse index and record birth-boundary for pruning.
                is_boundary = True
                rdig = (P, e2, S)
                if rdig not in self._pairs_rdigram:
                    self._pairs_rdigram[rdig] = set()
                self._pairs_rdigram[rdig].add(pair_key)
                self._pair_born[pair_key] = self._n_boundaries
            # (composition creation moved to digram-based trigger above)

            # If the right sub-pair (P →[e2]→ S) is a known composition,
            # mark it so the NEXT observe() records what follows it.
            # This fires on every occurrence (count >= 2), not only on creation.
            comp_key = (P, e2, S)
            if comp_key in self.rules_inv:
                self._pending_comp_ctx = self.rules_inv[comp_key]

        # ── Buffer management ─────────────────────────────────────────────────
        if is_boundary:
            # Emit everything up to and including P; S starts the next chunk.
            self._emit_chunk(list(self._buf))
            self._buf = [(S, edge_type)]
        else:
            self._buf.append((S, edge_type))
            # Compress the tail: if the last two entries (P, S) form a known
            # composition C, replace them with C.  Repeat until stable.
            # This feeds composition IDs back into the buffer so that the
            # *next* pair-check sees (Q, C, new_token) — enabling depth > 1.
            self._compress_buf_tail()

        return is_boundary

    def flush(self) -> None:
        """Emit the remaining buffer at end of a sequence.

        Call this after the last observe() call for each input sequence.
        Resets the buffer so the next sequence starts cleanly.
        """
        if self._buf:
            self._emit_chunk(list(self._buf))
            self._buf = []
        self._pending_comp_ctx = None  # no next token to record against

    def observe_sequence(self, data: Any, topology: Topology) -> None:
        """Process a full sequence using stream_tokens() from topology."""
        for value, etype in topology.stream_tokens(data):
            self.observe(value, etype)
        self.flush()

    def observe_edge(self, src_value: str, etype: int, tgt_value: str) -> None:
        """Record one explicit directed edge (for 2D+ topologies).

        Does not update the sequential chunk buffer.  Updates edge counts and
        the output index only.  Pair detection for 2D topologies is a future
        extension.
        """
        S = self._get_or_create_atom(src_value)
        T = self._get_or_create_atom(tgt_value)
        edge_key = (S, etype, T)
        new_count = self.edges.get(edge_key, 0) + 1
        self.edges[edge_key] = new_count
        self._inc_out(S, etype, T, new_count)

    # ── Composition ───────────────────────────────────────────────────────────

    def _create_composition(self, left: int, etype: int, right: int) -> int:
        """Create a Composition C = (left →[etype]→ right) if it does not exist.

        Returns the composition's symbol ID.
        """
        rule_key = (left, etype, right)
        existing = self.rules_inv.get(rule_key)
        if existing is not None:
            return existing

        # Determine the level: one above the higher of the two constituents
        l_level = self.symbols[left].level
        r_level = self.symbols[right].level
        c_level = max(l_level, r_level) + 1

        sid = len(self.symbols)
        comp = Composition(sid=sid, level=c_level, left=left, etype=etype, right=right)
        self.symbols.append(comp)
        self.rules[sid]          = rule_key
        self.rules_inv[rule_key] = sid

        # Retroactively rewrite earlier (left →[etype]→ right) in the buffer.
        self._rewrite_buf(left, etype, right, sid)
        # Prune all pair-table entries whose right sub-pair is (left, etype, right).
        # They are permanently dead: future occurrences get compressed to sid first.
        self._prune_rdigram(left, etype, right)
        return sid

    def _compress_buf_tail(self) -> None:
        """Greedily compress the buffer tail.

        While the last two entries (L →[e]→ R) in _buf match a known
        composition C, merge them: pop R, replace L with C (inheriting L's
        incoming edge).  Repeats until the tail pair is not a known
        composition.

        This is what allows depth > 1: once C = (L, e, R) exists and the
        buffer ends with [..., L, R], we replace the tail with C.  On the
        *next* observe() call the pair check sees (..., C, new_token), and
        if that triple appears twice a depth-2 composition is created.

        Runs in O(d) where d is the depth gained on this step (typically 1).
        """
        while len(self._buf) >= 2:
            right_id,  right_etype = self._buf[-1]   # right_etype = edge L→R
            left_id,   left_etype  = self._buf[-2]   # left_etype  = edge ?→L
            if right_etype is None:
                break
            comp_id = self.rules_inv.get((left_id, right_etype, right_id))
            if comp_id is None:
                break
            # Merge: discard R, replace L with the composition.
            # The composition inherits L's incoming edge (edge from L's
            # predecessor to the new composition symbol).
            self._buf.pop()
            self._buf[-1] = (comp_id, left_etype)

    def _rewrite_buf(self, left: int, etype: int, right: int, comp_id: int) -> None:
        """Retroactively replace (left →[etype]→ right) pairs in the buffer.

        Scans _buf[0 .. len-2] (up to but not including the current tail,
        which is the left constituent of the composition we just created).
        For each position i where buf[i] = (left, ...) and
        buf[i+1] = (right, etype), replace with (comp_id, buf[i][1]) and
        remove buf[i+1].

        After rewriting, runs _compress_buf_tail() to propagate any new
        compositions that the rewrite may have enabled.
        """
        n = len(self._buf) - 1   # exclude current tail (= left constituent)
        i = 0
        changed = False
        while i < n - 1:         # i+1 must also be before the tail
            if (self._buf[i][0] == left
                    and self._buf[i+1][0] == right
                    and self._buf[i+1][1] == etype):
                self._buf[i] = (comp_id, self._buf[i][1])
                self._buf.pop(i + 1)
                n -= 1
                changed = True
                # don't advance i — the new comp_id at position i might
                # itself form a composition with the next entry
            else:
                i += 1
        if changed:
            # Re-compress the entire buffer to propagate cascading merges.
            # We run _compress_buf_tail() only on the inner portion (stop
            # before the last entry, which is the current left constituent).
            # A full-buffer recompress is too expensive; local sweeps suffice.
            self._compress_buf_tail()

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_rdigram(self, P: int, etype: int, S: int) -> None:
        """Remove all pair-table entries whose right sub-pair is (P, etype, S).

        Called automatically from _create_composition().  Once Composition
        C = (P →[etype]→ S) exists, _compress_buf_tail() absorbs (P, S) into C
        before the pair-check step runs — so these pairs can never trigger a
        segment boundary again.  Removing them is provably correct (not an
        approximation) and frees memory in O(degree_rdigram).
        """
        rdig = (P, etype, S)
        keys = self._pairs_rdigram.pop(rdig, None)
        if not keys:
            return
        for k in keys:
            self.pairs.pop(k, None)
            self._pair_born.pop(k, None)
        self._n_pruned += len(keys)

    def prune(self, max_singleton_age: int = 500) -> int:
        """Prune stale singleton pairs (count=1, not seen for max_singleton_age boundaries).

        A pair with count=1 that has not been incremented in max_singleton_age
        boundaries is a "stale singleton".  Under a geometric recurrence model,
        P(recur) ≈ 1/(age+2).  For age >= 100 this is < 1%; the expected MDL
        benefit of retaining the entry is negligible compared to its storage cost.

        Side-effect: if a pruned triple does eventually recur it triggers one
        false boundary (the triple appears novel again).  This is an acceptable
        approximation with no impact on composition creation or prediction.

        Call at document boundaries or when memory pressure is detected.
        Returns the number of entries removed.
        """
        cutoff = self._n_boundaries - max_singleton_age
        dead = [k for k, cnt in self.pairs.items()
                if cnt == 1 and self._pair_born.get(k, 0) <= cutoff]
        for k in dead:
            del self.pairs[k]
            self._pair_born.pop(k, None)
            rdig = (k[2], k[3], k[4])
            s = self._pairs_rdigram.get(rdig)
            if s is not None:
                s.discard(k)
                if not s:
                    del self._pairs_rdigram[rdig]
        self._n_pruned += len(dead)
        return len(dead)

    # ── Segment boundary ──────────────────────────────────────────────────────

    def _emit_chunk(self, chunk: list[tuple[int, Optional[int]]]) -> None:
        """Emit a completed chunk to all registered callbacks."""
        self._n_boundaries += 1
        for cb in self._callbacks:
            cb(chunk, self)

    def on_segment(self, callback: Callable) -> None:
        """Register a callback: callback(chunk, graph) called on each boundary.

        chunk is a list of (symbol_id, incoming_edge_int | None).
        Use graph.symbols[sid] to look up symbol objects.
        """
        self._callbacks.append(callback)

    def on_observe(self, hook: Callable) -> None:
        """Register a pre-observation hook: hook(ctx_id, etype, tgt_id, graph).

        Called BEFORE any model update, so the hook sees the PRIOR distribution.
        This is the correct moment for computing prediction_error / free-energy.

        ctx_id  : int | None  — symbol_id of the previous token; None = first token
        etype   : int | None  — edge type from ctx to tgt; None = first token
        tgt_id  : int         — symbol_id of the current observation
        graph   : MorphismGraph — the graph (prior state, not yet updated)

        Hook must be side-effect-free w.r.t. the graph (read-only access only).
        """
        self._observe_hooks.append(hook)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_dist(self, src_id: int, etype: int) -> dict[int, float]:
        """Return P(next | src_id, etype) as a dict {tgt_id: probability}.

        Fast path: normalised edge counts from the output index.
        Returns an empty dict if src_id has never been observed as a source
        for edge type etype (callers should apply a back-off in that case).

        Hopf coproduct smoothing (BLUEPRINT §predict() step 4):
        When src_id is a Composition C = (left →[e]→ right), the direct
        distribution is blended with the right constituent's distribution:

            result = (1 - w) * direct + w * right_constituent

        where w = 1 / (1 + direct_count) decreases as more direct edges
        are observed from C.  This encodes the Hopf algebra identity
            Δ(C) = C ⊗ 1 + 1 ⊗ right
        as a convex smoothing: trust the composition's own history more as
        it accumulates evidence, but never discard the constituent prior.

        Only applies to Compositions; Atoms always return the raw distribution.
        Right constituent is read directly from _out (no recursion).
        """
        out_src = self._out.get(src_id)
        if out_src is None:
            return {}
        out_etype = out_src.get(etype)
        if not out_etype:
            return {}
        total = sum(out_etype.values())
        if total == 0:
            return {}
        direct = {tgt: cnt / total for tgt, cnt in out_etype.items()}

        # Hopf smoothing: only for Compositions (Atoms have no rule entry)
        rule = self.rules.get(src_id)
        if rule is None:
            return direct   # Atom: no smoothing

        _left, _e, right = rule

        # Read right-constituent distribution directly from _out (no recursion)
        right_emap = (self._out.get(right) or {}).get(etype) or {}
        if not right_emap:
            return direct   # Constituent has no data: no smoothing
        right_total = sum(right_emap.values())
        if right_total == 0:
            return direct

        constituent = {tgt: cnt / right_total for tgt, cnt in right_emap.items()}

        # Convex blend: w decreases as direct evidence accumulates
        w = 1.0 / (1.0 + total)
        all_keys = set(direct) | set(constituent)
        return {
            k: (1.0 - w) * direct.get(k, 0.0) + w * constituent.get(k, 0.0)
            for k in all_keys
        }

    def predict_dist_by_value(
        self, value: str, etype: int
    ) -> dict[str, float]:
        """Like predict_dist but returns {value_string: probability}."""
        sid = self.atoms.get(value)
        if sid is None:
            return {}
        raw = self.predict_dist(sid, etype)
        result: dict[str, float] = {}
        for tgt_id, prob in raw.items():
            sym = self.symbols[tgt_id]
            if isinstance(sym, Atom):
                result[sym.value] = prob
        return result

    # ── Decomposition (coproduct Δ) ───────────────────────────────────────────

    def decompose(self, sid: int) -> Optional[tuple[int, int, int]]:
        """Return (left_id, etype, right_id) for a Composition, None for Atoms."""
        return self.rules.get(sid)

    def generate(
        self, goal_id: int, target_level: int = 0
    ) -> list[int]:
        """Top-down expansion via coproduct Δ to target_level atoms.

        See BLUEPRINT.md §"generate() — top-down expansion".
        Returns a list of symbol IDs at target_level (or as low as possible).
        """
        sym = self.symbols[goal_id]
        if sym.level <= target_level:
            return [goal_id]
        rule = self.decompose(goal_id)
        if rule is None:
            return [goal_id]
        left_id, _etype, right_id = rule
        return self.generate(left_id, target_level) + self.generate(right_id, target_level)

    # ── Sense disambiguation ──────────────────────────────────────────────────

    def split_atom(
        self,
        atom_id: int,
        sense_a_etypes: set[int],
        sense_b_etypes: set[int],
    ) -> int:
        """Split atom_id into two atoms based on an edge-type partition.

        Creates a new Atom B (value = original_value + "__s2") and
        redistributes all edges that involve atom_id via sense_b_etypes:

          For each edge (src, e, tgt) where e ∈ sense_b_etypes:
            - If src == atom_id: new edge src = atom_B_id
            - If tgt == atom_id: new edge tgt = atom_B_id
            - If both (self-loop): both endpoints become atom_B_id

        Composition rules in self.rules that reference atom_id via a
        sense_b_etype are updated in-place to use atom_B_id.

        The pairs/digrams tables are NOT updated; they are approximations
        that will naturally self-correct as new observations arrive.
        The active buffer (_buf) is not updated; split_atom is called at
        segment boundaries, not mid-sequence.

        Returns atom_B_id.
        """
        sym = self.symbols[atom_id]
        if not isinstance(sym, Atom):
            raise ValueError(f"split_atom: symbol {atom_id} is not an Atom")

        # 1. Create new Atom B
        new_value = sym.value + "__s2"
        atom_B_id = len(self.symbols)
        atom_B    = Atom(sid=atom_B_id, level=0, value=new_value)
        self.symbols.append(atom_B)
        self.atoms[new_value] = atom_B_id

        # 2. Redistribute edges involving atom_id via sense_b_etypes.
        #    Build the move list BEFORE any modification to avoid dict-mutation.
        to_move: list[tuple[int, int, int, int, int, int]] = []
        for (src, e, tgt), cnt in self.edges.items():
            if e not in sense_b_etypes:
                continue
            if src != atom_id and tgt != atom_id:
                continue
            new_src = atom_B_id if src == atom_id else src
            new_tgt = atom_B_id if tgt == atom_id else tgt
            to_move.append((src, e, tgt, cnt, new_src, new_tgt))

        for src, e, tgt, cnt, new_src, new_tgt in to_move:
            # Update edges dict
            del self.edges[(src, e, tgt)]
            self.edges[(new_src, e, new_tgt)] = cnt

            # Update _out: remove OLD entry, add NEW entry
            src_map = self._out.get(src)
            if src_map is not None:
                e_map = src_map.get(e)
                if e_map is not None:
                    e_map.pop(tgt, None)
                    if not e_map:
                        del src_map[e]
                    if not src_map:
                        del self._out[src]

            if new_src not in self._out:
                self._out[new_src] = {}
            if e not in self._out[new_src]:
                self._out[new_src][e] = {}
            self._out[new_src][e][new_tgt] = cnt

        # 3. Update composition rules that reference atom_id via sense_b_etypes.
        rules_to_update = [
            (comp_id, left, e, right)
            for comp_id, (left, e, right) in self.rules.items()
            if e in sense_b_etypes and (left == atom_id or right == atom_id)
        ]
        for comp_id, left, e, right in rules_to_update:
            old_key  = (left, e, right)
            new_left  = atom_B_id if left  == atom_id else left
            new_right = atom_B_id if right == atom_id else right
            new_key  = (new_left, e, new_right)
            del self.rules[comp_id]
            del self.rules_inv[old_key]
            self.rules[comp_id] = new_key
            self.rules_inv[new_key] = comp_id
            comp_sym = self.symbols[comp_id]
            if isinstance(comp_sym, Composition):
                comp_sym.left  = new_left
                comp_sym.right = new_right

        return atom_B_id

    # ── Inspection ────────────────────────────────────────────────────────────

    def n_symbols(self)      -> int: return len(self.symbols)
    def n_atoms(self)        -> int: return len(self.atoms)
    def n_compositions(self) -> int: return len(self.rules)
    def n_edges(self)        -> int: return len(self.edges)
    def n_pairs(self)        -> int: return len(self.pairs)
    def n_digrams(self)      -> int: return len(self.digrams)
    def n_pruned(self)       -> int: return self._n_pruned

    def edge_count(self, src: int, etype: int, tgt: int) -> int:
        """Return the observed count for edge (src →[etype]→ tgt), or 0 if unseen."""
        return self.edges.get((src, etype, tgt), 0)

    def value_of(self, sid: int) -> str:
        """Return the string value for an Atom, or a compact notation for a Composition."""
        sym = self.symbols[sid]
        if isinstance(sym, Atom):
            return repr(sym.value)
        assert isinstance(sym, Composition)
        return f"C{sid}[{self.value_of(sym.left)}+{self.value_of(sym.right)}]"

    def summary(self) -> str:
        return (
            f"MorphismGraph("
            f"atoms={self.n_atoms()}, "
            f"compositions={self.n_compositions()}, "
            f"edges={self.n_edges()}, "
            f"pairs={self.n_pairs()}, "
            f"pruned={self._n_pruned}, "
            f"observations={self._n_obs}, "
            f"boundaries={self._n_boundaries})"
        )

    def __repr__(self) -> str:
        return self.summary()
