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
    value: str   # the raw observation string


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

    def __init__(self) -> None:
        # Symbol table — list-indexed by ID for O(1) access
        self.symbols:   list[Symbol]            = []
        self.atoms:     dict[str, int]          = {}   # value → atom ID

        # Edge and pair counts — tuple keys per DATA_FORMATS.md
        self.edges:     dict[tuple, int]        = {}   # (src, etype, tgt) → count
        self.pairs:     dict[tuple, int]        = {}   # (Q,e1,P,e2,S) → count

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

        # Running counters
        self._n_obs:        int = 0   # total observations
        self._n_boundaries: int = 0   # total segment boundaries emitted

        # Composition-level edge recording.
        # After observe() detects that the right sub-pair (P →[e2]→ S) matches a
        # known composition C, this is set to C's symbol ID.  The NEXT observe()
        # call records edge (C →[incoming_etype]→ next_token) into edges / _out,
        # enabling predict_dist(C_id, etype) for multi-level (trigram+) prediction.
        self._pending_comp_ctx: Optional[int] = None

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

        # ── Check pair (Q →[e1]→ P →[e2]→ S) ───────────────────────────────
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
                # First time this triple is seen → segment boundary
                is_boundary = True
            elif old_count == 1:
                # Second time → create composition for (P →[e2]→ S)
                self._create_composition(P, e2, S)

            # If the right sub-pair (P →[e2]→ S) is a known composition,
            # mark it so the NEXT observe() records what follows it.
            # This fires on every occurrence (count ≥ 2), not only on creation.
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
        self.rules[sid]        = rule_key
        self.rules_inv[rule_key] = sid
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

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_dist(self, src_id: int, etype: int) -> dict[int, float]:
        """Return P(next | src_id, etype) as a dict {tgt_id: probability}.

        Fast path: normalised edge counts from the output index.
        Returns an empty dict if src_id has never been observed as a source
        for edge type etype (callers should apply a back-off in that case).
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
        return {tgt: cnt / total for tgt, cnt in out_etype.items()}

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

    # ── Inspection ────────────────────────────────────────────────────────────

    def n_symbols(self)      -> int: return len(self.symbols)
    def n_atoms(self)        -> int: return len(self.atoms)
    def n_compositions(self) -> int: return len(self.rules)
    def n_edges(self)        -> int: return len(self.edges)
    def n_pairs(self)        -> int: return len(self.pairs)

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
            f"observations={self._n_obs}, "
            f"boundaries={self._n_boundaries})"
        )

    def __repr__(self) -> str:
        return self.summary()
