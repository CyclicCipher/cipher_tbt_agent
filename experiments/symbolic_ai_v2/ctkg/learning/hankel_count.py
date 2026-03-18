"""
Phase 1: real-valued co-occurrence matrix H[neighbourhood_hash, atom] = P(atom | neighbourhood).

HankelCount builds the co-occurrence matrix H incrementally from a stream of path-graph
sequences (lists of atom values).  Each entry H[context_key, atom] = P(atom | context)
is the conditional probability of observing `atom` at a position given that position's
r-hop neighbourhood pattern.

Neighbourhood canonicalisation (WL hash for path graphs):
    For atom at position i with radius r, the key is:
        'r{r}|{offset1},{atom1}|{offset2},{atom2}|...'
    where the pairs are all (signed_offset, atom) for j = i-r..i+r, j ≠ i,
    in ascending offset order.  Out-of-bounds positions use the token '<pad>'.

For path graphs this is equivalent to the standard (2r)-gram context window with
position information, and degenerates to the bigram left/right context at r=1.

-- Phase XXIII (Sprint A): AtomValue = NodeId (int) --

All atom values are now stored internally as opaque NodeId integers, encoded via
TOKEN_GRAPH at the update() boundary.  The public API is BACKWARD-COMPATIBLE:
    - update() / update_batch() still accept string sequences (encoded at boundary)
    - vocabulary() still returns list[str] (decoded from NodeId at read boundary)
    - get_distribution() still returns dict[str, float] (decoded at read boundary)
    - ContextKey (WL hash) stays as str — it is a derived structural key, not an
      atom identity.

This satisfies the Iron Law: no string identity stored above the character level.
Context keys embed decoded token strings only for human-readability of the hash;
all arithmetic (counting, distribution normalisation) operates on NodeId values.

See CTKG_ARCHITECTURE.md §Phase 1 for the full specification.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from experiments.symbolic_ai_v2.ctkg.core.node import NodeId, TOKEN_GRAPH


# Public type aliases (NodeId for atoms; ContextKey remains a derived str hash)
AtomValue = NodeId
ContextKey = str


@dataclass
class Context:
    """Typed representation of a single co-occurrence entry in the Hankel matrix.

    This is the structured form of a ContextKey string.  The canonical
    serialised form (used as dict key throughout) is produced by
    HankelCount._neighbourhood_key(), e.g. ``'r1|-1,succ|1,eq'``.

    Parameters
    ----------
    context_id:
        The canonical ContextKey string (WL-hash of the neighbourhood).
    position:
        Signed offset of the centre atom from an arbitrary reference point
        within its sequence.  Negative = left of centre; positive = right.
        When the Context is built from a specific token index i, position = i.
    atom:
        The atom value at the centre of this neighbourhood (NodeId).
    radius:
        The neighbourhood radius r used to compute the context.
    count:
        Accumulated raw co-occurrence count for this (context, atom) entry.
    """

    context_id: ContextKey
    position: int
    atom: AtomValue      # NodeId since Sprint A
    radius: int
    count: float = 0.0

    def __repr__(self) -> str:
        decoded = TOKEN_GRAPH.decode(self.atom) if isinstance(self.atom, int) else self.atom
        return (
            f"Context(r={self.radius}, pos={self.position}, "
            f"atom={decoded!r}, id={self.context_id!r}, count={self.count:.1f})"
        )


class HankelCount:
    """Sparse co-occurrence matrix H[neighbourhood_hash, atom] = P(atom | neighbourhood).

    Parameters
    ----------
    r_max:
        Maximum neighbourhood radius.  Neighbourhoods are collected for r = 1..r_max.
        A separate row is created for each (r, neighbourhood_pattern) pair.
        Default 3 (character/token/phrase levels).
    """

    def __init__(self, r_max: int = 3) -> None:
        self.r_max = r_max
        # _counts[context_key][atom_node_id] = raw integer count (bidirectional)
        # Sprint A: atom keys are NodeId (int), not str
        self._counts: dict[ContextKey, dict[AtomValue, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # _left_counts[left_key][atom_node_id] = raw count (left context only)
        self._left_counts: dict[ContextKey, dict[AtomValue, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._vocab: set[AtomValue] = set()   # set of NodeId
        self._n_sequences: int = 0

    # ------------------------------------------------------------------
    # Public API — writing
    # ------------------------------------------------------------------

    def update(self, sequence: Sequence[str]) -> None:
        """Process one path-graph sequence and update the co-occurrence counts.

        Parameters
        ----------
        sequence:
            An ordered list of atom strings, e.g.
            ['succ', '2', '3', 'eq', '2', '4'].
            Strings are encoded to NodeId at the boundary; all internal
            arithmetic uses NodeId.
        """
        seq_str = list(sequence)
        n = len(seq_str)
        self._n_sequences += 1

        # Encode strings → NodeIds for atom storage (Sprint A)
        seq_ids: list[NodeId] = [TOKEN_GRAPH.encode(tok) for tok in seq_str]

        for i, atom_id in enumerate(seq_ids):
            self._vocab.add(atom_id)
            for r in range(1, self.r_max + 1):
                # Context key uses decoded strings (human-readable WL hash)
                key = self._neighbourhood_key(seq_str, i, r)
                self._counts[key][atom_id] += 1
                lkey = self._left_key(seq_str, i, r)
                self._left_counts[lkey][atom_id] += 1

    def update_batch(self, sequences: Iterable[Sequence[str]]) -> None:
        """Convenience wrapper: call update() on each sequence in the iterable."""
        for seq in sequences:
            self.update(seq)

    # ------------------------------------------------------------------
    # Public API — reading
    # ------------------------------------------------------------------

    def get_distribution(self, neighbourhood_hash: ContextKey) -> dict[str, float]:
        """Row-normalised conditional distribution P(atom | neighbourhood_hash).

        Returns an empty dict if the neighbourhood has never been observed.

        Sprint A: internally keyed by NodeId; returned dict uses decoded strings
        for backward compatibility with all callers.
        """
        raw = self._counts.get(neighbourhood_hash, {})
        total = sum(raw.values())
        if total == 0:
            return {}
        return {
            TOKEN_GRAPH.decode(atom_id): cnt / total
            for atom_id, cnt in raw.items()
        }

    def get_left_distribution(self, left_key: ContextKey) -> dict[str, float]:
        """P(atom | left context only) — for autoregressive prediction.

        Uses the left-only index built during update().  Returns an empty dict
        if the left context has never been observed.

        Sprint A: internally keyed by NodeId; returned dict uses decoded strings.
        """
        raw = self._left_counts.get(left_key, {})
        total = sum(raw.values())
        if total == 0:
            return {}
        return {
            TOKEN_GRAPH.decode(atom_id): cnt / total
            for atom_id, cnt in raw.items()
        }

    def all_contexts(self) -> list[ContextKey]:
        """All observed neighbourhood hashes (across all radii)."""
        return list(self._counts.keys())

    def contexts_at_radius(self, r: int) -> list[ContextKey]:
        """All observed neighbourhood hashes for a specific radius."""
        prefix = f"r{r}|"
        return [k for k in self._counts if k.startswith(prefix)]

    def vocabulary(self) -> list[str]:
        """All atom values observed at least once, sorted.

        Sprint A: stored as NodeId internally; decoded to str at the boundary.
        """
        return sorted(TOKEN_GRAPH.decode(nid) for nid in self._vocab)

    def matrix(
        self, r: int | None = None
    ) -> tuple[list[ContextKey], list[str], np.ndarray]:
        """Return the full H matrix as a dense numpy array.

        Parameters
        ----------
        r:
            If given, restrict to contexts at that radius.
            If None, include all radii.

        Returns
        -------
        contexts:
            List of context keys (row labels).
        atoms:
            List of atom strings (column labels), sorted.
        H:
            Dense float64 array of shape (len(contexts), len(atoms)) with
            row-normalised probabilities.  Rows that sum to zero (shouldn't
            happen, but defensive) are left as zeros.
        """
        if r is not None:
            contexts = self.contexts_at_radius(r)
        else:
            contexts = self.all_contexts()

        atoms = self.vocabulary()  # list[str], sorted
        atom_idx = {a: j for j, a in enumerate(atoms)}
        H = np.zeros((len(contexts), len(atoms)), dtype=np.float64)

        for i, ctx in enumerate(contexts):
            raw = self._counts[ctx]
            total = sum(raw.values())
            if total == 0:
                continue
            for atom_id, cnt in raw.items():
                atom_str = TOKEN_GRAPH.decode(atom_id)
                j = atom_idx.get(atom_str)
                if j is not None:
                    H[i, j] = cnt / total

        return contexts, atoms, H

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the matrix dimensions and sparsity."""
        n_contexts = len(self._counts)
        n_atoms = len(self._vocab)
        total_nonzero = sum(len(v) for v in self._counts.values())
        total_cells = n_contexts * n_atoms if n_atoms else 0
        sparsity = (
            1.0 - total_nonzero / total_cells if total_cells > 0 else float("nan")
        )
        lines = [
            f"HankelCount(r_max={self.r_max})",
            f"  sequences processed : {self._n_sequences}",
            f"  distinct contexts   : {n_contexts}",
            f"  vocabulary size     : {n_atoms}",
            f"  non-zero entries    : {total_nonzero}",
            f"  sparsity            : {sparsity:.1%}",
        ]
        # per-radius breakdown
        for r in range(1, self.r_max + 1):
            n_r = len(self.contexts_at_radius(r))
            lines.append(f"  r={r} contexts         : {n_r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbourhood_key(seq: list[str], i: int, r: int) -> ContextKey:
        """Canonical WL hash for a path-graph neighbourhood.

        For position i with radius r, collects (offset, atom) for all positions
        j = i-r..i+r excluding j=i.  Out-of-bounds positions yield '<pad>'.
        The key format is:

            'r{r}|{offset},{atom}|{offset},{atom}|...'

        with pairs sorted by ascending offset.

        Sprint A note: the key uses decoded string tokens for human-readability
        of the WL hash.  Atom IDENTITY is stored as NodeId in _counts (not in
        the key string).
        """
        n = len(seq)
        parts: list[str] = []
        for offset in range(-r, r + 1):
            if offset == 0:
                continue
            j = i + offset
            atom = seq[j] if 0 <= j < n else "<pad>"
            parts.append(f"{offset},{atom}")
        inner = "|".join(parts)
        return f"r{r}|{inner}"

    @staticmethod
    def _left_key(seq: list[str], i: int, r: int) -> ContextKey:
        """Left-context-only key for autoregressive prediction.

        Identical to _neighbourhood_key but includes only negative offsets
        (positions i-r … i-1).  Right context is not included — this key
        matches during prediction when right tokens are unknown.

        Format:  'r{r}|{-r,atom}|...|{-1,atom}'
        """
        n = len(seq)
        parts: list[str] = []
        for offset in range(-r, 0):
            j = i + offset
            atom = seq[j] if 0 <= j < n else "<pad>"
            parts.append(f"{offset},{atom}")
        return f"r{r}|" + "|".join(parts)
