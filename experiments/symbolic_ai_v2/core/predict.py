"""Prediction: P(next token | context, edge_type).

predict()          — fast path (edge counts) + fallback (marginal), returns
                     a ranked list of (symbol_id, probability) pairs.
predict_by_value() — same but maps IDs back to string values.
perplexity()       — cross-entropy in bits/token on a test corpus.

Prediction strategy (see BLUEPRINT.md §"predict()"):
  1. Fast path: normalised edge counts from output index, O(degree).
  2. Type back-off (fallback): uniform over all observed targets for this
     edge type across all sources.  Applied when context has never been
     seen as a source for this edge type.

The CTKG-type back-off (Kneser-Ney-style smoothing using CTKG type hierarchy)
is a future extension once ctkg_live.py is fully operational.
"""

from __future__ import annotations

import math
from typing import Optional

from .morphism import MorphismGraph
from .topology import Topology


def predict(
    mg: MorphismGraph,
    context_id: int,
    etype: int,
    n_top: int = 10,
) -> list[tuple[int, float]]:
    """Return the top-n predicted next symbols as [(symbol_id, probability)].

    Uses the fast path (edge count distribution) if context_id has been seen.
    Falls back to the marginal distribution over all sources for etype if not.

    n_top = 0 returns the full distribution (may be large).
    """
    dist = mg.predict_dist(context_id, etype)

    if not dist:
        # Back-off: marginal over all sources for this edge type
        dist = _marginal_dist(mg, etype)

    if not dist:
        return []

    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    return ranked if n_top == 0 else ranked[:n_top]


def predict_by_value(
    mg: MorphismGraph,
    value: str,
    etype_name: str,
    topology: Topology,
    n_top: int = 10,
) -> list[tuple[str, float]]:
    """Convenience wrapper: returns [(string_value, probability)].

    Only atom symbols appear in the result (compositions are filtered out
    since they are internal abstractions, not surface observations).
    """
    etype = topology.registry.code(etype_name)
    sid   = mg.atoms.get(value)
    if sid is None:
        return []
    ranked_ids = predict(mg, sid, etype, n_top=0)
    result: list[tuple[str, float]] = []
    for tgt_id, prob in ranked_ids:
        sym = mg.symbols[tgt_id]
        from .morphism import Atom
        if isinstance(sym, Atom):
            result.append((sym.value, prob))
        if n_top > 0 and len(result) >= n_top:
            break
    return result


def perplexity(
    mg: MorphismGraph,
    sequences: list,
    topology: Topology,
) -> float:
    """Compute cross-entropy perplexity in bits/token on a list of sequences.

    Each sequence is passed through topology.stream_tokens().
    The first token of each sequence (edge_type = None) is skipped because
    there is no context from which to predict it.

    Returns bits/token.  Lower is better.  Baseline: log2(vocab_size).
    """
    etype_next = topology.registry.code("next") if "next" in topology.registry.names() else 0

    total_bits  = 0.0
    total_tokens = 0

    for seq in sequences:
        prev_id: Optional[int] = None
        for value, etype in topology.stream_tokens(seq):
            sid = mg.atoms.get(value)
            if sid is None:
                # Unseen atom: back-off to uniform over known atoms
                n_atoms = max(mg.n_atoms(), 1)
                bits = math.log2(n_atoms)
                if prev_id is not None:
                    total_bits  += bits
                    total_tokens += 1
                prev_id = None   # can't use unseen atom as context
                continue

            if prev_id is not None and etype is not None:
                # Predict sid given prev_id via etype
                dist = mg.predict_dist(prev_id, etype)
                if not dist:
                    dist = _marginal_dist(mg, etype)

                p = dist.get(sid, 0.0)
                if p <= 0.0:
                    # Assign a small probability mass for unseen transitions
                    n_tgts = max(len(dist) + 1, 1)
                    p = 1.0 / (n_tgts * 10)   # simple add-one-ish smoothing

                total_bits   += -math.log2(p)
                total_tokens += 1

            prev_id = sid

    if total_tokens == 0:
        return 0.0
    return total_bits / total_tokens


# ── Internal helpers ──────────────────────────────────────────────────────────

def _marginal_dist(mg: MorphismGraph, etype: int) -> dict[int, float]:
    """Marginal distribution P(tgt | etype) summed over all source symbols."""
    counts: dict[int, int] = {}
    for (src, et, tgt), cnt in mg.edges.items():
        if et == etype:
            counts[tgt] = counts.get(tgt, 0) + cnt
    total = sum(counts.values())
    if total == 0:
        return {}
    return {tgt: cnt / total for tgt, cnt in counts.items()}
