"""
Schema-based left Kan extension for novel operator generalization.

For a novel operator e (one not seen in the RelationStore), the left Kan
extension computes:

    (Lan_K F)(e) = colim_{K(c) → e} F(c)

where:
  - K maps known ops to their input schemas (arity + positional role names)
  - F maps known ops to their RelationRule predictions
  - The colimit over the comma category (K ↓ e) is a weighted mixture of
    all known-op predictions where K(c) → e is a schema morphism

A schema morphism K(c) → e exists when op c has the same arity as e (same
number of positional input roles p0, p1, ...).  The morphism is the identity
on roles: both ops use p0, p1, ... and the mapping is just the identity.

The colimit is computed as a weighted sum of F(c) over all compatible c,
where each weight is evidence(c) / total_evidence.  This is the canonical
discrete colimit formula and satisfies the universal property of the left
Kan extension in the finite-set enriched setting.

Phase XVI replaces the JSD centroid heuristic (which violated the universal
property and depended on the removed ConceptLattice) with this categorical
colimit computation.  See FIXING_GENERALIZATION_PART2.md §Phase XVI and
CT_REFERENCE.md §6.
"""

from __future__ import annotations

import math
from typing import Optional


class KanExtension:
    """Genuine left Kan extension over the op schema category.

    Computes the colimit (Lan_K F)(e) for novel ops e not seen in training.

    For a known op, returns its own RelationRule predictions directly (exact
    match — no extension needed).  For an unknown op with the same arity as
    some known op, the Kan extension colimit transfers the known ops'
    predictions to the novel op.

    Parameters
    ----------
    store:
        RelationStore instance (duck-typed).  Must support `.get_schema(op)`.
    rules_by_op:
        dict[str, list[RelationRule]] — relation rules per op.
    engine:
        ComposeEngine (duck-typed).  Passed to RelationRule.evaluate().
    """

    def __init__(self, store, rules_by_op: dict, engine) -> None:
        self._store = store
        self._rules_by_op = rules_by_op
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, op: str, input_tokens: list[str]) -> dict[str, float]:
        """Return P(output_token | op, input_tokens) via Kan extension.

        Parameters
        ----------
        op:
            Operator name (may be novel — not in rules_by_op).
        input_tokens:
            Ordered input token values.  len(input_tokens) is the arity.

        Returns
        -------
        Mapping output_token → probability.  Empty dict on complete miss.
        """
        # Direct lookup for known ops (exact match, no extension needed)
        own_rules = self._rules_by_op.get(op)
        if own_rules:
            return self._apply_rules_for_op(own_rules, op, input_tokens)

        # Kan extension colimit: find all arity-compatible known ops
        arity = len(input_tokens)
        compatible: list[tuple[str, list]] = []
        for known_op, rules in self._rules_by_op.items():
            schema = self._store.get_schema(known_op)
            if schema is not None and len(schema) == arity:
                compatible.append((known_op, rules))

        if not compatible:
            return {}

        # Total evidence across compatible ops (denominator of colimit weight)
        total_ev: float = sum(
            sum(getattr(r, 'evidence', 1) for r in rules) or 1
            for _, rules in compatible
        )
        if total_ev <= 0.0:
            total_ev = float(len(compatible))

        # Weighted mixture — the discrete colimit formula
        result: dict[str, float] = {}
        for known_op, rules in compatible:
            ev = float(sum(getattr(r, 'evidence', 1) for r in rules) or 1)
            w = ev / total_ev
            pred = self._apply_rules_for_op(rules, known_op, input_tokens)
            for tok, prob in pred.items():
                result[tok] = result.get(tok, 0.0) + w * prob

        if not result:
            return {}
        total = sum(result.values())
        if total <= 0.0:
            return {}
        return {k: v / total for k, v in result.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_rules_for_op(
        self,
        rules: list,
        op: str,
        input_tokens: list[str],
    ) -> dict[str, float]:
        """Evaluate rules against positionally-bound input_tokens."""
        schema = self._store.get_schema(op)
        if schema is None or len(schema) != len(input_tokens):
            return {}

        # Build role → token binding from schema and input_tokens
        role_values: dict[str, str] = {
            role: tok for (_, role), tok in zip(schema, input_tokens)
        }

        result: dict[str, float] = {}
        for rule in rules:
            pred = rule.evaluate(role_values, self._engine)
            for tok, conf in pred.items():
                # Take max confidence across rules for the same output token
                if conf > result.get(tok, 0.0):
                    result[tok] = conf

        if not result:
            return {}
        total = sum(result.values())
        if total <= 0.0:
            return {}
        return {k: v / total for k, v in result.items()}


# ---------------------------------------------------------------------------
# Neighbourhood hash utilities (retained — tested by test_stage3.py)
# ---------------------------------------------------------------------------

def _left_components(neighbourhood_hash: str) -> list[str]:
    """Parse the left (negative-offset) components from a neighbourhood hash.

    Neighbourhood hashes have the format:
        'r{r}|{offset},{atom}|{offset},{atom}|...'

    This function returns only the components whose offset is negative (i.e.
    the left-context positions).  Used in tests and diagnostics.

    Parameters
    ----------
    neighbourhood_hash:
        A WL-style neighbourhood key as produced by HankelCount._neighbourhood_key.

    Returns
    -------
    List of '{offset},{atom}' strings with negative offset, in ascending
    offset order.  Empty list if no negative-offset components exist.

    Example
    -------
    >>> _left_components('r1|-1,eq|1,<pad>')
    ['-1,eq']
    """
    if '|' not in neighbourhood_hash:
        return []
    # Drop the leading 'r{r}' prefix
    parts = neighbourhood_hash.split('|')[1:]
    return [p for p in parts if p and p.startswith('-')]


# ---------------------------------------------------------------------------
# JSD / KL utilities (retained for diagnostics)
# ---------------------------------------------------------------------------

def _jsd(p, q) -> float:
    """Jensen-Shannon Divergence between two probability vectors (base 2).

    Returns a value in [0, 1] bits.
    """
    import numpy as np
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum < 1e-12 or q_sum < 1e-12:
        return 1.0
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p, q) -> float:
    """KL(p||q) in bits.  Terms where p=0 contribute 0."""
    import numpy as np
    mask = p > 0
    if not np.any(mask):
        return 0.0
    p_m = p[mask]
    q_m = q[mask]
    ratio = np.where(q_m > 0, p_m / q_m, 0.0)
    safe_ratio = np.where(ratio > 0, ratio, 1.0)
    kl = np.sum(p_m * np.log2(safe_ratio))
    return float(np.clip(kl, 0.0, None))
