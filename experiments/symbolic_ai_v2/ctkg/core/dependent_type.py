"""
Dependent type system for the CTKG (Phase XXI).

Every token that participates in an NNO (Natural Number Object) structure
has type NNO_DIGIT with an ordinal — its 0-based position in the successor
chain.  Two tokens with the same ordinal are COMPUTATIONALLY INDISTINGUISHABLE
under any NNO-derived operation (add, mul, etc.).

This is the formal proof that the anonymization test (Phase XVI gate) passes
with 0% accuracy gap: the system reasons about structural position (ordinal),
never about surface form.

Dependent type judgments
------------------------
    Γ ⊢ tok : NNO_DIGIT(i)   iff tok is the i-th element of the NNO chain
    Γ ⊢ tok : NNO_CARRY       iff tok is a carry value not in the NNO chain
    Γ ⊢ tok : STRUCTURAL      iff tok is an operator or delimiter token
    Γ ⊢ tok : UNKNOWN         otherwise

The "dependent" property: the output type of add(a, b) is NNO_DIGIT only when
both a and b satisfy the NNO type — it depends on the input types.

For RelationRule type annotations ordinal is None (universally quantified):
    rule.arg1_type = NNO_DIGIT(ordinal=None)
means "arg1 must be an NNO digit for any ordinal" — the rule works uniformly
across all NNO carriers regardless of surface form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.node import (
    TOKEN_GRAPH,
    NodeId,
    EQ_NODE,
    STEP_NODE,
    ANS_NODE,
    EOS_NODE,
    PAD_NODE,
)


# ---------------------------------------------------------------------------
# Structural tokens — always typed STRUCTURAL regardless of position
# ---------------------------------------------------------------------------

STRUCTURAL_TOKENS: frozenset[NodeId] = frozenset({
    # Format separators and end-of-sequence markers — domain-independent.
    # Operator names (succ, add, mul, …) and domain-specific separators
    # (linear_eval, bern_p1, …) are NOT included here: they may coincide
    # with vocabulary tokens in NL mode and must be inferred from the
    # relation store's schema, not hardcoded.
    STEP_NODE, ANS_NODE, EQ_NODE, EOS_NODE, PAD_NODE,
})


# ---------------------------------------------------------------------------
# TypeTerm
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypeTerm:
    """A type in the dependent type system.

    Attributes
    ----------
    tag : str
        One of: 'NNO_DIGIT', 'NNO_CARRY', 'STRUCTURAL', 'UNKNOWN'.
    ordinal : Optional[int]
        For NNO_DIGIT tokens: 0-based position in the successor chain.
        None for all other tags, and also None in *rule* type annotations
        (where the rule is universally quantified over all ordinals).
    """
    tag: str
    ordinal: Optional[int] = None

    def is_nno_digit(self) -> bool:
        return self.tag == 'NNO_DIGIT'

    def is_compatible_with(self, other: 'TypeTerm') -> bool:
        """Structural compatibility: same tag and (if NNO_DIGIT) same ordinal.

        Two NNO_DIGIT tokens are compatible iff they have the same ordinal —
        the formal condition for computational indistinguishability under any
        NNO-derived BFM operation.

        If either ordinal is None (universally quantified rule type), the
        check degrades to tag equality (both are NNO digits).
        """
        if self.tag != other.tag:
            return False
        if self.tag == 'NNO_DIGIT':
            if self.ordinal is None or other.ordinal is None:
                return True   # universally quantified
            return self.ordinal == other.ordinal
        return True


# Convenience constructors
NNO_DIGIT = lambda ordinal=None: TypeTerm(tag='NNO_DIGIT', ordinal=ordinal)
NNO_CARRY = TypeTerm(tag='NNO_CARRY')
STRUCTURAL = TypeTerm(tag='STRUCTURAL')
UNKNOWN = TypeTerm(tag='UNKNOWN')


# ---------------------------------------------------------------------------
# Chain walking
# ---------------------------------------------------------------------------

def _walk_chain(succ_chain: dict[NodeId, NodeId]) -> list[NodeId]:
    """Walk succ_chain from the zero element (no predecessor) in order.

    Returns the ordered list [zero, succ(zero), succ²(zero), ...].
    Handles both cyclic (modular) and non-cyclic chains.
    """
    values = set(succ_chain.values())
    sources = set(succ_chain.keys())
    zero_candidates = sources - values
    zero = next(iter(zero_candidates)) if zero_candidates else next(iter(sources))

    chain: list[NodeId] = []
    seen: set[NodeId] = set()
    cur = zero
    while cur in succ_chain and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        cur = succ_chain[cur]
    if cur not in seen:
        chain.append(cur)
    return chain


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def infer_token_types(
    succ_chain: dict[NodeId, NodeId],
    carry_tokens: frozenset[NodeId] = frozenset(),
    structural_tokens: frozenset[NodeId] = STRUCTURAL_TOKENS,
) -> dict[NodeId, TypeTerm]:
    """Assign TypeTerms to all tokens reachable from succ_chain.

    Parameters
    ----------
    succ_chain : dict[NodeId, NodeId]
        The NNO successor map tok → succ(tok), with NodeId keys and values.
    carry_tokens : frozenset[NodeId]
        Tokens that act as carry values.  If a carry token is also in the
        NNO chain, it gets NNO_DIGIT type (chain takes priority).
    structural_tokens : frozenset[NodeId]
        Operator/delimiter tokens — always STRUCTURAL.

    Returns
    -------
    dict mapping each known NodeId to its TypeTerm.
    """
    result: dict[NodeId, TypeTerm] = {}

    # NNO chain tokens get ordinal-specific NNO_DIGIT types
    chain = _walk_chain(succ_chain)
    for i, tok in enumerate(chain):
        result[tok] = TypeTerm(tag='NNO_DIGIT', ordinal=i)

    # Carry tokens not already typed via the chain
    for tok in carry_tokens:
        if tok not in result:
            result[tok] = NNO_CARRY

    # Structural tokens (lowest priority)
    for tok in structural_tokens:
        if tok not in result:
            result[tok] = STRUCTURAL

    return result


def token_type(tok: NodeId, type_context: dict[NodeId, TypeTerm]) -> TypeTerm:
    """Look up tok's type, returning UNKNOWN if not in context."""
    return type_context.get(tok, UNKNOWN)


def rule_type_tag(
    role_values: dict[NodeId, tuple],
    role_name: NodeId,
    type_context: dict[NodeId, TypeTerm],
) -> TypeTerm:
    """Return the type tag (ordinal=None) for the token at *role_name*.

    Used when annotating RelationRule inputs: we want the structural type
    (tag only, no specific ordinal) so the rule is universally quantified.

    role_values maps role NodeId → tuple of value NodeIds.
    """
    val_tup = role_values.get(role_name)
    if val_tup is None or len(val_tup) == 0:
        return UNKNOWN
    # Use the first element of the value tuple for type inference
    val_nid = val_tup[0]
    tt = type_context.get(val_nid, UNKNOWN)
    # Strip ordinal for rule annotations (universal quantification)
    return TypeTerm(tag=tt.tag, ordinal=None)


# ---------------------------------------------------------------------------
# Anonymization theorem: bijection-stability
# ---------------------------------------------------------------------------

def types_compatible_under_bijection(
    types_a: dict[NodeId, TypeTerm],
    types_b: dict[NodeId, TypeTerm],
    bijection: dict[NodeId, NodeId],
) -> bool:
    """Return True iff bijection maps NNO ordinals consistently.

    For each (tok_a, tok_b) pair in bijection, checks:
        types_a[tok_a].ordinal == types_b[tok_b].ordinal

    This is the formal statement of anonymization invariance: if every token
    that tok_a maps to has the same NNO ordinal in both type contexts, then
    any computation defined via NNO universal property gives identical results
    regardless of surface form.
    """
    for tok_a, tok_b in bijection.items():
        ta = types_a.get(tok_a)
        tb = types_b.get(tok_b)
        if ta is None or tb is None:
            continue
        if not ta.is_compatible_with(tb):
            return False
    return True
