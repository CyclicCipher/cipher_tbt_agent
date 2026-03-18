"""
Context category C for sheaf-theoretic prediction dispatch (Phase XIX).

Objects of C are named discourse-state types (ContextId).  A morphism
c' → c in C means "c' is a refinement of c" (c' is more specific).

This forms a partial order:

    ANY
    ├── EQ      (prefix contains the 'eq' delimiter — equation format)
    └── TRACE   (prefix contains 'step' or 'ans' — trace format)
    └── INPUT   (no output delimiter seen yet — still in input phase)

A prediction rule is a *presheaf section* F: C^op → Set registered at
context c.  The restriction map ρ_{c'→c} is implicit: the rule fires
whenever the current context IS-A c (refines c).

For example:
  - Level 0.5 / 0.6 / 0.7 (FC, NNO, compose) are sections at ContextId.EQ:
    they only fire in eq-format sequences.
  - Level 1e (pullback) is a section at ContextId.TRACE (trace-format ops).
  - Level 1d (equalizer) is a section at ContextId.EQ.
  - Level 1b (chain rule) is a section at ContextId.ANY (handles all formats
    internally — the format switch use_eq IS the restriction map application).

This module replaces every bare ``'eq' in prefix`` / ``'step' in prefix``
string test in predict.py with a call to ``ContextCategory.classify()``,
eliminating ad-hoc format detection in favour of a principled context
specialisation hierarchy.

Sheaf condition (Phase XIX gate):
  Locally consistent sections (rules valid in overlapping contexts) are
  required to agree on their shared domain.  Enforced informally: Level 1b
  fires first for chain ops regardless of format; lower levels only fire for
  their registered contexts, so there is no ambiguity.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


# ---------------------------------------------------------------------------
# Context objects
# ---------------------------------------------------------------------------

class ContextId(str, Enum):
    """Named objects of the context category.

    Partial order (refinement / IS-A):

        ANY
        ├── EQ
        ├── TRACE
        └── INPUT
    """
    ANY   = "any"    # ⊤ — most general; no constraint on prefix format
    EQ    = "eq"     # prefix contains 'eq'  (plain equation format)
    TRACE = "trace"  # prefix contains 'step' or 'ans' (trace/Kleisli format)
    INPUT = "input"  # no output delimiter present yet (still reading inputs)


# Morphisms: _PARENTS[c] = direct parents (contexts that c refines).
_PARENTS: dict[ContextId, tuple[ContextId, ...]] = {
    ContextId.ANY:   (),
    ContextId.EQ:    (ContextId.ANY,),
    ContextId.TRACE: (ContextId.ANY,),
    ContextId.INPUT: (ContextId.ANY,),
}

# Memoised ancestor sets
_ANCESTOR_CACHE: dict[ContextId, FrozenSet[ContextId]] = {}


def _ancestors(ctx: ContextId) -> FrozenSet[ContextId]:
    """Return all contexts that *ctx* refines (inclusive)."""
    if ctx in _ANCESTOR_CACHE:
        return _ANCESTOR_CACHE[ctx]
    result: set[ContextId] = {ctx}
    for parent in _PARENTS[ctx]:
        result |= _ancestors(parent)
    frozen = frozenset(result)
    _ANCESTOR_CACHE[ctx] = frozen
    return frozen


# ---------------------------------------------------------------------------
# ContextCategory
# ---------------------------------------------------------------------------

class ContextCategory:
    """The context category C.

    Objects  = ContextId values (named discourse-state types).
    Morphisms c' → c = c' is a refinement of c (c' IS-A c).

    Prediction rules are presheaf sections registered at a context c.
    A rule fires for any current context ctx such that ctx refines c
    (i.e. ctx IS-A c).  The restriction map ρ_{c'→c} is implicit in
    how each rule handles the prefix — the rule at c is automatically
    "restricted" to c' by the fact that c' satisfies all constraints of c.

    Parameters
    ----------
    eq_token : str
        The token that separates input from output in eq-format sequences.
        Default 'eq'; may be an anonymous Unicode symbol.
    step_token : str
        Trace-format step delimiter.  Default 'step'.
    ans_token : str
        Trace-format answer delimiter.  Default 'ans'.

    Usage in predict.py::

        ctx = self._ctx_cat.classify(prefix)

        # Instead of: if 'eq' in prefix:
        if self._ctx_cat.is_refinement(ctx, ContextId.EQ):
            ...

        # The use_eq flag IS the restriction map — it specialises the
        # chain rule section at ANY to the EQ or TRACE sub-context:
        use_eq = self._ctx_cat.is_refinement(ctx, ContextId.EQ)
    """

    def __init__(
        self,
        eq_token: str = "eq",
        step_token: str = "step",
        ans_token: str = "ans",
    ) -> None:
        self._eq_token = eq_token
        self._step_token = step_token
        self._ans_token = ans_token

    def is_refinement(self, ctx: ContextId, of: ContextId) -> bool:
        """Return True iff *ctx* IS-A *of* (ctx refines of).

        Examples
        --------
        >>> cc = ContextCategory()
        >>> cc.is_refinement(ContextId.EQ, ContextId.ANY)
        True
        >>> cc.is_refinement(ContextId.TRACE, ContextId.EQ)
        False
        >>> cc.is_refinement(ContextId.EQ, ContextId.EQ)
        True
        """
        return of in _ancestors(ctx)

    def classify(self, prefix: list[str]) -> ContextId:
        """Map *prefix* to the most specific applicable context object.

        Classification rules (first match wins):

        1. If 'eq'   in prefix → EQ    (equation format; output after 'eq')
        2. If 'step' in prefix → TRACE (trace format; multi-step Kleisli chain)
        3. If 'ans'  in prefix → TRACE (trace format; past the answer delimiter)
        4. Otherwise           → INPUT (still reading input tokens)

        The returned ContextId is the *most specific* context for this prefix.
        Restriction maps are applied implicitly by callers via ``is_refinement``.
        """
        if not prefix:
            return ContextId.INPUT
        if self._eq_token in prefix:
            return ContextId.EQ
        if self._step_token in prefix or self._ans_token in prefix:
            return ContextId.TRACE
        return ContextId.INPUT
