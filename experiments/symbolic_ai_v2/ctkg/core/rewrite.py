"""
Rewrite rules and catamorphic reduction.

A RewriteRule is (lhs: Expr, rhs: Expr) where lhs may contain pattern variables.
cata_reduce applies a list of rules bottom-up until no rule fires (normal form).

This is the single recursive function that replaces all segment-type dispatch.
No special cases.  No hand-coded operator handlers.  One code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.term_algebra import Expr, match, substitute


@dataclass
class RewriteRule:
    """
    A single conditional rewrite rule: lhs → rhs.

    lhs may contain var() pattern nodes.
    rhs uses the same variable names; substitute(rhs, bindings) gives the result.

    algebra_name groups rules belonging to the same catamorphism or natural
    transformation (e.g. 'D_algebra', 'power_rule').

    evidence counts how many training examples support this rule.
    """
    lhs: Expr
    rhs: Expr
    algebra_name: str = ''
    evidence: int = 0

    def applies_to(self, expr: Expr) -> Optional[Expr]:
        """
        Try to apply this rule to expr.

        Returns the reduced expression if the rule fires, else None.
        """
        bindings = match(self.lhs, expr)
        if bindings is None:
            return None
        return substitute(self.rhs, bindings)

    def __repr__(self) -> str:
        name = f'[{self.algebra_name}] ' if self.algebra_name else ''
        return f'{name}{self.lhs} → {self.rhs}  (n={self.evidence})'


# ---------------------------------------------------------------------------
# cata_reduce — bottom-up rule application to normal form
# ---------------------------------------------------------------------------

_MAX_STEPS = 1000   # safety limit to detect infinite loops


def cata_reduce(expr: Expr, rules: list[RewriteRule],
                max_steps: int = _MAX_STEPS) -> Expr:
    """
    Reduce an expression to normal form by applying rules bottom-up.

    Algorithm:
      1. Recursively reduce all children (bottom-up / innermost-first).
      2. Try each rule on the (partially reduced) expression.
         The FIRST rule whose lhs matches fires; apply it and restart from step 1
         on the result (to enable cascading reductions).
      3. When no rule fires, the expression is in normal form.

    Rules are tried in the order given.  The caller is responsible for ordering
    (e.g. most-specific first, as in the SlotProgram specificity sort).

    max_steps is a safety limit against non-terminating rule sets.

    Returns the normal form (which equals expr if no rule ever fired).
    """
    steps = 0

    def _reduce(e: Expr) -> Expr:
        nonlocal steps
        if steps > max_steps:
            return e   # safety: abort silently on runaway rules

        # Step 1: reduce children
        if e.args:
            reduced_args = tuple(_reduce(a) for a in e.args)
            if reduced_args != e.args:
                e = Expr(head=e.head, args=reduced_args)

        # Step 2: try each rule on the reduced node
        for rule in rules:
            steps += 1
            result = rule.applies_to(e)
            if result is not None:
                # Rule fired — restart reduction on the result
                return _reduce(result)

        # Step 3: no rule fired — this node is in normal form
        return e

    return _reduce(expr)


def normalize(expr: Expr, rules: list[RewriteRule]) -> Expr:
    """Alias for cata_reduce to exhaustion — computes the normal form."""
    return cata_reduce(expr, rules)


# ---------------------------------------------------------------------------
# Rule specificity ordering (most-specific first)
# ---------------------------------------------------------------------------

def sort_by_specificity(rules: list[RewriteRule]) -> list[RewriteRule]:
    """
    Sort rules so the most specific (fewest pattern variables) fire first.

    A rule is more specific if its lhs contains more concrete (non-variable)
    nodes.  This mirrors the SlotProgram specificity sort in the old Predictor
    and ensures that special cases dominate general patterns.
    """
    from experiments.symbolic_ai_v2.ctkg.core.term_algebra import variables
    return sorted(rules, key=lambda r: len(variables(r.lhs)))
