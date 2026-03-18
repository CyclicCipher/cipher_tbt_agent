"""
Term algebra: initial algebra for expression trees.

Phase V additions:
- identify_variables   — maps each rule's pattern variables to their observed values
- unify_surface_forms  — discovers token equivalence classes from corpus structure

Phase XXIII: Expr.head is now a NodeId (opaque int) — identity defined by edges,
not by internal labels.  Constructors (atom, node, var) still accept str and encode
at the boundary via TOKEN_GRAPH.  All public APIs that return or receive variable
names (variables(), match(), substitute(), anti_unify()) use str-keyed dicts so that
callers need no changes.

Expr is the carrier of the free algebra over an operator signature.
Every rule, rewrite, and anti-unification result is an Expr.
No special cases; no token-width assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH, NodeId


@dataclass(frozen=True)
class Expr:
    """
    A node in an expression tree.

    Phase XXIII: head is a NodeId (int), not a str.  Use atom/node/var constructors
    (which encode str at the boundary) or TOKEN_GRAPH directly.

    - Leaf (atom):   atom('5')   → Expr(head=enc('5'), args=())
    - Internal node: node('add', e1, e2) → Expr(head=enc('add'), args=(e1, e2))
    - Pattern var:   var('V0')   → Expr(head=enc('V0'), args=(), is_var=True)

    Pattern variables match any sub-expression during rule application.
    Frozen so Expr can be used as a dict key or in a set.
    """
    head: NodeId
    args: tuple  # tuple[Expr, ...]
    is_var: bool = False

    def __repr__(self) -> str:
        head_str = TOKEN_GRAPH.decode(self.head)
        if self.is_var:
            return f'?{head_str}'
        if not self.args:
            return head_str
        inner = ', '.join(repr(a) for a in self.args)
        return f'{head_str}({inner})'


# ---------------------------------------------------------------------------
# Constructors  (str boundary — encode at entry)
# ---------------------------------------------------------------------------

def atom(tok: str) -> Expr:
    """Leaf node — an operator-0 symbol (digit, variable name, constant)."""
    return Expr(head=TOKEN_GRAPH.encode(tok), args=())


def node(head: str, *args: Expr) -> Expr:
    """Internal node — an operator applied to sub-expressions."""
    return Expr(head=TOKEN_GRAPH.encode(head), args=args)


def var(name: str) -> Expr:
    """Pattern variable — matches any sub-expression during rule matching."""
    return Expr(head=TOKEN_GRAPH.encode(name), args=(), is_var=True)


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

def size(e: Expr) -> int:
    """Number of nodes in the tree (including leaves)."""
    return 1 + sum(size(a) for a in e.args)


def depth(e: Expr) -> int:
    """Maximum depth of the tree (leaf = 0)."""
    if not e.args:
        return 0
    return 1 + max(depth(a) for a in e.args)


def is_ground(e: Expr) -> bool:
    """True if the expression contains no pattern variables."""
    if e.is_var:
        return False
    return all(is_ground(a) for a in e.args)


def variables(e: Expr) -> set[str]:
    """Return the set of variable names appearing in the expression."""
    if e.is_var:
        return {TOKEN_GRAPH.decode(e.head)}
    result: set[str] = set()
    for a in e.args:
        result |= variables(a)
    return result


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def match(pattern: Expr, expr: Expr,
          bindings: Optional[dict[str, Expr]] = None) -> Optional[dict[str, Expr]]:
    """
    Match a pattern (possibly containing var() nodes) against a ground expression.

    Returns a binding dict {var_name: Expr} on success, or None on failure.
    If a variable appears multiple times, all occurrences must match the same Expr.
    """
    if bindings is None:
        bindings = {}

    if pattern.is_var:
        vname = TOKEN_GRAPH.decode(pattern.head)
        existing = bindings.get(vname)
        if existing is None:
            bindings[vname] = expr
            return bindings
        # Variable already bound: must match the same expression
        return bindings if existing == expr else None

    # Non-variable: heads and arity must match
    if pattern.head != expr.head:
        return None
    if len(pattern.args) != len(expr.args):
        return None

    for p_arg, e_arg in zip(pattern.args, expr.args):
        bindings = match(p_arg, e_arg, bindings)
        if bindings is None:
            return None

    return bindings


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def substitute(expr: Expr, bindings: dict[str, Expr]) -> Expr:
    """
    Replace all var() nodes in expr with their values from bindings.
    Non-variable atoms and unbound variables are left unchanged.
    """
    if expr.is_var:
        return bindings.get(TOKEN_GRAPH.decode(expr.head), expr)
    if not expr.args:
        return expr
    new_args = tuple(substitute(a, bindings) for a in expr.args)
    # Avoid allocating a new object if nothing changed
    if new_args == expr.args:
        return expr
    return Expr(head=expr.head, args=new_args, is_var=False)


# ---------------------------------------------------------------------------
# Anti-unification (least general generalisation — Plotkin, 1970)
# ---------------------------------------------------------------------------

def _anti_unify(
    e1: Expr,
    e2: Expr,
    memo: dict[tuple[Expr, Expr], str],
    counter: list[int],
) -> tuple[Expr, dict[str, Expr], dict[str, Expr]]:
    """
    Internal recursive anti-unification with shared memo and counter.

    memo  : maps seen (e1, e2) pairs to the variable name assigned to them,
            ensuring the same pair at multiple positions gets the same variable.
    counter : single-element list holding the next fresh variable index.

    Returns (lgg, subst_for_e1, subst_for_e2).
    """
    # Equal expressions generalise to themselves
    if e1 == e2:
        return e1, {}, {}

    key = (e1, e2)
    if key in memo:
        vname = memo[key]
        return var(vname), {vname: e1}, {vname: e2}

    # Same head and same arity: recurse structurally
    if (not e1.is_var and not e2.is_var
            and e1.head == e2.head
            and len(e1.args) == len(e2.args)):
        gen_args: list[Expr] = []
        s1: dict[str, Expr] = {}
        s2: dict[str, Expr] = {}
        for a1, a2 in zip(e1.args, e2.args):
            g, gs1, gs2 = _anti_unify(a1, a2, memo, counter)
            gen_args.append(g)
            s1.update(gs1)
            s2.update(gs2)
        return Expr(head=e1.head, args=tuple(gen_args)), s1, s2

    # Incompatible structure: introduce a fresh variable
    vname = f'V{counter[0]}'
    counter[0] += 1
    memo[key] = vname
    return var(vname), {vname: e1}, {vname: e2}


def anti_unify(
    e1: Expr,
    e2: Expr,
) -> tuple[Expr, dict[str, Expr], dict[str, Expr]]:
    """
    Compute the least general generalisation of two expressions.

    Returns (lgg, subst1, subst2) where:
      - lgg is the most specific pattern P such that substitute(P, subst1) == e1
        and substitute(P, subst2) == e2
      - subst1, subst2 are the substitutions that recover the original expressions

    Example:
        anti_unify(pow(x, 2), pow(x, 3))
        → (pow(x, ?V0), {V0: 2}, {V0: 3})
    """
    memo: dict[tuple[Expr, Expr], str] = {}
    counter: list[int] = [0]
    return _anti_unify(e1, e2, memo, counter)


def anti_unify_list(
    exprs: list[Expr],
) -> tuple[Expr, list[dict[str, Expr]]]:
    """
    Compute the LGG of a list of expressions.

    Returns (lgg, substs) where substs[i] is the substitution such that
    substitute(lgg, substs[i]) == exprs[i] for all i.

    Uses fresh memos per fold step (to avoid cross-step variable aliasing)
    but shares the counter (to avoid variable name collisions).
    """
    if not exprs:
        raise ValueError("Cannot anti-unify empty list")
    if len(exprs) == 1:
        return exprs[0], [{}]

    counter: list[int] = [0]
    lgg = exprs[0]
    for e in exprs[1:]:
        memo: dict[tuple[Expr, Expr], str] = {}
        lgg, _, _ = _anti_unify(lgg, e, memo, counter)

    # Recover each example's substitution by matching lgg against it
    substs: list[dict[str, Expr]] = []
    for e in exprs:
        bindings = match(lgg, e)
        assert bindings is not None, (
            f"LGG does not match input — this is a bug in anti_unify_list.\n"
            f"  lgg = {lgg}\n  expr = {e}"
        )
        substs.append(bindings)

    return lgg, substs


# ---------------------------------------------------------------------------
# Skeleton (used for grouping by structural shape)
# ---------------------------------------------------------------------------

_PLACEHOLDER = atom('_')


def skeleton(e: Expr) -> Expr:
    """
    Replace all atoms with '_', preserving only the tree structure (operator arities).
    Used to group expressions by shape before anti-unification.

    skeleton(add(mul(2, x), 3)) == add(mul(_, _), _)
    """
    if not e.args:
        return _PLACEHOLDER
    return Expr(head=e.head, args=tuple(skeleton(a) for a in e.args))


# ---------------------------------------------------------------------------
# Phase V: Variable discovery and surface-form unification
# ---------------------------------------------------------------------------

def identify_variables(
    rules: "list",   # list[RewriteRule] — avoid circular import with rewrite.py
) -> dict:
    """
    Map each rule's pattern variable positions to the set of observed surface forms.

    For each RewriteRule, we examine the lhs pattern and return a dict:
        { (rule_index, var_name): frozenset[str] }

    A variable V is notation-independent if its observed values include multiple
    surface forms for the same underlying object (e.g. {'5', 'five', 'cinq'}).
    The caller can use this to build a normalization map via unify_surface_forms.

    Parameters
    ----------
    rules : list of RewriteRule objects (from rule_discover.discover_rules).
            Each rule is expected to have .lhs (Expr) and .rhs (Expr) fields.
            The evidence values are not used here.

    Returns
    -------
    dict mapping (rule_idx: int, var_name: str) → frozenset[str] of observed atom heads.

    Note: this function returns the *structural* variable positions, not the
    observed values (which require running the rules against a corpus).  Use
    parse_corpus + match to get values; see unify_surface_forms for the combined
    workflow.
    """
    from experiments.symbolic_ai_v2.ctkg.core.term_algebra import variables as _vars
    result: dict = {}
    for idx, rule in enumerate(rules):
        for vname in _vars(rule.lhs):
            result[(idx, vname)] = frozenset()
    return result


def unify_surface_forms(
    corpus: "list[list[str]]",   # list of token sequences
    rules: "list",               # list[RewriteRule]
    arities: dict,
) -> dict:
    """
    Discover a surface-form normalization map from the corpus and discovered rules.

    Algorithm
    ---------
    Two tokens a and b are surface-form equivalent if, for EVERY rule whose lhs
    pattern has a variable V, the following holds across all training examples:
        - In examples where V is bound to atom(a), substituting b for a yields
          an input that would bind V to atom(b), and the output changes in exactly
          the same way (the substituted output equals the original output with b
          substituted for a).

    In practice this reduces to:
        a ≡ b  iff  they appear in the same variable position across all training
                    examples AND they have the same distribution over output values.

    The canonical form of an equivalence class is the lexicographically first member.

    Returns
    -------
    dict: surface_form → canonical_form.  Tokens with no equivalences map to
    themselves (and may be omitted).  Apply this map via
    expr_parser.normalize_surface() before training and evaluation.
    """
    from experiments.symbolic_ai_v2.ctkg.core.expr_parser import parse_full
    from experiments.symbolic_ai_v2.ctkg.core.term_algebra import match as _match, variables as _vars

    # For each (rule, var_name), collect {input_atom → observed_output_atom_set}.
    # Two input atoms are equivalent if they map to the same output_atom_set.
    var_profiles: dict = {}   # (rule_idx, var_name) → {atom_head: frozenset[out_head]}

    for idx, rule in enumerate(rules):
        lvars = list(_vars(rule.lhs))
        if not lvars:
            continue
        profile: dict = {v: {} for v in lvars}   # var_name → {in_head: set[out_head]}

        for seq in corpus:
            inp_expr, out_expr = parse_full(seq, arities)
            if inp_expr is None or out_expr is None:
                continue
            bindings = _match(rule.lhs, inp_expr)
            if bindings is None:
                continue
            out_bindings = _match(rule.rhs, out_expr)
            for vn in lvars:
                in_val = bindings.get(vn)
                if in_val is None or in_val.args:
                    continue   # only track leaf (atom) values
                in_head = TOKEN_GRAPH.decode(in_val.head)
                out_heads: set = set()
                if out_bindings and vn in out_bindings:
                    ov = out_bindings.get(vn)
                    if ov is not None and not ov.args:
                        out_heads.add(TOKEN_GRAPH.decode(ov.head))
                if in_head not in profile[vn]:
                    profile[vn][in_head] = set()
                profile[vn][in_head] |= out_heads

        for vn, in_to_out in profile.items():
            var_profiles[(idx, vn)] = {
                k: frozenset(v) for k, v in in_to_out.items()
            }

    # Build equivalence classes: two atom heads are equivalent if they have
    # the same output-head profile across ALL variable positions they appear in.
    from collections import defaultdict
    profile_to_heads: dict = defaultdict(set)
    for (idx, vn), in_to_out in var_profiles.items():
        for in_head, out_heads in in_to_out.items():
            key = (idx, vn, out_heads)
            profile_to_heads[key].add(in_head)

    # Merge across positions: two heads are equivalent if they share a profile key
    # (union-find approach)
    parent: dict = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # canonical = lexicographically first
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for equiv_set in profile_to_heads.values():
        heads = sorted(equiv_set)
        for h in heads[1:]:
            union(heads[0], h)

    # Build the normalization map: head → canonical (lexicographically first in class)
    norm_map: dict = {}
    for (_, _, _), equiv_set in profile_to_heads.items():
        if len(equiv_set) > 1:
            canon = find(min(equiv_set))
            for h in equiv_set:
                if find(h) == canon and h != canon:
                    norm_map[h] = canon

    return norm_map
