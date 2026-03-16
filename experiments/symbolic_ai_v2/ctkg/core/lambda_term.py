"""
Lambda terms for trace-format prediction (Phase XX).

A LambdaTerm is a sequence of let-bindings:

    λp0. λp1. … λpN.
        let step = bfm_op1(arg1, arg2) in
        let ans  = bfm_op2(arg3, arg4) in
        …

where each arg_i is either an input role name (p0, p1, …) or a previously
computed output role name (step, step0, …).  This is a *partial evaluation
scheme*: as input tokens arrive one by one, variables are bound and β-redexes
are reduced.

The LambdaTerm representation replaces the ChainTable lookup for trace-format
ops.  The key difference is *generativity*: the lambda term can be evaluated
on any input (including OOD) by performing the BFM lookups at inference time,
whereas the ChainTable can only answer queries seen in training.

Creative transfer (Phase XX gate):
    A novel trace op not in training is handled by *structural transfer*: find
    all lambda terms in the library with the same arity and output-delimiter
    sequence, evaluate each, and return a distribution weighted by evidence.
    This is the mechanism that enables novel composition without retraining.

Synthesis algorithm:
    Each RelationRule `output_role = bfm_op(arg1, arg2)` is a one-step lambda
    abstraction `λarg1. λarg2. bfm_op(arg1, arg2)`.  A sequence of
    RelationRules in dependency order forms the body of the LambdaTerm.

See CTKG_ARCHITECTURE.md §Phase XX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
    Expr,
    atom,
    node,
    var,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LetStep:
    """One let-binding in the lambda term body.

    Represents: let *name* = *expr* in …
    *expr* may reference input variables (p0, p1, …) and earlier output
    role names (step, step0, ans).

    Phase XXII — probability monad:
        *confidence* is the empirical reliability of the RelationRule that
        generated this step (evidence / total_obs).  eval_term multiplies
        confidences across all steps (Kleisli composition in Dist): the
        resulting prediction has probability equal to the product of each
        step's individual confidence.
    """
    name: str             # output role name, e.g. 'step', 'step0', 'ans', 'eq'
    expr: Expr            # expression tree (var() nodes for role references)
    confidence: float = 1.0  # Phase XXII: from RelationRule.confidence


@dataclass
class LambdaTerm:
    """A lambda term synthesized for a trace-format operator.

    Attributes
    ----------
    op:
        The operator token this term was synthesized for (e.g. 'linear_trace').
    params:
        Ordered list of input role names — the lambda parameters
        (e.g. ['p0', 'p1', 'p2'] for positional ops).
    steps:
        Sequence of let-bindings in dependency order.  Each step's expr
        may reference earlier steps as variables.
    output_delims:
        The delimiter token that PRECEDES each step's value in the output
        sequence (e.g. 'step' for intermediate results, 'ans' for the final
        answer).  len(output_delims) == len(steps).
    evidence:
        Total evidence (training examples) used to synthesize the term.
    """
    op: str
    params: list[str]
    steps: list[LetStep]
    output_delims: list[str]
    evidence: int = 0

    def arity(self) -> int:
        """Number of input parameters."""
        return len(self.params)

    def output_signature(self) -> tuple[str, ...]:
        """Ordered tuple of output delimiter tokens (structural fingerprint)."""
        return tuple(self.output_delims)


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

def eval_expr(
    expr: Expr,
    env: dict[str, str],
    bfm: dict[str, dict[tuple, str]],
) -> Optional[str]:
    """Evaluate *expr* to a token string given variable bindings *env*.

    Rules:
      - var('x')    → env['x']  (None if not bound)
      - atom('5')   → env.get('5', '5')  (literal atom if not in env)
      - node(op, a, b)  → bfm[op][(eval(a), eval(b))]

    Returns None if any lookup misses.
    """
    if expr.is_var:
        return env.get(expr.head)
    if not expr.args:
        # Leaf atom: resolve as a variable binding if present, else literal
        return env.get(expr.head, expr.head)
    if len(expr.args) == 2:
        a_val = eval_expr(expr.args[0], env, bfm)
        b_val = eval_expr(expr.args[1], env, bfm)
        if a_val is None or b_val is None:
            return None
        return bfm.get(expr.head, {}).get((a_val, b_val))
    # Arity != 2 not supported in current BFM; return None
    return None


def eval_term(
    term: LambdaTerm,
    arg_tokens: list[str],
    bfm: dict[str, dict[tuple, str]],
    output_so_far: list[str],
) -> Optional[dict[str, float]]:
    """Evaluate *term* and return the next-token prediction.

    Parameters
    ----------
    term:
        The lambda term to evaluate.
    arg_tokens:
        Concrete token values for each parameter, in order.
        Must satisfy len(arg_tokens) == term.arity().
    bfm:
        Binary functional maps {op: {(a,b): result}}.
    output_so_far:
        Tokens already emitted in the output (everything after the first
        output delimiter in the prefix).

    Returns
    -------
    {next_token: 1.0}  if a unique next token is determined.
    {'<eos>': 1.0}     if the output is complete.
    None               if evaluation fails (BFM miss, wrong arity, etc.).
    """
    if len(arg_tokens) != term.arity():
        return None

    # Bind input variables
    env: dict[str, str] = dict(zip(term.params, arg_tokens))

    # Lazy streaming evaluation: advance through (delim, chars…) groups one at
    # a time, checking output_so_far for consistency and returning as soon as
    # we reach the next position to predict.  A step expression is only
    # evaluated when its delimiter has already been consumed (i.e. when we need
    # at least one character of the result), so BFM misses in later steps do
    # not prevent prediction of earlier delimiters.
    #
    # Phase XXII — Kleisli confidence propagation:
    #   acc_conf tracks the product of all step confidences consumed so far.
    #   The returned distribution {token: acc_conf} encodes the probability
    #   that the next token is correct given the full chain of rule applications.
    k = len(output_so_far)
    pos = 0  # current position within the virtual full-output sequence
    acc_conf: float = 1.0  # accumulated Kleisli confidence

    for step, delim in zip(term.steps, term.output_delims):
        # --- delimiter token ---
        if pos == k:
            return {delim: acc_conf}
        if output_so_far[pos] != delim:
            return None  # prefix mismatch
        pos += 1

        # --- evaluate the step expression (only now that delimiter is past) ---
        result_str = eval_expr(step.expr, env, bfm)
        if result_str is None:
            return None  # BFM miss on a needed step
        env[step.name] = result_str  # β-bind for subsequent steps
        acc_conf *= step.confidence  # Kleisli multiplication

        # --- character tokens of the result ---
        for ch in result_str:
            if pos == k:
                return {ch: acc_conf}
            if output_so_far[pos] != ch:
                return None  # prefix mismatch
            pos += 1

    # All steps consumed; if we are exactly at k the sequence is complete.
    if pos == k:
        return {'<eos>': acc_conf}
    return None


# ---------------------------------------------------------------------------
# Synthesis from RelationRules
# ---------------------------------------------------------------------------

def synthesize_from_rules(
    op: str,
    op_rules: list,     # list[RelationRule] — avoid circular import
    input_role_names: list[str],
) -> Optional[LambdaTerm]:
    """Synthesise a LambdaTerm from a sequence of RelationRules.

    Each RelationRule `output_role = bfm_op(arg1, arg2)` becomes a LetStep:
        LetStep(name=output_role, expr=node(bfm_op, var(arg1), var(arg2)))

    The resulting lambda term represents the complete computation for *op* as
    an explicit expression tree rather than an opaque lookup table.

    Parameters
    ----------
    op:
        Operator name.
    op_rules:
        RelationRules in dependency order (as returned by
        discover_relation_rules).  Must all belong to the same op.
    input_role_names:
        Ordered list of input role names.  For positional ops these are
        'p0', 'p1', 'p2', …; for separator ops the separator token names.

    Returns
    -------
    LambdaTerm on success, None if op_rules is empty or synthesis fails.
    """
    if not op_rules or not input_role_names:
        return None

    steps: list[LetStep] = []
    delims: list[str] = []
    total_evidence: int = 0

    for rr in op_rules:
        expr = node(rr.op_name, var(rr.arg1), var(rr.arg2))
        # Phase XXII: carry rule confidence into the LetStep for Kleisli propagation
        conf = getattr(rr, 'confidence', 1.0)
        steps.append(LetStep(name=rr.output_role, expr=expr, confidence=conf))
        # Determine the output delimiter for this step
        role = rr.output_role
        if role.startswith('step'):
            delims.append('step')
        else:
            delims.append(role)   # 'ans', 'eq', or custom
        total_evidence += getattr(rr, 'evidence', 0)

    if not steps:
        return None

    return LambdaTerm(
        op=op,
        params=list(input_role_names),
        steps=steps,
        output_delims=delims,
        evidence=total_evidence,
    )


def synthesize_library(
    relation_rules: dict[str, list],   # op → list[RelationRule]
    relation_store,                     # RelationStore
) -> dict[str, LambdaTerm]:
    """Build a library of LambdaTerms from all known RelationRules.

    Parameters
    ----------
    relation_rules:
        dict mapping op → [RelationRule, …] as produced by the Predictor's
        __init__ (only ops with full role coverage are included).
    relation_store:
        A RelationStore whose schemas map op → positional or separator schema.

    Returns
    -------
    dict op → LambdaTerm.  Only ops for which synthesis succeeds are included.
    """
    library: dict[str, LambdaTerm] = {}
    for op, rules in relation_rules.items():
        schema = relation_store.get_schema(op)
        if not schema:
            continue
        # Extract role names from schema entries
        input_role_names = [role for _, role in schema]
        term = synthesize_from_rules(op, rules, input_role_names)
        if term is not None:
            library[op] = term
    return library


# ---------------------------------------------------------------------------
# Prediction entry point (for predict.py integration)
# ---------------------------------------------------------------------------

_OUTPUT_DELIM_SET: frozenset[str] = frozenset({'step', 'ans', 'eq', '<eos>'})


def _split_prefix(prefix: list[str]) -> tuple[str, list[str], list[str]]:
    """Split *prefix* into (op, input_tokens, output_so_far).

    Works for any op and any format (step/ans or eq).  The split point is
    the first output delimiter token (step, ans, eq, <eos>).

    Returns (op, input_tokens, output_so_far).
    """
    if not prefix:
        return '', [], []
    op = prefix[0]
    split = len(prefix)
    for i, t in enumerate(prefix[1:], 1):
        if t in _OUTPUT_DELIM_SET:
            split = i
            break
    return op, prefix[1:split], prefix[split:]


def lambda_predict(
    prefix: list[str],
    lambda_library: dict[str, LambdaTerm],
    bfm: dict[str, dict[tuple, str]],
    allow_transfer: bool = True,
) -> Optional[dict[str, float]]:
    """Predict the next token using the lambda term library.

    Two-phase dispatch:

    Phase 1 — direct lookup:
        If *op* is in the library, evaluate its lambda term with the
        observed input tokens.

    Phase 2 — structural transfer (if allow_transfer=True):
        If *op* is NOT in the library (novel op), find all lambda terms
        with the same arity and output_signature and try each.  Returns a
        distribution weighted by evidence if multiple terms agree, or the
        single-term result if only one fires.  This is the creative transfer
        mechanism described in CTKG_ARCHITECTURE.md §Phase XX.

    Parameters
    ----------
    prefix:
        The current prefix (all tokens seen so far, including the op).
    lambda_library:
        dict op → LambdaTerm, produced by synthesize_library().
    bfm:
        Binary functional maps.
    allow_transfer:
        If False, skip Phase 2 (used in ablation tests).

    Returns
    -------
    {next_token: probability} or {'<eos>': 1.0}, or None on miss.
    """
    op, input_tokens, output_so_far = _split_prefix(prefix)
    if not op:
        return None

    # Phase 1: direct lookup for known op
    term = lambda_library.get(op)
    if term is not None:
        return eval_term(term, input_tokens, bfm, output_so_far)

    # Phase 2: creative transfer for novel op
    if not allow_transfer or not input_tokens:
        return None

    n_params = len(input_tokens)
    # Determine output_signature prefix already observed (from output_so_far)
    seen_delims: list[str] = [t for t in output_so_far if t in ('step', 'ans', 'eq')]

    candidates: list[LambdaTerm] = []
    for lib_term in lambda_library.values():
        if lib_term.arity() != n_params:
            continue
        # Require that the observed delimiters are a prefix of this term's signature
        sig = lib_term.output_signature()
        if len(seen_delims) > len(sig):
            continue
        if any(seen_delims[i] != sig[i] for i in range(len(seen_delims))):
            continue
        candidates.append(lib_term)

    if not candidates:
        return None

    # Try each candidate; weight results by evidence
    results: dict[str, float] = {}
    total_evidence = sum(max(c.evidence, 1) for c in candidates)
    for cterm in candidates:
        w = max(cterm.evidence, 1) / total_evidence
        result = eval_term(cterm, input_tokens, bfm, output_so_far)
        if result is not None:
            for tok, prob in result.items():
                results[tok] = results.get(tok, 0.0) + w * prob

    if not results:
        return None
    total = sum(results.values())
    return {tok: v / total for tok, v in results.items()}
