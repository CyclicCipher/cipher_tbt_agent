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

from experiments.symbolic_ai_v2.ctkg.core.node import (
    TOKEN_GRAPH,
    NodeId,
    OUTPUT_DELIMS,
    EOS_NODE,
)
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
    name: NodeId          # output role NodeId, e.g. STEP_NODE, ANS_NODE
    expr: Expr            # expression tree (var() nodes for role references)
    confidence: float = 1.0  # Phase XXII: from RelationRule.confidence


@dataclass
class LambdaTerm:
    """A lambda term synthesized for a trace-format operator.

    Attributes
    ----------
    op:
        The operator NodeId this term was synthesized for.
    params:
        Ordered list of input role NodeIds — the lambda parameters
        (e.g. [P0_NODE, P1_NODE, P2_NODE] for positional ops).
    steps:
        Sequence of let-bindings in dependency order.  Each step's expr
        may reference earlier steps as variables.
    output_delims:
        The delimiter NodeId that PRECEDES each step's value in the output
        sequence (e.g. STEP_NODE for intermediate results, ANS_NODE for the
        final answer).  len(output_delims) == len(steps).
    evidence:
        Total evidence (training examples) used to synthesize the term.
    """
    op: NodeId
    params: list[NodeId]
    steps: list[LetStep]
    output_delims: list[NodeId]
    evidence: int = 0

    def arity(self) -> int:
        """Number of input parameters."""
        return len(self.params)

    def output_signature(self) -> tuple[NodeId, ...]:
        """Ordered tuple of output delimiter NodeIds (structural fingerprint)."""
        return tuple(self.output_delims)


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

def eval_expr(
    expr: Expr,
    env: dict[NodeId, str],
    engine,
) -> Optional[str]:
    """Evaluate *expr* to a token string given variable bindings *env*.

    env maps role NodeId → string value (values remain strings since
    engine.compute() operates in the string domain at this boundary).

    Rules:
      - var(nid)         → env[nid]  (None if not bound)
      - atom(nid)        → env.get(nid) or TOKEN_GRAPH.decode(nid)
      - node(op, a, b)   → engine.compute(decode(op), eval(a), eval(b))

    Returns None if any lookup misses.
    Result is '\\x00'-joined for multi-token outputs.
    """
    if expr.is_var:
        return env.get(expr.head)
    if not expr.args:
        # Leaf atom: resolve as variable binding if present, else literal
        val = env.get(expr.head)
        if val is not None:
            return val
        return TOKEN_GRAPH.decode(expr.head)
    if len(expr.args) == 2:
        if engine is None:
            return None
        a_val = eval_expr(expr.args[0], env, engine)
        b_val = eval_expr(expr.args[1], env, engine)
        if a_val is None or b_val is None:
            return None
        result_tup = engine.compute(TOKEN_GRAPH.decode(expr.head), a_val, b_val)
        if result_tup is None:
            return None
        return '\x00'.join(result_tup)
    # Arity != 2 not supported; return None
    return None


def eval_term(
    term: LambdaTerm,
    arg_tokens: list[str],
    engine,
    output_so_far: list[NodeId],
) -> Optional[dict[NodeId, float]]:
    """Evaluate *term* and return the next-token prediction.

    Parameters
    ----------
    term:
        The lambda term to evaluate.
    arg_tokens:
        Concrete token string values for each parameter, in order.
        Must satisfy len(arg_tokens) == term.arity().
    engine:
        ComposeEngine (replaces BFM dict).
    output_so_far:
        NodeIds already emitted in the output (everything after the first
        output delimiter in the prefix).

    Returns
    -------
    {next_node_id: probability}  if a unique next token is determined.
    {EOS_NODE: 1.0}              if the output is complete.
    None                         if evaluation fails.
    """
    if len(arg_tokens) != term.arity():
        return None

    # Bind input variables: param NodeId → string value
    env: dict[NodeId, str] = dict(zip(term.params, arg_tokens))

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
        result_str = eval_expr(step.expr, env, engine)
        if result_str is None:
            return None  # engine miss on a needed step
        env[step.name] = result_str  # β-bind for subsequent steps
        acc_conf *= step.confidence  # Kleisli multiplication

        # --- token(s) of the result (split '\x00'-joined multi-token results) ---
        result_toks = result_str.split('\x00')
        for tok_str in result_toks:
            tok_nid = TOKEN_GRAPH.encode(tok_str)
            if pos == k:
                return {tok_nid: acc_conf}
            if output_so_far[pos] != tok_nid:
                return None  # prefix mismatch
            pos += 1

    # All steps consumed; if we are exactly at k the sequence is complete.
    if pos == k:
        return {EOS_NODE: acc_conf}
    return None


# ---------------------------------------------------------------------------
# Synthesis from RelationRules
# ---------------------------------------------------------------------------

def synthesize_from_rules(
    op: NodeId,
    op_rules: list,     # list[RelationRule] — avoid circular import
    input_role_names: list[NodeId],
) -> Optional[LambdaTerm]:
    """Synthesise a LambdaTerm from a sequence of RelationRules.

    Each RelationRule `output_role = bfm_op(arg1, arg2)` becomes a LetStep:
        LetStep(name=output_role, expr=node(bfm_op, var(arg1), var(arg2)))

    Parameters
    ----------
    op:
        Operator NodeId.
    op_rules:
        RelationRules in dependency order (as returned by
        discover_relation_rules).  All fields are NodeIds.
    input_role_names:
        Ordered list of input role NodeIds.
    """
    if not op_rules or not input_role_names:
        return None

    from experiments.symbolic_ai_v2.ctkg.core.node import STEP_NODE

    steps: list[LetStep] = []
    delims: list[NodeId] = []
    total_evidence: int = 0

    for rr in op_rules:
        # rr.op_name, rr.arg1, rr.arg2, rr.output_role are all NodeIds
        # Construct Expr directly — don't call node()/var() which expect strings
        expr = Expr(
            head=rr.op_name,
            args=(
                Expr(head=rr.arg1, args=(), is_var=True),
                Expr(head=rr.arg2, args=(), is_var=True),
            ),
        )
        conf = getattr(rr, 'confidence', 1.0)
        steps.append(LetStep(name=rr.output_role, expr=expr, confidence=conf))
        # Determine the output delimiter for this step
        role_str = TOKEN_GRAPH.decode(rr.output_role)
        if role_str.startswith('step'):
            delims.append(STEP_NODE)
        else:
            delims.append(rr.output_role)   # ANS_NODE, EQ_NODE, or custom
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
    relation_rules: dict[NodeId, list],   # op NodeId → list[RelationRule]
    relation_store,                        # RelationStore
) -> dict[NodeId, LambdaTerm]:
    """Build a library of LambdaTerms from all known RelationRules.

    Parameters
    ----------
    relation_rules:
        dict mapping op NodeId → [RelationRule, …].
    relation_store:
        A RelationStore whose schemas map op → positional or separator schema.

    Returns
    -------
    dict op NodeId → LambdaTerm.
    """
    library: dict[NodeId, LambdaTerm] = {}
    for op, rules in relation_rules.items():
        op_str = TOKEN_GRAPH.decode(op)
        schema = relation_store.get_schema(op_str)
        if not schema:
            continue
        # Roles are already NodeIds (relation_store.get_schema returns list[tuple[int, NodeId]])
        input_role_names = [role for _, role in schema]
        term = synthesize_from_rules(op, rules, input_role_names)
        if term is not None:
            library[op] = term
    return library


# ---------------------------------------------------------------------------
# Prediction entry point (for predict.py integration)
# ---------------------------------------------------------------------------

def _split_prefix(prefix: list[NodeId]) -> tuple[NodeId, list[str], list[NodeId]]:
    """Split *prefix* into (op_nid, input_token_strings, output_so_far).

    Works for any op and any format (step/ans or eq).  The split point is
    the first output delimiter NodeId (STEP_NODE, ANS_NODE, EQ_NODE, EOS_NODE).

    Returns (op_nid, input_token_strings, output_nids).
    Input tokens are decoded to strings for engine compatibility.
    """
    if not prefix:
        return 0, [], []
    op = prefix[0]
    split = len(prefix)
    for i, t in enumerate(prefix[1:], 1):
        if t in OUTPUT_DELIMS:
            split = i
            break
    input_nids = prefix[1:split]
    output_nids = prefix[split:]
    # Decode input NodeIds to strings for engine.compute() compatibility
    input_strs = [TOKEN_GRAPH.decode(n) for n in input_nids]
    return op, input_strs, output_nids


def lambda_predict(
    prefix: list[NodeId],
    lambda_library: dict[NodeId, LambdaTerm],
    engine,
    allow_transfer: bool = True,
) -> Optional[dict[NodeId, float]]:
    """Predict the next token using the lambda term library.

    Parameters
    ----------
    prefix:
        The current prefix as NodeIds (all tokens seen so far, including op).
    lambda_library:
        dict op NodeId → LambdaTerm, produced by synthesize_library().
    engine:
        ComposeEngine (replaces BFM dict).
    allow_transfer:
        If False, skip structural transfer (used in ablation tests).

    Returns
    -------
    {next_node_id: probability} or {EOS_NODE: 1.0}, or None on miss.
    """
    op, input_tokens, output_so_far = _split_prefix(prefix)
    if not op:
        return None

    # Phase 1: direct lookup for known op
    term = lambda_library.get(op)
    if term is not None:
        return eval_term(term, input_tokens, engine, output_so_far)

    # Phase 2: creative transfer for novel op
    if not allow_transfer or not input_tokens:
        return None

    n_params = len(input_tokens)
    # Determine output_signature prefix already observed (from output_so_far)
    seen_delims: list[NodeId] = [t for t in output_so_far if t in OUTPUT_DELIMS]

    candidates: list[LambdaTerm] = []
    for lib_term in lambda_library.values():
        if lib_term.arity() != n_params:
            continue
        sig = lib_term.output_signature()
        if len(seen_delims) > len(sig):
            continue
        if any(seen_delims[i] != sig[i] for i in range(len(seen_delims))):
            continue
        candidates.append(lib_term)

    if not candidates:
        return None

    # Try each candidate; weight results by evidence
    results: dict[NodeId, float] = {}
    total_evidence = sum(max(c.evidence, 1) for c in candidates)
    for cterm in candidates:
        w = max(cterm.evidence, 1) / total_evidence
        result = eval_term(cterm, input_tokens, engine, output_so_far)
        if result is not None:
            for nid, prob in result.items():
                results[nid] = results.get(nid, 0.0) + w * prob

    if not results:
        return None
    total = sum(results.values())
    return {nid: v / total for nid, v in results.items()}
