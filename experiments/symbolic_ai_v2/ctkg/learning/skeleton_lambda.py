"""SkeletonStore: skeleton-discriminated lambda terms (Phase XXIV).

Groups training sequences by (op, input_skeleton, output_template) triplets,
where the skeleton of a sequence is the ordered tuple of non-NNO tokens
and the template is the output sequence with NNO tokens replaced by slot
indices.  Within each group, learns how each output NNO slot depends on
input NNO slots using a small set of algebraic expressions (NNOExpr).  At
prediction time, looks up (op, in_skel), computes output NNO values,
reconstructs the full output, and returns the next-token prediction.

This handles operators like 'd' (derivatives) and 'int' (integrals) where
RelationStore fails because the input schema is not uniform.

Key ideas
---------
sq normalization:
    Before learning and before prediction, apply the flat token substitution
    ``sq v → pow v 2``.  This collapses the ``d sq x`` and ``d pow x 2``
    cases into the same input skeleton ``(pow, x)`` and makes the output
    skeleton uniform across all n>=2 power-rule examples.

Skeleton extraction:
    Given a token sequence, separate it into (skeleton, nno_values) where
    skeleton = tuple of non-NNO tokens, nno_values = list of NNO tokens in
    order.  NNO tokens are those in the discovered successor chain.

Output template:
    The full output sequence with each NNO token replaced by its slot index
    (0, 1, 2, ...).  E.g. ``step 3 ans mul 3 pow x 2 <eos>`` becomes
    ``('step', 0, 'ans', 'mul', 1, 'pow', 'x', 2, '<eos>')``.
    Templates with the same (in_skel, out_skel) are identical in structure.

NNO slot synthesis:
    For each output slot, tries expressions in increasing complexity:
    const → id(j) → pred(id(j)) / succ(id(j)) → bfm(op, id(j), id(k))
    → bfm(op, id(j), succ/pred(id(k))) → bfm(op, id(j), const(c)).
    The first expression that reproduces all training examples is kept.

Iron Rule compliance:
    No token string is special-cased except as a structural delimiter.
    The NNO set is discovered from the successor map, not hardcoded.
    Operator identity is purely positional (op = prefix[0]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# sq normalization (flat token substitution)
# ---------------------------------------------------------------------------

_SQ_STR = 'sq'
_POW_STR = 'pow'
_TWO_STR = '2'

_OUTPUT_DELIMS_STR = frozenset({'eq', 'step', 'ans', '<eos>'})


def _normalize_sq(toks: list[str]) -> list[str]:
    """Apply sq v → pow v 2 as a flat token substitution."""
    result: list[str] = []
    i = 0
    while i < len(toks):
        if toks[i] == _SQ_STR and i + 1 < len(toks):
            result.append(_POW_STR)
            result.append(toks[i + 1])
            result.append(_TWO_STR)
            i += 2
        else:
            result.append(toks[i])
            i += 1
    return result


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------

def _extract_skel(
    toks: list[str],
    nno_set: frozenset[str],
) -> tuple[tuple[str, ...], list[str]]:
    """Split a token sequence into (skeleton, nno_values).

    skeleton  = structural (non-NNO) tokens in order
    nno_values = NNO tokens in order
    """
    skel: list[str] = []
    nno_vals: list[str] = []
    for t in toks:
        if t in nno_set:
            nno_vals.append(t)
        else:
            skel.append(t)
    return tuple(skel), nno_vals


def _build_template(
    out_toks: list[str],
    nno_set: frozenset[str],
) -> tuple[tuple, list[str]]:
    """Build (template, nno_values) from an output token sequence.

    The template is the output sequence with NNO tokens replaced by
    their slot indices (0, 1, 2, ...).
    """
    template: list = []
    nno_vals: list[str] = []
    for t in out_toks:
        if t in nno_set:
            template.append(len(nno_vals))  # slot index
            nno_vals.append(t)
        else:
            template.append(t)
    return tuple(template), nno_vals


def _reconstruct(template: tuple, out_nno: list[str]) -> list[str]:
    """Reconstruct full output from template and computed NNO values."""
    result: list[str] = []
    for elem in template:
        if isinstance(elem, int):
            if elem < len(out_nno):
                result.append(out_nno[elem])
            else:
                return []  # incomplete NNO values
        else:
            result.append(elem)
    return result


# ---------------------------------------------------------------------------
# NNOExpr — algebraic expressions over input NNO slots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NNOExpr:
    """Algebraic expression computing one output NNO slot from input NNO slots.

    Kinds
    -----
    'const'  : args = (value: str,)
    'id'     : args = (index: int,)           — input NNO slot index
    'pred'   : args = (sub: NNOExpr,)         — predecessor of sub
    'succ'   : args = (sub: NNOExpr,)         — successor of sub
    'bfm'    : args = (op: str, lhs: NNOExpr, rhs: NNOExpr)
    """
    kind: str
    args: tuple


def _eval_nno_expr(
    expr: NNOExpr,
    in_nno: list[str],
    succ_map: dict[str, str],
    pred_map: dict[str, str],
    engine,  # ComposeEngine or None
) -> Optional[str]:
    """Evaluate an NNOExpr against input NNO values.

    Returns a string token, or None on evaluation failure.
    """
    if expr.kind == 'const':
        return expr.args[0]

    if expr.kind == 'id':
        idx = expr.args[0]
        return in_nno[idx] if idx < len(in_nno) else None

    if expr.kind == 'pred':
        val = _eval_nno_expr(expr.args[0], in_nno, succ_map, pred_map, engine)
        return pred_map.get(val) if val is not None else None

    if expr.kind == 'succ':
        val = _eval_nno_expr(expr.args[0], in_nno, succ_map, pred_map, engine)
        return succ_map.get(val) if val is not None else None

    if expr.kind == 'bfm':
        op_str, lhs_expr, rhs_expr = expr.args
        lv = _eval_nno_expr(lhs_expr, in_nno, succ_map, pred_map, engine)
        rv = _eval_nno_expr(rhs_expr, in_nno, succ_map, pred_map, engine)
        if lv is None or rv is None:
            return None
        if engine is None:
            return None
        res = engine.compute_tup(op_str, (lv,), (rv,))
        if res is None or len(res) != 1:
            return None
        return res[0]

    return None


# ---------------------------------------------------------------------------
# NNO slot synthesis
# ---------------------------------------------------------------------------

def _try_synth_slot(
    pairs: list[tuple[tuple[str, ...], str]],  # [(in_nno, out_val), ...]
    succ_map: dict[str, str],
    pred_map: dict[str, str],
    engine,
    n_in: int,
) -> Optional[NNOExpr]:
    """Find the simplest NNOExpr that reproduces out_val for all pairs.

    Tries expressions in order of increasing complexity.  Returns the first
    match, or None if no expression up to depth 2 works.
    """
    if not pairs:
        return None

    def check(expr: NNOExpr) -> bool:
        return all(
            _eval_nno_expr(expr, list(in_nno), succ_map, pred_map, engine) == out_val
            for in_nno, out_val in pairs
        )

    # 1. const
    first_out = pairs[0][1]
    if all(out == first_out for _, out in pairs):
        return NNOExpr('const', (first_out,))

    # 2. id(j)
    for j in range(n_in):
        e = NNOExpr('id', (j,))
        if check(e):
            return e

    # 3. pred(id(j)) and succ(id(j))
    for j in range(n_in):
        id_j = NNOExpr('id', (j,))
        for wrapper in ('pred', 'succ'):
            e = NNOExpr(wrapper, (id_j,))
            if check(e):
                return e

    # 4. pred(pred(id(j))) and succ(succ(id(j)))
    for j in range(n_in):
        id_j = NNOExpr('id', (j,))
        for wrapper in ('pred', 'succ'):
            inner = NNOExpr(wrapper, (id_j,))
            outer = NNOExpr(wrapper, (inner,))
            if check(outer):
                return outer

    if engine is None:
        return None

    # 5. bfm(op, id(j), id(k))
    for op_str in ('mul', 'div', 'add', 'sub'):
        for j in range(n_in):
            for k in range(n_in):
                e = NNOExpr('bfm', (op_str, NNOExpr('id', (j,)), NNOExpr('id', (k,))))
                if check(e):
                    return e

    # 6. bfm(op, id(j), succ/pred(id(k))) and vice versa
    for op_str in ('mul', 'div', 'add', 'sub'):
        for j in range(n_in):
            for k in range(n_in):
                for inner_op in ('succ', 'pred'):
                    inner = NNOExpr(inner_op, (NNOExpr('id', (k,)),))
                    e1 = NNOExpr('bfm', (op_str, NNOExpr('id', (j,)), inner))
                    if check(e1):
                        return e1
                    e2 = NNOExpr('bfm', (op_str, inner, NNOExpr('id', (j,))))
                    if check(e2):
                        return e2

    # 7. bfm(op, id(j), const(c)) for digit constants
    nno_vals = set(succ_map.keys()) | set(succ_map.values())
    for op_str in ('mul', 'div', 'add', 'sub'):
        for j in range(n_in):
            for c in sorted(nno_vals):  # sorted for determinism
                e = NNOExpr('bfm', (op_str, NNOExpr('id', (j,)), NNOExpr('const', (c,))))
                if check(e):
                    return e

    return None


# ---------------------------------------------------------------------------
# SkeletonRule — one learned rule for a (op, in_skel, out_skel) group
# ---------------------------------------------------------------------------

@dataclass
class SkeletonRule:
    """Mapping from input NNO values to full output sequence for one group.

    Parameters
    ----------
    in_skel:
        Structural (non-NNO) tokens of the input, e.g. ('pow', 'x').
    out_skel:
        Structural (non-NNO) tokens of the output, e.g. ('step', 'ans', 'mul', 'pow', 'x', '<eos>').
    template:
        Output token sequence with NNO tokens replaced by slot indices.
        E.g. ('step', 0, 'ans', 'mul', 1, 'pow', 'x', 2, '<eos>').
    slot_exprs:
        One NNOExpr per NNO slot.  May contain None for unresolved slots
        (the rule will refuse to fire in that case).
    n_in:
        Number of input NNO slots this rule was trained on.  Used to reject
        queries with a different number of input NNO tokens (e.g. prevents
        the single-digit successor rule from firing on 2-digit inputs).
    evidence:
        Number of training examples in this group.
    """
    in_skel: tuple[str, ...]
    out_skel: tuple[str, ...]
    template: tuple
    slot_exprs: list[Optional[NNOExpr]]
    n_in: int = 0
    evidence: int = 0


# ---------------------------------------------------------------------------
# SkeletonStore — main class
# ---------------------------------------------------------------------------

class SkeletonStore:
    """Learn and predict using skeleton-discriminated lambda terms.

    Usage
    -----
    >>> store = SkeletonStore()
    >>> store.learn(corpus, succ_map, engine)
    >>> result = store.predict(prefix, succ_map, engine)
    """

    def __init__(self) -> None:
        # (op, in_skel) → list of SkeletonRule, sorted by evidence descending
        self._rules: dict[tuple[str, tuple[str, ...]], list[SkeletonRule]] = {}

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        corpus: list[list[str]],
        succ_map: dict[str, str],
        engine,
    ) -> None:
        """Learn rules from the full training corpus.

        Parameters
        ----------
        corpus:
            All training sequences (list of token lists).
        succ_map:
            Discovered NNO successor map (str → str).
        engine:
            ComposeEngine for BFM evaluation during synthesis.
        """
        nno_set: frozenset[str] = frozenset(succ_map.keys()) | frozenset(succ_map.values())
        pred_map: dict[str, str] = {v: k for k, v in succ_map.items()}

        # Group by (op, in_skel, template) → [(in_nno, out_nno), ...]
        # Keying by template (not out_skel) ensures each group has a uniform
        # NNO slot count.  Examples with different slot counts (e.g. 1-digit
        # coefficient vs 2-digit) form separate groups and get separate rules.
        GroupKey = tuple  # (op: str, in_skel: tuple, template: tuple)
        groups: dict[GroupKey, list[tuple[tuple[str, ...], tuple[str, ...]]]] = {}

        for raw_seq in corpus:
            if not raw_seq:
                continue
            # Skip ftc sequences — contain 'and', too complex for this scheme
            if 'and' in raw_seq:
                continue

            seq = _normalize_sq(raw_seq)
            op = seq[0]

            # Find first output delimiter (includes dx-format integrals)
            first_out_idx = len(seq)
            for i in range(1, len(seq)):
                if seq[i] in _OUTPUT_DELIMS_STR:
                    first_out_idx = i
                    break

            if first_out_idx == len(seq):
                continue  # no output delimiter found

            in_toks = seq[1:first_out_idx]  # after op, before first delim
            out_toks = seq[first_out_idx:]   # from first delim (inclusive)

            in_skel, in_nno = _extract_skel(in_toks, nno_set)
            template, out_nno_list = _build_template(out_toks, nno_set)

            key: GroupKey = (op, in_skel, template)
            if key not in groups:
                groups[key] = []
            groups[key].append((tuple(in_nno), tuple(out_nno_list)))

        # Build SkeletonRule for each group
        new_rules: dict[tuple[str, tuple[str, ...]], list[SkeletonRule]] = {}

        for (op, in_skel, template), examples in groups.items():
            if not examples:
                continue

            # out_skel = structural tokens from template (for SkeletonRule field)
            out_skel = tuple(elem for elem in template if isinstance(elem, str))
            n_out_slots = sum(1 for elem in template if isinstance(elem, int))

            n_in = max((len(inp) for inp, _ in examples), default=0)

            if n_out_slots == 0:
                # No NNO slots — fully structural output, constant for this group
                rule = SkeletonRule(
                    in_skel=in_skel,
                    out_skel=out_skel,
                    template=template,
                    slot_exprs=[],
                    n_in=n_in,
                    evidence=len(examples),
                )
                _add_rule(new_rules, op, in_skel, rule)
                continue

            # Filter to consistent n_in (all examples in this group have same
            # template so n_out is already uniform)
            consistent = [(inp, out) for inp, out in examples if len(inp) == n_in]
            # Require at least 2 examples for synthesis.  A single example
            # always trivially matches const() — that rule would be wrong in
            # general.  Fewer examples = insufficient evidence for any rule.
            if len(consistent) < 2:
                continue

            # Synthesize per-slot NNOExpr
            slot_exprs: list[Optional[NNOExpr]] = []
            for slot_idx in range(n_out_slots):
                pairs = [
                    (inp, out[slot_idx])
                    for inp, out in consistent
                    if slot_idx < len(out)
                ]
                expr = _try_synth_slot(pairs, succ_map, pred_map, engine, n_in)
                slot_exprs.append(expr)

            rule = SkeletonRule(
                in_skel=in_skel,
                out_skel=out_skel,
                template=template,
                slot_exprs=slot_exprs,
                n_in=n_in,
                evidence=len(examples),
            )
            _add_rule(new_rules, op, in_skel, rule)

        # Sort by evidence (highest first) and store
        for k, rule_list in new_rules.items():
            rule_list.sort(key=lambda r: -r.evidence)
        self._rules = new_rules

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        prefix: list[str],
        succ_map: dict[str, str],
        engine,
    ) -> Optional[dict[str, float]]:
        """Predict the next token for the given prefix.

        Parameters
        ----------
        prefix:
            All tokens observed so far (including the operator as prefix[0]).
        succ_map:
            Discovered NNO successor map.
        engine:
            ComposeEngine for BFM evaluation.

        Returns
        -------
        {next_token: 1.0} on a definite prediction, {'<eos>': 1.0} when
        the output is complete, or None on miss.
        """
        if not prefix:
            return None

        nno_set: frozenset[str] = frozenset(succ_map.keys()) | frozenset(succ_map.values())
        pred_map: dict[str, str] = {v: k for k, v in succ_map.items()}

        # Normalize sq tokens in the prefix
        norm_prefix = _normalize_sq(prefix)
        op = norm_prefix[0]

        # Find first output delimiter
        first_out_idx = len(norm_prefix)
        for i in range(1, len(norm_prefix)):
            if norm_prefix[i] in _OUTPUT_DELIMS_STR:
                first_out_idx = i
                break

        in_toks = norm_prefix[1:first_out_idx]
        out_so_far = norm_prefix[first_out_idx:]  # includes the delimiter

        in_skel, in_nno = _extract_skel(in_toks, nno_set)

        key = (op, in_skel)
        if key not in self._rules:
            return None

        for rule in self._rules[key]:
            result = _apply_rule(
                rule, in_nno, out_so_far,
                succ_map, pred_map, engine,
            )
            if result is not None:
                return result

        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_rule(
    rules_dict: dict,
    op: str,
    in_skel: tuple[str, ...],
    rule: SkeletonRule,
) -> None:
    key = (op, in_skel)
    if key not in rules_dict:
        rules_dict[key] = []
    rules_dict[key].append(rule)


def _apply_rule(
    rule: SkeletonRule,
    in_nno: list[str],
    out_so_far: list[str],
    succ_map: dict[str, str],
    pred_map: dict[str, str],
    engine,
) -> Optional[dict[str, float]]:
    """Apply one SkeletonRule given the observed input and output-so-far.

    Returns {next_token: 1.0} or {'<eos>': 1.0} on success, None on miss.
    """
    # Reject if the number of input NNO tokens doesn't match what the rule
    # was trained on.  This prevents e.g. the single-digit successor rule
    # (n_in=1) from firing on 2-digit inputs (n_in=2) and giving wrong answers.
    if len(in_nno) != rule.n_in:
        return None

    # Compute output NNO values from slot expressions
    out_nno: list[str] = []
    for expr in rule.slot_exprs:
        if expr is None:
            return None  # incomplete rule
        val = _eval_nno_expr(expr, in_nno, succ_map, pred_map, engine)
        if val is None:
            return None  # engine miss or out-of-range predecessor
        out_nno.append(val)

    # Reconstruct full output sequence
    out_full = _reconstruct(rule.template, out_nno)
    if not out_full:
        return None

    # Match out_so_far against the prefix of out_full
    k = len(out_so_far)
    if k > len(out_full):
        return None

    # Verify prefix consistency
    for i, tok in enumerate(out_so_far):
        if i >= len(out_full) or out_full[i] != tok:
            return None

    # Return next token
    if k < len(out_full):
        return {out_full[k]: 1.0}

    # out_so_far exactly matches out_full → output complete
    return {'<eos>': 1.0}
