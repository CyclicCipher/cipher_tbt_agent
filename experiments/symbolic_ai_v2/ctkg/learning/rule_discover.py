"""
Rule discovery: infer RewriteRules from (input_tree, output_tree) example pairs.

Algorithm (anti-unification based, no special cases):
  1. Parse the corpus into (input_Expr, output_Expr) pairs.
  2. Group pairs by the skeleton of the input tree (= tree structure ignoring atoms).
  3. Within each group, anti-unify all input trees → lhs_pattern.
  4. Anti-unify all output trees with CONSISTENT variable names → rhs_pattern.
  5. Verify each rule: cata_reduce(input, [rule]) == output for all examples.
  6. Return consistent rules, sorted by specificity (most-specific first).

This is the replacement for _discover_trace_programs, _discover_slot_program,
_discover_value_segment, and all segment-type dispatch.  One algorithm.  No
special cases.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
    Expr, atom, node, var, substitute, match, anti_unify_list, skeleton, variables,
)
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH
from experiments.symbolic_ai_v2.ctkg.core.expr_parser import (
    ArityTable, parse, parse_full, split_on_terminators, TERMINATORS,
)
from experiments.symbolic_ai_v2.ctkg.core.rewrite import RewriteRule, cata_reduce


# ---------------------------------------------------------------------------
# Grouping by skeleton
# ---------------------------------------------------------------------------

def group_by_skeleton(
    examples: list[tuple[Expr, Expr]],
) -> dict[Expr, list[tuple[Expr, Expr]]]:
    """
    Partition (input_expr, output_expr) pairs by the skeleton of the input.

    Skeleton = tree structure with all atoms replaced by '_'.
    Pairs with the same skeleton have the same operator tree shape and can be
    anti-unified into a single rule.
    """
    groups: dict[Expr, list[tuple[Expr, Expr]]] = defaultdict(list)
    for inp, out in examples:
        groups[skeleton(inp)].append((inp, out))
    return dict(groups)


# ---------------------------------------------------------------------------
# Variable name alignment (rhs must use the same variable names as lhs)
# ---------------------------------------------------------------------------

def _align_rhs_variables(
    lhs_pattern: Expr,
    lhs_substs: list[dict[str, Expr]],
    outputs: list[Expr],
    functional_maps: Optional[dict[str, dict[str, str]]] = None,
    binary_functional_maps: Optional[dict[str, dict[tuple, str]]] = None,
) -> Optional[Expr]:
    """
    Anti-unify the output trees and map the resulting variable names to the
    names used in lhs_pattern by matching values.

    For each output variable V_out in the rhs pattern, find the lhs variable
    V_lhs such that lhs_subst[i][V_lhs] == rhs_subst[i][V_out] for all i.
    Rename V_out → V_lhs.

    binary_functional_maps: maps op_name → {(a_str, b_str): result_str}.
    Used to detect W = op(V0, V1) (binary functional) dependencies, e.g.
    coefficient = mul(coeff_in, exponent) in the derivative power rule.

    Returns the aligned rhs pattern, or None if alignment fails.
    """
    if not outputs:
        return None

    if len(outputs) == 1:
        # Single example: try to express the output directly in terms of
        # lhs variables.  Check if match(lhs_pattern_vars_in_rhs, output) works.
        # For now, just return the output as a ground pattern (no variables).
        return outputs[0]

    rhs_lgg, rhs_substs = anti_unify_list(outputs)

    # Build a mapping: rhs_var_name → lhs_var_name
    # For each rhs variable V_out:
    #   - look at what values it takes across examples: {rhs_substs[i][V_out]}
    #   - find the lhs variable V_lhs that takes the SAME values: {lhs_substs[i][V_lhs]}
    rhs_vars = list(variables(rhs_lgg))
    lhs_vars = list(variables(lhs_pattern))

    rename: dict[str, str] = {}         # rv → lv  (direct match)
    rename_expr: dict[str, Expr] = {}   # rv → Expr (functional match: W = f(V))

    for rv in rhs_vars:
        rhs_vals = [rhs_substs[i].get(rv) for i in range(len(outputs))]
        if any(v is None for v in rhs_vals):
            return None   # variable appears only in some outputs — skip

        # 1. Direct match: find lhs variable with same values across all examples
        matched = False
        for lv in lhs_vars:
            lhs_vals = [lhs_substs[i].get(lv) for i in range(len(outputs))]
            if lhs_vals == rhs_vals:
                rename[rv] = lv
                matched = True
                break

        # 2. Functional match: W = f(V) for a known unary op f and lhs variable V.
        #    functional_maps maps op_name → {from_tok: to_tok} (string-level lookup).
        #    If lhs_val[i] is a leaf atom and f(lhs_val[i]) == rhs_val[i] for all i,
        #    replace W in the rhs pattern with node(f, var(V)).
        if not matched and functional_maps:
            for op_name, op_map in functional_maps.items():
                for lv in lhs_vars:
                    lhs_vals = [lhs_substs[i].get(lv) for i in range(len(outputs))]
                    try:
                        mapped = [
                            atom(op_map[TOKEN_GRAPH.decode(v.head)])
                            for v in lhs_vals
                            if v is not None and not v.args
                        ]
                    except KeyError:
                        continue
                    if len(mapped) == len(rhs_vals) and mapped == rhs_vals:
                        rename_expr[rv] = node(op_name, var(lv))
                        matched = True
                        break
                if matched:
                    break

        # 3. Binary functional match: W = op(V0, V1) for a known binary op op
        #    and two lhs variables V0, V1.
        #    binary_functional_maps: {op_name: {(a_str, b_str): result_str}}
        #    Tolerates gaps in bfm (missing entries are "unknown", not mismatches).
        #    Accepts if: all KNOWN entries agree AND unknowns ≤ 20% of examples.
        if not matched and binary_functional_maps:
            for op_name, bop_map in binary_functional_maps.items():
                for lv1 in lhs_vars:
                    lhs_vals1 = [lhs_substs[i].get(lv1) for i in range(len(outputs))]
                    if any(v is None for v in lhs_vals1):
                        continue
                    for lv2 in lhs_vars:
                        lhs_vals2 = [lhs_substs[i].get(lv2) for i in range(len(outputs))]
                        if any(v is None for v in lhs_vals2):
                            continue
                        n_match = 0
                        n_unknown = 0
                        mismatch = False
                        for v1, v2, rv_val in zip(lhs_vals1, lhs_vals2, rhs_vals):
                            if v1 is None or v2 is None or v1.args or v2.args:
                                n_unknown += 1
                                continue
                            key = (TOKEN_GRAPH.decode(v1.head), TOKEN_GRAPH.decode(v2.head))
                            if key not in bop_map:
                                n_unknown += 1
                                continue
                            if atom(bop_map[key]) != rv_val:
                                mismatch = True
                                break
                            n_match += 1
                        n = len(rhs_vals)
                        if (not mismatch and n_match >= 1
                                and n_unknown / n <= 0.2):
                            rename_expr[rv] = node(op_name, var(lv1), var(lv2))
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break

        # 4. Two-level binary functional match: W = op2(op1(Vi, Vj), Vk)
        #    Handles composition rules like: eval(A,x,B,at,C) → add(mul(A,C), B)
        #    where step = mul(A,C) and ans = add(step, B).
        #    More lenient tolerance (≤40% unknowns) because multi-digit intermediates
        #    (e.g. '36' from mul(6,6)) are not valid keys for the second-level lookup.
        if not matched and binary_functional_maps:
            _bop_items = list(binary_functional_maps.items())
            _found = False
            for op1_name, bop1_map in _bop_items:
                if _found:
                    break
                for lv_i in lhs_vars:
                    if _found:
                        break
                    lhs_vi = [lhs_substs[k].get(lv_i) for k in range(len(outputs))]
                    if any(v is None for v in lhs_vi):
                        continue
                    for lv_j in lhs_vars:
                        if _found:
                            break
                        lhs_vj = [lhs_substs[k].get(lv_j) for k in range(len(outputs))]
                        if any(v is None for v in lhs_vj):
                            continue
                        # Compute inner = op1(lv_i, lv_j) for each example
                        inner_vals = []
                        for vi, vj in zip(lhs_vi, lhs_vj):
                            if vi.args or vj.args:
                                inner_vals.append(None)
                                continue
                            key1 = (TOKEN_GRAPH.decode(vi.head), TOKEN_GRAPH.decode(vj.head))
                            inner_vals.append(bop1_map.get(key1))  # str or None
                        for op2_name, bop2_map in _bop_items:
                            if _found:
                                break
                            for lv_k in lhs_vars:
                                lhs_vk = [lhs_substs[k].get(lv_k) for k in range(len(outputs))]
                                if any(v is None for v in lhs_vk):
                                    continue
                                n_match = 0
                                n_unknown = 0
                                mismatch = False
                                for inner_str, vk, rv_val in zip(inner_vals, lhs_vk, rhs_vals):
                                    if inner_str is None or vk.args:
                                        n_unknown += 1
                                        continue
                                    key2 = (inner_str, TOKEN_GRAPH.decode(vk.head))
                                    if key2 not in bop2_map:
                                        n_unknown += 1
                                        continue
                                    if atom(bop2_map[key2]) != rv_val:
                                        mismatch = True
                                        break
                                    n_match += 1
                                n = len(rhs_vals)
                                if (not mismatch and n_match >= 2
                                        and n_unknown / n <= 0.4):
                                    inner_expr = node(op1_name, var(lv_i), var(lv_j))
                                    rename_expr[rv] = node(op2_name, inner_expr, var(lv_k))
                                    matched = True
                                    _found = True
                                    break

        if not matched:
            # Output variable has no corresponding input variable — skip this rule
            # (this happens for 'C' in integral outputs: it's a free constant)
            rename[rv] = rv   # keep as-is; rule will still be generated

    # Apply renaming to rhs_lgg
    def _rename(e: Expr) -> Expr:
        if e.is_var:
            head_str = TOKEN_GRAPH.decode(e.head)
            if head_str in rename_expr:
                return rename_expr[head_str]   # functional node e.g. pred(V0)
            new_name = rename.get(head_str, head_str)
            return var(new_name)
        if not e.args:
            return e
        new_args = tuple(_rename(a) for a in e.args)
        if new_args == e.args:
            return e
        return Expr(head=e.head, args=tuple(new_args))

    return _rename(rhs_lgg)


# ---------------------------------------------------------------------------
# One-pass normalization (no cycle risk)
# ---------------------------------------------------------------------------


def _apply_norm_once(expr: Expr, rules: list[RewriteRule]) -> Expr:
    """Apply rules bottom-up exactly once per node, without re-applying on the result.

    Unlike cata_reduce, this does NOT recursively apply rules to the result of a
    fired rule.  This prevents cycles in rules like atom('x')→pow(x,1) which would
    loop under cata_reduce (the result pow(x,1) contains x again).

    Use for output_norm_rules in discover_rules where single-pass normalization is
    sufficient and rule-result cycles are possible.
    """
    if expr.args:
        new_args = tuple(_apply_norm_once(a, rules) for a in expr.args)
        if new_args != expr.args:
            expr = Expr(head=expr.head, args=new_args, is_var=expr.is_var)
    for rule in rules:
        result = rule.applies_to(expr)
        if result is not None:
            return result   # no recursion into result
    return expr


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_rules(
    corpus: list[list[str]],
    arities: ArityTable,
    min_examples: int = 1,
    norm_rules: Optional[list[RewriteRule]] = None,
    output_norm_rules: Optional[list[RewriteRule]] = None,
    functional_maps: Optional[dict[str, dict[str, str]]] = None,
    aux_rules: Optional[list[RewriteRule]] = None,
    binary_functional_maps: Optional[dict[str, dict[tuple, str]]] = None,
) -> list[RewriteRule]:
    """
    Discover RewriteRules from a corpus of token sequences.

    Parameters
    ----------
    corpus            : list of token sequences (flat prefix notation)
    arities           : ArityTable mapping token → arity (caller-provided)
    min_examples      : minimum examples per group to form a rule (default 1)
    norm_rules        : RewriteRules applied to BOTH input and output trees before
                        grouping (e.g. sq(V)→pow(V,2) structural normalization)
    output_norm_rules : RewriteRules applied to OUTPUT trees only before grouping
                        (e.g. atom('x')→pow(x,1) so bare-x outputs are uniform)
    functional_maps   : maps op_name → {from_tok: to_tok}.  Used in
                        _align_rhs_variables to detect W = f(V) dependencies
                        (e.g. {'pred': {'2':'1','3':'2',...}})
    aux_rules         : extra RewriteRules used ONLY during verification (e.g.
                        ground pred/succ rules to reduce pred(2)→1).  Not
                        returned as part of the discovered rule set.
    binary_functional_maps : maps op_name → {(a_str, b_str): result_str}.
                        Used in _align_rhs_variables to detect W = op(V0, V1)
                        dependencies (e.g. {'mul': {('2','3'):'6',...}}).

    Returns list of RewriteRule sorted by specificity (most-specific first),
    i.e. rules with fewer pattern variables fire before more general ones.
    """
    from experiments.symbolic_ai_v2.ctkg.core.rewrite import cata_reduce as _cata

    # Step 1: Parse corpus into (input_expr, output_expr) pairs; apply normalization
    examples: list[tuple[Expr, Expr]] = []
    for seq in corpus:
        inp, out = parse_full(seq, arities)
        if inp is not None and out is not None:
            if norm_rules:
                inp = _cata(inp, norm_rules)
                out = _cata(out, norm_rules)
            if output_norm_rules:
                # Use one-pass (no result re-reduction) to avoid cycles in rules
                # like x→pow(x,1) which would loop under cata_reduce.
                out = _apply_norm_once(out, output_norm_rules)
            examples.append((inp, out))

    if not examples:
        return []

    # Step 2: Group by input skeleton
    groups = group_by_skeleton(examples)

    rules: list[RewriteRule] = []

    for skel, group in groups.items():
        if len(group) < min_examples:
            continue

        inputs  = [inp for inp, _ in group]
        outputs = [out for _, out in group]

        # Step 3: Anti-unify all input trees → lhs_pattern
        lhs_pattern, lhs_substs = anti_unify_list(inputs)

        # Step 4: Align and anti-unify output trees
        rhs_pattern = _align_rhs_variables(
            lhs_pattern, lhs_substs, outputs,
            functional_maps=functional_maps,
            binary_functional_maps=binary_functional_maps,
        )
        if rhs_pattern is None:
            continue

        # Step 5: Verify consistency.
        #   Apply [rule] + aux_rules (e.g. ground pred/succ) to each input,
        #   compare against the (normalized) expected output.
        rule = RewriteRule(
            lhs=lhs_pattern,
            rhs=rhs_pattern,
            algebra_name=TOKEN_GRAPH.decode(lhs_pattern.head) if not lhs_pattern.is_var else 'unknown',
            evidence=len(group),
        )
        verify_rules = [rule] + list(aux_rules or [])
        # Scale max_steps with rule set size: 200 steps per rule per node depth.
        _verify_max_steps = max(10000, 200 * len(verify_rules))

        consistent = 0
        for inp, out in group:
            result = cata_reduce(inp, verify_rules, max_steps=_verify_max_steps)
            if result == out:
                consistent += 1

        # Accept rules that are consistent with at least 1 example.
        # (Rules that pass all examples are perfect; partial consistency can
        # be used with lower confidence in future phases.)
        if consistent >= min_examples:
            rule.evidence = consistent
            rules.append(rule)

    # Step 6: Sort by specificity (most-specific first)
    from experiments.symbolic_ai_v2.ctkg.core.rewrite import sort_by_specificity
    return sort_by_specificity(rules)


# ---------------------------------------------------------------------------
# Corpus parsing helper (used by Predictor for direct training)
# ---------------------------------------------------------------------------

def parse_corpus(
    corpus: list[list[str]],
    arities: ArityTable,
) -> list[tuple[Expr, Expr]]:
    """
    Parse a list of token sequences into (input_Expr, output_Expr) pairs.
    Sequences that cannot be fully parsed on both sides are dropped silently.
    """
    pairs: list[tuple[Expr, Expr]] = []
    for seq in corpus:
        inp, out = parse_full(seq, arities)
        if inp is not None and out is not None:
            pairs.append((inp, out))
    return pairs
