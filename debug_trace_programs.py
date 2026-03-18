"""Debug: trace _discover_trace_programs for eval — find why added=0."""
import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__))))

from experiments.symbolic_ai_v2.corpus.math_generator import (
    successor_seqs, addition_seqs, subtraction_seqs, multiplication_seqs,
    power_seqs, linear_eval_seqs, derivative_seqs, integral_seqs,
    power_trace_seqs, linear_eval_trace_seqs, algebra_trace_seqs,
    conservation_scenario_seqs, bernoulli_trace_seqs,
    derivative_trace_seqs, integral_trace_seqs,
)
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    build_free_category, build_fc_lookup, discover_binary_fold_rules,
    discover_compose_chains, build_unary_chain_maps, build_unary_carry_maps,
    complete_succ_map,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import (
    _compose, _lz_strip, _zpad, _discover_trace_programs, _discover_one_arity,
)
from collections import Counter

EOS = "<eos>"
def _strip_eos(seq):
    return seq[:-1] if seq and seq[-1] == EOS else seq

LEVELS = [
    successor_seqs, addition_seqs, subtraction_seqs, multiplication_seqs,
    power_seqs, linear_eval_seqs, derivative_seqs, integral_seqs,
    power_trace_seqs, linear_eval_trace_seqs, algebra_trace_seqs,
    conservation_scenario_seqs, bernoulli_trace_seqs,
    derivative_trace_seqs, integral_trace_seqs,
]
train_all = []
for fn in LEVELS:
    t, _ = fn()
    train_all.extend([_strip_eos(s) for s in t])

fc = build_free_category(train_all)
fc_lookup = build_fc_lookup(fc)
fold_rules = discover_binary_fold_rules(fc)

nno_candidates = fc.nno_candidates
best_nno = max(nno_candidates, key=lambda n: len(n.successor_map)) if nno_candidates else None
succ_carry = build_unary_carry_maps(fc)
succ_map = {}; carry_el = ""; carry_out = (); zero_digit = ""
if best_nno and best_nno.op in succ_carry:
    succ_map = complete_succ_map(best_nno.successor_map, best_nno.zero_candidate, succ_carry[best_nno.op][0])
    carry_el = succ_carry[best_nno.op][0]
    carry_out = succ_carry[best_nno.op][1]
    zero_digit = best_nno.zero_candidate

chain_rules_list = discover_compose_chains(train_all)
chain_rules = {cr.op_atom: cr for cr in chain_rules_list}

print(f"Processing order: {sorted(chain_rules.keys())}")

# Run _discover_trace_programs step by step, printing for eval
shared_cache = {}
programs = {}

for op_atom in sorted(chain_rules.keys()):
    cr = chain_rules[op_atom]
    if not cr.chain_table:
        continue

    parsed_1step = []
    parsed_2step = []

    for input_key, output_toks in cr.chain_table.items():
        if "ans" not in output_toks:
            continue
        ans_idx = output_toks.index("ans")
        ans_toks = tuple(output_toks[ans_idx + 1:])
        step_positions = [i for i, t in enumerate(output_toks[:ans_idx]) if t == "step"]
        a_ops = [tok[1:] for tok in input_key if tok.startswith("a") and len(tok) > 1 and tok[1:].isdigit()]
        if len(a_ops) < 2 or not ans_toks:
            continue
        if len(step_positions) == 1:
            step_toks = tuple(output_toks[step_positions[0] + 1:ans_idx])
            if step_toks:
                parsed_1step.append((a_ops, step_toks, ans_toks))
        elif len(step_positions) == 2:
            s1 = step_positions[0]; s2 = step_positions[1]
            s1_toks = tuple(output_toks[s1+1:s2]); s2_toks = tuple(output_toks[s2+1:ans_idx])
            if s1_toks and s2_toks:
                parsed_2step.append((a_ops, s1_toks, s2_toks, ans_toks))

    before_cache_size = len(shared_cache)
    before_programs = dict(programs)

    if len(parsed_1step) >= 3:
        n_counts = Counter(len(s[0]) for s in parsed_1step)
        for n, n_count in n_counts.items():
            if n < 2 or n_count < 3:
                continue
            _discover_one_arity(
                op_atom, n,
                [s for s in parsed_1step if len(s[0]) == n],
                fold_rules, fc_lookup, succ_map, carry_el, carry_out, zero_digit,
                shared_cache, programs,
            )

    added = len(programs) - len(before_programs)
    cache_added = len(shared_cache) - before_cache_size

    print(f"{op_atom}: 1step={len(parsed_1step)} 2step={len(parsed_2step)} "
          f"cache_before={before_cache_size} cache_after={len(shared_cache)} "
          f"added={added}")

    if op_atom == 'eval':
        # Debug: test step synthesis for eval n=3 with current cache state
        parsed_n3 = [s for s in parsed_1step if len(s[0]) == 3]
        print(f"  === EVAL n=3 debug ===")
        print(f"  Cache size: {len(shared_cache)}")
        for fop in fold_rules:
            for i in range(3):
                for j in range(3):
                    ev = sum(
                        1 for a_ops, step, _ in parsed_n3
                        if (lambda r, e: r is not None and _lz_strip(r) == _lz_strip(e))(
                            _compose(fop, ((a_ops[i],), (a_ops[j],)), fc_lookup, fold_rules,
                                     succ_map, carry_el, carry_out, zero_digit, shared_cache),
                            step
                        )
                    )
                    if ev == len(parsed_n3):
                        print(f"  STEP FOUND: {fop}(i={i},j={j}) ev={ev}")
                    elif ev > 200:
                        print(f"  NEAR: {fop}(i={i},j={j}) ev={ev}")

print(f"\nFinal programs: {list(programs.keys())}")
