"""Debug: trace exactly where eval synthesis fails in _discover_trace_programs."""
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
    complete_succ_map, build_adj_lookup,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import _compose, _lz_strip, _zpad
import collections

EOS = "<eos>"

def _strip_eos(seq):
    return seq[:-1] if seq and seq[-1] == EOS else seq

def build_train_all():
    LEVELS = [
        successor_seqs, addition_seqs, subtraction_seqs, multiplication_seqs,
        power_seqs, linear_eval_seqs, derivative_seqs, integral_seqs,
        power_trace_seqs, linear_eval_trace_seqs, algebra_trace_seqs,
        conservation_scenario_seqs, bernoulli_trace_seqs,
        derivative_trace_seqs, integral_trace_seqs,
    ]
    train_all = []
    for fn in LEVELS:
        train, _ = fn()
        train_all.extend([_strip_eos(s) for s in train])
    return train_all

print("Building corpus...")
train_all = build_train_all()
print(f"  {len(train_all)} train sequences")

print("Building free category...")
fc = build_free_category(train_all)
fc_lookup = build_fc_lookup(fc)
fold_rules = discover_binary_fold_rules(fc)
print(f"  FC edges: {len(fc.edges)}, fold_rules: {list(fold_rules.keys())}")

nno_candidates = fc.nno_candidates
best_nno = max(nno_candidates, key=lambda n: len(n.successor_map)) if nno_candidates else None
succ_carry = build_unary_carry_maps(fc)
succ_map = {}; carry_el = ""; carry_out = (); zero_digit = ""
if best_nno and best_nno.op in succ_carry:
    succ_map = complete_succ_map(best_nno.successor_map, best_nno.zero_candidate, succ_carry[best_nno.op][0])
    carry_el = succ_carry[best_nno.op][0]
    carry_out = succ_carry[best_nno.op][1]
    zero_digit = best_nno.zero_candidate
print(f"  zero_digit={zero_digit!r}, succ_map sample: {dict(list(succ_map.items())[:5])}")

print("Building chain rules...")
chain_rules_list = discover_compose_chains(train_all)
chain_rules = {cr.op_atom: cr for cr in chain_rules_list}
print(f"  Chain rules: {sorted(chain_rules.keys())}")

# Now specifically test eval
cr_eval = chain_rules.get('eval')
if cr_eval is None:
    print("ERROR: no chain rule for 'eval'")
    sys.exit(1)

print(f"\nEval chain_table size: {len(cr_eval.chain_table)}")

# Parse eval samples
parsed_1step = []
for input_key, output_toks in cr_eval.chain_table.items():
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

from collections import Counter
n_counts = Counter(len(s[0]) for s in parsed_1step)
print(f"Eval parsed_1step: {len(parsed_1step)}, n_counts: {dict(n_counts)}")

# Test with n=3
parsed_n3 = [s for s in parsed_1step if len(s[0]) == 3]
print(f"  n=3 samples: {len(parsed_n3)}")

# Show first few
for a_ops, step, ans in parsed_n3[:3]:
    print(f"  a_ops={a_ops}, step={step}, ans={ans}")

# Test step synthesis manually
cache = {}
print(f"\nTesting step synthesis for eval n=3:")
print(f"fold_rules: {list(fold_rules.keys())}")

for fop in fold_rules:
    for i in range(3):
        for j in range(3):
            ev = sum(
                1 for a_ops, step, _ in parsed_n3
                if (lambda r, e: r is not None and _lz_strip(r) == _lz_strip(e))(
                    _compose(fop, ((a_ops[i],), (a_ops[j],)), fc_lookup, fold_rules,
                             succ_map, carry_el, carry_out, zero_digit, cache),
                    step
                )
            )
            if ev > 0:
                print(f"  {fop}(i={i},j={j}): ev={ev}/{len(parsed_n3)}", "FOUND!" if ev == len(parsed_n3) else "")

# Test ans synthesis if step found
print("\nAssuming step=mul(0,2) found, testing ans synthesis:")
step_op, step_a0_idx, step_a1_idx = 'mul', (0,), (2,)
step_results = []
for a_ops, _, _ in parsed_n3:
    a0 = tuple(a_ops[i] for i in step_a0_idx)
    a1 = tuple(a_ops[i] for i in step_a1_idx)
    sr = _compose(step_op, (a0, a1), fc_lookup, fold_rules, succ_map, carry_el, carry_out, zero_digit, cache)
    step_results.append(sr)

none_count = sum(1 for sr in step_results if sr is None)
print(f"  Step results: {len(step_results)} total, {none_count} None")

def step_matches(result, expected):
    if result is None:
        return False
    return _lz_strip(result) == _lz_strip(expected)

for fop in fold_rules:
    for k in range(3):
        # step first
        ev = sum(
            1 for (a_ops, _, ans), sr in zip(parsed_n3, step_results)
            if sr is not None and step_matches(
                _compose(fop, (sr, (a_ops[k],)), fc_lookup, fold_rules,
                         succ_map, carry_el, carry_out, zero_digit, cache),
                ans
            )
        )
        if ev > 0:
            print(f"  fold({fop}, step_first, k={k}): ev={ev}/{len(parsed_n3)}", "FOUND!" if ev==len(parsed_n3) else "")
        # other first
        ev = sum(
            1 for (a_ops, _, ans), sr in zip(parsed_n3, step_results)
            if sr is not None and step_matches(
                _compose(fop, ((a_ops[k],), sr), fc_lookup, fold_rules,
                         succ_map, carry_el, carry_out, zero_digit, cache),
                ans
            )
        )
        if ev > 0:
            print(f"  fold({fop}, other_first, k={k}): ev={ev}/{len(parsed_n3)}", "FOUND!" if ev==len(parsed_n3) else "")

# Show any failures for the best candidate
print("\nSample failures for add(step, k=1):")
failures = []
for (a_ops, _, ans), sr in zip(parsed_n3, step_results):
    if sr is None:
        failures.append(f"  sr=None for a_ops={a_ops}, ans={ans}")
        continue
    result = _compose('add', (sr, (a_ops[1],)), fc_lookup, fold_rules, succ_map, carry_el, carry_out, zero_digit, cache)
    if result is None or _lz_strip(result) != _lz_strip(ans):
        failures.append(f"  a_ops={a_ops}, sr={sr}, a_ops[1]={a_ops[1]}, got={result}, want={ans}")
for f in failures[:5]:
    print(f)
print(f"  Total failures: {len(failures)}")
