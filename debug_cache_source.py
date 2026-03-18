"""Find the source of wrong cache entry ('add','3','5','5')."""
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
    build_unary_carry_maps, complete_succ_map,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import _compose, _lz_strip

EOS = "<eos>"
def _strip_eos(seq): return seq[:-1] if seq and seq[-1] == EOS else seq

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

TARGET_KEY = ('add', '3', '5', '5')

# Monkey-patch the cache dict to track writes
class TrackedDict(dict):
    def __setitem__(self, key, value):
        if key == TARGET_KEY:
            import traceback
            print(f"\n=== WRITE to {key} = {value} ===")
            traceback.print_stack(limit=6)
        super().__setitem__(key, value)

# Run _discover_one_arity for cs3/cs4 with tracked cache
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import discover_compose_chains
from experiments.symbolic_ai_v2.ctkg.inference.predict import _discover_one_arity
from collections import Counter

chain_rules_list = discover_compose_chains(train_all)
chain_rules_dict = {cr.op_atom: cr for cr in chain_rules_list}

tracked_cache = TrackedDict()
programs = {}

print("Running _discover_one_arity for cs1-cs4 with tracked cache...")
for op_atom in ['cs1', 'cs2', 'cs3', 'cs4']:
    cr = chain_rules_dict[op_atom]
    parsed_1step = []
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
    if len(parsed_1step) >= 3:
        n_counts = Counter(len(s[0]) for s in parsed_1step)
        for n, n_count in n_counts.items():
            if n < 2 or n_count < 3:
                continue
            print(f"  Running {op_atom} n={n} ({n_count} samples)...")
            _discover_one_arity(op_atom, n, [s for s in parsed_1step if len(s[0]) == n],
                               fold_rules, fc_lookup, succ_map, carry_el, carry_out, zero_digit,
                               tracked_cache, programs)
            if TARGET_KEY in tracked_cache:
                print(f"  After {op_atom} n={n}: key present = {tracked_cache[TARGET_KEY]}")
                break
    if TARGET_KEY in tracked_cache:
        break

if TARGET_KEY not in tracked_cache:
    print(f"\nKey NOT found after cs1-cs4!")
    print("Checking if it gets written during eval synthesis...")
    _discover_one_arity('eval', 3, None, fold_rules, fc_lookup, succ_map, carry_el, carry_out, zero_digit, tracked_cache, programs)

print(f"\nFinal cache[{TARGET_KEY}] = {tracked_cache.get(TARGET_KEY)}")
print(f"Correct value: {_compose('add', (('3','5'),('5',)), fc_lookup, fold_rules, succ_map, carry_el, carry_out, zero_digit, {})}")
