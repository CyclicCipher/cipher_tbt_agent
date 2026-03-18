"""Debug: find who wrote the wrong cache entry."""
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
    _compose, _lz_strip,
)

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

# Test 1: fresh cache - what does add(('3','5'),('5',)) give?
cache_fresh = {}
result_fresh = _compose('add', (('3','5'), ('5',)), fc_lookup, fold_rules,
                         succ_map, carry_el, carry_out, zero_digit, cache_fresh)
print(f"Fresh cache: add(('3','5'),('5',)) = {result_fresh}  (expected ('4','0'))")

# Test 2: first write ('3',) ('5','5') to cache, then test
cache_wrong = {}
result_wrong_order = _compose('add', (('3',), ('5','5')), fc_lookup, fold_rules,
                               succ_map, carry_el, carry_out, zero_digit, cache_wrong)
print(f"add(('3',),('5','5')) = {result_wrong_order}  (3+55=58? or...)")

# Now try the same key with the other interpretation
result_from_cache = _compose('add', (('3','5'), ('5',)), fc_lookup, fold_rules,
                              succ_map, carry_el, carry_out, zero_digit, cache_wrong)
print(f"After caching add(('3',),('5','5')), add(('3','5'),('5',)) = {result_from_cache}  (cache hit!)")

# Test 3: look at the specific cache key
print(f"\nCache key ('add','3','5','5') is in cache_wrong: {('add','3','5','5') in cache_wrong}")
print(f"Value: {cache_wrong.get(('add','3','5','5'))}")

# Now verify what cs3 step synthesis calls that could produce ('add','3','5','5')
# cs3: a_ops = [A, B, '0', D] where A,B in 1-8, D in 1-15
# 2-token group trials:
# (A, B) as arg0, single j:
#   j=2: add((A,B), ('0',)) → flat (A,B,'0')
#   j=3: add((A,B), (D,)) → flat (A,B,D)
# (B,'0') as arg0, j not in {1,2}:
#   j=0: add((B,'0'), (A,)) → flat (B,'0',A)
#   j=3: add((B,'0'), (D,)) → flat (B,'0',D)
# ('0',D) as arg0, j not in {2,3}:
#   j=0: add(('0',D), (A,)) → flat ('0',D,A)
#   j=1: add(('0',D), (B,)) → flat ('0',D,B)

# For ('3','5','5') = (A,B,D) or (B,'0',A) or ...
# (A,B,D): flat (A,B,D) = ('3','5','5') → A=3, B=5, D=5
print("\nChecking cs3 trial add((A,B),(D,)) for A=3,B=5,D=5:")
test_cache = {}
r = _compose('add', (('3','5'), ('5',)), fc_lookup, fold_rules,
             succ_map, carry_el, carry_out, zero_digit, test_cache)
print(f"  Fresh: add(('3','5'),('5',)) = {r}")

# But also: does cs3 try add((A,),(B,D)) = add(('3',),('5','5'))?
# Answer: No! The 2-token group trials use:
#   Group (gi, gi+1) as arg0, j as arg1 (single token)
# So the second arg is always SINGLE token. But the first arg can be 2-token.
# So cs3 would call add((A,B),(D,)) = add((3,5),(5,)) which IS the same as eval!
# That would correctly cache ('3','5','5') → 35+5=40=('4','0').

# Wait! cs3 in "2-token group" synthesis calls add((A,B),(D,)) with:
# - First arg = (a_ops[0], a_ops[1]) = (A, B)
# - Second arg = (a_ops[3],) = (D,)
# This is add(('3','5'),('5',)) if A=3,B=5,D=5 → correctly computes 40 → ('4','0')
# So this SHOULD populate cache[('add','3','5','5')] = ('4','0') CORRECTLY!

# Then when eval does add(('3','5'),('5',)) → same key → gets ('4','0').

# But the actual eval result showed ('6','7')!
# There must be ANOTHER computation that overwrites the cache AFTER cs3.

# What about cs2?
# cs2: ['cs2'] + _arg_digits(a) + _arg_v2(c) + _arg_v2(d)
# a_ops = [A, '0', C, '0', D] where _arg_v2(c) = ['a0','ac'], _arg_v2(d) = ['a0','ad']
# n=5 for cs2
# "2-token group" single+group:
#   gi=0: group(A,'0'), j can be 2,3,4 → add((A,'0'),(C,)), add((A,'0'),('0',)), add((A,'0'),(D,))
#   gi=1: group('0',C), j can be 0,3,4 → add(('0',C),(A,)), add(('0',C),('0',)), add(('0',C),(D,))
#   ...
# "both 2-token groups":
#   gi=0,gj=2: add((A,'0'),('0',C)) → flat(A,'0','0',C)
#   gi=0,gj=3: add((A,'0'),('0',D)) → flat(A,'0','0',D)
#   gi=1,gj=3: add(('0',C),('0',D)) → flat('0',C,'0',D)
#   ...

# None of these produce flat ('3','5','5') unless A='3', '0','5' or...
# Actually for "j as arg0, group as arg1" in the single+group loop:
#   j=0, gi=1: add((A,),('0',C)) → flat(A,'0',C)
#   j=0, gi=2: add((A,),('0',D)) → same
# Hmm. Let me look at cs1 also.

# Actually, let me just check: what does cs1 with specific values do?
# cs1: ['cs1'] + _arg_digits(b) + _arg_v2(c) + _arg_v2(d)
# a_ops = [B, '0', C, '0', D] for n=5
# In "j as arg0, group as arg1":
#   j=0, gi=2: add((B,),('0',D)) → flat(B,'0',D)
#   But j=0, gi=0: add((B,),(B,'0')) – wait gi=0 means group(B,'0'), and j not in {0,1}, so j∈{2,3,4}
# Hmm, cs1 is similar to cs2.

# Let me approach differently: just check if the CORRECT value gets into cache
# and then gets OVERWRITTEN by something wrong.
# Run cs3 synthesis up to the specific sample and check cache state.

print("\nChecking cs3 synthesis and cache entries for key ('add','3','5','5'):")
chain_rules_list = discover_compose_chains(train_all)
chain_rules_dict = {cr.op_atom: cr for cr in chain_rules_list}

cr_cs3 = chain_rules_dict['cs3']
parsed_cs3 = []
for input_key, output_toks in cr_cs3.chain_table.items():
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
            parsed_cs3.append((a_ops, step_toks, ans_toks))

# Run cs3 step synthesis manually, tracking the specific cache entry
test_cache2 = {}
TARGET_KEY = ('add', '3', '5', '5')

# The step synthesis loop for cs3 n=4:
n = 4
parsed_n4 = [s for s in parsed_cs3 if len(s[0]) == 4]
print(f"cs3 n=4 samples: {len(parsed_n4)}")

# Regular single-token loop:
for fop in fold_rules:
    for i in range(n):
        for j in range(n):
            for a_ops, step, _ in parsed_n4:
                r = _compose(fop, ((a_ops[i],), (a_ops[j],)), fc_lookup, fold_rules,
                             succ_map, carry_el, carry_out, zero_digit, test_cache2)
                if TARGET_KEY in test_cache2:
                    val = test_cache2[TARGET_KEY]
                    print(f"  KEY SET by {fop}(i={i},j={j}) for a_ops={a_ops} → {val}")
                    # Verify correct value
                    correct = _compose('add', (('3','5'), ('5',)), fc_lookup, fold_rules,
                                      succ_map, carry_el, carry_out, zero_digit, {})
                    print(f"  Correct value (fresh): {correct}")
                    break
            if TARGET_KEY in test_cache2:
                break
        if TARGET_KEY in test_cache2:
            break
    if TARGET_KEY in test_cache2:
        break

if TARGET_KEY not in test_cache2:
    print(f"  Key NOT set during cs3 regular step synthesis")

# 2-token group loop for cs3:
test_cache3 = {}
for fop in fold_rules:
    for gi in range(n - 1):
        for j in range(n):
            if j in (gi, gi + 1):
                continue
            for a_ops, step, _ in parsed_n4:
                # Group (gi, gi+1) as arg0, j as arg1
                r = _compose(fop, ((a_ops[gi], a_ops[gi+1]), (a_ops[j],)),
                            fc_lookup, fold_rules, succ_map, carry_el, carry_out, zero_digit, test_cache3)
                if TARGET_KEY in test_cache3:
                    val = test_cache3[TARGET_KEY]
                    print(f"  KEY SET (group_left) by {fop}(gi={gi},j={j}) for a_ops={a_ops}: flat=({a_ops[gi]},{a_ops[gi+1]},{a_ops[j]}) → {val}")
                    break
                # j as arg0, group as arg1
                r = _compose(fop, ((a_ops[j],), (a_ops[gi], a_ops[gi+1])),
                            fc_lookup, fold_rules, succ_map, carry_el, carry_out, zero_digit, test_cache3)
                if TARGET_KEY in test_cache3:
                    val = test_cache3[TARGET_KEY]
                    print(f"  KEY SET (group_right) by {fop}(j={j},gi={gi}) for a_ops={a_ops}: flat=({a_ops[j]},{a_ops[gi]},{a_ops[gi+1]}) → {val}")
                    break
            if TARGET_KEY in test_cache3:
                break
        if TARGET_KEY in test_cache3:
            break
    if TARGET_KEY in test_cache3:
        break

if TARGET_KEY not in test_cache3:
    print(f"  Key NOT set during cs3 2-token group step synthesis")

# What does cs1 do?
cr_cs1 = chain_rules_dict['cs1']
parsed_cs1 = []
for input_key, output_toks in cr_cs1.chain_table.items():
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
            parsed_cs1.append((a_ops, step_toks, ans_toks))
parsed_cs1_n5 = [s for s in parsed_cs1 if len(s[0]) == 5]
print(f"\ncs1 n=5 samples: {len(parsed_cs1_n5)}, first: {parsed_cs1_n5[0] if parsed_cs1_n5 else None}")

# Test if cs1 ans synthesis (which uses step_results from both-2-token-group step)
# computes add(step_result, ...) where step_result is multi-digit
test_cache4 = {}
n = 5
# First find step for cs1: should be add(gi=1,gj=3) = add(('0',C),('0',D)) ... actually:
# cs1 a_ops = [B, '0', C, '0', D] — BOTH 2-token groups:
# gi=0,gj=2: add(('B','0'),('0',C)) → flat (B,'0','0',C) ...
# gi=1,gj=3: add(('0',C),('0',D)) → step = C+D

# Let's see what both-2-token-group synthesis does
for fop in fold_rules:
    for gi in range(n - 1):
        for gj in range(gi + 2, n - 1):
            for a_ops, step, ans in parsed_cs1_n5[:50]:  # check subset
                r = _compose(fop, ((a_ops[gi], a_ops[gi+1]), (a_ops[gj], a_ops[gj+1])),
                            fc_lookup, fold_rules, succ_map, carry_el, carry_out, zero_digit, test_cache4)
                if TARGET_KEY in test_cache4:
                    val = test_cache4[TARGET_KEY]
                    print(f"  KEY SET in cs1 both-groups by {fop}(gi={gi},gj={gj}) for a_ops={a_ops} → {val}")
                    break
            if TARGET_KEY in test_cache4:
                break
        if TARGET_KEY in test_cache4:
            break
    if TARGET_KEY in test_cache4:
        break

if TARGET_KEY not in test_cache4:
    print(f"cs1 both-groups: Key NOT set for {TARGET_KEY}")
