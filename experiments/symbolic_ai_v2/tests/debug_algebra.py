"""Debug why algebra_trace shows 0% even though chain_table is populated."""
import sys, io, contextlib
sys.path.insert(0, '.')
from experiments.symbolic_ai_v2.tests.math_benchmark import (
    LEVELS, _strip_eos, _input_prefix, _ground_truth, _generate_n, _eval_level, R, K
)

train_all = []
per_level = {}
for name, gen_fn, is_trace in LEVELS:
    tr, te = gen_fn()
    tr = [_strip_eos(s) for s in tr]
    te = [_strip_eos(s) for s in te]
    train_all.extend(tr)
    per_level[name] = (tr, te, is_trace)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
    from experiments.symbolic_ai_v2.ctkg.learning.em_loop import em_loop
    from experiments.symbolic_ai_v2.ctkg.learning.mdl_prune import semantic_deduplicate
    from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
        discover_processes, discover_compose_chains, build_free_category, enrich_morphism_graph,
    )
    from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor

    hc = HankelCount(r_max=R)
    hc.update_batch(train_all)
    lattices, mg, _ = em_loop(corpus=train_all, r_levels=[R], n_em_max=3, lambda_productivity=0.1)
    lattice = lattices[0]
    mg = semantic_deduplicate(mg)
    fc = build_free_category(train_all)
    enrich_morphism_graph(fc, mg, lattice, hc)
    process_rules = discover_processes(train_all)
    chain_rules = discover_compose_chains(train_all)

    pred = Predictor(
        hankel=hc, lattice=lattice, morphism_graph=mg, process_rules=process_rules,
        chain_rules=chain_rules, k_neighbours=K, r=R, fc=fc,
    )

print(f"chain_op_atoms: {sorted(pred._chain_op_atoms)}")
linsolve_in = 'linsolve' in pred._chain_rules
print(f"'linsolve' in pred._chain_rules: {linsolve_in}")
if linsolve_in:
    cr = pred._chain_rules['linsolve']
    print(f"  chain_table entries: {len(cr.chain_table)}")

# Test algebra_trace
alg_train, alg_test, is_trace = per_level['algebra_trace']
seq = alg_train[0]
prefix = _input_prefix(seq, is_trace=True)
truth = _ground_truth(seq, is_trace=True)
print(f"\nalgebra_trace train[0]: {' '.join(seq)}")
print(f"prefix: {prefix}")
print(f"truth: {truth}")

# Step through generation
from experiments.symbolic_ai_v2.ctkg.core.working_memory import parse_chain_prefix
from experiments.symbolic_ai_v2.ctkg.inference.predict import _chain_predict

current = list(prefix)
for i, expected_tok in enumerate(truth):
    chain_state = parse_chain_prefix(current, pred._chain_op_atoms)
    dist = pred.predict_next(current)
    best = max(dist, key=dist.get) if dist else '?'
    # Also check chain_predict directly
    chain_result = None
    if chain_state and chain_state.op in pred._chain_rules:
        cr = pred._chain_rules[chain_state.op]
        chain_result = _chain_predict(cr, chain_state.input_tokens, chain_state.output_tokens)
    print(f"  step {i}: expected={expected_tok!r} got={best!r} chain={chain_result}")
    current.append(expected_tok)

# Quick accuracy check on first 10 train examples
print("\n=== algebra_trace train accuracy (first 10) ===")
correct = 0
total = 0
for seq in alg_train[:10]:
    prefix = _input_prefix(seq, is_trace=True)
    truth = _ground_truth(seq, is_trace=True)
    gen = _generate_n(pred, prefix, len(truth))
    if gen == truth:
        correct += 1
    else:
        print(f"  FAIL: prefix={prefix} expected={truth} got={gen}")
    total += 1
print(f"accuracy: {correct}/{total} = {100*correct/total:.1f}%")
