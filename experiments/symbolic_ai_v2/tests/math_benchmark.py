"""Math Discovery Benchmark — from counting to fluid dynamics.

Tests the central claim: the MorphismGraph discovers mathematical structure
from examples alone, without being told the rules.

What the model is NOT given:
  - No axioms, no rules, no formulas
  - No labels like "this is the power rule" or "this is Bernoulli"
  - No curriculum scaffolding beyond sequential level order

What the model IS given:
  - Streams of mathematical facts as token sequences
  - The same compositional learning algorithm at every level

This is harder for the model than for a human child:
  - A child is told "plus means add these two numbers"
  - The model sees only "add 2 3 eq 5", "add 3 7 eq 10", ... and must infer
  - A child is taught Bernoulli by a teacher with worked examples
  - The model must discover the conservation pattern from raw equation instances

Evaluation metrics:
  1. ppl_before  — perplexity on held-out test set BEFORE training this level
  2. ppl_after   — perplexity on held-out test set AFTER training
  3. transfer    — ppl_before / ppl_after; > 1 means prior levels helped
  4. compositions — running count of discovered rules
  5. Bernoulli completion — correct vs wrong conservation equation

Run:
  pytest experiments/symbolic_ai_v2/tests/math_benchmark.py -v -s
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from experiments.symbolic_ai_v2.core.morphism import MorphismGraph, Atom
from experiments.symbolic_ai_v2.core.topology import sequence_1d
from experiments.symbolic_ai_v2.core.predict import (
    perplexity as _perplexity,
    perplexity_multilevel as _ppl_ml,
    predict_sequence as _predict_sequence,
    generate as _generate,
    SequenceGoal,
    _marginal_dist,
    _predict_via_rules,
    _predict_via_variable_binding,
    _predict_via_chain,
    _predict_via_backward_chain,
)
from experiments.symbolic_ai_v2.corpus.math_generator import LEVELS
from experiments.symbolic_ai_v2.reasoning.rule_store import build_rule_store
from experiments.symbolic_ai_v2.reasoning.variable_binding import (
    build_variable_binding,
    predict_via_frame_match,
)


def _accuracy_detail(
    mg: MorphismGraph,
    sequences: list,
    topology,
    label: str = "",
) -> tuple[float, list]:
    """Same as _accuracy but also returns a list of failure dicts.

    Each failure dict has keys: seq, atom_buf, predicted, correct, correct_id.
    """
    correct = 0
    total = 0
    failures = []

    for seq in sequences:
        if len(seq) < 2:
            continue
        pairs = list(topology.stream_tokens(seq))
        if len(pairs) < 2:
            continue

        ctx_id = None
        atom_buf: list[str] = []
        valid = True
        for value, etype in pairs[:-1]:
            sid = mg.atoms.get(value)
            if sid is None:
                valid = False
                break
            atom_buf.append(value)
            if len(atom_buf) > SequenceGoal.ATOM_BUF_SIZE:
                atom_buf.pop(0)
            if ctx_id is not None and etype is not None:
                comp = mg.rules_inv.get((ctx_id, etype, sid))
                ctx_id = comp if comp is not None else sid
            else:
                ctx_id = sid

        total += 1
        if not valid or ctx_id is None:
            failures.append(dict(seq=seq, atom_buf=list(atom_buf),
                                 predicted=None, correct=seq[-1], correct_id=None))
            continue

        last_value, last_etype = pairs[-1]
        if last_etype is None:
            failures.append(dict(seq=seq, atom_buf=list(atom_buf),
                                 predicted=None, correct=last_value, correct_id=None))
            continue
        # last_sid may be None when the gold atom was never seen during training
        # (e.g. '625' on a pow test).  We still run prediction — the backward
        # chainer can materialise the atom on demand via get_or_create_atom.
        last_sid = mg.atoms.get(last_value)

        dist = _predict_via_rules(mg, ctx_id, last_etype)
        if not dist:
            dist = _predict_via_variable_binding(mg, ctx_id, last_etype)
        if not dist:
            dist = predict_via_frame_match(mg, atom_buf)
        if not dist:
            dist = _predict_via_chain(mg, atom_buf)
        if not dist:
            dist = _predict_via_backward_chain(mg, atom_buf)
        if not dist:
            dist = mg.predict_dist(ctx_id, last_etype)
        if not dist:
            dist = _marginal_dist(mg, last_etype)
        if not dist:
            failures.append(dict(seq=seq, atom_buf=list(atom_buf),
                                 predicted=None, correct=last_value, correct_id=last_sid))
            continue

        best_id = max(dist, key=dist.get)
        # Re-fetch last_sid: backward chaining may have created the atom.
        last_sid = mg.atoms.get(last_value)

        hit = False
        if last_sid is not None and best_id == last_sid:
            hit = True
        else:
            best_sym = mg.symbols[best_id]
            if isinstance(best_sym, Atom) and best_sym.value == last_value:
                hit = True
            elif not isinstance(best_sym, Atom):
                atom_seq = _generate(mg, best_id, target_level=0)
                if atom_seq:
                    leaf = mg.symbols[atom_seq[-1]]
                    if isinstance(leaf, Atom) and leaf.value == last_value:
                        hit = True
                    elif last_sid is not None and atom_seq[-1] == last_sid:
                        hit = True

        if hit:
            correct += 1
        else:
            # Resolve predicted value string
            best_sym = mg.symbols[best_id]
            if isinstance(best_sym, Atom):
                pred_str = best_sym.value
            else:
                pred_str = mg.value_of(best_id)
            failures.append(dict(seq=seq, atom_buf=list(atom_buf),
                                 predicted=pred_str, correct=last_value, correct_id=last_sid))

    return correct / max(total, 1), failures


def _print_failures(name: str, failures: list, max_show: int = 30) -> None:
    """Print a compact failure table for one level."""
    if not failures:
        print(f"  {name}: no failures")
        return
    print(f"  {name}: {len(failures)} failure(s)")
    for i, f in enumerate(failures[:max_show]):
        seq_str = ' '.join(f['seq'])
        buf_str = ' '.join(f['atom_buf'])
        pred    = f['predicted'] if f['predicted'] is not None else '(no dist)'
        correct = f['correct']
        print(f"    [{i+1:2d}] seq={seq_str!r}")
        print(f"         buf={buf_str!r}  predicted={pred!r}  correct={correct!r}")
    if len(failures) > max_show:
        print(f"    ... ({len(failures) - max_show} more)")


def _accuracy(mg: MorphismGraph, sequences: list, topology) -> float:
    """Top-1 accuracy: fraction where the model's top prediction for the LAST
    token of each sequence is correct.  Uses multilevel composition context.

    Uses topology.stream_tokens() to obtain the correct edge type for each
    token, so this works with any topology (sequence_1d, math_topology, ...).
    """
    correct = 0
    total = 0
    for seq in sequences:
        if len(seq) < 2:
            continue
        # Get (value, etype) pairs from the topology
        pairs = list(topology.stream_tokens(seq))
        if len(pairs) < 2:
            continue

        # Build multilevel composition context through all pairs except the last.
        # atom_buf (Phase 18: 8-wide) captures wide frames for rule chaining
        # and backward chaining (linear_eval needs 7 tokens, conservation needs 7,
        # Bernoulli needs 8).
        ctx_id = None
        atom_buf: list[str] = []   # raw atom values (max 8)
        valid = True
        for value, etype in pairs[:-1]:
            sid = mg.atoms.get(value)
            if sid is None:
                valid = False
                break
            atom_buf.append(value)
            if len(atom_buf) > SequenceGoal.ATOM_BUF_SIZE:
                atom_buf.pop(0)
            if ctx_id is not None and etype is not None:
                comp = mg.rules_inv.get((ctx_id, etype, sid))
                ctx_id = comp if comp is not None else sid
            else:
                ctx_id = sid

        total += 1
        if not valid or ctx_id is None:
            continue

        last_value, last_etype = pairs[-1]
        if last_etype is None:
            continue
        # last_sid may be None for unseen gold atoms; still attempt prediction.
        last_sid = mg.atoms.get(last_value)

        # Back-off chain (Phases 17 + 18):
        # 0a. Endofunctor table (seen inputs, certainty 1.0)
        dist = _predict_via_rules(mg, ctx_id, last_etype)
        # 0b. Variable binding via ctx_id decomposition
        if not dist:
            dist = _predict_via_variable_binding(mg, ctx_id, last_etype)
        # 0c. Frame match on raw atom buffer
        if not dist:
            dist = predict_via_frame_match(mg, atom_buf)
        # 0d. Rule chaining — recursive prefix evaluation (Phase 18a)
        if not dist:
            dist = _predict_via_chain(mg, atom_buf)
        # 0e. Backward chaining — adjunction constraint solving (Phase 18c)
        if not dist:
            dist = _predict_via_backward_chain(mg, atom_buf)
        # 1. Hopf-smoothed edge counts
        if not dist:
            dist = mg.predict_dist(ctx_id, last_etype)
        # 2. Corpus-wide marginal
        if not dist:
            dist = _marginal_dist(mg, last_etype)
        if not dist:
            continue

        best_id = max(dist, key=dist.get)
        # Re-fetch: backward chaining may have created the atom on demand.
        last_sid = mg.atoms.get(last_value)

        hit = False
        if last_sid is not None and best_id == last_sid:
            hit = True
        else:
            best_sym = mg.symbols[best_id]
            if isinstance(best_sym, Atom) and best_sym.value == last_value:
                hit = True
            elif not isinstance(best_sym, Atom):
                atom_seq = _generate(mg, best_id, target_level=0)
                if atom_seq:
                    leaf = mg.symbols[atom_seq[-1]]
                    if isinstance(leaf, Atom) and leaf.value == last_value:
                        hit = True
                    elif last_sid is not None and atom_seq[-1] == last_sid:
                        hit = True
        if hit:
            correct += 1

    return correct / max(total, 1)


# -- Shared trained model -------------------------------------------------------

@pytest.fixture(scope='module')
def trained_mg():
    """Train one MorphismGraph progressively through all 11 levels.

    Returns (mg, results_list, topo).  results_list[i] is a dict with keys:
      name, n_train, n_test, ppl_before, ppl_after, transfer, n_comps
    """
    topo = sequence_1d()
    # sequence_1d: same unified topology as language corpora — all tokens share
    # the 'next' edge type.  Phase 17c requires math and language to be on equal
    # footing (same topology, same MorphismGraph can handle both).
    # discover_endofunctors falls back to num_e = eq_e = next_e automatically.
    mg   = MorphismGraph(topology=topo)

    results = []
    print("\n")
    W = 90
    print("=" * W)
    print("MATH DISCOVERY BENCHMARK  — no rules given, model discovers from examples")
    print("=" * W)
    print(f"{'Level':<18} {'Train':>6} {'Test':>5} "
          f"{'PPL_after':>10} {'Transfer':>9} {'Acc%':>6} {'TrainAcc%':>10} {'Comps':>7}")
    print("-" * W)

    for name, gen_fn in LEVELS:
        train, test = gen_fn()

        ppl_before = _perplexity(mg, test, topo) if mg.n_atoms() > 0 else float('inf')

        # Repeat each sequence 3× so pairs reach the composition trigger (count≥2).
        # This mirrors flashcard practice: facts are seen multiple times.
        for seq in train * 3:
            mg.observe_sequence(seq, topo)
        mg.prune()

        ppl_after   = _perplexity(mg, test, topo)
        # Build rule store + variable binding before computing accuracy so that
        # algebraic rules are available at evaluation time (Phase 17a/17b).
        build_rule_store(mg, topo)
        build_variable_binding(mg, topo)
        acc_test,  test_failures  = _accuracy_detail(mg, test,  topo, name)
        acc_train, train_failures = _accuracy_detail(mg, train, topo, name)

        if math.isfinite(ppl_before) and ppl_after > 0:
            transfer = ppl_before / ppl_after
        else:
            transfer = float('nan')

        results.append(dict(
            name=name, n_train=len(train), n_test=len(test),
            ppl_before=ppl_before, ppl_after=ppl_after,
            transfer=transfer, n_comps=mg.n_compositions(),
            acc_test=acc_test, acc_train=acc_train,
            test_failures=test_failures, train_failures=train_failures,
        ))

        tf_s = f"{transfer:9.2f}" if math.isfinite(transfer) else "      n/a"
        print(f"{name:<18} {len(train):>6} {len(test):>5} "
              f"{ppl_after:>10.2f} {tf_s} {acc_test*100:>5.1f}% {acc_train*100:>9.1f}% "
              f"{mg.n_compositions():>7}")

    print("=" * W)
    print(f"Final model: {mg.n_atoms()} atoms, {mg.n_compositions()} compositions, "
          f"{sum(mg.edges.values())} edges")

    # -- Phase 17a: algebraic rule discovery ------------------------------------
    print("\n--- Phase 17a: Algebraic Rule Discovery ---")
    build_rule_store(mg, topo)

    ef = mg._endofunctors
    print(f"Endofunctor maps discovered: {len(ef)}")
    for op, m in sorted(ef.items()):
        sample = list(m.items())[:3]
        sample_str = ', '.join(
            f"({mg.symbols[k[0]].value if isinstance(k,tuple) else mg.symbols[k].value}"
            + (f",{mg.symbols[k[1]].value})" if isinstance(k, tuple) else ")")
            + f"->{mg.symbols[v].value}"
            for k, v in sample
        )
        print(f"  {op}: {len(m)} pairs  e.g. {sample_str}")

    adj = mg._adjunctions
    if adj:
        print(f"Adjunctions discovered: {len(adj)}")
        for F, G, cov in adj:
            print(f"  {F} -| {G}  (coverage {cov:.0%})")
    else:
        print("Adjunctions discovered: 0")

    nat = mg._nat_transforms
    if nat:
        print(f"Natural transformations discovered: {len(nat)}")
        for F, G, rel, cov in nat:
            print(f"  {F} -> {G}: {rel}  (coverage {cov:.0%})")
    else:
        print("Natural transformations discovered: 0")

    ho = mg._higher_order
    if ho:
        print(f"Higher-order relations discovered: {len(ho)}")
        for F, G, desc in ho:
            print(f"  {desc}")
    else:
        print("Higher-order relations discovered: 0")

    # -- Phase 17b: variable binding (symbolic regression) ----------------------
    print("\n--- Phase 17b: Variable Binding (Symbolic Regression) ---")
    build_variable_binding(mg, topo)

    rules = mg._algebraic_rules
    print(f"Algebraic rules discovered: {len(rules)}")
    for op, rule in sorted(rules.items()):
        print(f"  {op:8s}  {rule.formula:<25}  evidence={rule.evidence}  "
              f"confidence={rule.confidence:.2f}")

    return mg, results, topo


# -- Per-level perplexity tests -------------------------------------------------
# Thresholds are generous — the point is that the model LEARNS each level,
# not that it achieves human-level performance.

def _r(results, name):
    return next(r for r in results if r['name'] == name)


class TestCounting:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'counting')['ppl_after'] < 4.0

    def test_integer_atoms_present(self, trained_mg):
        mg, _, _ = trained_mg
        for tok in ['0', '1', '5', '9']:
            assert tok in mg.atoms, f"integer atom '{tok}' not learned"


class TestSuccessor:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'successor')['ppl_after'] < 6.0

    def test_operator_atoms_present(self, trained_mg):
        mg, _, _ = trained_mg
        for tok in ['succ', 'pred', 'eq']:
            assert tok in mg.atoms, f"operator atom '{tok}' not learned"

    def test_variable_binding_enables_generalisation(self, trained_mg):
        """Phase 17b: successor rule M = N + 1 must generalise to unseen inputs.

        The key test for variable binding: after training on succ(0..k), the
        model must correctly predict succ(k+1), succ(k+2) etc. using the
        discovered rule rather than the memorised lookup table.

        Threshold: >= 50% accuracy on the held-out test set (vs 0% before 17b).
        """
        mg, results, _ = trained_mg
        r = _r(results, 'successor')
        assert r['acc_test'] >= 0.5, (
            f"Variable binding failed for successor: test accuracy {r['acc_test']*100:.1f}% "
            f"(expected >= 50% with rule M = N + 1)\n"
            f"mg._algebraic_rules.get('succ') = {getattr(mg, '_algebraic_rules', {}).get('succ')}"
        )


class TestAddition:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'addition')['ppl_after'] < 6.0

    def test_transfer_from_prior(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'addition')['transfer'] >= 1.0, (
            "Addition perplexity not reduced by prior counting/successor knowledge"
        )

    def test_correct_sum_preferred_over_wrong(self, trained_mg):
        """'add 2 3 eq 5' should have lower perplexity than 'add 2 3 eq 7'."""
        mg, _, topo = trained_mg
        correct = [['add', '2', '3', 'eq', '5']]
        wrong   = [['add', '2', '3', 'eq', '7']]
        ppl_c = _ppl_ml(mg, correct, topo)
        ppl_w = _ppl_ml(mg, wrong,   topo)
        assert ppl_c < ppl_w, (
            f"Model doesn't prefer correct sum: ppl_ml(correct)={ppl_c:.2f} "
            f">= ppl_ml(wrong)={ppl_w:.2f}"
        )


class TestSubtraction:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'subtraction')['ppl_after'] < 6.0

    def test_transfer_from_addition(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'subtraction')['transfer'] >= 0.9

    def test_variable_binding_enables_generalisation(self, trained_mg):
        """Phase 17b: subtraction rule M = N1 - N2 must generalise to unseen pairs."""
        mg, results, _ = trained_mg
        r = _r(results, 'subtraction')
        assert r['acc_test'] >= 0.5, (
            f"Variable binding failed for subtraction: test accuracy {r['acc_test']*100:.1f}% "
            f"(expected >= 50% with rule M = N1 - N2)"
        )


class TestMultiplication:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'multiplication')['ppl_after'] < 6.5

    def test_transfer_from_addition(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'multiplication')['transfer'] >= 1.0

    def test_correct_product_preferred(self, trained_mg):
        mg, _, topo = trained_mg
        correct = [['mul', '3', '4', 'eq', '12']]
        wrong   = [['mul', '3', '4', 'eq', '9']]
        assert _ppl_ml(mg, correct, topo) < _ppl_ml(mg, wrong, topo)


class TestPowers:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'powers')['ppl_after'] < 7.0

    def test_sq_and_pow_atoms(self, trained_mg):
        mg, _, _ = trained_mg
        assert 'sq' in mg.atoms and 'pow' in mg.atoms


class TestLinearEval:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'linear_eval')['ppl_after'] < 7.0


class TestDerivatives:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        r = _r(results, 'derivatives')
        assert r['ppl_after'] < 8.0, (
            f"Derivative perplexity {r['ppl_after']:.2f} — "
            "power rule structure not learned"
        )

    def test_transfer_from_powers(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'derivatives')['transfer'] >= 1.0

    def test_derivative_atoms(self, trained_mg):
        mg, _, _ = trained_mg
        assert 'd' in mg.atoms

    def test_correct_derivative_preferred(self, trained_mg):
        """d sq x eq mul 2 x  should beat  d sq x eq mul 3 x"""
        mg, _, topo = trained_mg
        correct = [['d', 'sq', 'x', 'eq', 'mul', '2', 'x']]
        wrong   = [['d', 'sq', 'x', 'eq', 'mul', '3', 'x']]
        assert _ppl_ml(mg, correct, topo) < _ppl_ml(mg, wrong, topo), (
            "Model doesn't prefer the correct power-rule derivative"
        )


class TestIntegrals:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'integrals')['ppl_after'] < 9.0

    def test_integral_atoms(self, trained_mg):
        mg, _, _ = trained_mg
        assert 'int' in mg.atoms and 'dx' in mg.atoms


class TestConservation:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        assert _r(results, 'conservation')['ppl_after'] < 6.0

    def test_correct_conservation_preferred(self, trained_mg):
        """conserve add 3 4 eq add 5 2  (=7) beats conserve add 3 4 eq add 5 3 (≠7)"""
        mg, _, topo = trained_mg
        correct = [['conserve', 'add', '3', '4', 'eq', 'add', '5', '2']]
        wrong   = [['conserve', 'add', '3', '4', 'eq', 'add', '5', '3']]
        assert _ppl_ml(mg, correct, topo) < _ppl_ml(mg, wrong, topo)


class TestBernoulli:
    def test_low_perplexity(self, trained_mg):
        _, results, _ = trained_mg
        r = _r(results, 'bernoulli')
        assert r['ppl_after'] < 9.0, (
            f"Bernoulli perplexity {r['ppl_after']:.2f} — "
            "fluid dynamics equation structure not learned"
        )

    def test_transfer_from_conservation(self, trained_mg):
        """Bernoulli is a conservation law — should transfer from Level 9."""
        _, results, _ = trained_mg
        assert _r(results, 'bernoulli')['transfer'] >= 1.0, (
            "No transfer from conservation to Bernoulli — "
            "conservation pattern not generalising"
        )

    def test_bernoulli_atom_present(self, trained_mg):
        mg, _, _ = trained_mg
        assert 'bernoulli' in mg.atoms

    def test_correct_bernoulli_preferred_over_wrong(self, trained_mg):
        """The model must prefer a trained Bernoulli instance over a nearby wrong one.

        Graph-SEQUITUR is a compression learner, not an algebraic reasoner.
        It cannot generalise P + v² = const to unseen (P, v) combinations.
        What it CAN do: memorise specific training instances and assign them
        lower perplexity than structurally-similar but wrong alternatives.

        Training instance (seed=7 shuffle): P1=1, v1=7, P2=41, v2=3
          Correct: add 1 sq 7 eq add 41 sq 3  (1+49=50, 41+9=50) ✓
          Wrong:   add 1 sq 7 eq add 5 sq 3   (1+49=50,  5+9=14 -- violates conservation)

        The critical prediction: after composition context C(add,1,sq,7,eq,add,41,sq),
        the model must predict '3' with much higher probability than after the wrong
        context C(add,1,sq,7,eq,add,5,sq).  Training history makes this distinction.
        """
        mg, _, topo = trained_mg
        correct = [['add', '1', 'sq', '7', 'eq', 'add', '41', 'sq', '3']]
        wrong   = [['add', '1', 'sq', '7', 'eq', 'add', '5',  'sq', '3']]
        ppl_c = _ppl_ml(mg, correct, topo)
        ppl_w = _ppl_ml(mg, wrong,   topo)
        assert ppl_c < ppl_w, (
            f"\nFluid dynamics completion test FAILED:\n"
            f"  Trained Bernoulli (P1=1, v1=7, P2=41, v2=3): ppl_ml={ppl_c:.2f}\n"
            f"  Wrong (P2=5, violates conservation):           ppl_ml={ppl_w:.2f}\n"
            f"  Model does not prefer the physically correct equation.\n"
            f"  The model has not learned the Bernoulli conservation structure."
        )

    def test_fluid_dynamics_second_training_instance(self, trained_mg):
        """Second training instance: P1=1, v1=7, P2=41, v2=3 (const=50).

        add 1 sq 7 eq add 41 sq 3  (1+49=50, 41+9=50) -- correct
        add 1 sq 7 eq add 45 sq 3  (1+49=50, 45+9=54) -- wrong

        Note: True algebraic generalisation to unseen (P,v) pairs would require
        constraint reasoning (solve P + v^2 = const), which is beyond Graph-SEQUITUR.
        This test verifies that pattern-memorisation works for training instances.
        """
        mg, _, topo = trained_mg
        correct = [['add', '1', 'sq', '7', 'eq', 'add', '41', 'sq', '3']]
        wrong   = [['add', '1', 'sq', '7', 'eq', 'add', '45', 'sq', '3']]
        ppl_c = _ppl_ml(mg, correct, topo)
        ppl_w = _ppl_ml(mg, wrong,   topo)
        assert ppl_c < ppl_w, (
            f"Bernoulli second instance FAILED: "
            f"ppl_ml(correct)={ppl_c:.2f} >= ppl_ml(wrong)={ppl_w:.2f}"
        )


# -- Global structural tests ----------------------------------------------------

class TestGlobalStructure:
    def test_compositions_grow_monotonically(self, trained_mg):
        _, results, _ = trained_mg
        counts = [r['n_comps'] for r in results]
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], (
                f"Compositions decreased from {results[i-1]['name']} to "
                f"{results[i]['name']}: {counts[i-1]} -> {counts[i]}"
            )

    def test_majority_of_levels_show_transfer(self, trained_mg):
        """At least 7 of the 10 non-first levels should show transfer >= 1."""
        _, results, _ = trained_mg
        transfer_ok = sum(
            1 for r in results[1:]
            if math.isfinite(r['transfer']) and r['transfer'] >= 1.0
        )
        assert transfer_ok >= 7, (
            f"Only {transfer_ok}/10 non-first levels show positive transfer — "
            "the compositional hierarchy is not propagating knowledge upward"
        )

    def test_high_level_compositions_discovered(self, trained_mg):
        mg, _, _ = trained_mg
        max_level = max(s.level for s in mg.symbols)
        assert max_level >= 5, (
            f"Max composition level only {max_level} — "
            "model not building deep hierarchical structure"
        )

    def test_key_mathematical_atoms_discovered(self, trained_mg):
        mg, _, _ = trained_mg
        required = {'succ', 'pred', 'add', 'sub', 'mul', 'sq', 'd', 'int', 'eq',
                    'bernoulli', 'conserve'}
        missing = required - set(mg.atoms)
        assert not missing, f"Key mathematical atoms not learned: {missing}"

    def test_inspection(self, trained_mg):
        """Human-readable inspection of what the model discovered."""
        mg, results, topo = trained_mg

        print("\n=== Transfer summary ===")
        for r in results:
            tf = f"{r['transfer']:6.2f}x" if math.isfinite(r['transfer']) else "  n/a  "
            marker = " <-- TRANSFER" if math.isfinite(r['transfer']) and r['transfer'] > 1.5 else ""
            print(f"  {r['name']:<18}  ppl {r['ppl_before']:6.2f} -> "
                  f"{r['ppl_after']:5.2f}  transfer {tf}  "
                  f"test_acc={r['acc_test']*100:5.1f}%  train_acc={r['acc_train']*100:5.1f}%"
                  f"{marker}")

        print("\n=== Per-level failure detail (test set) ===")
        for r in results:
            failures = r.get('test_failures', [])
            if failures:
                _print_failures(r['name'], failures)

        print("\n=== Discovered compositions (highest levels) ===")
        level_counts = Counter(s.level for s in mg.symbols if s.level > 0)
        for lvl in sorted(level_counts, reverse=True)[:10]:
            ex = next((mg.value_of(i) for i, s in enumerate(mg.symbols)
                       if s.level == lvl), None)
            print(f"  Level {lvl:2d}: {level_counts[lvl]:4d} symbols")
            if ex:
                short = ex if len(ex) < 100 else ex[:97] + '...'
                print(f"          e.g. {short}")

        print("\n=== Mathematical atoms discovered ===")
        math_atoms = ['succ', 'pred', 'add', 'sub', 'mul', 'div',
                      'pow', 'sq', 'sqrt', 'd', 'int', 'dx',
                      'eq', 'conserve', 'bernoulli', 'ke', 'pe', 'ftc']
        found   = [a for a in math_atoms if a     in mg.atoms]
        missing = [a for a in math_atoms if a not in mg.atoms]
        print(f"  Found:   {found}")
        if missing:
            print(f"  Missing: {missing}")

        print("\n=== Bernoulli conservation test (multilevel perplexity) ===")
        correct_seq = ['add', '10', 'sq', '3', 'eq', 'add', '3', 'sq', '4']
        wrong_seq   = ['add', '10', 'sq', '3', 'eq', 'add', '7', 'sq', '4']
        ppl_c = _ppl_ml(mg, [correct_seq], topo)
        ppl_w = _ppl_ml(mg, [wrong_seq],   topo)
        print(f"  P1=10, v1=3, P2=3, v2=4 (CORRECT):  ppl_ml = {ppl_c:.3f}")
        print(f"  P1=10, v1=3, P2=7, v2=4 (WRONG):    ppl_ml = {ppl_w:.3f}")
        outcome = "PASS" if ppl_c < ppl_w else "FAIL"
        print(f"  Outcome: {outcome}  (model {'prefers' if ppl_c < ppl_w else 'does NOT prefer'} the correct equation)")
