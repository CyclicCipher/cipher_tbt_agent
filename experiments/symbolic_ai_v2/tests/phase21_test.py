"""Phase 21: NL word-problem generalisation + variadic fold.

Two new capabilities beyond Phase 20:

Level D — NL word-problem generalisation to unseen (N1, N2) pairs
  Training: 'alice has 3 apples bob gives her 4 how many eq 7'
             'add 3 4 eq 7'   (co-trained math fact)
  Test:     'dave has 5 coins carol gives her 3 how many eq ?'
             (N1=5, N2=3) not seen in word-problem training.
  Mechanism: predict_via_frame_match NL-numeral scan finds last two
             numerals before 'eq' and applies the discovered add rule.

Level E — Variadic fold (any arity)
  Training: 'vadd 1 2 eq 3', 'vadd 2 3 eq 5', ...  (arity-2, binary rule)
  Test:     'vadd 1 2 3 eq ?'   -> fold: add(add(1,2),3) = 6
             'vadd 2 3 4 5 eq ?' -> fold: add(add(add(2,3),4),5) = 14
  Mechanism: predict_via_frame_match variadic-fold check detects 'vadd'
             in buffer, extracts k>=3 numerals, folds with binary rule.

What the model is NOT given:
  - No mapping from NL words to math operators
  - No special fold/reduce primitive
  - No hardcoded sum formula for variadic case

Run:
  pytest experiments/symbolic_ai_v2/tests/phase21_test.py -v -s
"""
from __future__ import annotations

import pytest

from experiments.symbolic_ai_v2.core.morphism import MorphismGraph
from experiments.symbolic_ai_v2.core.topology import math_topology
from experiments.symbolic_ai_v2.core.predict import (
    generate_until_eos,
    perplexity as _perplexity,
    perplexity_multilevel as _ppl_ml,
)
from experiments.symbolic_ai_v2.reasoning.rule_store import build_rule_store
from experiments.symbolic_ai_v2.reasoning.variable_binding import build_variable_binding
from experiments.symbolic_ai_v2.corpus.qa_generator import (
    word_problem_level,
    math_cotraining_level,
    variadic_add_level,
    EOS,
)

REPS = 4


# ── Fixture: word problems with math_topology ─────────────────────────────────

@pytest.fixture(scope='module')
def wp_math_mg():
    """Word problems + math co-training using math_topology.

    Using math_topology (instead of sequence_1d) makes numerals get num_etype
    and 'eq' get eq_etype.  This enables:
    1. extract_binary_pairs to find (N1,N2)->M pairs from math facts
    2. predict_via_frame_match NL-numeral scan to generalise to novel (N1,N2)
       pairs in NL word problems — the last two numerals before 'eq' are the
       operands, and the discovered 'add' rule gives the result.

    Addition-only (n_sub=0) so the NL scan unambiguously applies the add rule.
    """
    topo = math_topology()
    mg   = MorphismGraph(topology=topo)

    math_facts          = math_cotraining_level(seed=42)
    wp_train, wp_test   = word_problem_level(seed=42, n_add=40, n_sub=0)

    for seq in (math_facts + wp_train) * REPS:
        mg.observe_sequence(seq, topo)
    mg.prune()
    build_rule_store(mg, topo)
    build_variable_binding(mg, topo)

    return mg, wp_train, wp_test, topo


# ── Fixture: variadic addition (same as Phase 20, re-used here) ───────────────

@pytest.fixture(scope='module')
def vadd_mg():
    topo = math_topology()
    mg   = MorphismGraph(topology=topo)

    train, _ = variadic_add_level(seed=42, max_terms=5, n_per_arity=15)
    for seq in train * REPS:
        mg.observe_sequence(seq, topo)
    mg.prune()
    build_rule_store(mg, topo)
    build_variable_binding(mg, topo)

    return mg, topo


# ── Level D: NL word-problem generalisation ───────────────────────────────────

class TestNLGeneralisation:
    def test_add_rule_discovered(self, wp_math_mg):
        """The add rule must be in algebraic_rules after math co-training."""
        mg, _, _, _ = wp_math_mg
        rules = getattr(mg, '_algebraic_rules', {})
        assert 'add' in rules, (
            f"add rule not discovered. Rules: {list(rules.keys())}"
        )

    def test_nl_atoms_learned(self, wp_math_mg):
        """Both NL tokens and math tokens must be present."""
        mg, _, _, _ = wp_math_mg
        for tok in ['alice', 'has', 'gives', 'how', 'many', 'add', 'eq']:
            assert tok in mg.atoms, f"Token '{tok}' not in atoms"

    def test_low_perplexity_on_word_problems(self, wp_math_mg):
        """Word problem test perplexity < 10.0 bits/token."""
        mg, _, wp_test, topo = wp_math_mg
        ppl = _perplexity(mg, wp_test, topo)
        assert ppl < 10.0, f"Word problem ppl {ppl:.2f} too high"

    def test_nl_generalises_unseen_pair(self, wp_math_mg):
        """Model predicts correct sum for a word problem (N1, N2) not in training.

        The NL-numeral scan in predict_via_frame_match finds the last two
        numerals before 'eq' and applies the discovered add rule.  This fires
        even for (N1, N2) pairs never seen in word-problem training.
        """
        mg, wp_train, _, topo = wp_math_mg

        # Find a test pair not in training. Use a held-out (N1, N2) pair.
        train_pairs = set()
        for seq in wp_train:
            if 'eq' in seq:
                # extract the two numerals before 'eq'
                eq_pos = seq.index('eq')
                nums = [t for t in seq[:eq_pos] if t.isdigit() or
                        (t.lstrip('-').isdigit())]
                if len(nums) >= 2:
                    try:
                        train_pairs.add((int(nums[-2]), int(nums[-1])))
                    except ValueError:
                        pass

        # Pick a pair outside training: (5, 6) = 11, within math_cotraining range
        # The model must use the algebraic rule, not memorisation.
        for n1, n2 in [(5, 6), (4, 7), (3, 8), (6, 5)]:
            if (n1, n2) not in train_pairs and n1 + n2 <= 12:
                target_sum = n1 + n2
                prompt = ['alice', 'has', str(n1), 'apples',
                          'bob', 'gives', 'her', str(n2),
                          'how', 'many', 'eq']
                gen = generate_until_eos(mg, prompt, topo, eos=EOS, max_steps=5)
                assert gen, f"No output for unseen pair ({n1},{n2})"
                assert gen[0] == str(target_sum), (
                    f"NL generalisation failed: alice has {n1} ... gives her {n2} eq\n"
                    f"  Expected: '{target_sum}'  Got: {gen}\n"
                    f"  (pair ({n1},{n2}) was not in word-problem training)"
                )
                return  # found and tested one unseen pair

        pytest.skip("Could not find a suitable unseen (N1, N2) pair")

    def test_nl_training_instance_recalled(self, wp_math_mg):
        """For a training word problem, generate_until_eos gives the correct sum."""
        mg, wp_train, _, topo = wp_math_mg
        for seq in wp_train:
            if 'eq' in seq and 'gives' in seq:
                eq_pos = seq.index('eq')
                prompt         = seq[:eq_pos + 1]
                correct_answer = seq[eq_pos + 1]
                gen = generate_until_eos(mg, prompt, topo, eos=EOS, max_steps=5)
                if gen and gen[0] == correct_answer:
                    return  # at least one training instance works
        pytest.fail("No training word problem recalled correctly")

    def test_nl_correct_preferred_over_wrong(self, wp_math_mg):
        """Correct NL answer has lower ppl_ml than a wrong answer."""
        mg, wp_train, _, topo = wp_math_mg
        add_seqs = [s for s in wp_train if 'gives' in s and 'eq' in s]
        if not add_seqs:
            pytest.skip("No addition training sequences")
        seq         = add_seqs[0]
        eq_pos      = seq.index('eq')
        prefix      = seq[:eq_pos]
        correct_ans = seq[eq_pos + 1]
        try:
            wrong_ans = str(int(correct_ans) + 1)
        except ValueError:
            pytest.skip("Answer is not an integer")
        ppl_c = _ppl_ml(mg, [prefix + ['eq', correct_ans, EOS]], topo)
        ppl_w = _ppl_ml(mg, [prefix + ['eq', wrong_ans,   EOS]], topo)
        assert ppl_c < ppl_w, (
            f"Correct answer '{correct_ans}' ppl={ppl_c:.2f} "
            f">= wrong '{wrong_ans}' ppl={ppl_w:.2f}"
        )


# ── Level E: Variadic fold ────────────────────────────────────────────────────

class TestVariadicFold:
    def test_vadd_binary_rule_discovered(self, vadd_mg):
        """Prerequisite: binary vadd rule M = N1 + N2 must exist."""
        mg, _ = vadd_mg
        rules = getattr(mg, '_algebraic_rules', {})
        assert 'vadd' in rules, f"vadd rule not found. Rules: {list(rules.keys())}"

    def test_fold_arity3(self, vadd_mg):
        """vadd 1 2 3 eq -> 6  (fold: 1+2=3, 3+3=6)."""
        mg, topo = vadd_mg
        gen = generate_until_eos(mg, ['vadd', '1', '2', '3', 'eq'],
                                 topo, eos=EOS, max_steps=5)
        assert gen, "No output for vadd 1 2 3 eq"
        assert gen[0] == '6', (
            f"vadd 1+2+3=6 expected, got {gen[0]!r} (full: {gen})"
        )

    def test_fold_arity4(self, vadd_mg):
        """vadd 2 3 4 5 eq -> 14."""
        mg, topo = vadd_mg
        gen = generate_until_eos(mg, ['vadd', '2', '3', '4', '5', 'eq'],
                                 topo, eos=EOS, max_steps=5)
        assert gen, "No output for vadd 2 3 4 5 eq"
        assert gen[0] == '14', (
            f"vadd 2+3+4+5=14 expected, got {gen[0]!r} (full: {gen})"
        )

    def test_fold_arity5(self, vadd_mg):
        """vadd 1 1 1 1 1 eq -> 5."""
        mg, topo = vadd_mg
        gen = generate_until_eos(mg, ['vadd', '1', '1', '1', '1', '1', 'eq'],
                                 topo, eos=EOS, max_steps=5)
        assert gen, "No output for vadd 1 1 1 1 1 eq"
        assert gen[0] == '5', (
            f"vadd 1+1+1+1+1=5 expected, got {gen[0]!r} (full: {gen})"
        )

    def test_fold_generalises_novel_triple(self, vadd_mg):
        """vadd fold works for a 3-term sum not seen in arity-3 training.

        The binary vadd rule was discovered from arity-2 examples.
        The fold applies it recursively — no arity-3 memorisation needed.
        """
        mg, topo = vadd_mg
        # (7, 5, 3) sum = 15; unlikely to be memorised from 15 arity-3 examples
        gen = generate_until_eos(mg, ['vadd', '7', '5', '3', 'eq'],
                                 topo, eos=EOS, max_steps=5)
        assert gen, "No output for vadd 7 5 3 eq"
        assert gen[0] == '15', (
            f"vadd 7+5+3=15 expected, got {gen[0]!r} (full: {gen})"
        )


# ── Combined inspection ───────────────────────────────────────────────────────

class TestPhase21Inspection:
    def test_inspection(self, wp_math_mg, vadd_mg):
        print("\n")
        print("=" * 70)
        print("PHASE 21 -- NL Generalisation + Variadic Fold")
        print("=" * 70)

        mg_wp, wp_train, wp_test, topo_wp = wp_math_mg
        ppl_wp  = _perplexity(mg_wp, wp_test, topo_wp)
        rules   = getattr(mg_wp, '_algebraic_rules', {})
        print(f"\n--- Level D: NL Word Problems (math_topology, co-trained) ---")
        print(f"  Atoms: {mg_wp.n_atoms()}  Comps: {mg_wp.n_compositions()}")
        print(f"  Test ppl: {ppl_wp:.2f}")
        print(f"  Rules: {sorted(rules.keys())}")

        test_cases_wp = [
            (['alice', 'has', '3', 'apples', 'bob', 'gives', 'her', '4',
              'how', 'many', 'eq'], '7',  'training pair (3,4)'),
            (['alice', 'has', '5', 'apples', 'bob', 'gives', 'her', '6',
              'how', 'many', 'eq'], '11', 'novel pair (5,6)'),
            (['dave',  'has', '7', 'coins',  'carol', 'gives', 'her', '2',
              'how', 'many', 'eq'], '9',  'novel pair + novel names (7,2)'),
        ]
        for prompt, expected, label in test_cases_wp:
            gen    = generate_until_eos(mg_wp, prompt, topo_wp, eos=EOS, max_steps=5)
            status = "OK" if gen and gen[0] == expected else "FAIL"
            print(f"  [{status}] {' '.join(prompt[-5:])} -> {gen!r}  "
                  f"(expected '{expected}', {label})")

        mg_va, topo_va = vadd_mg
        rules_va = getattr(mg_va, '_algebraic_rules', {})
        print(f"\n--- Level E: Variadic Fold ---")
        print(f"  Atoms: {mg_va.n_atoms()}  Comps: {mg_va.n_compositions()}")
        print(f"  Rules: {sorted(rules_va.keys())}")

        for prompt_vals, expected in [
            (['vadd', '2', '3', 'eq'],             '5'),
            (['vadd', '1', '2', '3', 'eq'],         '6'),
            (['vadd', '2', '3', '4', '5', 'eq'],    '14'),
            (['vadd', '1', '1', '1', '1', '1', 'eq'], '5'),
            (['vadd', '7', '5', '3', 'eq'],          '15'),
        ]:
            gen    = generate_until_eos(mg_va, prompt_vals, topo_va, eos=EOS, max_steps=5)
            status = "OK" if gen and gen[0] == expected else "FAIL"
            print(f"  [{status}] {' '.join(prompt_vals)} -> {gen!r}  "
                  f"(expected '{expected}')")

        print("=" * 70)
