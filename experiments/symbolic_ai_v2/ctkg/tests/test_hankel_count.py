"""Tests for ctkg/learning/hankel_count.py (Phase 1 — HankelCount).

The three required tests from the roadmap (Step 1.3):
1. At least 3 distinct neighbourhood patterns are discovered from the math corpus
   (operator context, left-operand context, right-operand/result context).
2. get_distribution for a digit-after-'succ' context gives high probability to
   digits (0–9), low probability to operators.
3. Streaming and batch results are identical.

Additional unit tests cover the neighbourhood key format and edge-case behaviour.
"""

from __future__ import annotations

import sys
import os

# Ensure the repository root is on sys.path for cross-package imports.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.corpus.math_generator import successor_seqs


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def succ_corpus() -> list[list[str]]:
    """All successor and predecessor sequences for n=0..49."""
    train, test = successor_seqs(n_max=50)
    return train + test  # use full corpus for pattern discovery tests


@pytest.fixture
def hc_trained(succ_corpus) -> HankelCount:
    hc = HankelCount(r_max=3)
    hc.update_batch(succ_corpus)
    return hc


# ---------------------------------------------------------------------------
# Unit tests: neighbourhood key format
# ---------------------------------------------------------------------------

class TestNeighbourhoodKey:
    def test_simple_bigram_r1(self):
        """r=1 key for position 1 in ['a', 'b', 'c'] should capture left and right."""
        seq = ['a', 'b', 'c']
        key = HankelCount._neighbourhood_key(seq, i=1, r=1)
        # should be r1|-1,a|1,c
        assert key == 'r1|-1,a|1,c'

    def test_left_edge_pad(self):
        """Position 0 with r=1: left neighbour is <pad>."""
        seq = ['x', 'y', 'z']
        key = HankelCount._neighbourhood_key(seq, i=0, r=1)
        assert key == 'r1|-1,<pad>|1,y'

    def test_right_edge_pad(self):
        """Position 2 (last) with r=1: right neighbour is <pad>."""
        seq = ['x', 'y', 'z']
        key = HankelCount._neighbourhood_key(seq, i=2, r=1)
        assert key == 'r1|-1,y|1,<pad>'

    def test_r2_key(self):
        """r=2 key should span positions i-2..i+2 (excluding i)."""
        seq = ['a', 'b', 'c', 'd', 'e']
        key = HankelCount._neighbourhood_key(seq, i=2, r=2)
        # offsets -2,-1,+1,+2 → a,b,d,e
        assert key == 'r2|-2,a|-1,b|1,d|2,e'

    def test_radius_prefix(self):
        """Key must start with 'r{r}|' so radius filtering works."""
        seq = ['p', 'q', 'r_atom']
        for r in range(1, 4):
            key = HankelCount._neighbourhood_key(seq, i=1, r=r)
            assert key.startswith(f'r{r}|'), f"Expected 'r{r}|' prefix, got: {key!r}"

    def test_single_token_sequence(self):
        """A sequence of length 1 has only pad neighbours."""
        seq = ['solo']
        key = HankelCount._neighbourhood_key(seq, i=0, r=1)
        assert '<pad>' in key
        assert 'solo' not in key  # centre position excluded


# ---------------------------------------------------------------------------
# Unit tests: update() and basic counting
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_empty_hc(self):
        hc = HankelCount()
        assert hc.all_contexts() == []
        assert hc.vocabulary() == []

    def test_single_sequence_adds_vocab(self):
        hc = HankelCount(r_max=1)
        hc.update(['a', 'b', 'c'])
        vocab = hc.vocabulary()
        assert 'a' in vocab
        assert 'b' in vocab
        assert 'c' in vocab

    def test_single_sequence_creates_contexts(self):
        hc = HankelCount(r_max=1)
        hc.update(['a', 'b', 'c'])
        # Each of 3 positions × 1 radius = 3 distinct context keys (in general)
        assert len(hc.all_contexts()) >= 1

    def test_distribution_sums_to_one(self):
        hc = HankelCount(r_max=2)
        for _ in range(10):
            hc.update(['x', 'y', 'z', 'x', 'y'])
        for ctx in hc.all_contexts():
            dist = hc.get_distribution(ctx)
            if dist:
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-9, f"Distribution for {ctx!r} sums to {total}"

    def test_unknown_context_returns_empty(self):
        hc = HankelCount()
        assert hc.get_distribution('nonexistent_context') == {}


# ---------------------------------------------------------------------------
# Unit tests: matrix()
# ---------------------------------------------------------------------------

class TestMatrix:
    def test_matrix_shape(self):
        hc = HankelCount(r_max=2)
        hc.update(['a', 'b', 'c', 'b', 'a'])
        contexts, atoms, H = hc.matrix()
        assert H.shape == (len(contexts), len(atoms))

    def test_matrix_rows_sum_to_one(self):
        hc = HankelCount(r_max=1)
        hc.update(['p', 'q', 'p', 'q', 'p'])
        _, _, H = hc.matrix()
        row_sums = H.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-9, f"Row sums to {s}, expected 1.0"

    def test_matrix_r_filter(self):
        hc = HankelCount(r_max=3)
        hc.update(['a', 'b', 'c', 'd', 'e'])
        for r in range(1, 4):
            contexts_r, atoms_r, H_r = hc.matrix(r=r)
            assert H_r.shape[0] == len(contexts_r)
            # All returned contexts should belong to radius r
            for ctx in contexts_r:
                assert ctx.startswith(f'r{r}|')

    def test_matrix_consistency_with_get_distribution(self):
        """matrix() and get_distribution() must agree on every row."""
        hc = HankelCount(r_max=1)
        for seq in [['a', 'b'], ['b', 'c'], ['a', 'c'], ['b', 'a']]:
            hc.update(seq)
        contexts, atoms, H = hc.matrix()
        atom_idx = {a: j for j, a in enumerate(atoms)}
        for i, ctx in enumerate(contexts):
            dist = hc.get_distribution(ctx)
            for atom, prob in dist.items():
                j = atom_idx[atom]
                assert abs(H[i, j] - prob) < 1e-12


# ---------------------------------------------------------------------------
# Roadmap test 3: streaming == batch
# ---------------------------------------------------------------------------

class TestStreamingBatchEquivalence:
    def test_streaming_equals_batch(self, succ_corpus):
        """update() one-by-one must produce identical result to update_batch()."""
        hc_stream = HankelCount(r_max=3)
        for seq in succ_corpus:
            hc_stream.update(seq)

        hc_batch = HankelCount(r_max=3)
        hc_batch.update_batch(succ_corpus)

        assert set(hc_stream.all_contexts()) == set(hc_batch.all_contexts())
        assert hc_stream.vocabulary() == hc_batch.vocabulary()

        for ctx in hc_stream.all_contexts():
            d_stream = hc_stream.get_distribution(ctx)
            d_batch = hc_batch.get_distribution(ctx)
            assert set(d_stream.keys()) == set(d_batch.keys()), (
                f"Context {ctx!r}: atom sets differ"
            )
            for atom in d_stream:
                assert abs(d_stream[atom] - d_batch[atom]) < 1e-12, (
                    f"Context {ctx!r}, atom {atom!r}: "
                    f"stream={d_stream[atom]:.6f} batch={d_batch[atom]:.6f}"
                )


# ---------------------------------------------------------------------------
# Roadmap test 1: at least 3 distinct neighbourhood patterns
# ---------------------------------------------------------------------------

class TestDistinctNeighbourhoodPatterns:
    def test_at_least_3_distinct_contexts_at_r1(self, hc_trained):
        """The math corpus must produce at least 3 distinct r=1 neighbourhood patterns.

        In succ/pred sequences at r=1 we expect at minimum:
          - context right-of-operator (left neighbour is 'succ' or 'pred')
          - context left-of-'eq'
          - context right-of-'eq'
        """
        r1_contexts = hc_trained.contexts_at_radius(1)
        assert len(r1_contexts) >= 3, (
            f"Expected at least 3 r=1 contexts, got {len(r1_contexts)}: {r1_contexts[:10]}"
        )

    def test_operator_context_exists(self, hc_trained):
        """There must be at least one context whose key contains 'succ' or 'pred'."""
        contexts = hc_trained.all_contexts()
        operator_contexts = [c for c in contexts if 'succ' in c or 'pred' in c]
        assert len(operator_contexts) > 0, (
            "No context contains 'succ' or 'pred' — operator context not found"
        )

    def test_eq_context_exists(self, hc_trained):
        """There must be at least one context whose key contains 'eq'."""
        contexts = hc_trained.all_contexts()
        eq_contexts = [c for c in contexts if ',eq' in c or '|eq' in c]
        assert len(eq_contexts) > 0, "No context containing 'eq' found"


# ---------------------------------------------------------------------------
# Roadmap test 2: digit-after-succ context assigns high probability to digits
# ---------------------------------------------------------------------------

class TestDigitAfterSuccContext:
    DIGITS = set('0123456789')
    OPERATORS = {'succ', 'pred', 'add', 'sub', 'mul', 'eq'}

    def test_succ_context_dominated_by_digits(self, hc_trained):
        """In the r=1 context immediately right of 'succ', digits should dominate.

        The position immediately after 'succ' in 'succ N eq N+1' has left
        neighbour 'succ' and right neighbour = first digit of N.  We look for
        contexts with left-neighbour = 'succ' and check that digit atoms get
        collectively higher probability than operators.
        """
        # Find contexts where left-1 neighbour is 'succ' (offset -1 = 'succ')
        succ_left_contexts = [
            c for c in hc_trained.contexts_at_radius(1)
            if '-1,succ' in c
        ]
        assert len(succ_left_contexts) > 0, (
            "No r=1 context found with left neighbour 'succ'"
        )
        for ctx in succ_left_contexts:
            dist = hc_trained.get_distribution(ctx)
            digit_mass = sum(p for a, p in dist.items() if a in self.DIGITS)
            op_mass = sum(p for a, p in dist.items() if a in self.OPERATORS)
            assert digit_mass > op_mass, (
                f"Context {ctx!r}: digit mass {digit_mass:.3f} should exceed "
                f"operator mass {op_mass:.3f}. Distribution: {dist}"
            )

    def test_eq_context_contains_digits_and_operators(self, hc_trained):
        """Contexts immediately left of 'eq' should see both digits and operators.

        In 'succ N eq N+1' at r=1, the position right before 'eq' has right
        neighbour 'eq'.  The atom at that position is always a digit (last digit
        of N).  So get_distribution should give high probability to digits.
        """
        eq_right_contexts = [
            c for c in hc_trained.contexts_at_radius(1)
            if '1,eq' in c
        ]
        assert len(eq_right_contexts) > 0, (
            "No r=1 context with right neighbour 'eq'"
        )
        for ctx in eq_right_contexts:
            dist = hc_trained.get_distribution(ctx)
            digit_mass = sum(p for a, p in dist.items() if a in self.DIGITS)
            # In succ/pred corpus, position left of 'eq' is always a digit
            assert digit_mass > 0.5, (
                f"Context {ctx!r}: expected digit mass > 0.5, got {digit_mass:.3f}"
            )


# ---------------------------------------------------------------------------
# Smoke test: summary() doesn't crash
# ---------------------------------------------------------------------------

def test_summary_runs(hc_trained):
    s = hc_trained.summary()
    assert 'HankelCount' in s
    assert 'sequences' in s
    assert 'r_max=3' in s
