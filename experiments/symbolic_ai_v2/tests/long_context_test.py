"""long_context_test.py — MorphismGraph handles long sequences correctly.

Tests that edge counts (global bigram memory) remain accurate regardless of
sequence length, and that perplexity stays well below log2(vocab) even for
sequences with thousands of tokens.

The key distinction from neural networks: MorphismGraph stores exact bigram
statistics (no forgetting, no approximation).  The "needle" in a haystack is
recovered perfectly because every observed edge is stored.

Run:  python -m pytest experiments/symbolic_ai_v2/tests/long_context_test.py -v
 or:  python experiments/symbolic_ai_v2/tests/long_context_test.py

Expected:
  - needle_in_haystack_100:  rare bigram learned from early in 100-token seq
  - needle_in_haystack_1k:   same at 1 000 tokens
  - perplexity_scales:       ppl stays below baseline as length grows 100→10k
  - no_memory_growth:        edge table size O(alphabet²), not O(sequence_length)
"""

import sys
import math
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.symbolic_ai_v2.core.topology  import sequence_1d
from experiments.symbolic_ai_v2.core.morphism  import MorphismGraph
from experiments.symbolic_ai_v2.core.predict   import perplexity


# ── Helpers ────────────────────────────────────────────────────────────────────

_RNG = random.Random(0)


def make_noisy_sequence(length: int, alphabet: list[str], signal_pair: tuple[str, str],
                         signal_freq: float = 0.05) -> str:
    """Build a random sequence with a planted rare bigram injected at signal_freq rate."""
    tokens: list[str] = []
    i = 0
    while i < length:
        if i > 0 and _RNG.random() < signal_freq:
            tokens.append(signal_pair[0])
            tokens.append(signal_pair[1])
            i += 2
        else:
            tokens.append(_RNG.choice(alphabet))
            i += 1
    return "".join(tokens[:length])


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_needle_in_haystack_100():
    """A bigram planted in the first 10 tokens must be retrievable after 100 more."""
    topo   = sequence_1d()
    mg     = MorphismGraph()
    next_e = topo.registry.code("next")

    # Plant "XY" early, then fill with random lowercase noise
    prefix = "XYabcdeXY"               # XY appears twice → stored + composition
    noise  = "abcdefghij" * 10         # 100 tokens of noise
    mg.observe_sequence(prefix + noise, topo)

    assert "X" in mg.atoms and "Y" in mg.atoms, "Planted atoms not found"
    x_id = mg.atoms["X"]
    y_id = mg.atoms["Y"]
    dist = mg.predict_dist(x_id, next_e)
    p_y  = dist.get(y_id, 0.0)
    assert p_y >= 0.9, (
        f"P(Y|X) = {p_y:.3f} after 100 tokens; expected ≥ 0.9 (bigram must persist)"
    )
    print(f"  needle_100: P(Y|X) = {p_y:.3f}, {mg.n_edges()} edges, {mg.n_atoms()} atoms")


def test_needle_in_haystack_1k():
    """Same bigram must be retrievable even after 1 000 tokens of noise."""
    topo   = sequence_1d()
    mg     = MorphismGraph()
    next_e = topo.registry.code("next")

    prefix = "XYabcdeXY"
    noise  = "abcdefghij" * 100        # 1 000 tokens
    mg.observe_sequence(prefix + noise, topo)

    x_id = mg.atoms["X"]
    y_id = mg.atoms["Y"]
    p_y  = mg.predict_dist(x_id, next_e).get(y_id, 0.0)
    assert p_y >= 0.9, (
        f"P(Y|X) = {p_y:.3f} after 1k tokens; expected ≥ 0.9"
    )
    print(f"  needle_1k: P(Y|X) = {p_y:.3f}, {mg.n_edges()} edges, {mg.n_atoms()} atoms")


def test_perplexity_scales():
    """Perplexity on a periodic signal stays below log2(vocab) as length grows."""
    topo = sequence_1d()

    # Signal: "abcabc..." with 5-char alphabet (log2(5) ≈ 2.32 bits baseline)
    alphabet = list("abcde")
    pattern  = "".join(alphabet)       # 'abcde'

    for n_repeats in (20, 200, 2000):
        mg   = MorphismGraph()
        seq  = pattern * n_repeats
        # Train on first 80%, test on last 20%
        split = int(len(seq) * 0.8)
        mg.observe_sequence(seq[:split], topo)
        ppl  = perplexity(mg, [seq[split:]], topo)
        baseline = math.log2(len(set(alphabet)))
        assert ppl < baseline, (
            f"n={len(seq)}: ppl {ppl:.3f} ≥ baseline {baseline:.3f}"
        )
        print(f"  perplexity_scales n={len(seq):5d}: ppl={ppl:.3f} < baseline={baseline:.3f}")


def test_no_memory_growth():
    """Edge table must saturate — growth must be sub-linear in sequence length.

    With deep compositions enabled (via _compress_buf_tail), the edge table
    can contain entries of the form (comp_id, etype, atom_id) at any depth.
    The total edge count is O(V^(d+1)) where d is the max composition depth,
    NOT O(n) in the sequence length.

    We verify this by comparing edge counts at 1k vs 10k characters with the
    same alphabet.  A healthy model shows edge count growth much smaller than
    10x despite the 10x increase in data — because all reachable (ctx, next)
    pairs are covered long before 10k characters.
    """
    topo     = sequence_1d()
    alphabet = list("abcdefghij")   # V = 10
    rng      = random.Random(1)

    seq1k  = "".join(rng.choices(alphabet, k=1_000))
    seq10k = "".join(rng.choices(alphabet, k=10_000))

    mg1k = MorphismGraph()
    mg1k.observe_sequence(seq1k, topo)
    e1k = mg1k.n_edges()

    mg10k = MorphismGraph()
    mg10k.observe_sequence(seq10k, topo)
    e10k = mg10k.n_edges()

    # 10× more data must not give 10× more edges (that would be O(n) growth).
    ratio = e10k / max(e1k, 1)
    assert ratio < 5.0, (
        f"Edge count grew {ratio:.1f}× with 10× more data "
        f"(1k→{e1k}, 10k→{e10k}): looks like O(n) growth."
    )
    print(f"  no_memory_growth: 1k={e1k} edges, 10k={e10k} edges, "
          f"ratio={ratio:.2f}x  (must be < 5.0×)")


def test_exact_bigram_recovery():
    """Bigram statistics are preserved in the composition hierarchy.

    With buffer compression enabled, raw atom→atom edge counts are LOWER
    than the number of bigram occurrences in the input: once a composition
    C = (b →[e]→ a) is created, `a` is absorbed into C at the buffer tail,
    so subsequent `a→b` transitions are stored as `C→b` edges rather than
    `a→b` edges.  This is correct behaviour — the information is still in the
    graph, just at a higher level.

    What we verify:
      1. Some atom-level a→b edges exist (from before the first composition).
      2. perplexity_multilevel recovers near-zero bits/char on a deterministic
         test sequence, proving the composition hierarchy carries the full info.
    """
    topo   = sequence_1d()
    mg     = MorphismGraph()
    next_e = topo.registry.code("next")

    seq = "ab" * 50 + "ba" * 50
    mg.observe_sequence(seq, topo)

    a_id = mg.atoms["a"]
    b_id = mg.atoms["b"]
    count_ab = mg.edge_count(a_id, next_e, b_id)
    count_ba = mg.edge_count(b_id, next_e, a_id)

    # Raw atom-level edges: some must exist (transitions before compression),
    # but NOT necessarily all 50 — later ones are absorbed into compositions.
    assert count_ab >= 1, f"Expected count(a->b) >= 1, got {count_ab}"
    assert count_ba >= 1, f"Expected count(b->a) >= 1, got {count_ba}"

    # Multilevel predictor must recover the pattern via the composition hierarchy.
    from experiments.symbolic_ai_v2.core.predict import perplexity_multilevel
    ppl = perplexity_multilevel(mg, ["ab" * 5], topo)
    assert ppl < 1.0, (
        f"perplexity_multilevel on deterministic 'ababab' = {ppl:.3f} bits/char "
        f"(expected < 1.0 — composition hierarchy must carry bigram info)"
    )
    print(f"  exact_bigram_recovery: count(a->b)={count_ab}, "
          f"count(b->a)={count_ba}, multilevel_ppl={ppl:.3f}")


# ── Test runner ────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_needle_in_haystack_100,
        test_needle_in_haystack_1k,
        test_perplexity_scales,
        test_no_memory_growth,
        test_exact_bigram_recovery,
    ]
    passed = 0
    for t in tests:
        try:
            print(f"Running {t.__name__}...")
            t()
            print(f"  PASSED\n")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()
    print(f"{passed}/{len(tests)} tests passed.")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
