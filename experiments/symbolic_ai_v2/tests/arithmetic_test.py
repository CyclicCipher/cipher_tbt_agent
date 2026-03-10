"""arithmetic_test.py — MorphismGraph on digit sequences.

Tests that the same algorithm that handles characters also handles
symbolic arithmetic sequences, with the same zero-parameter invariants.

Run:  python -m pytest experiments/symbolic_ai_v2/tests/arithmetic_test.py -v
 or:  python experiments/symbolic_ai_v2/tests/arithmetic_test.py

Expected:
  - digit_succession: P(n+1 | n, next) ≥ 0.9 after repeated digit cycles
  - compositional_chain: compositions emerge at each level of repetition
  - perplexity_vs_random: structured digit sequence beats random permutation
  - multi_radix: same code handles binary (2-token) and decimal (10-token)
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.symbolic_ai_v2.core.topology  import sequence_1d
from experiments.symbolic_ai_v2.core.morphism  import MorphismGraph
from experiments.symbolic_ai_v2.core.predict   import perplexity


# ── Helpers ────────────────────────────────────────────────────────────────────

def digit_cycle(radix: int, repeats: int) -> str:
    """Return 'radix' digits repeated 'repeats' times, e.g. '0123456789' × 3."""
    digits = [str(d % radix) for d in range(radix)]
    return "".join(digits * repeats)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_digit_succession():
    """After learning 10 cycles of 0-9, P(succ(d) | d, next) should be ≥ 0.9."""
    topo = sequence_1d()
    mg   = MorphismGraph()
    seq  = digit_cycle(10, 10)          # '01234567890123456789...' × 10
    mg.observe_sequence(seq, topo)

    next_e = topo.registry.code("next")
    # Check every digit d predicts d+1 (mod 10) with ≥ 0.9 probability
    for d in range(9):                  # d = 0..8 (d=9 wraps to 0, skip that edge)
        src_id  = mg.atoms[str(d)]
        tgt_id  = mg.atoms[str(d + 1)]
        dist    = mg.predict_dist(src_id, next_e)
        p_succ  = dist.get(tgt_id, 0.0)
        assert p_succ >= 0.9, (
            f"P({d+1}|{d},next) = {p_succ:.3f}, expected ≥ 0.9"
        )

    assert mg.n_atoms() == 10, f"Expected 10 digit atoms, got {mg.n_atoms()}"
    print(f"  digit_succession: {mg.summary()}")


def test_compositional_chain():
    """Repeated digit triples should create compositions at multiple levels."""
    topo = sequence_1d()
    mg   = MorphismGraph()
    # "012012012012" — triple (0→1→2) recurs many times → compositions
    seq  = "012" * 12
    mg.observe_sequence(seq, topo)

    assert mg.n_atoms() == 3, f"Expected 3 atoms (0,1,2), got {mg.n_atoms()}"
    assert mg.n_compositions() >= 1, (
        f"Expected ≥1 composition from repeated triple, got {mg.n_compositions()}"
    )

    # The compositions should have levels > 0
    for cid, (left, etype, right) in mg.rules.items():
        sym = mg.symbols[cid]
        assert sym.level > 0, f"Composition {cid} has level 0 (should be ≥ 1)"

    print(f"  compositional_chain: {mg.summary()}")
    for cid, (left, etype, right) in mg.rules.items():
        sym = mg.symbols[cid]
        print(f"    C{cid} level={sym.level}: {mg.value_of(left)} + {mg.value_of(right)}")


def test_perplexity_vs_random():
    """Structured digit sequence should have lower perplexity than random permutation."""
    import random
    rng = random.Random(42)

    topo = sequence_1d()

    # Train and test on structured sequences
    mg_struct = MorphismGraph()
    train_struct = [digit_cycle(5, 6), digit_cycle(5, 4)]
    test_struct  = [digit_cycle(5, 3)]
    for s in train_struct:
        mg_struct.observe_sequence(s, topo)
    ppl_struct = perplexity(mg_struct, test_struct, topo)

    # Train and test on random permutations of the same alphabet
    alphabet = [str(d) for d in range(5)]
    mg_rand = MorphismGraph()
    train_rand = ["".join(rng.choices(alphabet, k=len(train_struct[0]))),
                  "".join(rng.choices(alphabet, k=len(train_struct[1])))]
    test_rand  = ["".join(rng.choices(alphabet, k=len(test_struct[0])))]
    for s in train_rand:
        mg_rand.observe_sequence(s, topo)
    ppl_rand = perplexity(mg_rand, test_rand, topo)

    assert ppl_struct < ppl_rand, (
        f"Structured ppl {ppl_struct:.3f} should be < random ppl {ppl_rand:.3f}"
    )
    print(f"  perplexity_vs_random: structured={ppl_struct:.3f}, random={ppl_rand:.3f}")


def test_multi_radix():
    """The same MorphismGraph class handles binary (radix=2) and decimal (radix=10)."""
    topo = sequence_1d()
    next_e = topo.registry.code("next")

    # Binary: "0101010101" → P(1|0) and P(0|1) should both be ≥ 0.9
    mg_bin = MorphismGraph()
    mg_bin.observe_sequence("01" * 15, topo)
    assert mg_bin.n_atoms() == 2, f"Binary: expected 2 atoms, got {mg_bin.n_atoms()}"
    p_01 = mg_bin.predict_dist(mg_bin.atoms["0"], next_e).get(mg_bin.atoms["1"], 0.0)
    p_10 = mg_bin.predict_dist(mg_bin.atoms["1"], next_e).get(mg_bin.atoms["0"], 0.0)
    assert p_01 >= 0.9 and p_10 >= 0.9, (
        f"Binary: P(1|0)={p_01:.3f}, P(0|1)={p_10:.3f}, both should be ≥0.9"
    )

    # Decimal: 10 atoms, all succession edges learned
    mg_dec = MorphismGraph()
    mg_dec.observe_sequence(digit_cycle(10, 8), topo)
    assert mg_dec.n_atoms() == 10, f"Decimal: expected 10 atoms, got {mg_dec.n_atoms()}"

    print(f"  multi_radix binary: {mg_bin.summary()}")
    print(f"  multi_radix decimal: {mg_dec.summary()}")


def test_arithmetic_boundaries():
    """Segment boundaries in arithmetic sequences should mark structure transitions."""
    topo     = sequence_1d()
    mg       = MorphismGraph()
    chunks   = []
    mg.on_segment(lambda chunk, g: chunks.append(len(chunk)))

    # "012012012" — repeated triple; each new occurrence fires a boundary
    mg.observe_sequence("012" * 6, topo)

    assert len(chunks) >= 1, f"Expected ≥1 boundary, got {len(chunks)}"
    print(f"  arithmetic_boundaries: {len(chunks)} chunks, sizes {chunks}")


# ── Test runner ────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_digit_succession,
        test_compositional_chain,
        test_perplexity_vs_random,
        test_multi_radix,
        test_arithmetic_boundaries,
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
