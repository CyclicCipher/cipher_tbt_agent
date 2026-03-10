"""topology_test.py — same core code handles 1D sequence and 2D grid.

Run:  python -m pytest experiments/symbolic_ai_v2/tests/topology_test.py -v
 or:  python experiments/symbolic_ai_v2/tests/topology_test.py

Expected:
  - sequence_1d: compositions discovered for repeated trigrams
  - grid_2d:     edge counts populated from a 3×3 grid
  - Same MorphismGraph code serves both topologies with zero changes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.symbolic_ai_v2.core.topology  import sequence_1d, grid_2d
from experiments.symbolic_ai_v2.core.morphism  import MorphismGraph


def test_sequence_1d_basic():
    """Compositions should appear after a triple recurs twice."""
    topo = sequence_1d()
    mg   = MorphismGraph()

    # "ababab" — the triple (a,next,b,next,a) appears twice → composition for (b→a)
    mg.observe_sequence("ababab", topo)

    assert mg.n_atoms() == 2, f"Expected 2 atoms (a,b), got {mg.n_atoms()}"
    assert mg.n_edges() >= 2, f"Expected edges (a→b, b→a), got {mg.n_edges()}"
    assert mg.n_compositions() >= 1, (
        f"Expected ≥1 composition from repeated triple, got {mg.n_compositions()}"
    )
    print(f"  sequence_1d: {mg.summary()}")


def test_sequence_1d_boundaries():
    """Segment boundaries should fire when a new triple is first seen."""
    topo = sequence_1d()
    mg   = MorphismGraph()
    boundaries = []
    mg.on_segment(lambda chunk, g: boundaries.append(len(chunk)))

    mg.observe_sequence("abcd", topo)

    # Every new triple triggers a boundary; 'abcd' has 2 triples: (a,b,c) and (b,c,d)
    assert len(boundaries) >= 1, f"Expected ≥1 boundary, got {len(boundaries)}"
    print(f"  sequence_1d boundaries: {len(boundaries)} chunks, sizes {boundaries}")


def test_sequence_1d_predict():
    """After learning 'ababab', P(b | a, next) should be high."""
    topo = sequence_1d()
    mg   = MorphismGraph()
    mg.observe_sequence("ababababab", topo)

    a_id   = mg.atoms["a"]
    next_e = topo.registry.code("next")
    dist   = mg.predict_dist(a_id, next_e)

    assert dist, "predict_dist returned empty dict"
    top_tgt, top_prob = max(dist.items(), key=lambda kv: kv[1])
    top_val = mg.symbols[top_tgt].value  # type: ignore[attr-defined]
    assert top_val == "b", f"Expected top prediction 'b', got '{top_val}' (p={top_prob:.3f})"
    assert top_prob > 0.9, f"P(b|a) should be >0.9 after 'ababababab', got {top_prob:.3f}"
    print(f"  sequence_1d predict: P(b|a, next) = {top_prob:.3f}")


def test_sequence_1d_perplexity():
    """Perplexity on a repeated sequence should be well below log2(vocab)."""
    import math
    from experiments.symbolic_ai_v2.core.predict import perplexity

    topo  = sequence_1d()
    mg    = MorphismGraph()
    train = ["abcabcabcabc", "abcabcabc"]
    test  = ["abcabc"]
    for seq in train:
        mg.observe_sequence(seq, topo)

    ppl = perplexity(mg, test, topo)
    vocab_size = mg.n_atoms()
    baseline   = math.log2(vocab_size) if vocab_size > 1 else 1.0

    assert ppl < baseline, (
        f"Perplexity {ppl:.3f} should be < log2({vocab_size}) = {baseline:.3f}"
    )
    print(f"  sequence_1d perplexity: {ppl:.3f} bits/token (baseline {baseline:.3f})")


def test_grid_2d_edge_counts():
    """grid_2d stream_edges should populate edge counts for all spatial pairs."""
    topo = grid_2d()
    mg   = MorphismGraph()

    # 3×3 grid of distinct values
    grid = [
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g", "h", "i"],
    ]
    for src_val, etype, tgt_val in topo.stream_edges(grid):
        mg.observe_edge(src_val, etype, tgt_val)

    # Every pixel should have outgoing edges to its neighbors
    assert mg.n_atoms() == 9, f"Expected 9 atoms for 3×3 grid, got {mg.n_atoms()}"
    assert mg.n_edges() > 0, "Expected edges from grid_2d"

    # 'a' at (0,0) has 2 neighbors: right→'b', down→'d'
    a_id   = mg.atoms["a"]
    rc     = topo.registry.code("right")
    dc     = topo.registry.code("down")
    assert mg.edge_count(a_id, rc, mg.atoms["b"]) == 1, "Expected edge a→right→b"
    assert mg.edge_count(a_id, dc, mg.atoms["d"]) == 1, "Expected edge a→down→d"
    print(f"  grid_2d: {mg.n_edges()} directed edges for 3×3 grid")


def test_same_algorithm_both_topologies():
    """The MorphismGraph class is identical; only the Topology changes."""
    topo1d = sequence_1d()
    topo2d = grid_2d()
    mg1    = MorphismGraph()
    mg2    = MorphismGraph()

    mg1.observe_sequence("hello world hello world", topo1d)

    grid = [["X", "Y"], ["Z", "W"]]
    for src, et, tgt in topo2d.stream_edges(grid):
        mg2.observe_edge(src, et, tgt)

    # Both use the same class with no domain-specific code
    assert type(mg1) is type(mg2) is MorphismGraph
    print(f"  1D: {mg1.summary()}")
    print(f"  2D: {mg2.summary()}")


# ── Test runner ───────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_sequence_1d_basic,
        test_sequence_1d_boundaries,
        test_sequence_1d_predict,
        test_sequence_1d_perplexity,
        test_grid_2d_edge_counts,
        test_same_algorithm_both_topologies,
    ]
    passed = 0
    for t in tests:
        try:
            print(f"Running {t.__name__}...")
            t()
            print(f"  PASSED\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
    print(f"{passed}/{len(tests)} tests passed.")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
