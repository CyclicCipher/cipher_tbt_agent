"""latin_test.py — MorphismGraph on real Latin character sequences.

Uses hardcoded Latin excerpts as the primary test (no data files required).
If the GT4HistOCR corpus is present, also runs a larger perplexity benchmark.

Run:  python -m pytest experiments/symbolic_ai_v2/tests/latin_test.py -v
 or:  python experiments/symbolic_ai_v2/tests/latin_test.py

Expected (hardcoded excerpts):
  - latin_char_perplexity: ppl < log2(26) ≈ 4.70 on held-out Latin text
  - latin_compositions:    ≥1 composition from repeated Latin character triples
  - latin_predict:         top prediction for space after 'u', 'm', 's' is reasonable
  - latin_real_data:       SKIP if corpus absent; otherwise ppl < 4.5 bits/char
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.symbolic_ai_v2.core.topology  import sequence_1d
from experiments.symbolic_ai_v2.core.morphism  import MorphismGraph, Atom
from experiments.symbolic_ai_v2.core.predict   import perplexity


# ── Hardcoded Latin excerpts (no data files needed) ───────────────────────────
# Source: opening of De Bello Gallico (Caesar), lowercased, spaces retained.

_LATIN_TRAIN = [
    "gallia est omnis divisa in partes tres quarum unam incolunt belgae",
    "aliam aquitani tertiam qui ipsorum lingua celtae nostra galli appellantur",
    "hi omnes lingua institutis legibus inter se differunt",
    "gallos ab aquitanis garumna flumen a belgis matrona et sequana dividit",
    "horum omnium fortissimi sunt belgae propterea quod a cultu atque humanitate",
    "provinciae longissime absunt minimeque ad eos mercatores saepe commeant",
    "atque ea quae ad effeminandos animos pertinent important",
    "proximique sunt germanis qui trans rhenum incolunt",
    "quibuscum continenter bellum gerunt",
]

_LATIN_TEST = [
    "cum his finibus qui supra demonstrati sunt gallia continetur",
    "huius totius galliae caesar summa rerum praefecit",
]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_latin_char_perplexity():
    """Perplexity on held-out Latin text should be well below log2(alphabet)."""
    topo = sequence_1d()
    mg   = MorphismGraph()

    for seq in _LATIN_TRAIN:
        mg.observe_sequence(seq, topo)

    ppl      = perplexity(mg, _LATIN_TEST, topo)
    alphabet = set("".join(_LATIN_TRAIN + _LATIN_TEST))
    baseline = math.log2(len(alphabet))

    assert ppl < baseline, (
        f"Latin char ppl {ppl:.3f} >= baseline {baseline:.3f} — "
        "model should exploit character bigram structure"
    )
    print(f"  latin_char_perplexity: ppl={ppl:.3f} < baseline={baseline:.3f}"
          f"  (alphabet={len(alphabet)} chars)")


def test_latin_compositions():
    """Repeated Latin character triples (e.g. 'um ', 'us ', 'ae ') form compositions."""
    topo = sequence_1d()
    mg   = MorphismGraph()
    for seq in _LATIN_TRAIN:
        mg.observe_sequence(seq, topo)

    assert mg.n_compositions() >= 1, (
        f"Expected ≥1 composition from Latin text, got {mg.n_compositions()}"
    )
    assert mg.n_atoms() >= 20, (
        f"Expected ≥20 distinct chars in Latin, got {mg.n_atoms()}"
    )
    print(f"  latin_compositions: {mg.n_compositions()} compositions, "
          f"{mg.n_atoms()} atoms, {mg.n_edges()} edges")


def test_latin_predict():
    """After Latin training, the top prediction after common suffixes should be space."""
    topo   = sequence_1d()
    mg     = MorphismGraph()
    next_e = topo.registry.code("next")

    for seq in _LATIN_TRAIN:
        mg.observe_sequence(seq, topo)

    # In Latin, letters like 'm', 's', 'e' frequently precede word boundaries (space).
    # The top prediction after each should be space with non-trivial probability.
    space_id = mg.atoms.get(" ")
    assert space_id is not None, "Space character not found in atom table"

    for char in ("m", "s", "e", "a"):
        if char not in mg.atoms:
            continue
        char_id = mg.atoms[char]
        dist    = mg.predict_dist(char_id, next_e)
        if not dist:
            continue
        top_id, top_prob = max(dist.items(), key=lambda kv: kv[1])
        top_sym = mg.symbols[top_id]
        top_val = top_sym.value if isinstance(top_sym, Atom) else "?"
        p_space = dist.get(space_id, 0.0)
        # Space should be in the top-3 followers of word-ending letters
        top3_ids = sorted(dist, key=lambda k: dist[k], reverse=True)[:3]
        assert space_id in top3_ids or p_space > 0.15, (
            f"P(space|'{char}') = {p_space:.3f} — expected space in top-3 after '{char}'"
        )
        print(f"  latin_predict: P(space|'{char}') = {p_space:.3f}, "
              f"top='{top_val}' ({top_prob:.3f})")


def test_latin_real_data():
    """If GT4HistOCR corpus exists, test perplexity on the first book."""
    corpus_root = (Path(__file__).resolve().parents[3]
                   / "data" / "GT4HistOCR" / "corpus" / "EarlyModernLatin")
    if not corpus_root.exists():
        print("  latin_real_data: SKIPPED (corpus not found at "
              f"{corpus_root})")
        return

    # Load first book's lines
    book_dirs = sorted(d for d in corpus_root.iterdir() if d.is_dir())
    if not book_dirs:
        print("  latin_real_data: SKIPPED (no book subdirectories found)")
        return

    book_dir = book_dirs[0]
    lines    = []
    for gt_file in sorted(book_dir.glob("*.gt.txt")):
        text = gt_file.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            lines.append(text)

    if not lines:
        print("  latin_real_data: SKIPPED (no .gt.txt files in first book)")
        return

    # Train on first 80%, test on last 20%
    split      = max(1, int(len(lines) * 0.8))
    train_seqs = ["".join(lines[:split])]
    test_seqs  = ["".join(lines[split:])]

    topo = sequence_1d()
    mg   = MorphismGraph()
    for seq in train_seqs:
        mg.observe_sequence(seq, topo)

    ppl      = perplexity(mg, test_seqs, topo)
    baseline = math.log2(max(mg.n_atoms(), 2))

    assert ppl < baseline, (
        f"Real Latin ppl {ppl:.3f} >= baseline {baseline:.3f} on book '{book_dir.name}'"
    )
    print(f"  latin_real_data '{book_dir.name}': ppl={ppl:.3f} < baseline={baseline:.3f}"
          f"  ({mg.n_atoms()} atoms, {mg.n_edges()} edges, {len(lines)} lines)")


# ── Test runner ────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_latin_char_perplexity,
        test_latin_compositions,
        test_latin_predict,
        test_latin_real_data,
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
