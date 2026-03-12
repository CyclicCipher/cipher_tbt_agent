"""phase_g_test.py — Phase G: SpectralPredictor perplexity target.

One test (ROADMAP_REDESIGN §III.8):
  test_spectral_ppl_below_2 — SpectralPredictor(k_max=6) achieves
  < 2.0 bits/char on the held-out Latin corpus.

The MorphismGraph baseline comparison test has been removed following
MorphismGraph deprecation (ROADMAP_REDESIGN §IV.5).  The comparison result
(SpectralPredictor 1.94 < MorphismGraph bigram 3.49) is documented in
ROADMAP_REDESIGN §III.8.

Corpus: experiments/symbolic_ai_v2/corpus/latin books/ (always present,
no external data files required).  Uses an 80/20 character split over all
16 books concatenated.

This test is SLOW (train ≈ 20s, ppl ≈ 3s) and is marked no_ait to disable
the ActiveInference autouse fixture (now a no-op; kept for documentation).

Run:
  ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/phase_g_test.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

try:
    pytestmark = pytest.mark.no_ait
except AttributeError:
    pass  # running without pytest

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "latin books"
TRAIN_FRAC  = 0.8
K_MAX       = 6          # minimum k_max needed to clear the 2.0 bit target
PPL_TARGET  = 2.0        # bits/char — Phase G goal


# ── Shared fixture ────────────────────────────────────────────────────────────

def _load_split() -> tuple[list[str], list[str]]:
    """Return (train_chars, test_chars) as flat char lists."""
    assert CORPUS_DIR.exists(), f"Corpus dir not found: {CORPUS_DIR}"
    files = sorted(CORPUS_DIR.glob("*.txt"))
    assert files, "No .txt files in latin books corpus"
    all_text = "".join(
        f.read_text(encoding="utf-8", errors="replace") for f in files
    )
    split = int(len(all_text) * TRAIN_FRAC)
    return list(all_text[:split]), list(all_text[split:])


_TRAIN: list[str] | None = None
_TEST:  list[str] | None = None
_SP    = None  # SpectralPredictor, lazily trained once


def _get_predictor():
    """Lazily train the SpectralPredictor (expensive — reused across tests)."""
    global _TRAIN, _TEST, _SP
    if _SP is not None:
        return _SP, _TRAIN, _TEST
    from experiments.symbolic_ai_v2.core.spectral_predict import SpectralPredictor
    _TRAIN, _TEST = _load_split()
    _SP = SpectralPredictor.train([_TRAIN], k_max=K_MAX)
    return _SP, _TRAIN, _TEST


# ── Test 1: SpectralPredictor < 2.0 bits/char ────────────────────────────────

def test_spectral_ppl_below_2():
    """SpectralPredictor(k_max=6) must achieve < 2.0 bits/char on the
    held-out Latin corpus (Phase G target, ROADMAP_REDESIGN §III.8)."""
    sp, _train, test = _get_predictor()
    ppl = sp.perplexity(test)
    print(f"\n  spectral ppl = {ppl:.4f} bits/char  (target < {PPL_TARGET})")
    assert ppl < PPL_TARGET, (
        f"SpectralPredictor ppl {ppl:.4f} >= {PPL_TARGET} bits/char "
        f"(k_max={K_MAX}, rank={sp.ss.rank}, "
        f"train={len(_train):,} chars, test={len(test):,} chars)"
    )


