"""Language corpus benchmark — CTKG Predictor on real Latin text.

Trains the CTKG Predictor on a sample of Caesar's De Bello Gallico
(word-level tokenization), then evaluates next-word prediction on a
held-out test split.

Design:
  - Tokenize: lowercase, strip punctuation → word sequences.
    Sentences (or fixed-window sliding sequences) are the training units.
  - Train/test split: first 80% sentences for training, last 20% for test.
  - Metric:
      accuracy   — fraction of test positions where argmax prediction = truth
      random_acc — 1 / |vocab| (expected accuracy under uniform random)
      lift       — accuracy / random_acc
  - Pass criterion: lift >= 2.0  (predict at least 2× better than chance)
    Also reports: top-5 accuracy and per-position entropy of predictions.

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/corpus_benchmark.py
"""

from __future__ import annotations

import os
import re
import sys
import random

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import discover_concepts
from experiments.symbolic_ai_v2.ctkg.learning.morphism_discover import discover_morphisms
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import discover_processes
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the Latin text corpus
_CORPUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "corpus", "latin books"
)
# Use Caesar's Gallic Wars (largest single text, clean prose)
_CORPUS_FILE = os.path.join(
    _CORPUS_DIR,
    "COMMENTARIORUM LIBRI VII DE BELLO GALLICO CUM A. HIRTI SUPPLEMENTO.txt",
)

R = 3           # context radius: captures trigram context (left-3 to right-3)
K = 5           # k-nearest for Kan extension
MAX_SEQS = 4000 # cap on training sequences (sentences) to keep runtime reasonable
TRAIN_FRAC = 0.8
MIN_LEN = 4     # minimum tokens per sentence to include
MAX_LEN = 30    # maximum tokens per sentence (longer sentences truncated)
RANDOM_SEED = 42

# Pass criterion
MIN_LIFT = 2.0


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def _tokenize_text(text: str) -> list[list[str]]:
    """Convert raw Latin text to a list of token sequences (one per sentence).

    Splitting on sentence-final punctuation (.!?) and paragraph breaks.
    Each sentence is lowercased and stripped of non-alphanumeric characters.
    """
    # Split on sentence boundaries: periods, exclamation/question marks
    # and blank lines (paragraph breaks)
    raw_sentences = re.split(r"[.!?]\s+|\n\n+", text)
    sequences: list[list[str]] = []
    for sent in raw_sentences:
        # Lowercase and remove non-alphanumeric (keep spaces)
        cleaned = _PUNCT_RE.sub(" ", sent.lower())
        tokens = cleaned.split()
        # Filter very short tokens (Roman numerals like 'i', 'v' are fine)
        tokens = [t for t in tokens if len(t) >= 1]
        if len(tokens) < MIN_LEN:
            continue
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
        sequences.append(tokens)
    return sequences


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> bool:
    """Run the corpus benchmark.  Returns True if the lift criterion is met."""

    # ---- Load and tokenize corpus ----
    if not os.path.exists(_CORPUS_FILE):
        print(f"ERROR: corpus file not found: {_CORPUS_FILE}")
        return False

    with open(_CORPUS_FILE, encoding="utf-8", errors="replace") as f:
        text = f.read()

    all_seqs = _tokenize_text(text)
    print(f"Corpus: {len(text):,} chars, {len(all_seqs):,} sentences")

    # Shuffle with fixed seed for reproducibility
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_seqs)

    # Cap total sequences
    if len(all_seqs) > MAX_SEQS:
        all_seqs = all_seqs[:MAX_SEQS]

    n_train = int(len(all_seqs) * TRAIN_FRAC)
    train_seqs = all_seqs[:n_train]
    test_seqs  = all_seqs[n_train:]

    vocab: set[str] = set()
    for seq in train_seqs:
        vocab.update(seq)
    vocab_size = len(vocab)

    print(f"Train: {len(train_seqs):,} seqs | Test: {len(test_seqs):,} seqs")
    print(f"Vocab: {vocab_size:,} unique tokens")
    print(f"r={R}, k={K}")
    print()

    # ---- Build HankelCount on training sequences ----
    hc = HankelCount(r_max=R)
    hc.update_batch(train_seqs)

    # ---- FCA ----
    lattices = discover_concepts(
        hankel=hc,
        r_levels=[R],
        lambda_productivity=0.1,
        merge_threshold=0.15,
        min_support=2.0,
    )
    lattice = lattices[0]
    print(f"FCA concepts: {len(lattice.concepts)}")

    # ---- Morphism discovery ----
    mg = discover_morphisms(train_seqs, hc, lattice, r=R)

    # ---- Process rules (none expected for natural language) ----
    process_rules = discover_processes(train_seqs, op_atoms=[])

    # ---- Build Predictor ----
    pred = Predictor(
        hankel=hc,
        lattice=lattice,
        morphism_graph=mg,
        process_rules=process_rules,
        k_neighbours=K,
        r=R,
    )

    # ---- Evaluate on test sequences ----
    # For each test sentence, predict each token given its left context.
    # We do NOT feed right context (consistent with how generate() works).
    n_correct_top1 = 0
    n_correct_top5 = 0
    n_total = 0
    entropy_sum = 0.0

    for seq in test_seqs:
        for pos in range(1, len(seq)):
            prefix = seq[:pos]
            truth  = seq[pos]

            dist = pred.predict_next(prefix)
            if not dist:
                continue

            # Top-1
            best = max(dist, key=lambda x: dist[x])
            if best == truth:
                n_correct_top1 += 1

            # Top-5
            top5 = sorted(dist, key=lambda x: dist[x], reverse=True)[:5]
            if truth in top5:
                n_correct_top5 += 1

            # Entropy of the predicted distribution
            total_w = sum(dist.values())
            if total_w > 0:
                import math
                for w in dist.values():
                    p = w / total_w
                    if p > 0:
                        entropy_sum -= p * math.log2(p)

            n_total += 1

    if n_total == 0:
        print("ERROR: no test positions evaluated")
        return False

    acc_top1  = n_correct_top1 / n_total
    acc_top5  = n_correct_top5 / n_total
    random_acc = 1.0 / vocab_size
    lift       = acc_top1 / random_acc if random_acc > 0 else float("inf")
    avg_entropy = entropy_sum / n_total

    print(f"{'Metric':<22}  {'Value':>10}")
    print("-" * 36)
    print(f"{'Test positions':<22}  {n_total:>10,}")
    print(f"{'Top-1 accuracy':<22}  {acc_top1:>9.3%}")
    print(f"{'Top-5 accuracy':<22}  {acc_top5:>9.3%}")
    print(f"{'Random baseline':<22}  {random_acc:>9.4%}  (1/{vocab_size})")
    print(f"{'Lift (top-1/random)':<22}  {lift:>10.2f}x")
    print(f"{'Avg pred entropy':<22}  {avg_entropy:>9.3f} bits")
    print()

    passed = lift >= MIN_LIFT
    if passed:
        print(f"RESULT: PASS  (lift {lift:.2f}x >= {MIN_LIFT:.1f}x threshold)")
    else:
        print(f"RESULT: FAIL  (lift {lift:.2f}x < {MIN_LIFT:.1f}x threshold)")
    return passed


if __name__ == "__main__":
    ok = run_benchmark()
    sys.exit(0 if ok else 1)
