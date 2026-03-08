"""train_lang.py -- Build character bigram language model from GT4HistOCR text.

Reads all .gt.txt ground-truth transcription files, normalises the text
(same transforms as ocr_test.py), and trains a character-level bigram
language model.

No tokenization.  The model learns directly from raw character sequences,
capturing the graphemic regularities of Early Modern Latin and German Fraktur
without any linguistic knowledge engineered in:
  - Latin: Q always followed by U, AE/OE are common digraphs, endings in -um/-us/-is
  - Fraktur: SCH/ST/SP clusters, umlauts follow specific consonants
  - Shared: sentence-initial capitalisation patterns

The resulting LanguagePrior is used by ocr_test.py (--lang-prior) to
re-score GlyphReader output via Viterbi decoding.  This corrects systematic
visual errors by asking: "given what I see AND what characters typically
follow each other in this language, what is the most likely character here?"

Usage
-----
    # Train on EarlyModernLatin (default)
    python train_lang.py

    # Train on German Fraktur
    python train_lang.py --corpus RIDGES-Fraktur

    # Train on all corpora (combined model)
    python train_lang.py --corpus all

    # Limit to N lines (fast smoke test)
    python train_lang.py --n 500

    # Custom output path
    python train_lang.py --output lang_prior_latin.pkl

Then use in evaluation:
    python ocr_test.py --lang-prior lang_prior.pkl
    python ocr_test.py --reader glyph_reader_hist.pkl --lang-prior lang_prior.pkl
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_DATA_DIR = os.path.join(_HERE, "data", "GT4HistOCR", "corpus")

_ALL_CORPORA = [
    "EarlyModernLatin",
    "RIDGES-Fraktur",
    "Kallimachos",
    "dta19",
    "RefCorpus-ENHG-Incunabula",
]

# Import shared normalisation from ocr_test
from ocr_test import find_pairs, normalise_gt          # type: ignore[import]
from modalities.language_prior import LanguagePrior    # type: ignore[import]


# ---------------------------------------------------------------------------
# Collect GT text from corpus
# ---------------------------------------------------------------------------

def collect_texts(
    pairs:   List,
    verbose: bool = True,
) -> List[str]:
    """Read and normalise GT text from all (image, gt) pairs.

    Returns list of normalised strings (one per line).
    """
    texts = []
    n_empty = 0
    for img_path, gt_path in pairs:
        try:
            with open(gt_path, encoding="utf-8", errors="replace") as f:
                raw = f.read().strip()
            norm = normalise_gt(raw)
            if norm:
                texts.append(norm)
            else:
                n_empty += 1
        except Exception:
            n_empty += 1

    if verbose:
        n_chars = sum(len(t) for t in texts)
        print(f"  {len(texts)} lines collected ({n_empty} empty/missing), "
              f"{n_chars:,} characters total")

    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import io
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "utf-8").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Build a character bigram language model from GT4HistOCR "
            "transcription text.  Saves a LanguagePrior for use with "
            "ocr_test.py --lang-prior."
        )
    )
    parser.add_argument(
        "--corpus",
        default="EarlyModernLatin",
        help=(
            f"Corpus name, full path, or 'all'.  "
            f"Names: {', '.join(_ALL_CORPORA)}.  "
            "Default: EarlyModernLatin"
        ),
    )
    parser.add_argument("--n",       type=int, default=None,
                        help="Max line pairs per corpus (None=all).")
    parser.add_argument("--output",  default="lang_prior.pkl",
                        help="Output .pkl path (default: lang_prior.pkl).")
    parser.add_argument("--min-word-count", type=int, default=2,
                        help="Min word occurrences for vocabulary (default 2).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show top bigrams per corpus.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve corpus path(es)
    # ------------------------------------------------------------------
    if args.corpus.lower() == "all":
        corpus_dirs = [os.path.join(_DATA_DIR, c) for c in _ALL_CORPORA
                       if os.path.isdir(os.path.join(_DATA_DIR, c))]
        if not corpus_dirs:
            print(f"ERROR: no corpora found under {_DATA_DIR!r}")
            sys.exit(1)
    else:
        d = args.corpus
        if not os.path.isdir(d):
            d = os.path.join(_DATA_DIR, args.corpus)
        if not os.path.isdir(d):
            print(f"ERROR: corpus not found: {args.corpus!r}")
            print(f"  Available: "
                  f"{os.listdir(_DATA_DIR) if os.path.isdir(_DATA_DIR) else '(missing)'}")
            sys.exit(1)
        corpus_dirs = [d]

    # ------------------------------------------------------------------
    # Collect all GT text
    # ------------------------------------------------------------------
    all_texts: List[str] = []

    for corpus_dir in corpus_dirs:
        corpus_name = os.path.basename(corpus_dir)
        print(f"\n{'='*60}")
        print(f"  Corpus: {corpus_name}")
        print(f"{'='*60}")

        pairs = find_pairs(corpus_dir, max_n=args.n, shuffle=True, seed=42)
        print(f"  Found {len(pairs)} line pairs.")
        if not pairs:
            continue

        texts = collect_texts(pairs, verbose=True)
        all_texts.extend(texts)

    if not all_texts:
        print("\nERROR: no text collected.  Check data directory:")
        print(f"  {_DATA_DIR}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Train language model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training LanguagePrior on {len(all_texts):,} lines ...")
    print(f"{'='*60}")

    t0 = time.time()
    prior = LanguagePrior()
    prior.train(all_texts, min_word_count=args.min_word_count, verbose=True)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")
    print(f"  {prior.summary()}")

    # ------------------------------------------------------------------
    # Diagnostics: show top bigrams for a few common characters
    # ------------------------------------------------------------------
    print("\n  Top bigrams for common characters:")
    for sample_ch in list("etaoinsr "):
        top = prior.top_bigrams(sample_ch, n=6)
        top_str = "  ".join(
            f"{repr(c2)}:{p:.2f}" for c2, p in top
            if c2 not in ("\x02", "\x03")
        )
        print(f"    after {repr(sample_ch)}: {top_str}")

    # ------------------------------------------------------------------
    # Quick Viterbi smoke test
    # ------------------------------------------------------------------
    print("\n  Viterbi smoke test (pure language prior, alpha=0.0):")
    # Feed each character as a uniform visual observation -> LM drives sequence
    test_prefix = "qui"
    # Build candidates: all chars equally likely visually
    all_chars = prior._chars or []
    uniform = [(c, 1.0 / max(1, len(all_chars))) for c in all_chars]
    # For the known prefix, set the correct char to high confidence
    candidates = []
    for ch in test_prefix:
        cands = [(ch, 0.9)] + [(c, 0.01) for c in all_chars[:5] if c != ch]
        candidates.append(cands)
    # Then 3 unknown positions (uniform visual)
    for _ in range(3):
        candidates.append(uniform[:8])
    result = prior.viterbi(candidates, alpha=0.3)
    print(f"    Prefix {test_prefix!r} + 3 LM-predicted chars -> {repr(result[:10])}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = (os.path.join(_HERE, args.output)
                   if not os.path.isabs(args.output) else args.output)
    prior.save(output_path)
    print(f"\n  Saved -> {output_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Language model complete:")
    print(f"    {prior.summary()}")
    print(f"    Output: {output_path}")
    print(f"\n  To use in OCR evaluation:")
    print(f"    python ocr_test.py --lang-prior {args.output}")
    print(f"    python ocr_test.py --reader glyph_reader_hist.pkl "
          f"--lang-prior {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
