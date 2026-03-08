"""train_ocr.py -- Train GlyphReader on real GT4HistOCR historical font images.

Uses forced character alignment: for each GT line where the vertical-projection
segment count closely matches the ground-truth character count, we label each
segment with its corresponding GT character.  This builds a reader trained on
the actual historical font appearance rather than PIL's tiny bitmap font.

Expected result: CER drops from ~0.95 (PIL-trained) to ~0.20-0.40 on
EarlyModernLatin (Latin Roman serif, 1471-1686 CE).

Usage
-----
    # Train on EarlyModernLatin (default, ~10k lines, ~2-5 min on CPU)
    python train_ocr.py

    # Train on a different corpus
    python train_ocr.py --corpus RIDGES-Fraktur

    # Train on all corpora
    python train_ocr.py --corpus all

    # Limit lines used (fast smoke test: ~100 clean lines)
    python train_ocr.py --n 500

    # Custom output path
    python train_ocr.py --output glyph_reader_hist.pkl

    # Immediately evaluate after training
    python train_ocr.py && python ocr_test.py --reader glyph_reader_hist.pkl

Forced alignment algorithm
--------------------------
For each (image, GT text) pair:
  1. Normalise GT text:  long-s -> s, ligatures -> ASCII, strip combining marks.
  2. Extract non-space GT chars -> M characters.
  3. Load line image, normalise height to TARGET_H pixels.
  4. Segment by vertical projection gap detection -> N char-segments.
  5. If |N - M| / max(1, M) <= match_tolerance (default 0.20):
       pair segment[i] <-> gt_char[i] for i in range(min(N, M)).
  6. Accumulate pixel patches per GT character label.
After all lines:
  7. For each character with >= min_char_count patches: compute pixel centroid.
  8. Build GlyphReader with one centroid per character class.
  9. Save as glyph_reader_hist.pkl (or --output path).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_DATA_DIR  = os.path.join(_HERE, "data", "GT4HistOCR", "corpus")
_TARGET_H  = 24    # normalise line height (match GlyphReader training scale)
_MIN_SEG_W = 3     # minimum character segment width after normalisation

_ALL_CORPORA = [
    "EarlyModernLatin",
    "RIDGES-Fraktur",
    "Kallimachos",
    "dta19",
    "RefCorpus-ENHG-Incunabula",
]

# Re-use normalise_gt and segment_line_image from ocr_test.py
from ocr_test import (                   # type: ignore[import]
    find_pairs,
    normalise_gt,
    segment_line_image,
)
from modalities.glyph_reader import GlyphReader
from modalities.visual_symbol import _to_gray_f32  # type: ignore[import]


# ---------------------------------------------------------------------------
# Collect labelled patches via forced alignment
# ---------------------------------------------------------------------------

def collect_patches(
    pairs:             List[Tuple[str, str]],
    match_tolerance:   float = 0.20,
    min_gt_chars:      int   = 3,
    max_gt_chars:      int   = 200,
    patch_size:        int   = 16,
    verbose:           bool  = True,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, int]]:
    """Run forced alignment over ``pairs`` and collect labeled patches.

    Parameters
    ----------
    pairs            List of (image_path, gt_path) tuples.
    match_tolerance  Max allowed |N-M|/M fraction before discarding a line.
    min_gt_chars     Lines shorter than this are skipped (noise).
    max_gt_chars     Lines longer than this are skipped (very wide lines).
    patch_size       Patch size expected by GlyphReader (default 16).
    verbose          Print progress every 1000 lines.

    Returns
    -------
    patches_by_char  Dict[char -> list of (patch_size, patch_size) float32 arrays]
    stats            Dict of diagnostic counters
    """
    from PIL import Image  # type: ignore[import]

    patches_by_char: Dict[str, List[np.ndarray]] = defaultdict(list)
    stats: Dict[str, int] = Counter()

    t0 = time.time()
    for line_idx, (img_path, gt_path) in enumerate(pairs):
        if verbose and line_idx > 0 and line_idx % 1000 == 0:
            elapsed = time.time() - t0
            n_total = sum(len(v) for v in patches_by_char.values())
            print(f"  [{line_idx}/{len(pairs)}] patches={n_total:,}  "
                  f"elapsed={elapsed:.0f}s  "
                  f"clean={stats['clean']}  skipped={stats['skipped']}")

        # ------------------------------------------------------------------
        # 1. Load and normalise GT text
        # ------------------------------------------------------------------
        try:
            with open(gt_path, encoding="utf-8", errors="replace") as f:
                gt_raw = f.read().strip()
        except Exception:
            stats["skipped"] += 1
            continue

        gt_norm = normalise_gt(gt_raw)
        gt_chars_no_space = [c for c in gt_norm if c != " " and c != "\t"]
        M = len(gt_chars_no_space)

        if M < min_gt_chars or M > max_gt_chars:
            stats["skipped"] += 1
            continue

        # ------------------------------------------------------------------
        # 2. Load and height-normalise image
        # ------------------------------------------------------------------
        try:
            img = Image.open(img_path)
        except Exception:
            stats["skipped"] += 1
            continue

        if img.mode != "L":
            img = img.convert("L")

        H, W = img.height, img.width
        if H == 0 or W == 0:
            stats["skipped"] += 1
            continue

        scale = _TARGET_H / H
        new_w = max(patch_size, int(W * scale))
        img_small = img.resize((new_w, _TARGET_H), Image.LANCZOS)
        gray = _to_gray_f32(np.array(img_small))   # (_TARGET_H, new_w) float32

        # ------------------------------------------------------------------
        # 3. Segment into character runs
        # ------------------------------------------------------------------
        segs = segment_line_image(gray, min_seg_w=_MIN_SEG_W)
        char_segs = [(x0, x1) for x0, x1, is_space in segs if not is_space]
        N = len(char_segs)

        if N == 0:
            stats["skipped"] += 1
            continue

        # ------------------------------------------------------------------
        # 4. Alignment check
        # ------------------------------------------------------------------
        deviation = abs(N - M) / max(1, M)
        if deviation > match_tolerance:
            stats["poor_match"] += 1
            continue

        # ------------------------------------------------------------------
        # 5. Forced alignment: pair segment[i] <-> gt_char[i]
        #    Use the shorter of (N, M) to avoid index errors
        # ------------------------------------------------------------------
        n_pairs = min(N, M)
        for i in range(n_pairs):
            x0, x1 = char_segs[i]
            ch = gt_chars_no_space[i]

            # Crop segment, resize to patch_size x patch_size
            seg = gray[:, x0:x1]
            if seg.size == 0:
                continue

            from PIL import Image as _Im
            seg_img = _Im.fromarray(
                (seg * 255).clip(0, 255).astype(np.uint8), mode="L"
            )
            patch_img = seg_img.resize((patch_size, patch_size), _Im.LANCZOS)
            patch = np.array(patch_img, dtype=np.float32) / 255.0

            patches_by_char[ch].append(patch)

        stats["clean"] += 1

    if verbose:
        elapsed = time.time() - t0
        n_total = sum(len(v) for v in patches_by_char.values())
        n_chars = len(patches_by_char)
        print(f"\n  Collection done in {elapsed:.1f}s:")
        print(f"    Lines clean    = {stats['clean']}")
        print(f"    Lines skipped  = {stats['skipped']}")
        print(f"    Lines mismatch = {stats['poor_match']}")
        print(f"    Total patches  = {n_total:,} across {n_chars} unique characters")

    return dict(patches_by_char), dict(stats)


# ---------------------------------------------------------------------------
# Build a GlyphReader from labelled patches
# ---------------------------------------------------------------------------

def build_reader_from_patches(
    patches_by_char: Dict[str, List[np.ndarray]],
    patch_size:      int   = 16,
    quant_bits:      int   = 3,
    min_char_count:  int   = 5,
    verbose:         bool  = True,
) -> GlyphReader:
    """Build a GlyphReader whose centroids come from real historical patches.

    Parameters
    ----------
    patches_by_char  Dict[char -> list of patches] from collect_patches().
    patch_size       Patch size (must match GlyphReader.patch_size).
    quant_bits       Quantisation bits for hash lookup table.
    min_char_count   Discard characters with fewer than this many patches.
    verbose          Print per-character summary.

    Returns
    -------
    GlyphReader instance with _trained=True.
    """
    from modalities.visual_symbol import _quantize  # type: ignore[import]

    reader = GlyphReader(
        patch_size = patch_size,
        quant_bits = quant_bits,
        n_clusters = len(patches_by_char),
    )

    if verbose:
        print(f"\n  Building reader from {len(patches_by_char)} character classes ...")

    char_to_cid: Dict[str, int] = {}
    centroids:   Dict[int, np.ndarray] = {}
    labels:      Dict[int, str]        = {}
    label_conf:  Dict[int, float]      = {}
    hash_to_cluster: Dict[str, int]    = {}

    n_flat = patch_size * patch_size
    cid = 0

    char_items = sorted(patches_by_char.items(), key=lambda x: -len(x[1]))
    for ch, plist in char_items:
        if len(plist) < min_char_count:
            if verbose:
                print(f"    skip {ch!r}: only {len(plist)} patches (< {min_char_count})")
            continue

        # Compute mean patch (centroid)
        stack = np.stack(plist, axis=0)        # (n, patch_size, patch_size)
        centroid = stack.reshape(len(plist), -1).mean(axis=0).astype(np.float32)
        if len(centroid) != n_flat:
            continue

        char_to_cid[ch] = cid
        centroids[cid]  = centroid
        labels[cid]     = ch
        label_conf[cid] = 1.0

        # Build hash -> cluster map from all patches for this char
        for p in plist:
            h_key = _quantize(_to_gray_f32(p), quant_bits)
            # Majority vote: if already assigned, keep this char if it's more frequent
            # (first-come wins for speed; can refine later)
            if h_key not in hash_to_cluster:
                hash_to_cluster[h_key] = cid

        if verbose:
            print(f"    char={ch!r}  patches={len(plist):5d}  cid={cid}")

        cid += 1

    reader._centroids       = centroids
    reader._labels          = labels
    reader._label_conf      = label_conf
    reader._hash_to_cluster = hash_to_cluster
    reader._trained         = True

    if verbose:
        n_labelled = len(labels)
        print(f"\n  Reader built: {n_labelled} character classes, "
              f"{len(hash_to_cluster)} hash entries")

    return reader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _safe(s: str, maxlen: int = 60) -> str:
    """Truncate and ASCII-encode string for safe printing on any terminal."""
    return s[:maxlen].encode("ascii", "replace").decode("ascii")


def main() -> None:
    # Force UTF-8 stdout on Windows so non-ASCII GT text doesn't crash prints.
    import io
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "utf-8").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Train GlyphReader on real GT4HistOCR historical font images "
            "via forced character alignment."
        )
    )
    parser.add_argument(
        "--corpus",
        default="EarlyModernLatin",
        help=(
            "Corpus name, full path, or 'all'.  "
            f"Names: {', '.join(_ALL_CORPORA)}.  "
            "Default: EarlyModernLatin"
        ),
    )
    parser.add_argument("--n",              type=int,   default=None,
                        help="Max line images per corpus (None=all).")
    parser.add_argument("--output",         default="glyph_reader_hist.pkl",
                        help="Output .pkl path (default: glyph_reader_hist.pkl).")
    parser.add_argument("--match-tolerance", type=float, default=0.20,
                        help="Max allowed |N-M|/M fraction (default 0.20).")
    parser.add_argument("--min-char-count", type=int,   default=5,
                        help="Discard chars with fewer patches than this (default 5).")
    parser.add_argument("--patch-size",     type=int,   default=16,
                        help="Patch size (must match GlyphReader; default 16).")
    parser.add_argument("--verbose",        action="store_true",
                        help="Print per-character summary.")
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
        print(f"Using {len(corpus_dirs)} corpora: {[os.path.basename(c) for c in corpus_dirs]}")
    else:
        d = args.corpus
        if not os.path.isdir(d):
            d = os.path.join(_DATA_DIR, args.corpus)
        if not os.path.isdir(d):
            print(f"ERROR: corpus not found: {args.corpus!r}")
            print(f"  Available under {_DATA_DIR}: "
                  f"{os.listdir(_DATA_DIR) if os.path.isdir(_DATA_DIR) else '(missing)'}")
            sys.exit(1)
        corpus_dirs = [d]

    # ------------------------------------------------------------------
    # Collect patches from all corpora
    # ------------------------------------------------------------------
    all_patches: Dict[str, List[np.ndarray]] = defaultdict(list)

    for corpus_dir in corpus_dirs:
        corpus_name = os.path.basename(corpus_dir)
        print(f"\n{'='*60}")
        print(f"  Corpus: {corpus_name}")
        print(f"{'='*60}")

        pairs = find_pairs(corpus_dir, max_n=args.n, shuffle=True, seed=42)
        print(f"  Found {len(pairs)} line image pairs.")

        if not pairs:
            print(f"  Skipping (no pairs).")
            continue

        patches, stats = collect_patches(
            pairs,
            match_tolerance = args.match_tolerance,
            patch_size      = args.patch_size,
            verbose         = True,
        )

        for ch, plist in patches.items():
            all_patches[ch].extend(plist)

    if not all_patches:
        print("\nERROR: no patches collected.  Check that the data directory exists:")
        print(f"  {_DATA_DIR}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build reader from collected patches
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Building GlyphReader from {len(all_patches)} character classes ...")
    print(f"{'='*60}")

    reader = build_reader_from_patches(
        all_patches,
        patch_size     = args.patch_size,
        min_char_count = args.min_char_count,
        verbose        = args.verbose,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = os.path.join(_HERE, args.output) if not os.path.isabs(args.output) else args.output
    reader.model_path = output_path
    reader.save(output_path)
    print(f"\n  Saved -> {output_path}")

    # ------------------------------------------------------------------
    # Quick self-evaluation: read back and check a few lines
    # ------------------------------------------------------------------
    print(f"\n  Quick self-check (10 random lines) ...")
    from ocr_test import read_line, cer, normalise_gt  # type: ignore[import]

    sample_corpus = corpus_dirs[0]
    sample_pairs  = find_pairs(sample_corpus, max_n=50, shuffle=True, seed=99)[:10]

    n_ok = 0
    for img_path, gt_path in sample_pairs:
        try:
            with open(gt_path, encoding="utf-8", errors="replace") as f:
                gt_norm = normalise_gt(f.read().strip())
            pred = read_line(img_path, reader)
            c = cer(pred, gt_norm)
            status = "OK" if c < 0.50 else "  "
            print(f"    [{status}] CER={c:.2f}  GT={_safe(gt_norm, 40)!r}")
            print(f"           PRED={_safe(pred, 40)!r}")
            if c < 0.50:
                n_ok += 1
        except Exception as e:
            print(f"    [ERR] {_safe(str(e), 80)}")

    print(f"\n  {n_ok}/10 lines with CER < 0.50")

    # Summary
    n_classes = len(reader._labels)
    print(f"\n{'='*60}")
    print(f"  Training complete:")
    print(f"    Character classes : {n_classes}")
    print(f"    Output            : {output_path}")
    print(f"\n  To evaluate:")
    print(f"    python ocr_test.py --reader {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
