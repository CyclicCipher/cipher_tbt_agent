"""ocr_test.py -- Real-image OCR validation on GT4HistOCR.

Evaluates the learned GlyphReader on the GT4HistOCR benchmark:
  https://zenodo.org/record/1344132

Dataset structure (in data/GT4HistOCR/corpus/):
  EarlyModernLatin/  -- Latin Roman type, 10,288 line images (.bin.png)
  RIDGES-Fraktur/    -- German Fraktur,   12,588 line images (.bin.png)
  Kallimachos/       -- Early Fraktur,     4,803 line images (.nrm.png)
  dta19/             -- 19th-c. German,    ??? line images (.nrm.png)
  RefCorpus-ENHG-Incunabula/ -- Incunabula (.nrm.png)

Each sample: <id>.bin.png (or .nrm.png) + <id>.gt.txt
The image is a single text line; the GT file is the corresponding Unicode text.

Standard metrics reported:
  CER  Character Error Rate = edit_distance(pred, gt) / len(gt)   [lower is better]
  WER  Word Error Rate      = word_edit_dist / n_gt_words          [lower is better]
  ACC  1 - CER              [higher is better]

Pipeline:
  1. Normalise line image height to ~24px (matching GlyphReader training range)
  2. Segment characters by vertical-projection gap detection
  3. For each segment: resize to patch_size x patch_size, call GlyphReader.read_patch()
  4. Insert spaces for large inter-segment gaps
  5. Compare to GT; report per-corpus CER/WER

Usage
-----
    # Evaluate on EarlyModernLatin (most tractable: Latin Roman type)
    python ocr_test.py

    # Evaluate on a specific corpus
    python ocr_test.py --corpus EarlyModernLatin
    python ocr_test.py --corpus RIDGES-Fraktur

    # Limit samples (fast smoke test)
    python ocr_test.py --n 200

    # Show before/after calibration comparison
    python ocr_test.py --calibrate

    # Read a single line image
    python ocr_test.py --line data/GT4HistOCR/corpus/EarlyModernLatin/1668-Leviathan-Hobbes/00002.bin.png

    # Use a non-default GlyphReader (e.g. one trained on real historical fonts)
    python ocr_test.py --reader glyph_reader_hist.pkl

    # Run Phase O POS clustering on all read text
    python ocr_test.py --pos

Training on real historical fonts (forced alignment)
-----------------------------------------------------
The baseline PIL-trained reader gives CER~0.95 because it learned from tiny
bitmap fonts.  To train on the actual GT4HistOCR images:

    python train_ocr.py                          # EarlyModernLatin (default)
    python train_ocr.py --corpus all             # all 5 corpora
    python train_ocr.py --n 2000                 # limit lines (faster)

Then evaluate:
    python ocr_test.py --reader glyph_reader_hist.pkl
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_DATA_DIR   = os.path.join(_HERE, "data", "GT4HistOCR", "corpus")
_TARGET_H   = 24   # normalised line height in pixels (chars will be ~18px)
_MIN_SEG_W  = 3    # minimum segment width in pixels (after normalisation) to be a character


# ===========================================================================
# Dataset loader
# ===========================================================================

def find_pairs(corpus_dir: str, max_n: Optional[int] = None,
               shuffle: bool = True, seed: int = 42) -> List[Tuple[str, str]]:
    """Return list of (image_path, gt_path) pairs from a corpus directory.

    Supports both .bin.png and .nrm.png image variants.
    """
    pairs = []
    for ext in ("bin.png", "nrm.png"):
        images = glob.glob(os.path.join(corpus_dir, "**", f"*.{ext}"), recursive=True)
        for img_path in images:
            gt_path = img_path.replace(f".{ext}", ".gt.txt")
            if os.path.exists(gt_path):
                pairs.append((img_path, gt_path))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    if max_n is not None:
        pairs = pairs[:max_n]

    return pairs


# ===========================================================================
# GT text normalisation (map historical chars to modern equivalents)
# ===========================================================================

_GT_NORMALIZE = str.maketrans({
    "\u017f": "s",    # long-s -> s
    "\u00e6": "ae",   # ae ligature -> ae
    "\u00c6": "Ae",   # AE ligature -> Ae
    "\u0153": "oe",   # oe ligature -> oe
    "\u0152": "Oe",   # OE ligature -> Oe
    "\u1e9b": "s",    # dot-above long-s -> s
    "\u016b": "u",    # u-macron -> u
    "\u0101": "a",    # a-macron -> a
    "\u0113": "e",    # e-macron -> e
    "\u014d": "o",    # o-macron -> o
    "\u012b": "i",    # i-macron -> i
    "\u0303": "",     # combining tilde (dead key) -> drop
    "\u0306": "",     # combining breve -> drop
    "\u0307": "",     # combining dot above -> drop
    # Common OCR post-processing: remove soft hyphens
    "\u00ad": "",     # soft hyphen
})


def normalise_gt(text: str) -> str:
    """Normalise GT text to ASCII-compatible form for evaluation."""
    return text.translate(_GT_NORMALIZE).strip()


# ===========================================================================
# Character segmentation from vertical projection
# ===========================================================================

def segment_line_image(
    gray: np.ndarray,          # 2D float32 [0,1], 0=ink, 1=background
    min_seg_w: int = _MIN_SEG_W,
    gap_threshold: float = 0.04,   # ink density below this = inter-char gap
    space_gap_px: int = 5,         # gaps wider than this -> insert a space
) -> List[Tuple[int, int, bool]]:
    """Segment a line image into character columns by vertical projection.

    Returns list of (x_start, x_end, is_space) tuples.
    - (x0, x1, False): a character segment occupying columns x0..x1
    - (x0, x1, True):  a wide gap suggesting a word boundary (space)
    """
    H, W = gray.shape[:2]
    # Ink density per column: 1 = all ink, 0 = all background
    # In GT4HistOCR: 0=ink(black), 1=background(white) in the raw uint8 image.
    # After _to_gray_f32 normalisation: same convention preserved.
    # Ink density = fraction of pixels that are DARK (< 0.5 after normalisation).
    ink_density = (gray < 0.5).mean(axis=0)   # shape: (W,)

    # Binary: column is "inked" if density > gap_threshold
    inked = ink_density > gap_threshold

    segments: List[Tuple[int, int, bool]] = []
    i = 0
    while i < W:
        if inked[i]:
            # Start of a character: find its end
            j = i
            while j < W and inked[j]:
                j += 1
            if j - i >= min_seg_w:
                segments.append((i, j, False))
            i = j
        else:
            # Gap: find its width
            j = i
            while j < W and not inked[j]:
                j += 1
            gap_w = j - i
            if gap_w >= space_gap_px and segments:
                # Wide gap = word space
                segments.append((i, j, True))
            i = j

    return segments


# ===========================================================================
# Read a single line image -> string
# ===========================================================================

def _load_gray_segments(
    image_path: str,
    glyph_reader,
    target_h: int = _TARGET_H,
):
    """Shared image loading + segmentation used by read_line and read_line_viterbi.

    Returns (segs, gray, patch_size) or (None, None, None) on error.
    segs: List[(x0, x1, is_space)]
    gray: (target_h, W) float32 image
    """
    from PIL import Image
    from modalities.visual_symbol import _to_gray_f32

    patch_size = glyph_reader.patch_size

    try:
        img = Image.open(image_path)
    except Exception:
        return None, None, None

    if img.mode != "L":
        img = img.convert("L")

    H, W = img.height, img.width
    if H == 0 or W == 0:
        return None, None, None

    scale = target_h / H
    new_w = max(patch_size, int(W * scale))
    img_small = img.resize((new_w, target_h), Image.LANCZOS)

    from modalities.visual_symbol import _to_gray_f32 as _tgf
    gray = _tgf(np.array(img_small))
    segs = segment_line_image(gray, min_seg_w=_MIN_SEG_W)
    return segs, gray, patch_size


def read_line(
    image_path: str,
    glyph_reader,
    target_h: int = _TARGET_H,
    lang_prior=None,
    alpha: float = 0.6,
) -> str:
    """Read a GT4HistOCR line image; return the predicted string.

    Steps:
      1. Load grayscale
      2. Resize to normalise line height to target_h pixels
      3. Segment by vertical projection
      4. For each character segment: resize to patch_size x patch_size, read_patch()
         (or read_patch_topk() + Viterbi if lang_prior is provided)
      5. Insert spaces for wide gaps

    Parameters
    ----------
    lang_prior  Optional LanguagePrior instance.  If provided, uses Viterbi
                decoding to combine GlyphReader visual scores with character
                bigram transition probabilities.  This corrects systematic
                misreads using knowledge of Early Modern Latin / German Fraktur.
    alpha       Visual weight for Viterbi (0=pure language, 1=pure visual).
                Default 0.6 (favour visual but allow language corrections).
    """
    from PIL import Image

    segs, gray, patch_size = _load_gray_segments(image_path, glyph_reader, target_h)
    if segs is None:
        return ""

    chars: List[str] = []

    if lang_prior is None:
        # -------------------------------------------------------------------
        # Greedy top-1 decoding (no language prior)
        # -------------------------------------------------------------------
        for x0, x1, is_space in segs:
            if is_space:
                chars.append(" ")
                continue
            seg = gray[:, x0:x1]
            if seg.size == 0:
                continue
            seg_img = Image.fromarray(
                (seg * 255).clip(0, 255).astype(np.uint8), mode="L"
            )
            patch_img = seg_img.resize((patch_size, patch_size), Image.LANCZOS)
            patch = np.array(patch_img, dtype=np.uint8)
            result = glyph_reader.read_patch(patch)
            if result.char and result.char != " ":
                chars.append(result.char)
    else:
        # -------------------------------------------------------------------
        # Viterbi decoding with language prior
        # Groups non-space segments within each word, then spaces between words.
        # -------------------------------------------------------------------
        word_candidates: List[List[Tuple[str, float]]] = []  # current word
        for x0, x1, is_space in segs:
            if is_space:
                # Flush current word through Viterbi
                if word_candidates:
                    chars.append(lang_prior.viterbi(word_candidates, alpha=alpha))
                    word_candidates = []
                chars.append(" ")
                continue
            seg = gray[:, x0:x1]
            if seg.size == 0:
                continue
            seg_img = Image.fromarray(
                (seg * 255).clip(0, 255).astype(np.uint8), mode="L"
            )
            patch_img = seg_img.resize((patch_size, patch_size), Image.LANCZOS)
            patch = np.array(patch_img, dtype=np.uint8)
            topk = glyph_reader.read_patch_topk(patch, k=8)
            if topk:
                word_candidates.append(topk)
        # Flush final word
        if word_candidates:
            chars.append(lang_prior.viterbi(word_candidates, alpha=alpha))

    return "".join(chars).strip()


# ===========================================================================
# Edit distance and metrics
# ===========================================================================

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (character-level)."""
    if a == b:
        return 0
    if not b:
        return len(a)
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def word_edit_distance(a_words: List[str], b_words: List[str]) -> int:
    """Word-level Levenshtein distance."""
    a, b = a_words, b_words
    if not b:
        return len(a)
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for wa in a:
        curr = [prev[0] + 1]
        for j, wb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if wa == wb else 1)))
        prev = curr
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    """Character Error Rate = edit_distance / max(1, len(gt))."""
    return edit_distance(pred, gt) / max(1, len(gt))


def wer(pred: str, gt: str) -> float:
    """Word Error Rate = word_edit_distance / max(1, n_gt_words)."""
    pw = pred.split()
    gw = gt.split()
    return word_edit_distance(pw, gw) / max(1, len(gw))


# ===========================================================================
# Evaluate GlyphReader on a corpus
# ===========================================================================

def evaluate_corpus(
    corpus_dir: str,
    glyph_reader,
    n: Optional[int] = None,
    calibrate_frac: float = 0.1,
    verbose: bool = False,
    show_n_errors: int = 5,
    lang_prior=None,
    alpha: float = 0.6,
) -> Dict:
    """Evaluate GlyphReader on a corpus; return metrics dict.

    Parameters
    ----------
    corpus_dir      Path to a corpus directory (e.g. .../EarlyModernLatin)
    glyph_reader    Trained GlyphReader instance (will be calibrated in-place)
    n               Max number of line images to evaluate (None = all)
    calibrate_frac  Fraction of lines used for calibration (not evaluated)
    verbose         Print per-book breakdown
    show_n_errors   Number of worst-CER examples to show
    lang_prior      Optional LanguagePrior for Viterbi decoding (see read_line)
    alpha           Visual weight for Viterbi (default 0.6)
    """
    from PIL import Image
    from modalities.visual_symbol import _to_gray_f32

    pairs = find_pairs(corpus_dir, max_n=n)
    if not pairs:
        print(f"  No pairs found in {corpus_dir!r}")
        return {"ok": False}

    n_calib = max(1, int(len(pairs) * calibrate_frac))
    calib_pairs = pairs[:n_calib]
    eval_pairs  = pairs[n_calib:]

    print(f"  {len(pairs)} line images found ({n_calib} for calibration, "
          f"{len(eval_pairs)} for evaluation)")

    # ------------------------------------------------------------------
    # Calibrate on a random subset of the corpus
    # ------------------------------------------------------------------
    if n_calib > 0 and hasattr(glyph_reader, "calibrate"):
        print(f"  Calibrating GlyphReader on {n_calib} lines ...")
        t0 = time.time()
        # Build one large calibration frame by stacking line images vertically
        cal_images = []
        for img_path, _ in calib_pairs[:min(n_calib, 100)]:
            try:
                img = Image.open(img_path)
                if img.mode != "L":
                    img = img.convert("L")
                # Normalise height
                scale = _TARGET_H / max(1, img.height)
                img = img.resize((max(1, int(img.width * scale)), _TARGET_H),
                                 Image.LANCZOS)
                cal_images.append(np.array(img))
            except Exception:
                continue
        if cal_images:
            # Pad to same width and stack
            max_w = max(im.shape[1] for im in cal_images)
            padded = [np.pad(im, ((0, 0), (0, max_w - im.shape[1])), constant_values=255)
                      for im in cal_images]
            cal_frame = np.concatenate(padded, axis=0)
            # Convert to RGB for calibrate() which expects RGB
            cal_frame_rgb = np.stack([cal_frame]*3, axis=-1)
            glyph_reader.calibrate(cal_frame_rgb, verbose=False)
        print(f"  Calibration done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"  Evaluating {len(eval_pairs)} lines ...")
    t0 = time.time()

    total_cer = 0.0
    total_wer = 0.0
    n_lines   = 0
    per_book: Dict[str, List[float]] = defaultdict(list)
    worst_examples: List[Tuple[float, str, str, str]] = []  # (cer, pred, gt, path)

    for img_path, gt_path in eval_pairs:
        try:
            with open(gt_path, encoding="utf-8", errors="replace") as f:
                gt_raw = f.read().strip()
            gt_norm = normalise_gt(gt_raw)
            if not gt_norm:
                continue

            pred = read_line(img_path, glyph_reader, lang_prior=lang_prior, alpha=alpha)
            c = cer(pred, gt_norm)
            w = wer(pred, gt_norm)

            total_cer += c
            total_wer += w
            n_lines   += 1

            # Track per-book (parent directory name)
            book = os.path.basename(os.path.dirname(img_path))
            per_book[book].append(c)

            # Track worst examples
            if len(worst_examples) < show_n_errors or c > worst_examples[0][0]:
                worst_examples.append((c, pred[:60], gt_norm[:60], img_path))
                worst_examples.sort(reverse=True)
                worst_examples = worst_examples[:show_n_errors]

        except Exception as e:
            if verbose:
                print(f"    [warn] {os.path.basename(img_path)}: {e}")
            continue

    elapsed = time.time() - t0

    if n_lines == 0:
        print("  No lines evaluated successfully.")
        return {"ok": False}

    mean_cer = total_cer / n_lines
    mean_wer = total_wer / n_lines
    acc = 1.0 - mean_cer

    print(f"\n  Results ({n_lines} lines, {elapsed:.1f}s):")
    print(f"    CER  = {mean_cer:.3f}  ({mean_cer:.1%})")
    print(f"    WER  = {mean_wer:.3f}  ({mean_wer:.1%})")
    print(f"    ACC  = {acc:.3f}  ({acc:.1%})")

    # Per-book breakdown
    if verbose and per_book:
        print(f"\n  Per-book CER:")
        for book in sorted(per_book, key=lambda b: np.mean(per_book[b])):
            cers = per_book[book]
            print(f"    {book[:45]:45s}: CER={np.mean(cers):.3f} (n={len(cers)})")

    # Worst examples
    if worst_examples:
        print(f"\n  Worst {len(worst_examples)} lines (highest CER):")
        for c_val, pred, gt, path in worst_examples:
            print(f"    CER={c_val:.2f}  GT={gt!r}")
            print(f"           PRED={pred!r}")

    return {
        "ok":       True,
        "n_lines":  n_lines,
        "cer":      mean_cer,
        "wer":      mean_wer,
        "acc":      acc,
        "elapsed":  elapsed,
        "per_book": {k: float(np.mean(v)) for k, v in per_book.items()},
    }


# ===========================================================================
# Phase O: POS clustering on read text
# ===========================================================================

def run_pos_clustering(all_text: str, n_clusters: int = 8) -> None:
    """Run Phase O distributional POS clustering on the aggregated read text."""
    print("\n  --- Phase O: POS clustering on read text ---")
    words = all_text.lower().split()
    print(f"  Tokens: {len(words)}, unique: {len(set(words))}")

    if len(words) < 100:
        print("  Too few tokens for POS clustering.")
        return

    try:
        from synthesis import discover_categories_from_dists
    except ImportError:
        print("  synthesis.py not found -- skipping POS clustering.")
        return

    # Build forward bigram distributions
    vocab = list(set(words))
    word_idx = {w: i for i, w in enumerate(vocab)}
    forward = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    for w1, w2 in zip(words, words[1:]):
        forward[word_idx[w1], word_idx[w2]] += 1.0

    # Normalise rows
    row_sums = forward.sum(axis=1, keepdims=True).clip(min=1)
    forward /= row_sums

    # Only cluster words that appear >= 3 times (meaningful distributions)
    counts = Counter(words)
    valid = [w for w in vocab if counts[w] >= 3]
    if len(valid) < n_clusters:
        print(f"  Not enough high-freq words ({len(valid)}) for {n_clusters} clusters.")
        return

    valid_idx = np.array([word_idx[w] for w in valid])
    dists = forward[valid_idx]

    labels = discover_categories_from_dists(dists, n_clusters=n_clusters)

    cluster_members: Dict[int, List] = defaultdict(list)
    for word, lbl in zip(valid, labels):
        cluster_members[lbl].append((counts[word], word))

    print(f"  {n_clusters} clusters from {len(valid)} frequent words:")
    for cid in sorted(cluster_members, key=lambda c: -len(cluster_members[c])):
        members = sorted(cluster_members[cid], reverse=True)[:8]
        words_str = ", ".join(w for _, w in members)
        print(f"    C{cid} ({len(cluster_members[cid])} words): {words_str}")


# ===========================================================================
# Single-line demo
# ===========================================================================

def demo_single_line(line_path: str, glyph_reader, lang_prior=None, alpha: float = 0.6) -> None:
    """Read a single GT4HistOCR line image and show result vs GT."""
    gt_path = (line_path.replace(".bin.png", ".gt.txt")
                        .replace(".nrm.png", ".gt.txt"))
    gt = ""
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8", errors="replace") as f:
            gt = normalise_gt(f.read().strip())

    pred_base = read_line(line_path, glyph_reader)

    print(f"\n  Image: {os.path.basename(line_path)}")
    print(f"  GT:   {gt!r}")
    print(f"  PRED (greedy): {pred_base!r}")
    if gt:
        print(f"  CER (greedy)={cer(pred_base, gt):.3f}  WER={wer(pred_base, gt):.3f}")

    if lang_prior is not None:
        pred_vit = read_line(line_path, glyph_reader, lang_prior=lang_prior, alpha=alpha)
        print(f"  PRED (viterbi, alpha={alpha}): {pred_vit!r}")
        if gt:
            print(f"  CER (viterbi)={cer(pred_vit, gt):.3f}  WER={wer(pred_vit, gt):.3f}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR evaluation on GT4HistOCR (real historical document benchmark)"
    )
    parser.add_argument(
        "--corpus",
        default="EarlyModernLatin",
        help=(
            "Corpus name or full path. "
            "Names: EarlyModernLatin, RIDGES-Fraktur, Kallimachos, dta19, "
            "RefCorpus-ENHG-Incunabula. "
            f"Resolved relative to {_DATA_DIR}. "
            "Default: EarlyModernLatin"
        ),
    )
    parser.add_argument("--n",          type=int, default=None,
                        help="Max line images to use (None=all). "
                             "Default: all (can be slow).")
    parser.add_argument("--reader",     default="glyph_reader.pkl",
                        help="Path to pre-trained GlyphReader .pkl "
                             "(default: glyph_reader.pkl)")
    parser.add_argument("--calibrate",  action="store_true",
                        help="Show before/after calibration comparison")
    parser.add_argument("--line",       default=None, metavar="PATH",
                        help="Read a single line image and exit")
    parser.add_argument("--verbose",    action="store_true",
                        help="Show per-book breakdown")
    parser.add_argument("--pos",        action="store_true",
                        help="Run Phase O POS clustering on all read text")
    parser.add_argument("--lang-prior", default=None, metavar="PATH",
                        help="Path to a LanguagePrior .pkl (from train_lang.py). "
                             "Enables Viterbi decoding: combines GlyphReader "
                             "visual scores with character bigram transitions.")
    parser.add_argument("--alpha",      type=float, default=0.6,
                        help="Visual weight for Viterbi decoding [0,1]. "
                             "1.0=pure visual, 0.0=pure language. Default 0.6.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve corpus path
    # ------------------------------------------------------------------
    corpus_dir = args.corpus
    if not os.path.isdir(corpus_dir):
        corpus_dir = os.path.join(_DATA_DIR, args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"ERROR: corpus not found: {corpus_dir!r}")
        print(f"  Available: {os.listdir(_DATA_DIR) if os.path.isdir(_DATA_DIR) else '(data dir missing)'}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load or train GlyphReader
    # ------------------------------------------------------------------
    from modalities.glyph_reader import GlyphReader

    if os.path.exists(args.reader):
        print(f"Loading GlyphReader from {args.reader!r} ...")
        try:
            gr = GlyphReader.load(args.reader)
        except Exception as e:
            print(f"  Load failed ({e}) -- training from scratch ...")
            gr = GlyphReader()
            gr.train(verbose=True)
    else:
        print(f"{args.reader!r} not found -- training GlyphReader from scratch ...")
        gr = GlyphReader()
        gr.train(verbose=True)

    # ------------------------------------------------------------------
    # Load language prior (optional)
    # ------------------------------------------------------------------
    lang_prior = None
    if args.lang_prior:
        from modalities.language_prior import LanguagePrior
        lp_path = args.lang_prior
        if not os.path.isabs(lp_path):
            lp_path = os.path.join(_HERE, lp_path)
        if os.path.exists(lp_path):
            print(f"Loading LanguagePrior from {lp_path!r} ...")
            lang_prior = LanguagePrior.load(lp_path)
            print(f"  {lang_prior.summary()}")
        else:
            print(f"WARNING: lang-prior not found: {lp_path!r} -- running without")

    # ------------------------------------------------------------------
    # Single-line demo mode
    # ------------------------------------------------------------------
    if args.line:
        demo_single_line(args.line, gr, lang_prior=lang_prior, alpha=args.alpha)
        return

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    corpus_name = os.path.basename(corpus_dir)
    print(f"\n{'='*60}")
    print(f"  OCR Evaluation: {corpus_name}")
    print(f"{'='*60}")

    if args.calibrate:
        # Run without calibration first
        import copy
        gr_base = copy.deepcopy(gr)
        print("\n  [Before calibration]")
        res_base = evaluate_corpus(corpus_dir, gr_base, n=args.n,
                                   calibrate_frac=0.0, verbose=False,
                                   lang_prior=lang_prior, alpha=args.alpha)
        print("\n  [After calibration]")
        res_cal = evaluate_corpus(corpus_dir, gr, n=args.n,
                                  calibrate_frac=0.1, verbose=args.verbose,
                                  lang_prior=lang_prior, alpha=args.alpha)
        print(f"\n  Calibration improvement:")
        print(f"    CER: {res_base['cer']:.3f} -> {res_cal['cer']:.3f}  "
              f"(delta {res_base['cer']-res_cal['cer']:+.3f})")
        print(f"    WER: {res_base['wer']:.3f} -> {res_cal['wer']:.3f}  "
              f"(delta {res_base['wer']-res_cal['wer']:+.3f})")
        result = res_cal
    else:
        result = evaluate_corpus(corpus_dir, gr, n=args.n,
                                 calibrate_frac=0.1, verbose=args.verbose,
                                 lang_prior=lang_prior, alpha=args.alpha)

    # ------------------------------------------------------------------
    # Phase O POS clustering (optional)
    # ------------------------------------------------------------------
    if args.pos and result.get("ok"):
        # Re-read all text for clustering
        print("\n  Re-reading corpus for POS clustering ...")
        pairs = find_pairs(corpus_dir, max_n=args.n)
        all_text_parts = []
        for img_path, _ in pairs:
            t = read_line(img_path, gr)
            if t.strip():
                all_text_parts.append(t)
        run_pos_clustering(" ".join(all_text_parts))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if result.get("ok"):
        print(f"  {corpus_name}: CER={result['cer']:.3f}  "
              f"WER={result['wer']:.3f}  ACC={result['acc']:.3f}")
        ok = result["cer"] < 1.20   # pipeline ran and produced output (CER>1 = inserting chars)
        print(f"  {'PASS' if ok else 'FAIL'}")
    else:
        print("  FAIL (no results)")
        ok = False
    print(f"{'='*60}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
