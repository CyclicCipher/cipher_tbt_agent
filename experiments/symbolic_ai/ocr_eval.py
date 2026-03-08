"""ocr_eval.py -- Evaluate the symbolic AI's OCR capability.

Loads a pre-trained checkpoint (produced by discover_chars.py --align --save)
and tests the AI's glyph_reads_as mapping on held-out line images.

This demonstrates the transfer: the AI was trained once, saved as a checkpoint,
and now a FRESH instance can read characters without any retraining.

Usage
-----
    # Train first (if not already done):
    python discover_chars.py --corpus EarlyModernLatin --align --save ocr_knowledge.json

    # Evaluate on 200 held-out lines:
    python ocr_eval.py --checkpoint ocr_knowledge.json --n 200

    # Evaluate on a different corpus:
    python ocr_eval.py --checkpoint ocr_knowledge.json --corpus RIDGES-Fraktur --n 100
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

import io
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', 'utf-8').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from ctkg.parser import parse_file, merge
from engine import SymbolicAI
from ocr_test import find_pairs, normalise_gt          # type: ignore[import]
from modalities.visual_symbol import _to_gray_f32, _extract_patches, _quantize


_DATA_DIR = os.path.join(_HERE, 'data', 'GT4HistOCR', 'corpus')
_PATCH_SIZE = 16
_QUANT_BITS = 3


def _banner(title: str) -> None:
    print(f'\n{"=" * 62}')
    print(f'  {title}')
    print('=' * 62)


def _cer(pred: str, ref: str) -> float:
    """Character Error Rate via edit distance."""
    if not ref:
        return 0.0 if not pred else 1.0
    # Dynamic programming edit distance
    m, n = len(pred), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / n


def evaluate(
    ai:         SymbolicAI,
    pairs:      list,
    verbose:    bool = True,
) -> dict:
    """Evaluate glyph_reads_as on held-out (image, GT text) pairs.

    For each line:
      1. Extract patch hashes from the image.
      2. Quantize each patch -> glyph_hash.
      3. Look up the glyph cluster (from Phase 2 training, via ExampleStore).
      4. Ask the AI: glyph_reads_as(glyph_cluster) -> predicted char.
      5. Compare predicted char sequence to GT text.

    The glyph cluster lookup (step 3) uses the next_glyph ExampleStore to find
    which cluster each hash was assigned to during induce_hierarchy. This is
    a limitation of the current approach: hashes not seen during training return
    None (out-of-vocabulary glyph). Coverage metric shows how often this happens.
    """
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print('  ERROR: PIL not available. Use venv Python.')
        return {}

    # Build hash -> glyph_cluster lookup from the next_glyph ExampleStore.
    # During Phase 2 training, we called ai.teach('next_glyph', (hash,), (hash_next,)).
    # After induce_hierarchy, each hash was assigned to a cluster.
    # We recover this mapping from the ExampleStore: the cluster assignment is
    # stored in the 'next_glyph__cat' concept added by induce_hierarchy.
    #
    # If induce_hierarchy produced concept cat_C0, cat_C1, ..., the assignment
    # dict is in result['assignment']. But we didn't save that directly.
    # Instead: try ai.ask('next_glyph__cat', (hash,)) if it exists, else fallback.
    #
    # For now: use the glyph_reads_as ExampleStore directly — it maps
    # (glyph_cid,) -> (char,). We can invert: any glyph_cid seen during
    # alignment is queryable. Hashes not seen in alignment are OOV.

    glyph_reads_as_store = ai.stores.get('glyph_reads_as')
    if glyph_reads_as_store is None or len(glyph_reads_as_store) == 0:
        print('  ERROR: no glyph_reads_as examples found in checkpoint.')
        print('  Run: python discover_chars.py --align --save <checkpoint>')
        return {}

    # Build glyph_cid -> most_frequent_char from ExampleStore.
    cid_to_chars: dict = collections.defaultdict(collections.Counter)
    for inputs, outputs in glyph_reads_as_store.examples:
        cid_to_chars[inputs[0]][outputs[0]] += 1
    cid_to_char = {cid: ctr.most_common(1)[0][0] for cid, ctr in cid_to_chars.items()}

    # Build glyph_hash -> glyph_cid from next_glyph ExampleStore.
    # During training we stored (hash,) -> (hash_next,) — which doesn't encode
    # the cluster assignment directly. The cluster assignment lives in the
    # cat_C* concepts added by induce_hierarchy.
    # Shortcut: use ai.ask() which will consult the ExampleStore for any stored concept.
    # If 'cat_C0' etc. exist in the AI, we can build hash->cluster from them.
    hash_to_cid: dict = {}
    for cname, store in ai.stores.items():
        if cname.startswith('cat_C') and cname.replace('cat_C', '').isdigit():
            cid = int(cname.replace('cat_C', ''))
            for inputs, _ in store.examples:
                if inputs:
                    hash_to_cid[inputs[0]] = cid

    if not hash_to_cid:
        print('  NOTE: No cat_C* stores found. Glyph cluster assignment unavailable.')
        print('  Evaluation will show OOV=100% — run --align first.')
    else:
        print(f'  Hash->cluster map: {len(hash_to_cid):,} hashes -> {len(set(hash_to_cid.values()))} clusters')

    # Evaluate.
    n_lines = n_skipped = 0
    total_cer = 0.0
    n_in_vocab = n_total_patches = 0
    confusion: dict = collections.Counter()  # (predicted, actual) pairs

    for img_path, gt_path in pairs:
        # Load GT text.
        try:
            with open(gt_path, encoding='utf-8', errors='replace') as f:
                raw = f.read().strip()
            gt = normalise_gt(raw)
            if not gt:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue

        # Load image.
        for candidate in (
            img_path,
            img_path.replace('.nrm.png', '.bin.png'),
            img_path.replace('.bin.png', '.nrm.png'),
        ):
            if os.path.exists(candidate):
                img_path = candidate
                break
        else:
            n_skipped += 1
            continue

        try:
            img     = Image.open(img_path).convert('L')
            arr     = np.array(img, dtype=np.float32) / 255.0
            gray    = arr if arr.ndim == 2 else _to_gray_f32(arr)
            patches = _extract_patches(gray, _PATCH_SIZE, _PATCH_SIZE)
            if not patches:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue

        # Predict character for each patch.
        predicted_chars = []
        for patch in patches:
            h   = _quantize(_to_gray_f32(patch), _QUANT_BITS)
            cid = hash_to_cid.get(h)
            n_total_patches += 1
            if cid is not None and cid in cid_to_char:
                predicted_chars.append(cid_to_char[cid])
                n_in_vocab += 1
            else:
                predicted_chars.append('?')  # OOV

        pred_str = ''.join(c for c in predicted_chars if c != '?')
        gt_chars = [c for c in gt if c != ' ']
        gt_str   = ''.join(gt_chars)

        # Confusion tracking (only for aligned positions).
        if len(predicted_chars) == len(gt_chars):
            for p, a in zip(predicted_chars, gt_chars):
                if p != '?' and p != a:
                    confusion[(p, a)] += 1

        cer = _cer(pred_str, gt_str)
        total_cer += cer
        n_lines += 1

    if n_lines == 0:
        print('  ERROR: no lines evaluated.')
        return {}

    avg_cer = total_cer / n_lines
    coverage = n_in_vocab / max(1, n_total_patches)

    _banner('Results')
    print(f'\n  Lines evaluated:  {n_lines} ({n_skipped} skipped)')
    print(f'  Avg CER:          {avg_cer:.3f}  (lower is better; 1.0 = no correct chars)')
    print(f'  Glyph coverage:   {coverage:.1%}  (patches with known cluster)')
    print(f'  Total patches:    {n_total_patches:,}  ({n_in_vocab:,} in-vocab)')

    if confusion:
        print(f'\n  Top-10 most confused char pairs (predicted -> actual):')
        for (pred, actual), cnt in confusion.most_common(10):
            print(f'    {repr(pred)} -> {repr(actual)}  ({cnt}x)')

    return {
        'cer': avg_cer,
        'coverage': coverage,
        'n_lines': n_lines,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description='Evaluate symbolic AI OCR from a pre-trained checkpoint.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--checkpoint', required=True, metavar='PATH',
                   help='Checkpoint file from discover_chars.py --save.')
    p.add_argument('--corpus', default='EarlyModernLatin',
                   help='Corpus to evaluate on.')
    p.add_argument('--n', type=int, default=200,
                   help='Number of lines to evaluate (default 200).')
    p.add_argument('--seed', type=int, default=99,
                   help='Random seed for line selection (default 99, different from training).')
    args = p.parse_args()

    # Resolve corpus.
    d = args.corpus if os.path.isdir(args.corpus) else os.path.join(_DATA_DIR, args.corpus)
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {args.corpus!r}')
        sys.exit(1)

    pairs = find_pairs(d, max_n=args.n, shuffle=True, seed=args.seed)
    print(f'  Corpus: {args.corpus}  ({len(pairs)} pairs, seed={args.seed})')

    # Build a fresh symbolic AI — loads OCR knowledge from checkpoint.
    _banner('Loading fresh symbolic AI + OCR checkpoint')
    arith_ctkg = os.path.join(_HERE, '..', 'ctkg', 'domains', 'arithmetic.ctkg')
    ocr_ctkg   = os.path.join(_HERE, '..', 'ctkg', 'domains', 'ocr.ctkg')

    graph = parse_file(arith_ctkg)
    if os.path.exists(ocr_ctkg):
        ocr_graph = parse_file(ocr_ctkg)
        from ctkg.parser import merge as ctkg_merge
        graph = ctkg_merge(graph, ocr_graph)
        print('  Loaded ocr.ctkg structural skeleton.')
    else:
        print('  NOTE: ocr.ctkg not found — using arithmetic CTKG only.')

    ai = SymbolicAI(graph)

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_HERE, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f'ERROR: checkpoint not found: {ckpt_path!r}')
        print('  Run: python discover_chars.py --align --save ocr_knowledge.json')
        sys.exit(1)

    ai.load_checkpoint(ckpt_path)
    print(f'  Checkpoint loaded: {ckpt_path}')

    # Show what was loaded.
    learned = [name for name in ai.stores if len(ai.stores[name]) > 0]
    print(f'  Learned concepts: {", ".join(learned[:8])}{"..." if len(learned) > 8 else ""}')

    # Evaluate.
    _banner(f'Evaluating on {args.corpus}')
    results = evaluate(ai, pairs)

    if results:
        _banner('Summary')
        print(f"""
  Checkpoint:  {args.checkpoint}
  Corpus:      {args.corpus}
  Lines:       {results["n_lines"]}
  CER:         {results["cer"]:.3f}
  Coverage:    {results["coverage"]:.1%}

  This is a FRESH AI instance with NO training — just loading the checkpoint.
  The symbolic AI learned to read once; the knowledge transfers instantly.
""")


if __name__ == '__main__':
    main()
