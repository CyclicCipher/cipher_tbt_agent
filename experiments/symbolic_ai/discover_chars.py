"""discover_chars.py -- Phase S.1: Character and glyph category discovery.

The symbolic AI learns the structure of Early Modern Latin and German Fraktur
by observing raw character sequences from GT4HistOCR ground-truth text.

This is Phase O applied at the CHARACTER level rather than the word level.
The same discover_categories_from_dists() machinery finds phonological
categories (vowels, fricatives, word-boundary chars, etc.) without being
told they exist.

After discovering char categories, the AI discovers glyph categories from
raw image patch sequences (no GT labels needed), then learns the alignment:
which visual glyph category corresponds to which character category.

Three phases:
  Phase 1 (--chars):  Discover character categories from raw GT text.
  Phase 2 (--glyphs): Discover glyph categories from raw line images.
  Phase 3 (--align):  Learn glyph→char alignment from paired data.

Usage
-----
    # Quick smoke test (200 lines):
    python discover_chars.py --n 200

    # Full corpus character discovery:
    python discover_chars.py --corpus EarlyModernLatin

    # Full pipeline (all three phases):
    python discover_chars.py --corpus EarlyModernLatin --align
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

from ctkg.parser import parse_file
from engine import SymbolicAI
from ocr_test import find_pairs, normalise_gt          # type: ignore[import]
from modalities.visual_symbol import _to_gray_f32, _extract_patches, _quantize


_DATA_DIR = os.path.join(_HERE, 'data', 'GT4HistOCR', 'corpus')

_ALL_CORPORA = [
    'EarlyModernLatin',
    'RIDGES-Fraktur',
    'Kallimachos',
    'dta19',
    'RefCorpus-ENHG-Incunabula',
]

_PATCH_SIZE = 16
_QUANT_BITS = 3


def _banner(title: str) -> None:
    print(f'\n{"=" * 62}')
    print(f'  {title}')
    print('=' * 62)


def _resolve_corpus(corpus_arg: str) -> list[str]:
    """Resolve --corpus argument to a list of directory paths."""
    if corpus_arg.lower() == 'all':
        dirs = [os.path.join(_DATA_DIR, c) for c in _ALL_CORPORA
                if os.path.isdir(os.path.join(_DATA_DIR, c))]
        if not dirs:
            print(f'ERROR: no corpora found under {_DATA_DIR!r}')
            sys.exit(1)
        return dirs
    d = corpus_arg if os.path.isdir(corpus_arg) else os.path.join(_DATA_DIR, corpus_arg)
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {corpus_arg!r}')
        sys.exit(1)
    return [d]


# ---------------------------------------------------------------------------
# Phase 1: Character category discovery
# ---------------------------------------------------------------------------

def phase1_chars(
    ai:       SymbolicAI,
    pairs:    list,
    n_clusters: int = 12,
    verbose:  bool = True,
) -> dict:
    """Teach the AI raw character sequences; induce char categories.

    For each adjacent character pair (ch_i, ch_i+1) in every GT line:
        ai.teach('next_char', (ch_i,), (ch_i+1,))

    Then induce_hierarchy() clusters characters by their forward
    distributions -- discovering phonological/morphological structure
    without labels.
    """
    _banner('Phase 1: Character Category Discovery')
    print('  Streaming GT text characters to the symbolic AI ...')

    # Register the concept if not already present.
    if 'next_char' not in ai.stores:
        ai.add_concept(
            name='next_char', domain='language',
            description='predict next character given current character',
            input_type=['char'], output_type=['char'], tier='theorem',
        )

    n_lines = n_chars = n_empty = 0
    for _, gt_path in pairs:
        try:
            with open(gt_path, encoding='utf-8', errors='replace') as f:
                raw = f.read().strip()
            text = normalise_gt(raw)
            if not text:
                n_empty += 1
                continue
        except Exception:
            n_empty += 1
            continue

        n_lines += 1
        for i in range(len(text) - 1):
            ch, ch_next = text[i], text[i + 1]
            ai.teach('next_char', (ch,), (ch_next,))
            n_chars += 1

    print(f'  {n_lines} lines processed ({n_empty} empty), {n_chars:,} char pairs taught')

    if n_chars == 0:
        print('  ERROR: no character pairs collected.')
        return {}

    # Consolidate to get unique distribution counts.
    ai.freq_consolidate('next_char')
    store = ai.stores['next_char']
    print(f'  ExampleStore: {len(store):,} examples, '
          f'{len({inp for inp, _ in store.examples}):,} unique chars')

    # Discover categories.
    _banner('Phase 1.2: Inducing Character Categories')
    result = ai.induce_hierarchy('next_char', n_clusters=n_clusters, min_examples=2)
    if 'error' in result:
        print(f'  ERROR: {result["error"]}')
        return {}

    clusters   = result['clusters']
    assignment = result['assignment']

    # Display discovered categories.
    print(f'\n  Discovered {result["n_clusters"]} character categories:\n')
    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        # Sort by frequency (most common first).
        chars_str = '  '.join(repr(c) for c in sorted(members)[:20])
        print(f'  C{cid} ({len(members)} chars): {chars_str}')

    # Show transition matrix between categories.
    _banner('Phase 1.3: Discovered Character Grammar')
    print('  P(next_category | curr_category)\n')
    k = result['n_clusters']
    trans = [[0] * k for _ in range(k)]
    for inputs, outputs in store.examples:
        c1 = assignment.get(inputs[0])
        c2 = assignment.get(outputs[0])
        if c1 is not None and c2 is not None:
            trans[c1][c2] += 1

    for i in range(k):
        total = sum(trans[i])
        if total == 0:
            continue
        row = [v / total for v in trans[i]]
        top2 = sorted(range(k), key=lambda j: -row[j])[:2]
        row_str = ''.join(f'{p:4.0%} ' for p in row)
        arrow = '-> ' + ' '.join(f'C{j}' for j in top2)
        members_preview = ''.join(sorted(clusters.get(i, []))[:6])
        print(f'  C{i} [{members_preview:<6}] {row_str} {arrow}')

    return {'clusters': clusters, 'assignment': assignment}


# ---------------------------------------------------------------------------
# Phase 2: Glyph category discovery
# ---------------------------------------------------------------------------

def phase2_glyphs(
    ai:       SymbolicAI,
    pairs:    list,
    n_clusters: int = 64,
    verbose:  bool = True,
) -> dict:
    """Teach the AI raw image patch sequences; induce glyph categories.

    For each line image, extract patches using the same _extract_patches()
    + _quantize() machinery as VisualSymbolLearner (Phase R6), then:
        ai.teach('next_glyph', (hash_i,), (hash_i+1,))

    induce_hierarchy() discovers visual glyph categories from the
    statistics of which patch types follow which -- no GT labels needed.
    """
    _banner('Phase 2: Glyph Category Discovery')
    print('  Streaming image patch sequences to the symbolic AI ...')

    if 'next_glyph' not in ai.stores:
        ai.add_concept(
            name='next_glyph', domain='vision',
            description='predict next glyph patch given current glyph patch',
            input_type=['patch_hash'], output_type=['patch_hash'], tier='theorem',
        )

    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print('  ERROR: PIL not available.  Run from venv.')
        return {}

    n_lines = n_patches = n_empty = 0
    for img_path, _ in pairs:
        # Try .bin.png first, then .nrm.png, then the path as-is.
        for candidate in (
            img_path,
            img_path.replace('.nrm.png', '.bin.png'),
            img_path.replace('.bin.png', '.nrm.png'),
        ):
            if os.path.exists(candidate):
                img_path = candidate
                break
        else:
            n_empty += 1
            continue

        try:
            img   = Image.open(img_path).convert('L')
            arr   = np.array(img, dtype=np.float32) / 255.0
            gray  = _to_gray_f32(arr) if arr.ndim > 2 else arr
            patches = _extract_patches(gray, _PATCH_SIZE, _PATCH_SIZE)
            if not patches:
                n_empty += 1
                continue
        except Exception:
            n_empty += 1
            continue

        hashes = [_quantize(_to_gray_f32(p), _QUANT_BITS) for p in patches]
        for i in range(len(hashes) - 1):
            ai.teach('next_glyph', (hashes[i],), (hashes[i + 1],))
            n_patches += 1
        n_lines += 1

    print(f'  {n_lines} images processed ({n_empty} failed), {n_patches:,} patch pairs taught')

    if n_patches == 0:
        print('  ERROR: no patch pairs collected.')
        return {}

    ai.freq_consolidate('next_glyph')
    store = ai.stores['next_glyph']
    n_unique = len({inp for inp, _ in store.examples})
    print(f'  ExampleStore: {len(store):,} examples, {n_unique:,} unique patch hashes')

    _banner('Phase 2.2: Inducing Glyph Categories')
    result = ai.induce_hierarchy('next_glyph', n_clusters=n_clusters, min_examples=3)
    if 'error' in result:
        print(f'  ERROR: {result["error"]}')
        return {}

    clusters = result['clusters']
    print(f'  Discovered {result["n_clusters"]} glyph categories '
          f'from {n_unique:,} unique patch types')
    for cid in sorted(clusters.keys())[:8]:
        print(f'  G{cid}: {len(clusters[cid])} patch types')
    if len(clusters) > 8:
        print(f'  ... ({len(clusters) - 8} more)')

    return {'clusters': clusters, 'assignment': result['assignment']}


# ---------------------------------------------------------------------------
# Phase 3: Alignment — which glyph cluster reads as which char
# ---------------------------------------------------------------------------

def phase3_align(
    ai:              SymbolicAI,
    pairs:           list,
    char_assignment: dict,
    glyph_assignment: dict,
    verbose:         bool = True,
) -> None:
    """Teach glyph→char alignment from paired (image, GT text) data.

    For each line where segment count ≈ GT char count (forced alignment):
        ai.teach('glyph_reads_as', (glyph_cluster,), (char,))

    The AI learns which visual category corresponds to which character.
    This is the only step that uses paired labels.
    """
    _banner('Phase 3: Glyph→Char Alignment')
    print('  Teaching the AI which glyph cluster reads as which character ...')

    if 'glyph_reads_as' not in ai.stores:
        ai.add_concept(
            name='glyph_reads_as', domain='reading',
            description='map visual glyph category to character',
            input_type=['glyph_cluster'], output_type=['char'], tier='theorem',
        )

    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print('  ERROR: PIL not available.')
        return

    n_aligned = n_skipped = n_taught = 0

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
            img  = Image.open(img_path).convert('L')
            arr  = np.array(img, dtype=np.float32) / 255.0
            gray = arr if arr.ndim == 2 else _to_gray_f32(arr)
            patches = _extract_patches(gray, _PATCH_SIZE, _PATCH_SIZE)
            if not patches:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue

        # Forced alignment: accept only if counts are close.
        gt_chars = [c for c in gt if c != ' ']
        if abs(len(patches) - len(gt_chars)) > 0.25 * max(len(gt_chars), 1):
            n_skipped += 1
            continue

        # Teach alignment for each aligned pair.
        limit = min(len(patches), len(gt_chars))
        for i in range(limit):
            h = _quantize(_to_gray_f32(patches[i]), _QUANT_BITS)
            glyph_cid = glyph_assignment.get((h,))
            if glyph_cid is not None:
                ai.teach('glyph_reads_as', (glyph_cid,), (gt_chars[i],))
                n_taught += 1

        n_aligned += 1

    print(f'  {n_aligned} lines aligned, {n_skipped} skipped, {n_taught:,} pairs taught')

    if n_taught == 0:
        print('  ERROR: no alignment pairs collected.')
        return

    ai.freq_consolidate('glyph_reads_as')

    # Show what the AI learned: which glyph cluster → which char.
    store = ai.stores['glyph_reads_as']
    # Build glyph_cluster → {char: count}
    glyph_to_chars: dict = collections.defaultdict(collections.Counter)
    for inputs, outputs in store.examples:
        glyph_to_chars[inputs[0]][outputs[0]] += 1

    _banner('Phase 3.2: Learned Glyph→Char Mapping')
    print(f'\n  {len(glyph_to_chars)} glyph clusters mapped to characters:\n')
    for gcid in sorted(glyph_to_chars.keys())[:20]:
        counter = glyph_to_chars[gcid]
        total   = sum(counter.values())
        top     = counter.most_common(3)
        top_str = '  '.join(f'{repr(c)}:{n/total:.0%}' for c, n in top)
        print(f'  G{gcid}: {top_str}  ({total} obs)')
    if len(glyph_to_chars) > 20:
        print(f'  ... ({len(glyph_to_chars) - 20} more glyph clusters)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Symbolic AI character and glyph category discovery from GT4HistOCR.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--corpus', default='EarlyModernLatin',
                   help=f'Corpus name or "all". Names: {", ".join(_ALL_CORPORA)}.')
    p.add_argument('--n',         type=int, default=None,
                   help='Max line pairs (None=all).')
    p.add_argument('--n_char_clusters', type=int, default=12,
                   help='Number of character clusters (default 12).')
    p.add_argument('--n_glyph_clusters', type=int, default=64,
                   help='Number of glyph clusters (default 64).')
    p.add_argument('--align', action='store_true',
                   help='Run all three phases (char + glyph + alignment).')
    p.add_argument('--chars_only', action='store_true',
                   help='Only run Phase 1 (character discovery).')
    p.add_argument('--save', metavar='PATH', default=None,
                   help='Save AI checkpoint after training (e.g. ocr_knowledge.json).')
    p.add_argument('--load', metavar='PATH', default=None,
                   help='Load existing checkpoint before training (skip already-done phases).')
    args = p.parse_args()

    corpus_dirs = _resolve_corpus(args.corpus)

    # Collect pairs from all corpora.
    all_pairs = []
    for d in corpus_dirs:
        name = os.path.basename(d)
        pairs = find_pairs(d, max_n=args.n, shuffle=True, seed=42)
        print(f'  {name}: {len(pairs)} pairs')
        all_pairs.extend(pairs)

    if not all_pairs:
        print('ERROR: no pairs found.')
        sys.exit(1)

    print(f'\n  Total: {len(all_pairs)} line pairs from {len(corpus_dirs)} corpus/corpora')

    # Build the symbolic AI (arithmetic CTKG as base, same as discover_test.py).
    arith_ctkg = os.path.join(_HERE, '..', 'ctkg', 'domains', 'arithmetic.ctkg')
    ai = SymbolicAI(parse_file(arith_ctkg))

    # Optionally restore a previous checkpoint (skip already-learned phases).
    if args.load:
        load_path = args.load if os.path.isabs(args.load) else os.path.join(_HERE, args.load)
        if os.path.exists(load_path):
            ai.load_checkpoint(load_path)
            print(f'\n  Loaded checkpoint: {load_path}')
        else:
            print(f'\n  WARNING: --load path not found: {load_path}')

    # Phase 1: character categories.
    char_result = phase1_chars(ai, all_pairs, n_clusters=args.n_char_clusters)

    if args.chars_only or not args.align:
        _banner('Summary')
        if char_result:
            print(f'  Character categories discovered: {len(char_result["clusters"])}')
            print(f'  Symbolic AI CTKG extended with "next_char" concept.')
        if args.save:
            save_path = args.save if os.path.isabs(args.save) else os.path.join(_HERE, args.save)
            ai.save_checkpoint(save_path)
            print(f'  Checkpoint saved -> {save_path}')
        return

    # Phase 2: glyph categories.
    glyph_result = phase2_glyphs(ai, all_pairs, n_clusters=args.n_glyph_clusters)

    if not glyph_result:
        print('Phase 2 failed -- cannot run alignment.')
        return

    # Phase 3: alignment.
    phase3_align(
        ai,
        all_pairs,
        char_assignment  = char_result.get('assignment', {}),
        glyph_assignment = glyph_result.get('assignment', {}),
    )

    _banner('Summary')
    print(f"""
  The symbolic AI has autonomously discovered:
    {len(char_result.get("clusters", {}))} character categories from raw GT text
    {len(glyph_result.get("clusters", {}))} glyph categories from raw images
    Alignment: which visual cluster reads as which character

  No external OCR.  No pre-built HMM.  No statistical model outside
  the symbolic AI.  Structure from distributions alone.
""")
    if args.save:
        save_path = args.save if os.path.isabs(args.save) else os.path.join(_HERE, args.save)
        ai.save_checkpoint(save_path)
        print(f'  Checkpoint saved -> {save_path}')
        print(f'  Load in any new AI:')
        print(f'    ai = SymbolicAI(parse_file("arithmetic.ctkg"))')
        print(f'    ai.load_checkpoint("{args.save}")')
        print(f'    char = ai.ask("glyph_reads_as", (glyph_cluster_id,))')


if __name__ == '__main__':
    main()
