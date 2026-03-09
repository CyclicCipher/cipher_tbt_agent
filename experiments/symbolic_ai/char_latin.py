"""char_latin.py — Character-level sequence learning on EarlyModernLatin corpus.

What structure emerges when we treat individual characters (not words) as tokens?

Expected discoveries:
  - Vowels (a, e, i, o, u) cluster together
  - Consonant families: nasals (m, n), liquids (r, l), stops (p, t, c, k)
  - Space clusters with word-boundary patterns
  - Latin morphological affixes (-us, -um, -ae, -is) create characteristic
    bigram patterns that the learner discovers without any linguistic knowledge

Usage:
    python char_latin.py                  # default: first 2000 files, n_clusters=12
    python char_latin.py --n_files 5000   # more data
    python char_latin.py --n_clusters 8   # fewer clusters
    python char_latin.py --all            # all 10K files (slow)
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# UTF-8 stdout (Windows fix)
if (hasattr(sys.stdout, 'buffer') and
        getattr(sys.stdout, 'encoding', 'utf-8').lower() != 'utf-8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

from sequence_pipeline import SequenceLearner

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

CORPUS_DIR = os.path.join(_HERE, 'data', 'GT4HistOCR', 'corpus', 'EarlyModernLatin')


def _normalise(text: str) -> str:
    """Normalize Early Modern Latin text to a clean character set.

    - Lowercase
    - Long s (ſ) → s
    - ae/oe ligatures → ae/oe
    - u-tilde (ũ) → un  (Latin abbreviation for 'un')
    - Strip combining diacritics via NFD decomposition
    - Keep: a-z, space; drop everything else (punctuation, Greek, numerals)
    """
    # Ligature substitutions BEFORE NFD (NFD won't decompose these cleanly)
    text = text.replace('ſ', 's')
    text = text.replace('æ', 'ae')
    text = text.replace('Æ', 'ae')
    text = text.replace('œ', 'oe')
    text = text.replace('Œ', 'oe')
    text = text.replace('ũ', 'un')
    text = text.replace('ã', 'an')
    text = text.replace('õ', 'on')
    text = text.replace('ñ', 'n')
    text = text.lower()
    # NFD: decompose combining characters, then keep only base ASCII
    nfd = unicodedata.normalize('NFD', text)
    result = []
    for ch in nfd:
        cat = unicodedata.category(ch)
        if cat == 'Mn':   # combining mark → skip
            continue
        if ch == ' ':
            result.append(' ')
        elif 'a' <= ch <= 'z':
            result.append(ch)
        # else: drop (punctuation, Greek, digits, control chars)
    # Collapse multiple spaces
    out = ' '.join(''.join(result).split())
    return out


def load_sequences(corpus_dir: str, n_books: int | None = None,
                   per_line: bool = False,
                   min_chars: int = 10) -> list[list[str]]:
    """Load .gt.txt files and return character-level sequences.

    Default (per_line=False): weld all lines from the same book into one
    sequence, preserving within-book character adjacency.  Lines are joined
    with a single space (natural word boundary between OCR lines).  Books
    (subdirectories) are kept as SEPARATE sequences — a character from
    Tortellius (1471) will never be adjacent to one from Beauvais (1476).

    per_line=True: each file is its own sequence (no cross-line bigrams).

    Args:
        corpus_dir: Path to EarlyModernLatin corpus root.
        n_books:    Limit to first N books (None = all).
        per_line:   If True, treat each .gt.txt line as a separate sequence.
        min_chars:  Discard sequences shorter than this.
    """
    import collections

    pattern = os.path.join(corpus_dir, '**', '*.gt.txt')
    all_paths = sorted(glob.glob(pattern, recursive=True))

    if per_line:
        # --- Old per-line mode (no welding) ----------------------------------
        paths = all_paths
        sequences = []
        n_skipped = 0
        for path in paths:
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    raw = f.read().strip()
            except OSError:
                n_skipped += 1
                continue
            norm = _normalise(raw)
            if len(norm) < min_chars:
                n_skipped += 1
                continue
            sequences.append(list(norm))
        print(f'[per-line] Loaded {len(sequences):,} sequences ({n_skipped} skipped)')
        if sequences:
            lengths = [len(s) for s in sequences]
            print(f'Lengths: min={min(lengths)}, avg={sum(lengths)//len(lengths)}, '
                  f'max={max(lengths)}')
        return sequences

    # --- Book-welding mode (default) -----------------------------------------
    # Group files by their immediate parent directory (= book directory).
    # Sort files within each book by filename (they are numbered 00001, 00002…)
    # so that the text is reconstructed in reading order.
    book_files: dict[str, list[str]] = collections.defaultdict(list)
    for path in all_paths:
        book_dir = os.path.dirname(path)
        book_files[book_dir].append(path)

    book_dirs = sorted(book_files.keys())
    if n_books is not None:
        book_dirs = book_dirs[:n_books]

    sequences = []
    for book_dir in book_dirs:
        files = sorted(book_files[book_dir])   # numerical filename order
        book_chars: list[str] = []
        for path in files:
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    raw = f.read().strip()
            except OSError:
                continue
            norm = _normalise(raw)
            if not norm:
                continue
            if book_chars:
                # Separate consecutive lines with a space (word boundary)
                book_chars.append(' ')
            book_chars.extend(list(norm))

        if len(book_chars) < min_chars:
            continue
        sequences.append(book_chars)

    print(f'[book-welded] Loaded {len(sequences)} books '
          f'({len(book_dirs)} requested)')
    if sequences:
        lengths = [len(s) for s in sequences]
        avg = sum(lengths) // len(lengths)
        print(f'Book lengths (chars): min={min(lengths):,}, avg={avg:,}, '
              f'max={max(lengths):,}')
        for i, book_dir in enumerate(book_dirs[:len(sequences)]):
            print(f'  {os.path.basename(book_dir)}: {lengths[i]:,} chars')
    return sequences


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _cluster_label(chars: list[str]) -> str:
    """Human-readable label for a character cluster."""
    # Separate space from letters
    has_space = ' ' in chars
    letters = sorted(c for c in chars if c != ' ')
    label = ''.join(letters)
    if has_space:
        label = f'[SPC]+{label}' if label else '[SPC]'
    return label


def report(learner: SequenceLearner) -> None:
    """Print cluster assignments and top successor distributions."""
    print('\n' + '=' * 60)
    print('CHARACTER CLUSTERS (by distributional context)')
    print('=' * 60)

    # Group tokens by cluster
    cluster_chars: dict[int, list[str]] = {}
    for tok, cid in sorted(learner.assignment.items()):
        cluster_chars.setdefault(cid, []).append(tok)

    for cid in sorted(cluster_chars):
        chars = cluster_chars[cid]
        label = _cluster_label(chars)
        print(f'  C{cid:2d}  {label}')

    print()
    print('TOP SUCCESSORS PER CLUSTER')
    print('-' * 60)

    # Build cluster → successor-cluster distribution from next_token
    ai = learner.ai
    for cid in sorted(cluster_chars):
        chars = cluster_chars[cid]
        # Aggregate successor distributions over all chars in this cluster
        succ_counts: dict[str, float] = {}
        for ch in chars:
            try:
                dist = ai.ask('next_token', (str(ch),))
            except Exception:
                continue
            for tok, prob in (dist.items() if isinstance(dist, dict) else []):
                succ_counts[tok] = succ_counts.get(tok, 0.0) + prob

        if not succ_counts:
            continue

        total = sum(succ_counts.values())
        succ_norm = {k: v / total for k, v in succ_counts.items()}
        top = sorted(succ_norm.items(), key=lambda x: -x[1])[:8]
        top_str = '  '.join(f'{repr(t)}:{p:.2f}' for t, p in top)

        label = _cluster_label(chars)
        print(f'  C{cid:2d} {label:20s}  →  {top_str}')

    print()
    print('SAMPLE PREDICTIONS')
    print('-' * 60)
    test_pairs = [
        ('a', 'n'),   # common Latin suffix "-an-"
        ('e', 's'),   # "-es" plural/genitive
        ('u', 's'),   # "-us" nominative
        ('i', 'n'),   # "-in-" common sequence
        ('q', 'u'),   # "qu" digraph (always together)
        (' ', 'a'),   # word starting with 'a'
        ('e', 't'),   # "et" (and)
    ]
    for c1, c2 in test_pairs:
        pred = learner.predict((c1, c2))
        if pred is not None:
            print(f'  predict({repr(c1)}, {repr(c2)}) → {repr(pred)}')

    print()
    print('CLUSTER VOCABULARY')
    print('-' * 60)
    print(f'  Unique characters: {len(learner.assignment)}')
    print(f'  Clusters:          {len(cluster_chars)}')
    vowels = set('aeiou')
    v_clusters = set()
    c_clusters: dict[int, list[str]] = {}
    for cid, chars in cluster_chars.items():
        letters = [c for c in chars if c != ' ']
        if any(c in vowels for c in letters):
            v_clusters.add(cid)
        c_clusters[cid] = letters
    print(f'  Clusters with vowels: {sorted(v_clusters)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_books', type=int, default=None,
                        help='Limit to first N books (default: all 12 books)')
    parser.add_argument('--per_line', action='store_true',
                        help='One sequence per OCR line instead of per book')
    parser.add_argument('--n_clusters', type=int, default=12,
                        help='Number of character clusters (default: 12)')
    parser.add_argument('--corpus_dir', default=CORPUS_DIR,
                        help='Path to EarlyModernLatin corpus directory')
    parser.add_argument('--phases', default='e1e3',
                        help='Sequence learner phases (default: e1e3)')
    args = parser.parse_args()

    print(f'Corpus: {args.corpus_dir}')
    print(f'Mode:   {"per-line" if args.per_line else "book-welded"}')
    print(f'Books:  {args.n_books or "all"}')
    print(f'Clusters: {args.n_clusters}')
    print()

    sequences = load_sequences(args.corpus_dir, n_books=args.n_books,
                               per_line=args.per_line)
    if not sequences:
        print('ERROR: No sequences loaded. Check corpus path.')
        sys.exit(1)

    print(f'\nFitting SequenceLearner (n_clusters={args.n_clusters}, phases={args.phases})...')
    learner = SequenceLearner(n_clusters=args.n_clusters)
    learner.fit(sequences, phases=args.phases, verbose=True)

    report(learner)


if __name__ == '__main__':
    main()
