"""char_latin.py — Character-level relational learning on EarlyModernLatin corpus.

Hypothesis: language is better represented as a graph than a purely linear
sequence.  We test this by giving the relational pipeline multiple relation
types (forward/backward/skip-gram) and asking:
  - Do 'next' and 'prev' cluster together? (bidirectional symmetry → graph)
  - Do 'next' and 'skip2f' cluster together? (long-range vs local → graph nodes)
  - What graph (C_i → C_j) emerges between character categories?

Relations supplied:
  next   (char_i, 'next',   char_{i+1})   immediate successor
  prev   (char_i, 'prev',   char_{i-1})   immediate predecessor
  skip2f (char_i, 'skip2f', char_{i+2})   two-step lookahead
  skip2b (char_i, 'skip2b', char_{i-2})   two-step lookbehind

If 'next' and 'prev' collapse to one relation cluster → local adjacency is
direction-agnostic → graph structure.  If they stay separate → linear sequence.

Usage:
    python char_latin.py                  # all books, n_clusters=8, all relations
    python char_latin.py --n_clusters 12  # finer clustering
    python char_latin.py --n_books 3      # quick test with 3 books
    python char_latin.py --relations next prev   # forward+backward only
"""
from __future__ import annotations

import argparse
import collections
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

from relational_pipeline import (
    RelationalLearner, RelationClusterer, SecondOrderGrammar,
    GeometryDetector, RelationalParadigmDiscoverer, RelationalSenseSplitter,
)

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

CORPUS_DIR = os.path.join(_HERE, 'data', 'GT4HistOCR', 'corpus', 'EarlyModernLatin')


def _normalise(text: str) -> str:
    """Normalize Early Modern Latin text to a-z + space only."""
    text = text.replace('ſ', 's')
    text = text.replace('æ', 'ae').replace('Æ', 'ae')
    text = text.replace('œ', 'oe').replace('Œ', 'oe')
    text = text.replace('ũ', 'un').replace('ã', 'an')
    text = text.replace('õ', 'on').replace('ñ', 'n')
    text = text.lower()
    nfd = unicodedata.normalize('NFD', text)
    result = []
    for ch in nfd:
        if unicodedata.category(ch) == 'Mn':
            continue
        if ch == ' ':
            result.append(' ')
        elif 'a' <= ch <= 'z':
            result.append(ch)
    return ' '.join(''.join(result).split())


def load_sequences(corpus_dir: str, n_books: int | None = None,
                   min_chars: int = 10) -> list[list[str]]:
    """Load GT text files, welding lines within each book into one sequence.

    Books (subdirectories) are kept SEPARATE — no cross-book adjacency.
    Files are loaded in sorted filename order (= page/line reading order).
    """
    pattern = os.path.join(corpus_dir, '**', '*.gt.txt')
    all_paths = sorted(glob.glob(pattern, recursive=True))

    book_files: dict[str, list[str]] = collections.defaultdict(list)
    for path in all_paths:
        book_files[os.path.dirname(path)].append(path)

    book_dirs = sorted(book_files.keys())
    if n_books is not None:
        book_dirs = book_dirs[:n_books]

    sequences = []
    for book_dir in book_dirs:
        chars: list[str] = []
        for path in sorted(book_files[book_dir]):
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    raw = f.read().strip()
            except OSError:
                continue
            norm = _normalise(raw)
            if not norm:
                continue
            if chars:
                chars.append(' ')
            chars.extend(list(norm))
        if len(chars) >= min_chars:
            sequences.append(chars)

    print(f'Loaded {len(sequences)} books from {corpus_dir}')
    if sequences:
        lengths = [len(s) for s in sequences]
        avg = sum(lengths) // len(lengths)
        print(f'Book lengths: min={min(lengths):,}  avg={avg:,}  max={max(lengths):,}')
        for i, bd in enumerate(book_dirs[:len(sequences)]):
            print(f'  {os.path.basename(bd)}: {lengths[i]:,} chars')
    return sequences


# ---------------------------------------------------------------------------
# Triple construction
# ---------------------------------------------------------------------------

_OFFSETS = {
    'next':   +1,
    'prev':   -1,
    'skip2f': +2,
    'skip2b': -2,
}


def build_triples(sequences: list[list[str]],
                  relations: list[str]) -> list[tuple[str, str, str]]:
    """Convert character sequences to (atom, relation, atom) triples.

    Only relations in `relations` are included.  Offsets that fall outside
    the sequence boundary are silently skipped — no cross-book adjacency.
    """
    triples: list[tuple[str, str, str]] = []
    for seq in sequences:
        n = len(seq)
        for i in range(n):
            for rel in relations:
                off = _OFFSETS[rel]
                j = i + off
                if 0 <= j < n:
                    triples.append((seq[i], rel, seq[j]))
    return triples


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _label(chars: list[str]) -> str:
    has_spc = ' ' in chars
    letters = ''.join(sorted(c for c in chars if c != ' '))
    return (f'[SPC]+{letters}' if has_spc and letters
            else '[SPC]' if has_spc else letters)


def report(learner: RelationalLearner,
           rel_clusterer: RelationClusterer,
           grammar: SecondOrderGrammar,
           triples: list[tuple[str, str, str]]) -> None:

    K = max(learner.assignment.values()) + 1 if learner.assignment else 0

    # ---- Character category clusters -----------------------------------------
    print('\n' + '=' * 65)
    print('L1  CHARACTER CATEGORIES  (atoms clustered by relational context)')
    print('=' * 65)
    cluster_chars: dict[int, list[str]] = collections.defaultdict(list)
    for tok, cid in sorted(learner.assignment.items()):
        cluster_chars[cid].append(tok)
    for cid in sorted(cluster_chars):
        print(f'  C{cid:2d}  {_label(cluster_chars[cid])}')

    # ---- Relation clusters ---------------------------------------------------
    print()
    print('=' * 65)
    print('L2  RELATION CLUSTERS  (do next/prev/skip collapse to one?)')
    print('=' * 65)
    rel_cluster_groups: dict[int, list[str]] = collections.defaultdict(list)
    for rel, cid in sorted(rel_clusterer.assignment.items()):
        rel_cluster_groups[cid].append(rel)
    for rcid in sorted(rel_cluster_groups):
        members = rel_cluster_groups[rcid]
        print(f'  R{rcid}: {members}')

    if len(rel_cluster_groups) == 1:
        print('\n  ► ALL relations collapsed to one cluster.')
        print('    The data is symmetric: forward/backward/skip carry the')
        print('    same distributional signal.  GRAPH structure confirmed.')
    elif len(rel_cluster_groups) == len(rel_clusterer.assignment):
        print('\n  ► Every relation stayed distinct.')
        print('    Forward ≠ backward ≠ skip.  LINEAR sequence structure.')
    else:
        merged = [v for v in rel_cluster_groups.values() if len(v) > 1]
        print(f'\n  ► Partial collapse: {len(merged)} merged groups.')
        for g in merged:
            print(f'    {g} are equivalent.')

    # ---- Character transition graph ------------------------------------------
    print()
    print('=' * 65)
    print('L1  CHARACTER GRAPH  (C_i → C_j transition matrix via "next")')
    print('=' * 65)
    # Build count matrix from triples with relation='next'
    trans = [[0] * K for _ in range(K)]
    for a, rel, b in triples:
        if rel == 'next':
            ca = learner.assignment.get(str(a))
            cb = learner.assignment.get(str(b))
            if ca is not None and cb is not None:
                trans[ca][cb] += 1

    print(f'  {"":5s}' + ''.join(f' C{j:<3d}' for j in range(K)))
    for i in range(K):
        row_total = sum(trans[i])
        if row_total == 0:
            continue
        row = [trans[i][j] / row_total for j in range(K)]
        cells = ''.join(f' {p:4.0%}' if p >= 0.01 else '    .' for p in row)
        label = _label(cluster_chars.get(i, ['?']))
        print(f'  C{i:<2d} [{label:10s}]{cells}')

    # ---- Second-order grammar ------------------------------------------------
    print()
    print('=' * 65)
    print('L4  SECOND-ORDER GRAMMAR  (which relations follow which?)')
    print('=' * 65)
    for rel, dist in sorted(grammar.next_rel_dist.items()):
        top = sorted(dist.items(), key=lambda x: -x[1])[:4]
        top_str = '  '.join(f'{r2}({p:.2f})' for r2, p in top)
        print(f'  after {rel:8s}: {top_str}')

    # ---- Graph summary -------------------------------------------------------
    print()
    print('=' * 65)
    print('SUMMARY')
    print('=' * 65)
    n_rel_clusters = len(rel_cluster_groups)
    print(f'  Character categories:   {len(cluster_chars)}')
    print(f'  Relation clusters:      {n_rel_clusters}  '
          f'(from {len(rel_clusterer.assignment)} relations)')
    print(f'  Graph nodes:            {len(cluster_chars)} character categories')
    # Count non-trivial edges (> 5% probability in next-relation matrix)
    n_edges = sum(1 for i in range(K) for j in range(K)
                  if trans[i][j] > 0 and sum(trans[i]) > 0
                  and trans[i][j] / sum(trans[i]) >= 0.05)
    print(f'  Graph edges (≥5%):      {n_edges}')
    print(f'  Avg degree:             {n_edges / max(len(cluster_chars), 1):.1f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_books', type=int, default=None,
                        help='Limit to first N books (default: all)')
    parser.add_argument('--corpus_dir', default=CORPUS_DIR)
    parser.add_argument('--relations', nargs='+',
                        default=['next', 'prev', 'skip2f', 'skip2b'],
                        choices=['next', 'prev', 'skip2f', 'skip2b'],
                        help='Relations to include (default: all four)')
    args = parser.parse_args()

    print(f'Corpus:    {args.corpus_dir}')
    print(f'Books:     {args.n_books or "all"}')
    print(f'Relations: {args.relations}')
    print()

    sequences = load_sequences(args.corpus_dir, n_books=args.n_books)
    if not sequences:
        print('ERROR: No sequences loaded.')
        sys.exit(1)

    print(f'\nBuilding triples...')
    triples = build_triples(sequences, args.relations)
    print(f'  {len(triples):,} triples  '
          f'({len(set((a, r) for a, r, b in triples)):,} unique (atom,relation) pairs)')

    print(f'\nL1: Fitting RelationalLearner (auto K)...')
    learner = RelationalLearner()
    learner.fit(triples, verbose=True)

    print(f'\nL2: Fitting RelationClusterer...')
    rel_clusterer = RelationClusterer()
    rel_clusterer.fit(learner, triples, verbose=True)

    print(f'\nL4: Fitting SecondOrderGrammar...')
    grammar = SecondOrderGrammar()
    grammar.fit(triples, verbose=True)

    print(f'\nR0: Detecting geometry...')
    geo = GeometryDetector()
    geo.fit(learner, rel_clusterer)

    print(f'\nR1: Discovering paradigmatic roles (E4)...')
    paradigm = RelationalParadigmDiscoverer()
    paradigm.fit(learner, triples, verbose=True)

    print(f'\nR2: Sense disambiguation (E5)...')
    sense = RelationalSenseSplitter()
    sense.fit(sequences, verbose=True)

    report(learner, rel_clusterer, grammar, triples)
    print()
    print(geo.report())
    print()
    print(paradigm.report(learner))
    print()
    print(sense.report())


if __name__ == '__main__':
    main()
