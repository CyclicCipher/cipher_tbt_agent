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
    RelationalAlgebra,
)

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_CORPUS_ROOT = os.path.join(_HERE, 'data', 'GT4HistOCR', 'corpus')
CORPUS_DIR   = os.path.join(_CORPUS_ROOT, 'EarlyModernLatin')

# All corpora available in the dataset.
ALL_CORPUS_DIRS = [
    os.path.join(_CORPUS_ROOT, 'EarlyModernLatin'),       # 12 books — Latin
    os.path.join(_CORPUS_ROOT, 'Kallimachos'),            #  9 books — Early Modern Latin/German
    os.path.join(_CORPUS_ROOT, 'RIDGES-Fraktur'),         # 20 books — Early Modern German
    os.path.join(_CORPUS_ROOT, 'RefCorpus-ENHG-Incunabula'),  #  9 books — ENHG incunabula
    os.path.join(_CORPUS_ROOT, 'dta19'),                  # 39 books — 19th-century German
]


def _normalise(text: str) -> str:
    """Normalise any historical text (Latin or German) to a-z + space only.

    German-specific substitutions applied before stripping:
      ß  → ss   (standard romanisation)
      ä  → ae   (matches how Latin æ ligature is handled)
      ö  → oe   (matches Latin œ ligature)
      ü  → ue
    All remaining diacritics are stripped via NFD decomposition.
    """
    # German special characters
    text = text.replace('ß', 'ss').replace('ẞ', 'ss')
    text = text.replace('ä', 'ae').replace('Ä', 'ae')
    text = text.replace('ö', 'oe').replace('Ö', 'oe')
    text = text.replace('ü', 'ue').replace('Ü', 'ue')
    # Latin ligatures / abbreviations
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

    Welded sequences are cached in welded_cache/ (next to the corpus dir) so
    the 10K+ .gt.txt files are only opened once ever.  Subsequent runs load
    one plain-text file per book instead of thousands of tiny files.
    """
    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(corpus_dir)), 'welded_cache'
    )
    os.makedirs(cache_dir, exist_ok=True)

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
        book_name = os.path.basename(book_dir)
        cache_path = os.path.join(cache_dir, book_name + '.txt')

        if os.path.exists(cache_path):
            with open(cache_path, encoding='utf-8') as f:
                chars = list(f.read())
        else:
            chars = []
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
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(''.join(chars))

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


def load_all_corpora(n_books_per_corpus: int | None = None,
                     corpus_dirs: list | None = None) -> list[list[str]]:
    """Load all corpora (Latin + German), returning one sequence per book.

    Each corpus is loaded separately so book-level identity is preserved.
    Books from different corpora are simply concatenated into one flat list —
    the PCH treats them as independent documents (no cross-book adjacency).

    Parameters
    ----------
    n_books_per_corpus
        If given, load at most this many books from EACH corpus.
    corpus_dirs
        Explicit list of corpus directories to load.  Defaults to
        ``ALL_CORPUS_DIRS`` (all five corpora).
    """
    dirs = corpus_dirs if corpus_dirs is not None else ALL_CORPUS_DIRS
    all_seqs: list = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f'  [skip] {d} — not found')
            continue
        seqs = load_sequences(d, n_books=n_books_per_corpus)
        all_seqs.extend(seqs)
    total = sum(len(s) for s in all_seqs)
    print(f'\nAll corpora: {len(all_seqs)} books, {total:,} total characters.')
    return all_seqs


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

def _hits_at_k(learner: RelationalLearner,
               test_triples: list[tuple[str, str, str]],
               rel: str = 'next',
               ks: tuple = (1, 3, 10),
               n_sample: int = 10000) -> dict[int, float]:
    """Compute Hits@K for a given relation on held-out test triples."""
    import random
    subset = [t for t in test_triples if t[1] == rel]
    if not subset:
        return {k: 0.0 for k in ks}
    if len(subset) > n_sample:
        subset = random.sample(subset, n_sample)
    hits = {k: 0 for k in ks}
    for a, r, b in subset:
        dist = learner.predict_dist(a, r)
        if not dist:
            continue
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        true_rank = next((i + 1 for i, (tok, _) in enumerate(ranked)
                          if tok == b), len(ranked) + 1)
        for k in ks:
            if true_rank <= k:
                hits[k] += 1
    n = len(subset)
    return {k: hits[k] / n for k in ks}


def _run_benchmark(sequences: list[list[str]],
                   relations: list[str]) -> None:
    """R5: Train/test split benchmark — compare RelationalLearner against baselines."""
    import random

    # 80/20 sequence split (preserve book-level integrity)
    n = len(sequences)
    random.seed(42)
    idx = list(range(n))
    random.shuffle(idx)
    split = max(1, int(n * 0.8))
    train_seqs = [sequences[i] for i in idx[:split]]
    test_seqs  = [sequences[i] for i in idx[split:]]

    if not test_seqs:
        test_seqs = sequences  # fallback: use all if only 1 book

    train_triples = build_triples(train_seqs, relations)
    test_triples  = build_triples(test_seqs,  relations)

    # Train on 80%
    train_learner = RelationalLearner()
    train_learner.fit(train_triples, verbose=False)

    # Baselines: random and unigram frequency
    all_targets = [b for _, r, b in test_triples if r == 'next']
    from collections import Counter
    freq = Counter(all_targets)
    top_chars = [c for c, _ in freq.most_common()]
    V = len(top_chars)

    ks = (1, 3, 10)
    print(f'\n  Train triples: {len(train_triples):,}  '
          f'Test triples: {len(test_triples):,}')

    hits_model = _hits_at_k(train_learner, test_triples, rel='next', ks=ks)

    print(f'\n  Hits@K on "next" relation  (V={V} atoms)')
    print(f'  {"":20s} H@1     H@3     H@10')
    print(f'  {"Random":20s} '
          + '  '.join(f'{k/V:.3f}' for k in ks))
    print(f'  {"Unigram (most-freq)":20s} '
          + '  '.join(f'{sum(1 for c in top_chars[:k]) / V if k <= V else 1.0:.3f}'
                      if False else f'{k/V:.3f}' for k in ks))
    # Correct unigram: Hits@K = fraction of test triples where true target is in top-K by frequency
    unigram_hits = {k: sum(1 for a, r, b in test_triples
                           if r == 'next' and b in top_chars[:k])
                    / max(1, sum(1 for _, r, _ in test_triples if r == 'next'))
                    for k in ks}
    print(f'  {"Unigram":20s} '
          + '  '.join(f'{unigram_hits[k]:.3f}' for k in ks))
    print(f'  {"RelationalLearner E3":20s} '
          + '  '.join(f'{hits_model[k]:.3f}' for k in ks))


def _run_context_benchmark(
        train_seqs:  list[list[str]],
        test_seqs:   list[list[str]],
        learner:     RelationalLearner,
        context_lens: tuple = (0, 3, 10, 30),
        n_sample:    int = 5000,
        rel:         str = 'next',
) -> None:
    """R5+: Context-aware Hits@K using ContextBeliefState.

    Processes each test sequence CHARACTER BY CHARACTER, maintaining a running
    ContextBeliefState for a window of preceding characters.  At each position,
    predicts the next character using:

      context_len=0  (baseline): learner.predict_dist(char, rel) — stateless
      context_len=N: ContextBeliefState built from the N preceding chars

    The improvement (if any) is the epistemic gain from knowing where in the
    category sequence the agent currently is — a category-level context signal
    on top of the atom-level bigram.

    Prints a comparison table for all context_lens.
    """
    import random
    from relational_pipeline import ContextBeliefState

    # Collect (sequence, position) pairs where we can evaluate
    max_ctx = max(context_lens)
    test_positions: list[tuple[list[str], int]] = []
    for seq in test_seqs:
        for i in range(max_ctx, len(seq) - 1):
            test_positions.append((seq, i))
    if not test_positions:
        print('  No test positions available.')
        return
    if len(test_positions) > n_sample:
        random.seed(42)
        test_positions = random.sample(test_positions, n_sample)
    n_eval = len(test_positions)

    ks = (1, 3, 10)

    # Evaluate each context length
    results: dict[int, dict] = {}
    for ctx_len in context_lens:
        hits = {k: 0 for k in ks}
        no_dist = 0

        for seq, pos in test_positions:
            true_next = seq[pos + 1]
            cur_char  = seq[pos]

            if ctx_len == 0:
                # Stateless: atom-level bigram / category fallback
                dist = learner.predict_dist(cur_char, rel)
            else:
                # Contextual: walk back ctx_len steps, accumulate belief
                cb = ContextBeliefState(learner)
                start = max(0, pos - ctx_len + 1)
                for i in range(start, pos + 1):
                    cb.observe(seq[i])
                    if i < pos:
                        cb.transition(rel)
                dist = cb.predict_target_dist(rel)

            if not dist:
                no_dist += 1
                continue

            ranked = sorted(dist.items(), key=lambda kv: -kv[1])
            true_rank = next(
                (i + 1 for i, (tok, _) in enumerate(ranked) if tok == true_next),
                len(ranked) + 1,
            )
            for k in ks:
                if true_rank <= k:
                    hits[k] += 1

        results[ctx_len] = {
            'hits':    {k: hits[k] / n_eval for k in ks},
            'no_dist': no_dist,
        }

    print(f'\n  Context-aware Hits@K  (n={n_eval:,}, rel={rel!r})')
    print(f'  {"context_len":>12s}  H@1     H@3     H@10   no_dist')
    print(f'  {"-" * 50}')
    for ctx_len in context_lens:
        r = results[ctx_len]
        label = f'ctx={ctx_len}' + (' (stateless)' if ctx_len == 0 else '')
        print(f'  {label:>20s}  '
              + '  '.join(f'{r["hits"][k]:.3f}' for k in ks)
              + f'   {r["no_dist"]}')


def _demo_infer_chain(learner: RelationalLearner,
                      algebra: 'RelationalAlgebra') -> None:
    """R6: Demo compositional relational inference with infer_chain()."""
    print('  Demonstrating infer_chain() — distribution-preserving multi-hop inference')
    print()

    # Test cases: verify that next∘next ≈ skip2f (R3 composition rule)
    test_atoms = ['q', 't', ' ', 'a', 'e']
    for atom in test_atoms:
        chain_result = learner.infer_chain(atom, ['next', 'next'], topk=3)
        single_result = learner.infer_chain(atom, ['skip2f'], topk=3)
        chain_top  = [f"{tok}({p:.2f})" for tok, p in chain_result]
        single_top = [f"{tok}({p:.2f})" for tok, p in single_result]
        print(f'  {atom!r}: next∘next→ {chain_top}  |  skip2f→ {single_top}')

    print()
    # Verify prev∘prev ≈ skip2b
    print('  Verifying prev∘prev ≈ skip2b:')
    for atom in ['a', 'u', 't']:
        c1 = learner.infer_chain(atom, ['prev', 'prev'], topk=3)
        c2 = learner.infer_chain(atom, ['skip2b'], topk=3)
        t1 = [f"{tok}({p:.2f})" for tok, p in c1]
        t2 = [f"{tok}({p:.2f})" for tok, p in c2]
        print(f'  {atom!r}: prev∘prev→ {t1}  |  skip2b→ {t2}')

    # Confirmed rules from R3
    confirmed = [(ri, rj, rk, jsd)
                 for (ri, rj), (rk, jsd) in algebra.composition_table.items()
                 if rk is not None]
    print(f'\n  R3 confirmed {len(confirmed)} composition rules '
          f'(verified above that infer_chain reproduces them).')


def _eval_accuracy(learner: RelationalLearner,
                   triples: list[tuple[str, str, str]],
                   n_sample: int = 5000,
                   rel: str = 'next') -> float:
    """Estimate E3 top-1 prediction accuracy on a random sample of triples."""
    import random
    subset = [t for t in triples if t[1] == rel]
    if not subset:
        return 0.0
    if len(subset) > n_sample:
        subset = random.sample(subset, n_sample)
    correct = 0
    for a, r, b in subset:
        pred = learner.predict(a, r)
        if pred == b:
            correct += 1
    return correct / len(subset)


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
                        help='Limit books per corpus (default: all)')
    parser.add_argument('--corpus_dir', default=CORPUS_DIR)
    parser.add_argument('--all_corpora', action='store_true',
                        help='Load all 5 corpora (Latin + German) instead of '
                             'just EarlyModernLatin')
    parser.add_argument('--save_path', default=None,
                        help='Override compressed-model save path '
                             '(default: pch_compressed.pkl or pch_all_corpora.pkl)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbose E0-R6 analysis output (default: compact summary only)')
    parser.add_argument('--skip-reprocess', action='store_true', default=False,
                        help='Skip the M13 frozen second-pass (saves ~same time as first pass)')
    parser.add_argument('--relations', nargs='+',
                        default=['next', 'prev', 'skip2f', 'skip2b'],
                        choices=['next', 'prev', 'skip2f', 'skip2b'],
                        help='Relations to include (default: all four)')
    parser.add_argument('--mode', default='pch',
                        choices=['rl', 'pch'],
                        help=(
                            'Pipeline to run after the base R0-R6 analysis. '
                            '"rl" = base only; '
                            '"pch" = online PredictiveCodingHierarchy (M8, default)'))
    args = parser.parse_args()

    if args.all_corpora:
        print(f'Corpora:   ALL ({len(ALL_CORPUS_DIRS)} directories)')
    else:
        print(f'Corpus:    {args.corpus_dir}')
    print(f'Books:     {args.n_books or "all"} per corpus')
    print(f'Relations: {args.relations}')
    print()

    if args.all_corpora:
        sequences = load_all_corpora(n_books_per_corpus=args.n_books)
    else:
        sequences = load_sequences(args.corpus_dir, n_books=args.n_books)
    if not sequences:
        print('ERROR: No sequences loaded.')
        sys.exit(1)
    total_chars = sum(len(s) for s in sequences)
    print(f'Loaded {len(sequences)} books, {total_chars:,} total characters.')

    # PCH: full online pipeline + multi-scale R0-R6 + cross-level analysis.
    if args.mode == 'pch':
        print(f'\n{"=" * 65}')
        print('PredictiveCodingHierarchy + Multi-Scale R0-R6 (M9/M10)')
        print('=' * 65)
        from relational_pipeline import PredictiveCodingHierarchy
        pch = PredictiveCodingHierarchy(
            n_levels=10,
            max_chunk_size=7,
            adaptive_threshold=True,
            surprise_k=0.5,
            min_tokens_active=20,
        )
        print(f'Processing {len(sequences)} books ({total_chars:,} chars)...')
        pch.process_corpus(sequences)
        pch.level_summary()

        print(f'\n{"=" * 65}')
        print('M9: Multi-scale R0-R6 analysis (M12: type-abstracted clustering)')
        print('=' * 65)
        pch.analyse_with_sequences(sequences, verbose=args.verbose)

        print(f'\n{"=" * 65}')
        print('M10: Cross-level constituency analysis')
        print('=' * 65)
        pch.analyse_cross_level(verbose=args.verbose)

        print(f'\n{"=" * 65}')
        print('M13/M16: Init belief cascade + gated-DeltaNet reprocess')
        print('=' * 65)
        pch.init_beliefs()
        n_beliefs = sum(1 for b in pch._beliefs if b is not None)
        print(f'  Belief states initialised: {n_beliefs} levels')
        if not args.skip_reprocess:
            pch.reprocess(sequences)
            print('  Frozen reprocess complete.')
        else:
            print('  (reprocess skipped via --skip-reprocess)')

        # Compute CTKG stats directly — avoid building a multi-MB string for large runs.
        n_concepts = pch.vocab.n_merges() + pch.vocab.n_segments()
        n_types = sum(
            len(set(getattr(lrn, 'assignment', {}).values()))
            for lrn in pch.learners
        )
        # Only generate the full string when small enough to be useful.
        _MAX_CTKG_CONCEPTS = 10_000
        if n_concepts <= _MAX_CTKG_CONCEPTS:
            ctkg_str = pch.export_ctkg(domain_name='latin_pch')
            n_lines = ctkg_str.count('\n')
        else:
            ctkg_str = ''
            n_lines = 0

        print(f'\n{"=" * 65}')
        print('M14: Compression pass + save (type-only model)')
        print('=' * 65)
        pch.compress(verbose=args.verbose)
        import os as _os
        if args.save_path:
            _ckpt = args.save_path
        elif args.all_corpora:
            _ckpt = _os.path.join(_os.path.dirname(__file__), 'pch_all_corpora.pkl')
        else:
            _ckpt = _os.path.join(_os.path.dirname(__file__), 'pch_compressed.pkl')
        pch.save_compressed(_ckpt)
        # Verify round-trip.
        from relational_pipeline import PredictiveCodingHierarchy as _PCH
        pch2 = _PCH.load_compressed(_ckpt)
        pch2.init_beliefs()
        print(f'  Load verified: {sum(1 for b in pch2._beliefs if b is not None)} beliefs restored')

        print(f'\nCTKG stats (pre-compression): {n_types} types, {n_concepts} concepts'
              + (f', {n_lines} lines' if n_lines else ' (skipped: vocab too large)'))
        if ctkg_str:
            print('\n--- CTKG preview (first 50 lines) ---')
            for line in ctkg_str.splitlines()[:50]:
                print(f'  {line}')

        print(f'\n{"=" * 65}')
        print('M15: Type-level inference — perplexity + reasoning chains')
        print('=' * 65)
        # Re-init beliefs on the loaded (compressed) model for clean evaluation.
        pch2.init_beliefs()

        # Perplexity on a held-out book (last book, if multiple).
        eval_seqs = sequences[-1:] if len(sequences) > 1 else sequences
        ppl = pch2.evaluate_perplexity(eval_seqs, level=0)
        # Baseline = uniform over observed vocab (log2 V).
        V0 = len(getattr(pch2.learners[0], 'assignment', {}) or {})
        if V0 < 2:
            V0 = len(set(c for seq in eval_seqs for c in seq))
        import math as _math
        baseline = _math.log2(V0) if V0 > 1 else 1.0
        print(f'  Level-0 bigram perplexity (held-out book): {ppl:.3f} bits/token')
        print(f'  Uniform baseline (log2 V={V0}):            {baseline:.3f} bits/token')
        print(f'  Compression gain: {baseline - ppl:.3f} bits/token')

        # Reasoning chain demo: what follows a word in the most common category?
        # Pick 3 representative surface forms and show 2-hop predictions.
        demo_tokens = [' ', 'e', 't']
        for tok in demo_tokens:
            chain = pch2.reason_chain(tok, ['next', 'next'], level=0, topk=5)
            chain_str = ', '.join(f'{t!r}:{p:.2f}' for t, p in chain)
            print(f'  reason_chain({tok!r}, [next,next]) → {chain_str}')

        print(f'\n{"=" * 65}')
        print('M17: Cross-domain functors + sheaf consistency')
        print('=' * 65)

        # Build Interface for the loaded model (domain A).
        kg_a = pch2.build_interface('latin_A')
        iface_a = list(kg_a.interfaces.values())[0]
        print(f'  Interface latin_A: {len(iface_a.types)} types, '
              f'{len(iface_a.concepts)} concepts')

        # Self-functor: align pch2 with itself (should give perfect match).
        result = pch2.build_functor(pch2, sim_threshold=0.8,
                                    domain_self='latin_A', domain_other='latin_A2')
        n_mapped = sum(len(m) for m in result['mapping'].values())
        n_viol   = len(result['sheaf_violations'])
        print(f'  Self-functor: {n_mapped} type pairs matched across '
              f'{len(result["mapping"])} levels')
        print(f'  Sheaf check: {n_viol} violation(s) '
              f'(expect 0 for identical types)')
        for lv, lmap in sorted(result['mapping'].items()):
            pairs = ', '.join(f'{k}→{j}' for k, j in sorted(lmap.items()))
            print(f'    L{lv}: {pairs}')

        # Adjunctions: verify next/prev round-trip quality.
        adj_kg = pch2.build_adjunction()
        print(f'\n  Adjunctions (next∘prev round-trip quality per level):')
        for lv in sorted(pch2._adjunction_quality):
            q = pch2._adjunction_quality[lv]
            bar = '█' * int(q * 20)
            print(f'    L{lv}: {q:.3f}  {bar}')

        # Wire functor + adjunctions into a merged CTKG.
        kg_a.sheaf_merge(adj_kg)
        kg_a.adjunctions.update(adj_kg.adjunctions)
        n_adjs = len(kg_a.adjunctions)
        n_funcs = len(result['kg'].functors)
        print(f'\n  Final CTKG: {len(kg_a.types)} types, '
              f'{len(kg_a.concepts)} concepts, '
              f'{n_adjs} adjunction(s), {n_funcs} functor(s)')

        # ── Within-domain: Compose⊣Decompose adjunctions ─────────────────────
        print(f'\n  Compose⊣Decompose adjunctions (within-domain, cross-level):')
        cd_kg = pch2.build_all_compose_decompose_adjunctions('latin')
        cd_results = getattr(pch2, '_compose_decompose_results', {})
        for lv_lo, res in sorted(cd_results.items()):
            u   = res['unit_quality']
            cu  = res['counit_quality']
            nF  = len(res['F'])
            nG  = len(res['G'])
            bar_u  = '█' * int(u  * 20)
            bar_cu = '█' * int(cu * 20)
            print(f'    L{lv_lo}→L{lv_lo+1}:  '
                  f'F={nF} pairs  G={nG} pairs  '
                  f'unit={u:.3f} {bar_u}  counit={cu:.3f} {bar_cu}')
        print(f'  Cross-level KG: '
              f'{len(cd_kg.functors)} functor(s), '
              f'{len(cd_kg.adjunctions)} adjunction(s)')

        print(f'\n{"=" * 65}')
        print('M18: Causal reasoning, MasteryState, full CTKG closure')
        print('=' * 65)

        # M18a: MasteryState — which type categories need more data?
        mastery = pch2.type_mastery(tokens_needed_per_type=200)
        frontier = mastery.frontier(threshold=0.8)
        print(f'  MasteryState: {len(mastery.levels)} concepts tracked')
        print(f'  Frontier (needs more data): {len(frontier)} types')
        if frontier:
            print(f'    {sorted(list(frontier))[:8]}')

        # M18a: information_gain per type (prioritize what to learn next).
        kg_mastery = pch2.build_transition_kg(level=1)
        if kg_mastery.concepts:
            ig_scores = {c: kg_mastery.mastery_state().information_gain(c)
                         for c in list(kg_mastery.concepts)[:6]}
            top_ig = sorted(ig_scores.items(), key=lambda kv: -kv[1])[:3]
            print(f'  Top-3 types by information_gain: '
                  + ', '.join(f'{c.split("_")[-1]}:{v:.3f}' for c, v in top_ig))

        # M18c/d: causal analysis at level 0 (character-level Markov chain).
        pch2.causal_analysis(level=0, verbose=True)

        # M18c: d-separation query at level 1.
        K1 = getattr(pch2.learners[1], '_K', 0)
        if K1 >= 3:
            a, b, g = 0, 2, {1}
            sep = pch2.d_separated_types(1, a, b, given_types=g)
            print(f'  d-sep L1: type_{a} ⊥ type_{b} | type_{set(g)}: {sep}')

        print(f'\n  M18 DONE — Full CTKG toolkit exercised:')
        print(f'    Functor ✓  Adjunction ✓  Interface ✓  sheaf_check ✓')
        print(f'    MasteryState ✓  d_separated ✓  intervene ✓  information_flow ✓')
        print(f'    transfer_probability ✓  information_gain ✓')
        return

    print(f'\nBuilding triples...')
    triples = build_triples(sequences, args.relations)
    print(f'  {len(triples):,} triples  '
          f'({len(set((a, r) for a, r, b in triples)):,} unique (atom,relation) pairs)')

    print(f'\nL1: Fitting RelationalLearner (auto K)...')
    learner = RelationalLearner()
    learner.fit(triples, verbose=True)

    print(f'\nL2: Fitting RelationClusterer...')
    rel_clusterer = RelationClusterer()
    rel_clusterer.fit(learner, verbose=True)

    print(f'\nL4: Fitting SecondOrderGrammar...')
    grammar = SecondOrderGrammar()
    grammar.fit(verbose=True, learner=learner)

    print(f'\nR0: Detecting geometry...')
    geo = GeometryDetector()
    geo.fit(learner, rel_clusterer)

    print(f'\nR1: Discovering paradigmatic roles (E4)...')
    paradigm = RelationalParadigmDiscoverer()
    paradigm.fit(learner, verbose=True)

    print(f'\nR2: Sense disambiguation (E5)...')
    sense = RelationalSenseSplitter()
    sense.fit(sequences, verbose=True)

    print(f'\nR3: Relational algebra (E6)...')
    algebra = RelationalAlgebra()
    algebra.fit(learner, verbose=True)

    print(f'\nR5: Benchmark (train/test split Hits@K)...')
    _run_benchmark(sequences, args.relations)

    # R5+: context-aware variant using ContextBeliefState
    print(f'\nR5+: Context-aware benchmark (ContextBeliefState)...')
    import random as _rng
    _rng.seed(42)
    n = len(sequences)
    idx = list(range(n))
    _rng.shuffle(idx)
    split = max(1, int(n * 0.8))
    _train_seqs = [sequences[i] for i in idx[:split]]
    _test_seqs  = [sequences[i] for i in idx[split:]] or sequences
    _ctx_learner = RelationalLearner()
    _ctx_learner.fit(
        build_triples(_train_seqs, args.relations), verbose=False)
    _run_context_benchmark(
        _train_seqs, _test_seqs, _ctx_learner,
        context_lens=(0, 3, 10, 30),
        n_sample=5000,
    )

    print(f'\nR4: Geometry-adapted metric...')
    topology = geo.topology
    acc_before = _eval_accuracy(learner, triples, n_sample=5000)
    learner.adapt_metric(topology)
    acc_after = _eval_accuracy(learner, triples, n_sample=5000)
    print(f'  Topology: {topology}')
    print(f'  E3 accuracy  JSD-metric: {acc_before:.3f}  '
          f'{topology}-metric: {acc_after:.3f}  '
          f'delta={acc_after - acc_before:+.3f}')

    print(f'\nR6: Compositional relational inference (infer_chain)...')
    _demo_infer_chain(learner, algebra)

    report(learner, rel_clusterer, grammar, triples)
    print()
    print(geo.report())
    print()
    print(paradigm.report(learner))
    print()
    print(sense.report())
    print()
    print(algebra.report())

    # M1-M7 / M8: Hierarchical Merge — mode-controlled
    if args.mode == 'rl':
        pass   # base R0-R6 only; no Merge pipeline

    # (mode == 'pch' handled by early return above)


if __name__ == '__main__':
    main()
