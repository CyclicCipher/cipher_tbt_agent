"""e5_polysemy.py — Phase E5: Polysemy Detection and Word Sense Disambiguation

Polysemy = one surface form, multiple distinct meanings.
    "bank" → financial institution  vs  river bank
    "est"  → copula (is)  vs  abbreviation marker  vs  discourse particle

Signal: bimodal next-word distribution.
    A polysemous word has clustered successor distributions — mass falls into
    2+ groups with a gap between them.
    A high-frequency function word has a FLAT successor distribution (high
    entropy, but uniformly distributed — NOT bimodal).

Architecture:
    1. detect_polysemy: compute entropy H(ask_dist('next_word_hier', (w,)))
       AND cluster entropy H(successor cluster distribution).
       High entropy + non-uniform cluster distribution = polysemy candidate.

    2. split_senses: for each occurrence of word w as w2 in trigram (w1, w2, w3),
       record the preceding cluster c1 = assignment[w1].
       k-means cluster the c1 distribution → n_senses sense classes.
       Each sense class = "word w in [cluster] contexts".

    3. word_sense_given_ctx: (c1, w2) → sense_id
       Replaces E1's word_pos with sense-specific mapping.
       Enables: if "est" in financial context → sense_0 distributions
                if "est" in geographic context → sense_1 distributions

    4. Evaluation: compare next-word prediction with and without sense splitting
       for the polysemous words. With sense splitting, word_given_cat uses
       sense-specific entries, giving cleaner distributions.

Usage:
    python e5_polysemy.py --corpus EarlyModernLatin --n_train 5000
    python e5_polysemy.py --corpus EarlyModernLatin --n_train 5000 --top_n 30
    python e5_polysemy.py --corpus EarlyModernLatin --words est in de non
"""
from __future__ import annotations

import argparse
import collections
import math
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

from discover_structure import run_pipeline, _stream_texts, _banner, _DATA_DIR
from language_pipeline import (
    build_assignment, train_chain,
    predict_chain, logprob_chain,
    logprob_flat,
)

try:
    from ocr_test import find_pairs
except ImportError:
    import glob as _glob, random as _random
    def find_pairs(d, max_n=None, shuffle=False, seed=0):
        pairs = []
        for gt in _glob.glob(os.path.join(d, '**', '*.gt.txt'), recursive=True):
            pairs.append((None, gt))
        if shuffle:
            rng = _random.Random(seed)
            rng.shuffle(pairs)
        return pairs[:max_n] if max_n else pairs


# ---------------------------------------------------------------------------
# Polysemy detection
# ---------------------------------------------------------------------------

def _entropy(dist: dict) -> float:
    """Shannon entropy (bits) of a probability distribution."""
    h = 0.0
    for p in dist.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def _cluster_entropy(dist: dict, assignment: dict) -> float:
    """Entropy of the cluster distribution induced by a word distribution.

    Maps word probabilities onto cluster IDs and computes entropy.
    Low cluster entropy → mass concentrated in few syntactic classes
                        → the word is "focused" (low polysemy risk from entropy).
    High cluster entropy → spreads across many clusters
                        → potential polysemy OR common function word.
    """
    cluster_probs: dict = collections.defaultdict(float)
    for out_tup, prob in dist.items():
        w = out_tup[0] if isinstance(out_tup, tuple) else str(out_tup)
        cid = assignment.get(w)
        if cid is not None:
            cluster_probs[cid] += prob
    total = sum(cluster_probs.values())
    if total < 1e-12:
        return 0.0
    norm = {k: v / total for k, v in cluster_probs.items()}
    return _entropy(norm)


def detect_polysemy(
    ai,
    assignment: dict,
    word_freqs: dict,
    min_freq:   int   = 20,
    top_n:      int   = 30,
    verbose:    bool  = True,
) -> list[tuple[str, float, float, float]]:
    """Detect polysemous words by high-entropy bimodal successor distribution.

    Polysemy score = word_entropy × cluster_entropy / log2(|vocab|)
        - word_entropy:    H(P(next_word | w))    — how spread the successors are
        - cluster_entropy: H(P(next_cluster | w)) — how spread across syntactic classes

    High WORD entropy alone = function word (many successors, all equiprobable).
    High CLUSTER entropy alone = also consistent with function words.
    The PRODUCT of both penalises words that are merely common, and rewards
    words whose successors straddle multiple syntactic classes.

    Returns: [(word, word_entropy, cluster_entropy, score), ...] sorted by score.
    """
    _banner('Phase E5: Detecting polysemous words')

    candidates = []
    n_checked  = 0

    for word, freq in word_freqs.items():
        if freq < min_freq:
            continue
        dist = ai.ask_dist('next_word_hier', (word,))
        if dist is None:
            continue

        h_word    = _entropy(dist)
        h_cluster = _cluster_entropy(dist, assignment)
        # Polysemy score: high when BOTH word and cluster entropy are high.
        # Normalise word entropy by its maximum (log2 of distinct successors).
        n_succ    = len(dist)
        max_h     = math.log2(max(n_succ, 2))
        score     = (h_word / max_h) * h_cluster if max_h > 0 else 0.0

        candidates.append((word, h_word, h_cluster, score))
        n_checked += 1

    # Sort by score descending
    candidates.sort(key=lambda x: -x[3])

    if verbose:
        print(f'  Words checked (freq≥{min_freq}): {n_checked:,}')
        print(f'  Top {min(top_n, len(candidates))} polysemy candidates:')
        print()
        print(f'  {"Word":<20} {"Freq":>6} {"H(word)":>8} {"H(cluster)":>11} {"Score":>7}')
        print('  ' + '-' * 56)
        for word, h_w, h_c, score in candidates[:top_n]:
            freq = word_freqs.get(word, 0)
            print(f'  {word:<20} {freq:>6} {h_w:>8.2f} {h_c:>11.2f} {score:>7.3f}')

    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Sense splitting
# ---------------------------------------------------------------------------

def split_senses(
    word:        str,
    train_texts: list[str],
    assignment:  dict,
    n_senses:    int  = 2,
    verbose:     bool = True,
) -> dict[int, dict]:
    """Split a polysemous word into n_senses context-conditional sense clusters.

    For each occurrence of `word` as w2 in training trigram (w1, w2, w3):
        context = (c1, c3) = (assignment[w1], assignment[w3])

    Cluster contexts via k-means on cluster-ID co-occurrence vectors.
    Returns {sense_id: {context_pair: count}} for n_senses senses.

    Algorithm: k-means on 1-hot (c1, c3) vectors.
    Simple and fast for small K (K²=144 max dimensions).
    """
    _banner(f'Phase E5: Splitting senses of {word!r} (n_senses={n_senses})')

    # Collect (c1, c3) contexts where `word` appeared as w2
    contexts: list[tuple] = []
    for text in train_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            if tokens[i + 1] != word:
                continue
            c1 = assignment.get(tokens[i])
            c3 = assignment.get(tokens[i + 2])
            if c1 is not None and c3 is not None:
                contexts.append((c1, c3))

    if not contexts:
        if verbose:
            print(f'  No training contexts found for {word!r}.')
        return {}

    if verbose:
        print(f'  Occurrences as middle word: {len(contexts)}')

    # All unique K values
    K = max(max(c1, c3) for c1, c3 in contexts) + 1

    # Simple k-means on (c1, c3) co-occurrence indicator
    # Each context is a K²-dim 0/1 vector; cluster by most frequent context bucket
    import random
    rng = random.Random(42)

    # Initialise: assign each context to a random sense
    labels = [rng.randrange(n_senses) for _ in contexts]

    def _centroid(sense_labels, sense_id):
        # Average (c1, c3) distribution for this sense
        freq: dict = collections.Counter()
        total = 0
        for ctx, lbl in zip(contexts, sense_labels):
            if lbl == sense_id:
                freq[ctx] += 1
                total += 1
        if total == 0:
            return {}
        return {ctx: cnt / total for ctx, cnt in freq.items()}

    for _iter in range(20):
        # Compute centroids
        centroids = [_centroid(labels, s) for s in range(n_senses)]
        # Reassign: closest centroid by co-occurrence (dot product = agreement)
        new_labels = []
        for ctx in contexts:
            best_s = 0
            best_score = -1.0
            for s, centroid in enumerate(centroids):
                score = centroid.get(ctx, 0.0)
                if score > best_score:
                    best_score = score
                    best_s = s
            new_labels.append(best_s)
        if new_labels == labels:
            break
        labels = new_labels

    # Build result: {sense_id: {(c1,c3): count}}
    senses: dict = {s: collections.Counter() for s in range(n_senses)}
    for ctx, lbl in zip(contexts, labels):
        senses[lbl][ctx] += 1

    if verbose:
        for s, counter in sorted(senses.items()):
            top_ctx = counter.most_common(5)
            ctx_str = '  '.join(f'(c{c1},c{c3}):{n}' for (c1, c3), n in top_ctx)
            print(f'  Sense {s} ({sum(counter.values())} occurrences): {ctx_str}')

    return senses


def _sense_description(sense_counter: dict, clusters: dict) -> str:
    """Human-readable sense description from (c1,c3) context counter."""
    # Dominant preceding cluster
    c1_freq: dict = collections.Counter()
    for (c1, c3), cnt in sense_counter.items():
        c1_freq[c1] += cnt
    dominant_c1 = c1_freq.most_common(1)[0][0] if c1_freq else -1
    # Sample members of dominant preceding cluster
    members = clusters.get(dominant_c1, [])
    sample = members[:4]
    return f'c{dominant_c1} context: {" ".join(repr(w) for w in sample)}'


# ---------------------------------------------------------------------------
# Sense-aware prediction
# ---------------------------------------------------------------------------

def train_sense_concepts(
    ai,
    train_texts:    list[str],
    assignment:     dict,
    polysemous_map: dict,     # {word: {sense_id: {(c1,c3): count}}}
    verbose:        bool = True,
) -> None:
    """Train sense-specific word_given_cat entries for polysemous words.

    For each polysemous word w2 in a trigram (w1, w2, w3):
        c1 = assignment[w1], c3 = assignment[w3]
        sense = polysemous_map[w2]: lookup (c1, c3) → sense_id
        Teach: word_given_cat_sense(c1, c2=sense_pseudo_id, c3) → w3

    This allows E1 chain to use sense-specific w3 distributions.
    The sense_pseudo_id is K * sense + c2 (encodes sense into the cluster index).
    """
    _banner('Phase E5: Training sense-aware concepts')

    # Build (c1, c3) → sense_id lookup per polysemous word
    sense_lookup: dict = {}    # {word: {(c1,c3): sense_id}}
    for word, senses in polysemous_map.items():
        lookup = {}
        for sense_id, counter in senses.items():
            for ctx in counter:
                # Last-write wins for ties (deterministic)
                if ctx not in lookup or senses[sense_id][ctx] > senses[lookup[ctx]][ctx]:
                    lookup[ctx] = sense_id
        sense_lookup[word] = lookup

    # Register sense-aware concept
    if 'word_given_sense' not in ai.stores:
        ai.add_concept(
            name='word_given_sense', domain='language',
            description='Sense-aware: (c1, sense_label, c3) → w3',
            input_type=['cat', 'sense', 'cat'], output_type=['word'], tier='theorem',
        )

    n_used = n_skip = 0
    for text in train_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            if w2 not in sense_lookup:
                continue   # only train on polysemous words
            c1 = assignment.get(w1)
            c3 = assignment.get(w3)
            if c1 is None or c3 is None:
                n_skip += 1
                continue
            sense_id = sense_lookup[w2].get((c1, c3), 0)
            sense_label = f'{w2}_{sense_id}'
            ai.teach('word_given_sense', (str(c1), sense_label, str(c3)), (w3,))
            n_used += 1

    if verbose:
        n_ex = len(ai.stores.get('word_given_sense',
                                  type('', (), {'examples': []})()).examples)
        print(f'  Sense-aware examples trained: {n_ex:,}  (skipped: {n_skip:,})')


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def polysemy_report(
    ai,
    candidates:  list,         # from detect_polysemy
    assignment:  dict,
    clusters:    dict,
    train_texts: list[str],
    n_senses:    int = 2,
    show_top:    int = 8,
) -> dict[str, dict]:
    """For the top polysemy candidates, split senses and show interpretation."""
    _banner('Phase E5: Polysemy report (top candidates)')

    polysemous_map: dict = {}

    for word, h_w, h_c, score in candidates[:show_top]:
        print(f'\n  ── {word!r}  (H_word={h_w:.2f}, H_cluster={h_c:.2f}, score={score:.3f}) ──')
        senses = split_senses(
            word, train_texts, assignment, n_senses=n_senses, verbose=True)
        if senses:
            polysemous_map[word] = senses
            for s_id, counter in sorted(senses.items()):
                desc = _sense_description(counter, clusters)
                print(f'    Sense {s_id}: {desc}')

    return polysemous_map


# ---------------------------------------------------------------------------
# Evaluation: does sense splitting improve next-word prediction?
# ---------------------------------------------------------------------------

def evaluate_sense_prediction(
    ai,
    test_texts:     list[str],
    assignment:     dict,
    train_pairs:    set,         # (w2, w3) pairs seen in training
    sense_lookup:   dict,        # {word: {(c1,c3): sense_id}}
    verbose:        bool = True,
) -> dict:
    """Compare standard E1 vs sense-aware prediction on polysemous words.

    For trigrams where w2 is polysemous (in sense_lookup):
        Standard E1: predict w3 via word_given_cat(c1, c2, c3)
        Sense-aware: predict w3 via word_given_sense(c1, sense_label, c3)

    Shows whether sense splitting improves accuracy on ambiguous words.
    """
    _banner('Phase E5: Sense-aware vs E1 prediction on polysemous words')

    r = dict(
        total=0,
        e1_correct=0,    e1_answered=0,
        sense_correct=0, sense_answered=0,
        unseen_total=0,
        e1_unseen_correct=0,    e1_unseen_answered=0,
        sense_unseen_correct=0, sense_unseen_answered=0,
    )

    for text in test_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            if w2 not in sense_lookup:
                continue    # only evaluate on polysemous words

            c1 = assignment.get(w1)
            c3_true = assignment.get(w3)
            is_unseen = (w2, w3) not in train_pairs

            r['total'] += 1
            if is_unseen:
                r['unseen_total'] += 1

            # E1 standard
            e1_pred = predict_chain(ai, w1, w2, assignment)
            if e1_pred is not None:
                r['e1_answered'] += 1
                if e1_pred == w3:
                    r['e1_correct'] += 1
                    if is_unseen:
                        r['e1_unseen_correct'] += 1
                if is_unseen:
                    r['e1_unseen_answered'] += 1

            # Sense-aware
            if c1 is not None and c3_true is not None:
                sense_id    = sense_lookup[w2].get((c1, c3_true), 0)
                sense_label = f'{w2}_{sense_id}'
                sense_pred_tup = ai.ask('word_given_sense', (str(c1), sense_label, str(c3_true)))
                if sense_pred_tup is not None:
                    sense_pred = sense_pred_tup[0] if isinstance(sense_pred_tup, tuple) else str(sense_pred_tup)
                    r['sense_answered'] += 1
                    if sense_pred == w3:
                        r['sense_correct'] += 1
                        if is_unseen:
                            r['sense_unseen_correct'] += 1
                    if is_unseen:
                        r['sense_unseen_answered'] += 1

    if verbose:
        T  = r['total']
        UT = r['unseen_total']

        def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'

        ea = r['e1_answered'];    ec = r['e1_correct']
        sa = r['sense_answered']; sc = r['sense_correct']
        eua = r['e1_unseen_answered'];    euc = r['e1_unseen_correct']
        sua = r['sense_unseen_answered']; suc = r['sense_unseen_correct']

        W = 12
        LW = 36

        def row(label, ev, sv):
            return f'  {label:<{LW}} {ev:>{W}} {sv:>{W}}'

        print(f'\n  Trigrams with polysemous w2: {T:,}  (unseen pairs: {UT:,})')
        print()
        print(row('Metric', 'E1 (standard)', 'Sense-aware'))
        print('  ' + '-' * (LW + 2 * W + 2))
        print(row('Coverage',                pct(ea, T),   pct(sa, T)))
        print(row('Top-1 accuracy (answ.)',  pct(ec, ea),  pct(sc, sa)))
        print(row('Top-1 accuracy (total)',  pct(ec, T),   pct(sc, T)))
        print()
        print(f'  {"=== UNSEEN (w2,w3) PAIRS ==="}')
        print('  ' + '-' * (LW + 2 * W + 2))
        print(row('Coverage',                pct(eua, UT), pct(sua, UT)))
        print(row('Acc. of answered',        pct(euc, eua), pct(suc, sua)))
        print(row('Acc. of total',           pct(euc, UT), pct(suc, UT)))
        print()

        if T > 0:
            sense_acc = sc / max(sa, 1)
            e1_acc    = ec / max(ea, 1)
            if sense_acc > e1_acc:
                print(f'  RESULT: Sense-aware BETTER on polysemous words '
                      f'({pct(sc, sa)} vs {pct(ec, ea)}). Polysemy hypothesis validated.')
            else:
                print(f'  RESULT: E1 matches sense-aware. Polysemy splitting did not help '
                      f'— senses may overlap or corpus too small.')

    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Phase E5: Polysemy Detection and Word Sense Disambiguation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--corpus',     default='EarlyModernLatin')
    p.add_argument('--n_train',    type=int,   default=5000)
    p.add_argument('--n_test',     type=int,   default=None)
    p.add_argument('--split',      type=float, default=0.8)
    p.add_argument('--n_clusters', type=int,   default=12)
    p.add_argument('--min_freq',   type=int,   default=20,
                   help='Min word frequency to consider for polysemy (default: 20)')
    p.add_argument('--top_n',      type=int,   default=15,
                   help='Number of top polysemy candidates to report (default: 15)')
    p.add_argument('--n_senses',   type=int,   default=2,
                   help='Number of senses to split each polysemous word into (default: 2)')
    p.add_argument('--words',      nargs='+', metavar='WORD',
                   help='Force split these specific words (in addition to auto-detected)')
    p.add_argument('--load',       metavar='PATH')
    p.add_argument('--save',       metavar='PATH')
    p.add_argument('--seed',       type=int,   default=42)
    args = p.parse_args()

    _banner('Phase E5: Polysemy Detection and Word Sense Disambiguation')
    print(f'  Corpus:      {args.corpus}')
    print(f'  Train lines: {args.n_train}')
    print(f'  K clusters:  {args.n_clusters}')
    print(f'  min_freq:    {args.min_freq}')
    print(f'  n_senses:    {args.n_senses}')

    # ---- Corpus ----
    d = (args.corpus if os.path.isdir(args.corpus)
         else os.path.join(_DATA_DIR, args.corpus))
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {args.corpus!r}')
        sys.exit(1)

    total_n = args.n_train + (args.n_test or max(args.n_train // 4, 100))
    pairs   = find_pairs(d, max_n=total_n, shuffle=True, seed=args.seed)
    texts   = _stream_texts(pairs)

    n_train     = min(args.n_train, int(len(texts) * args.split))
    n_test_size = args.n_test or max(n_train // 4, 50)
    train_texts = texts[:n_train]
    test_texts  = texts[n_train: n_train + n_test_size]
    print(f'  Lines: {len(train_texts)} train  {len(test_texts)} test')

    # ---- Discovery + E1 ----
    print('\n  Running multi-scale discovery...')
    ai, _ = run_pipeline(
        texts=train_texts, n_levels=3, max_merges=500,
        save_path=None, load_path=args.load, verbose=False,
    )

    _banner('Building E1 word cluster assignment')
    assignment, clusters = build_assignment(ai, n_clusters=args.n_clusters)
    print(f'  Vocab: {len(assignment):,}   Clusters: {len(clusters)}')

    train_chain(ai, train_texts, assignment, verbose=False)

    # Word frequencies for polysemy detection
    word_freqs: collections.Counter = collections.Counter()
    for text in train_texts:
        word_freqs.update(text.split())

    # Train pairs for seen/unseen split
    train_pairs: set = set()
    for text in train_texts:
        toks = text.split()
        for i in range(len(toks) - 1):
            train_pairs.add((toks[i], toks[i + 1]))

    # ---- Phase E5: detect polysemy ----
    candidates = detect_polysemy(
        ai, assignment, dict(word_freqs),
        min_freq=args.min_freq, top_n=args.top_n, verbose=True,
    )

    # Add forced words at front
    if args.words:
        forced = [(w, 0.0, 0.0, 0.0) for w in args.words if w not in {c[0] for c in candidates}]
        candidates = forced + candidates

    if not candidates:
        print('\n  No polysemy candidates found. Try --min_freq lower or larger corpus.')
        return

    # ---- Polysemy report: sense splitting ----
    polysemous_map = polysemy_report(
        ai, candidates, assignment, clusters, train_texts,
        n_senses=args.n_senses, show_top=min(args.top_n, 8),
    )

    if not polysemous_map:
        print('\n  No senses split successfully.')
        return

    # ---- Train sense-aware concepts ----
    train_sense_concepts(ai, train_texts, assignment, polysemous_map, verbose=True)

    # ---- Build sense lookup for evaluation ----
    sense_lookup: dict = {}
    for word, senses in polysemous_map.items():
        lookup = {}
        for sense_id, counter in senses.items():
            for ctx in counter:
                if ctx not in lookup or senses[sense_id][ctx] > senses.get(lookup.get(ctx, -1), {}).get(ctx, 0):
                    lookup[ctx] = sense_id
        sense_lookup[word] = lookup

    # ---- Evaluate sense-aware prediction ----
    evaluate_sense_prediction(
        ai, test_texts, assignment, train_pairs, sense_lookup, verbose=True)

    if args.save:
        ai.save_checkpoint(args.save)
        print(f'\n  Checkpoint saved: {args.save}')


if __name__ == '__main__':
    main()
