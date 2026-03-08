"""language_pipeline.py — Category-chain word prediction (language.ctkg Phase E1).

Tests the central thesis of language.ctkg: using category-compressed context
(distributional POS-like clusters) generalises to unseen word pairs far better
than flat word bigrams.

Architecture (mirrors language.ctkg):
    word_pos        : word → cluster_id           (from induce_hierarchy_bidir)
    next_cat        : cat × cat → cat             (K^2 entries, K=n_clusters)
    word_given_cat  : cat × cat × cat → word      (K^3 entries — generalises)
    next_word(w1,w2) = word_given_cat(
                           word_pos(w1), word_pos(w2),
                           next_cat(word_pos(w1), word_pos(w2)))

Compression example (K=12, V=5000):
    Flat bigram:     V^2 = 25,000,000 entries
    Category chain:  K^3 = 1,728 entries  (~14,000x fewer)
    word_given_cat covers the K^3 combinations; next_word_hier covers the V^2.
    Most POS trigrams are seen during training even if word trigrams aren't.

Key test:
    Evaluate on trigrams where (w_{t-1}, w_t) was NEVER seen in training.
    Flat bigram: cannot predict (returns None).
    Category chain: can still route via (cat_{t-1}, cat_t) → cat_{t+1} → word.

Usage:
    # Basic: 2000 train lines, 500 test lines, 12 word clusters
    python language_pipeline.py --corpus EarlyModernLatin --n_train 2000 --n_test 500

    # More clusters for richer categories
    python language_pipeline.py --corpus EarlyModernLatin --n_train 2000 --n_clusters 20

    # Save AI checkpoint for reuse
    python language_pipeline.py --corpus EarlyModernLatin --n_train 2000 --save chain.pkl
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

from ctkg.parser import parse_file, merge as ctkg_merge
from engine import SymbolicAI
from discover_structure import (
    run_pipeline, _stream_texts, _banner, _DATA_DIR, _DEFAULT_CLUSTERS, _DEFAULT_PMI,
)
from synthesis import discover_categories_from_dists

try:
    from ocr_test import find_pairs, normalise_gt  # type: ignore[import]
except ImportError:
    import glob, random as _random
    def find_pairs(d, max_n=None, shuffle=False, seed=0):
        pairs = []
        for gt in glob.glob(os.path.join(d, '**', '*.gt.txt'), recursive=True):
            pairs.append((None, gt))
        if shuffle:
            rng = _random.Random(seed)
            rng.shuffle(pairs)
        return pairs[:max_n] if max_n else pairs
    def normalise_gt(raw: str) -> str:
        return raw.strip()


# ---------------------------------------------------------------------------
# Step 1: Build word cluster assignment
# ---------------------------------------------------------------------------

def build_assignment(
    ai:         SymbolicAI,
    n_clusters: int = 12,
) -> dict[str, int]:
    """Recover word → cluster_id mapping by re-running bidir clustering on the AI.

    Requires that 'next_word_hier' and 'prev_word_hier' are already trained
    (i.e. run_pipeline with n_levels >= 3 has been called).
    """
    result = ai.induce_hierarchy_bidir(
        'next_word_hier', 'prev_word_hier', n_clusters=n_clusters,
    )
    if 'error' in result:
        print(f'  WARNING: {result["error"]}')
        return {}
    assignment = result.get('assignment', {})
    clusters   = result.get('clusters', {})
    return assignment, clusters


# ---------------------------------------------------------------------------
# Step 2: Train chain concepts
# ---------------------------------------------------------------------------

def train_chain(
    ai:         SymbolicAI,
    texts:      list[str],
    assignment: dict[str, int],
    verbose:    bool = True,
) -> tuple[int, int]:
    """Train next_cat and word_given_cat from texts + cluster assignment.

    Registers both concepts in the AI if not already present.
    Uses trigrams: for each (w1, w2, w3) in the text:
        c1 = assignment[w1], c2 = assignment[w2], c3 = assignment[w3]
        teach next_cat:       (c1, c2) → c3
        teach word_given_cat: (c1, c2, c3) → w3

    Args:
        texts:      Normalised text lines (training portion only).
        assignment: word → cluster_id from build_assignment.
        verbose:    Print progress.

    Returns:
        (n_trigrams_used, n_trigrams_skipped) — skipped when any word is OOV.
    """
    _banner('Training category chain (next_cat + word_given_cat)')

    for cname, desc, in_types, out_types in [
        ('next_cat',       'Category bigram: (cat1,cat2) → cat3',
         ['cat', 'cat'], ['cat']),
        ('word_given_cat', 'Word given category context: (cat1,cat2,cat3) → word',
         ['cat', 'cat', 'cat'], ['word']),
    ]:
        if cname not in ai.stores:
            ai.add_concept(
                name=cname, domain='language',
                description=desc,
                input_type=in_types, output_type=out_types, tier='theorem',
            )

    n_used = 0
    n_skip = 0
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            c1 = assignment.get(w1)
            c2 = assignment.get(w2)
            c3 = assignment.get(w3)
            if c1 is None or c2 is None or c3 is None:
                n_skip += 1
                continue
            c1s, c2s, c3s = str(c1), str(c2), str(c3)
            ai.teach('next_cat',       (c1s, c2s),       (c3s,))
            ai.teach('word_given_cat', (c1s, c2s, c3s),  (w3,))
            n_used += 1

    if verbose:
        n_cat_entries  = len(ai.stores.get('next_cat',       type('', (), {'examples': []})()).examples)
        n_word_entries = len(ai.stores.get('word_given_cat', type('', (), {'examples': []})()).examples)
        n_cat_unique   = len({inp for inp, _ in ai.stores['next_cat'].examples})       if 'next_cat'       in ai.stores else 0
        n_word_unique  = len({inp for inp, _ in ai.stores['word_given_cat'].examples}) if 'word_given_cat' in ai.stores else 0
        print(f'  Trigrams used:  {n_used:,}   skipped (OOV): {n_skip:,}')
        print(f'  next_cat:       {n_cat_entries:,} examples  '
              f'{n_cat_unique:,} unique (c1,c2) inputs')
        print(f'  word_given_cat: {n_word_entries:,} examples  '
              f'{n_word_unique:,} unique (c1,c2,c3) inputs')

    return n_used, n_skip


# ---------------------------------------------------------------------------
# Step 3: Prediction functions
# ---------------------------------------------------------------------------

def predict_chain(
    ai:         SymbolicAI,
    w1:         str,
    w2:         str,
    assignment: dict[str, int],
) -> str | None:
    """Predict w3 via the category chain: (w1,w2) → (c1,c2) → c3 → w3."""
    c1 = assignment.get(w1)
    c2 = assignment.get(w2)
    if c1 is None or c2 is None:
        return None
    c3_tup = ai.ask('next_cat', (str(c1), str(c2)))
    if c3_tup is None:
        return None
    c3 = c3_tup[0] if isinstance(c3_tup, tuple) else str(c3_tup)
    w3_tup = ai.ask('word_given_cat', (str(c1), str(c2), c3))
    if w3_tup is None:
        return None
    return w3_tup[0] if isinstance(w3_tup, tuple) else str(w3_tup)


def predict_flat(
    ai: SymbolicAI,
    w2: str,
) -> str | None:
    """Predict next word using the flat word bigram (next_word_hier)."""
    result = ai.ask('next_word_hier', (w2,))
    if result is None:
        return None
    return result[0] if isinstance(result, tuple) else str(result)


def logprob_chain(
    ai:         SymbolicAI,
    w1:         str,
    w2:         str,
    w3:         str,
    assignment: dict[str, int],
) -> float | None:
    """Log₂ probability of w3 given (w1, w2) via chain. None if OOV."""
    c1 = assignment.get(w1)
    c2 = assignment.get(w2)
    c3 = assignment.get(w3)
    if c1 is None or c2 is None or c3 is None:
        return None
    # P(c3 | c1, c2)
    cat_dist = ai.ask_dist('next_cat', (str(c1), str(c2)))
    if cat_dist is None:
        return None
    c3_tup = (str(c3),)
    p_c3 = cat_dist.get(c3_tup, 0.0)
    if p_c3 < 1e-12:
        return None
    # P(w3 | c1, c2, c3)
    word_dist = ai.ask_dist('word_given_cat', (str(c1), str(c2), str(c3)))
    if word_dist is None:
        return None
    p_w3 = word_dist.get((w3,), 0.0)
    if p_w3 < 1e-12:
        return None
    return math.log2(p_c3 * p_w3)


def logprob_flat(
    ai: SymbolicAI,
    w2: str,
    w3: str,
) -> float | None:
    """Log₂ probability of w3 given w2 via flat bigram. None if OOV."""
    dist = ai.ask_dist('next_word_hier', (w2,))
    if dist is None:
        return None
    p = dist.get((w3,), 0.0)
    if p < 1e-12:
        return None
    return math.log2(p)


# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    ai:          SymbolicAI,
    test_texts:  list[str],
    assignment:  dict[str, int],
    train_pairs: set[tuple[str, str]],
    verbose:     bool = True,
) -> dict:
    """Evaluate category chain vs flat bigram on test trigrams.

    Args:
        train_pairs: Set of (w2, w3) pairs seen in training, for seen/unseen split.

    Returns dict with keys:
        total, chain_correct, flat_correct, chain_answered, flat_answered,
        unseen_total, chain_unseen_correct, flat_unseen_correct,
        chain_unseen_answered, flat_unseen_answered,
        chain_logloss, flat_logloss  (only over answered positions)
    """
    _banner('Evaluation: category chain vs flat bigram')

    r = dict(
        total=0,
        chain_correct=0, flat_correct=0,
        chain_answered=0, flat_answered=0,
        unseen_total=0,
        chain_unseen_correct=0, flat_unseen_correct=0,
        chain_unseen_answered=0, flat_unseen_answered=0,
        chain_logloss=0.0, flat_logloss=0.0,
        chain_logloss_n=0, flat_logloss_n=0,
    )

    for text in test_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            is_unseen = (w2, w3) not in train_pairs

            r['total'] += 1
            if is_unseen:
                r['unseen_total'] += 1

            # Chain prediction.
            chain_pred = predict_chain(ai, w1, w2, assignment)
            if chain_pred is not None:
                r['chain_answered'] += 1
                if chain_pred == w3:
                    r['chain_correct'] += 1
                    if is_unseen:
                        r['chain_unseen_correct'] += 1
                if is_unseen:
                    r['chain_unseen_answered'] += 1

            # Flat bigram prediction.
            flat_pred = predict_flat(ai, w2)
            if flat_pred is not None:
                r['flat_answered'] += 1
                if flat_pred == w3:
                    r['flat_correct'] += 1
                    if is_unseen:
                        r['flat_unseen_correct'] += 1
                if is_unseen:
                    r['flat_unseen_answered'] += 1

            # Log-probabilities for perplexity.
            lp_chain = logprob_chain(ai, w1, w2, w3, assignment)
            if lp_chain is not None:
                r['chain_logloss'] -= lp_chain   # nats accumulate as positive loss
                r['chain_logloss_n'] += 1

            lp_flat = logprob_flat(ai, w2, w3)
            if lp_flat is not None:
                r['flat_logloss'] -= lp_flat
                r['flat_logloss_n'] += 1

    if verbose:
        T  = r['total']
        CA = r['chain_answered']
        FA = r['flat_answered']
        CC = r['chain_correct']
        FC = r['flat_correct']
        UT = r['unseen_total']
        CUA = r['chain_unseen_answered']
        FUA = r['flat_unseen_answered']
        CUC = r['chain_unseen_correct']
        FUC = r['flat_unseen_correct']

        def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'

        chain_ppl = 2 ** (r['chain_logloss'] / max(r['chain_logloss_n'], 1))
        flat_ppl  = 2 ** (r['flat_logloss']  / max(r['flat_logloss_n'],  1))

        print(f'\n  Test trigrams:   {T:,}  (unseen pairs: {UT:,} = {pct(UT, T)})')
        print()
        print(f'  {"Metric":<32} {"Chain":>10} {"Flat":>10}')
        print(f'  {"-"*52}')
        print(f'  {"Coverage (answered/total)":<32} '
              f'{pct(CA,T):>10} {pct(FA,T):>10}')
        print(f'  {"Top-1 accuracy (of answered)":<32} '
              f'{pct(CC,CA):>10} {pct(FC,FA):>10}')
        print(f'  {"Top-1 accuracy (of total)":<32} '
              f'{pct(CC,T):>10} {pct(FC,T):>10}')
        print()
        print(f'  === UNSEEN WORD PAIRS ===')
        print(f'  {"Coverage (answered/unseen)":<32} '
              f'{pct(CUA,UT):>10} {pct(FUA,UT):>10}')
        print(f'  {"Top-1 accuracy (of answered)":<32} '
              f'{pct(CUC,CUA):>10} {pct(FUC,FUA):>10}')
        print(f'  {"Top-1 accuracy (of unseen total)":<32} '
              f'{pct(CUC,UT):>10} {pct(FUC,UT):>10}')
        print()
        print(f'  Perplexity (2^avg_logloss): chain={chain_ppl:.1f}  flat={flat_ppl:.1f}')
        print(f'  (lower = better;  N answered: chain={r["chain_logloss_n"]:,}  '
              f'flat={r["flat_logloss_n"]:,})')
        print()

        # Summary verdict
        chain_acc_unseen = CUC / max(CUA, 1)
        flat_acc_unseen  = FUC / max(FUA, 1)
        if chain_acc_unseen > flat_acc_unseen and CUA > 0:
            print(f'  RESULT: Category chain is BETTER on unseen pairs '
                  f'({pct(CUC,CUA)} vs {pct(FUC,FUA)}). Thesis supported.')
        elif CUA == 0:
            print(f'  RESULT: Chain answered no unseen pairs — corpus too small or '
                  f'n_clusters too high.')
        else:
            print(f'  RESULT: Flat bigram matches or exceeds chain on unseen pairs. '
                  f'Consider more clusters or larger corpus.')

    return r


# ---------------------------------------------------------------------------
# Phase E2: Context-sensitive (trigram) clustering
# ---------------------------------------------------------------------------

def build_trigram_assignment(
    train_texts:     list[str],
    base_assignment: dict[str, int],
    n_clusters:      int = 12,
    min_examples:    int = 3,
    verbose:         bool = True,
) -> tuple[dict, dict]:
    """Phase E2: context-sensitive clustering.

    Clusters (c_prev, word) pairs by their next-word distributions,
    where c_prev = base_assignment[prev_word] is the E1 base cluster.

    The same word gets DIFFERENT cluster assignments depending on what
    cluster preceded it — this is the context-sensitive representation.
    Example: "bank" in DET context → financial-bank cluster;
             "bank" in ADJ context → terrain-bank cluster.

    Returns:
        (context_assignment, ctx_clusters) where:
        context_assignment: {(c_prev, word): ctx_cluster_id}
        ctx_clusters:       {ctx_cluster_id: [(c_prev, word), ...]}
    """
    import collections

    _banner('Phase E2: Building context-sensitive cluster assignment')

    # Accumulate (c_prev, word) → Counter(next_word) from trigrams.
    raw: dict = collections.defaultdict(collections.Counter)
    for text in train_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            c1 = base_assignment.get(w1)
            if c1 is None or w2 not in base_assignment:
                continue
            raw[(c1, w2)][w3] += 1

    if not raw:
        if verbose:
            print('  WARNING: no context-word pairs found')
        return {}, {}

    # Convert to discover_categories_from_dists format.
    # Outer key is ((c_prev, word),) — a 1-tuple wrapping the compound atom.
    # Inner key is (next_word,) — standard single-element tuple.
    dists:  dict = {}
    counts: dict = {}
    for key, counter in raw.items():
        total = sum(counter.values())
        if total < 1:
            continue
        atom = (key,)  # ((c_prev, word),)
        dists[atom]  = {(w,): cnt / total for w, cnt in counter.items()}
        counts[atom] = total

    if verbose:
        print(f'  Context-word pairs (c_prev, word): {len(dists):,}')
        print(f'  Target clusters: {n_clusters}')

    raw_assignment = discover_categories_from_dists(
        dists, counts, n_clusters=n_clusters, min_examples=min_examples,
    )
    if not raw_assignment:
        if verbose:
            print('  WARNING: context clustering returned no assignment')
        return {}, {}

    # Convert: {((c_prev, word),): cluster_id} → {(c_prev, word): cluster_id}
    clusters_raw: dict = collections.defaultdict(list)
    context_assignment: dict = {}
    for atom_key, cid in raw_assignment.items():
        key = atom_key[0]  # (c_prev, word)
        context_assignment[key] = cid
        clusters_raw[cid].append(key)

    # Renumber clusters by size (largest = C0).
    by_size = sorted(clusters_raw.items(), key=lambda kv: -len(kv[1]))
    renumber = {old: new for new, (old, _) in enumerate(by_size)}
    context_assignment = {k: renumber[v] for k, v in context_assignment.items()}
    ctx_clusters = {renumber[old]: members for old, members in clusters_raw.items()}

    if verbose:
        print(f'  Context clusters formed: {len(ctx_clusters)}')
        for cid in sorted(ctx_clusters.keys())[:4]:
            sample = ctx_clusters[cid][:5]
            sample_str = ' '.join(f'(c{c},{w!r})' for c, w in sample)
            print(f'    CC{cid:02d} ({len(ctx_clusters[cid])} members): {sample_str}')
        if len(ctx_clusters) > 4:
            print(f'    ... {len(ctx_clusters) - 4} more clusters')

    return context_assignment, ctx_clusters


def train_chain_ctx(
    ai:                 'SymbolicAI',
    texts:              list[str],
    base_assignment:    dict[str, int],
    context_assignment: dict,
    verbose:            bool = True,
) -> tuple[int, int]:
    """Phase E2: Train next_cat_ctx and word_given_cat_ctx.

    Uses context-sensitive clusters where:
        c2_ctx = context_assignment[(c1, w2)]  — E2 cluster for w2 given c1.
        c3_ctx = context_assignment[(c2_base, w3)] — E2 cluster for w3 given c2_base.

    Training format per trigram (w1, w2, w3):
        c1      = base_assignment[w1]
        c2_ctx  = context_assignment[(c1, w2)]
        c2_base = base_assignment[w2]
        c3_ctx  = context_assignment[(c2_base, w3)]
        teach: next_cat_ctx(c1, c2_ctx) → c3_ctx
        teach: word_given_cat_ctx(c1, c2_ctx, c3_ctx) → w3
    """
    _banner('Phase E2: Training context-sensitive chain (next_cat_ctx + word_given_cat_ctx)')

    for cname, desc, in_types, out_types in [
        ('next_cat_ctx',
         'Context-sensitive category transition: (c1, c2_ctx) → c3_ctx',
         ['cat', 'cat_ctx'], ['cat_ctx']),
        ('word_given_cat_ctx',
         'Word given context-sensitive categories: (c1, c2_ctx, c3_ctx) → word',
         ['cat', 'cat_ctx', 'cat_ctx'], ['word']),
    ]:
        if cname not in ai.stores:
            ai.add_concept(
                name=cname, domain='language', description=desc,
                input_type=in_types, output_type=out_types, tier='theorem',
            )

    n_used = n_skip = 0
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            c1      = base_assignment.get(w1)
            c2_base = base_assignment.get(w2)
            c2_ctx  = context_assignment.get((c1, w2))     if c1      is not None else None
            c3_ctx  = context_assignment.get((c2_base, w3)) if c2_base is not None else None
            if c1 is None or c2_ctx is None or c3_ctx is None:
                n_skip += 1
                continue
            s1, s2, s3 = str(c1), str(c2_ctx), str(c3_ctx)
            ai.teach('next_cat_ctx',       (s1, s2),       (s3,))
            ai.teach('word_given_cat_ctx', (s1, s2, s3),   (w3,))
            n_used += 1

    if verbose:
        nc = len(ai.stores['next_cat_ctx'].examples)       if 'next_cat_ctx'       in ai.stores else 0
        nw = len(ai.stores['word_given_cat_ctx'].examples) if 'word_given_cat_ctx' in ai.stores else 0
        print(f'  Trigrams used: {n_used:,}  skipped (OOV/OOC): {n_skip:,}')
        print(f'  next_cat_ctx:       {nc:,} examples')
        print(f'  word_given_cat_ctx: {nw:,} examples')

    return n_used, n_skip


def predict_chain_ctx(
    ai:                 'SymbolicAI',
    w1:                 str,
    w2:                 str,
    base_assignment:    dict[str, int],
    context_assignment: dict,
) -> str | None:
    """Phase E2: Predict w3 via context-sensitive chain, falling back to E1.

    If (c1, w2) is in context_assignment, uses E2 (next_cat_ctx + word_given_cat_ctx).
    Otherwise falls back to E1 (next_cat + word_given_cat) to preserve coverage.
    """
    c1     = base_assignment.get(w1)
    c2_ctx = context_assignment.get((c1, w2)) if c1 is not None else None

    if c2_ctx is not None:
        # E2 path: context-sensitive clusters available
        c3_tup = ai.ask('next_cat_ctx', (str(c1), str(c2_ctx)))
        if c3_tup is not None:
            c3 = c3_tup[0] if isinstance(c3_tup, tuple) else str(c3_tup)
            w3_tup = ai.ask('word_given_cat_ctx', (str(c1), str(c2_ctx), c3))
            if w3_tup is not None:
                return w3_tup[0] if isinstance(w3_tup, tuple) else str(w3_tup)

    # E1 fallback: use base cluster of w2
    return predict_chain(ai, w1, w2, base_assignment)


def logprob_chain_ctx(
    ai:                 'SymbolicAI',
    w1:                 str,
    w2:                 str,
    w3:                 str,
    base_assignment:    dict[str, int],
    context_assignment: dict,
) -> float | None:
    """Log₂ probability of w3 given (w1, w2) via E2 chain with E1 fallback."""
    c1      = base_assignment.get(w1)
    c2_base = base_assignment.get(w2)
    c2_ctx  = context_assignment.get((c1, w2))      if c1      is not None else None
    c3_ctx  = context_assignment.get((c2_base, w3)) if c2_base is not None else None

    if c2_ctx is not None and c3_ctx is not None:
        cat_dist = ai.ask_dist('next_cat_ctx', (str(c1), str(c2_ctx)))
        if cat_dist is not None:
            p_c3 = cat_dist.get((str(c3_ctx),), 0.0)
            if p_c3 >= 1e-12:
                word_dist = ai.ask_dist('word_given_cat_ctx',
                                        (str(c1), str(c2_ctx), str(c3_ctx)))
                if word_dist is not None:
                    p_w3 = word_dist.get((w3,), 0.0)
                    if p_w3 >= 1e-12:
                        return math.log2(p_c3 * p_w3)

    # E1 fallback
    return logprob_chain(ai, w1, w2, w3, base_assignment)


def evaluate_all(
    ai:                 'SymbolicAI',
    test_texts:         list[str],
    base_assignment:    dict[str, int],
    context_assignment: dict,
    train_pairs:        set[tuple[str, str]],
    verbose:            bool = True,
) -> dict:
    """Evaluate flat bigram, E1 chain, and E2 context-sensitive chain together.

    Extends evaluate() to include E2 metrics alongside E1 and flat bigram.
    """
    _banner('Evaluation: flat bigram vs E1 chain vs E2 context-sensitive chain')

    r = dict(
        total=0,
        # Flat bigram
        flat_correct=0, flat_answered=0,
        flat_unseen_correct=0, flat_unseen_answered=0,
        flat_logloss=0.0, flat_logloss_n=0,
        # E1 category chain
        chain_correct=0, chain_answered=0,
        chain_unseen_correct=0, chain_unseen_answered=0,
        chain_logloss=0.0, chain_logloss_n=0,
        # E2 context-sensitive chain
        ctx_correct=0, ctx_answered=0,
        ctx_unseen_correct=0, ctx_unseen_answered=0,
        ctx_logloss=0.0, ctx_logloss_n=0,
        unseen_total=0,
    )

    for text in test_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            is_unseen = (w2, w3) not in train_pairs

            r['total'] += 1
            if is_unseen:
                r['unseen_total'] += 1

            # --- Flat bigram ---
            flat_pred = predict_flat(ai, w2)
            if flat_pred is not None:
                r['flat_answered'] += 1
                if flat_pred == w3:
                    r['flat_correct'] += 1
                    if is_unseen:
                        r['flat_unseen_correct'] += 1
                if is_unseen:
                    r['flat_unseen_answered'] += 1
            lp = logprob_flat(ai, w2, w3)
            if lp is not None:
                r['flat_logloss'] -= lp; r['flat_logloss_n'] += 1

            # --- E1 category chain ---
            chain_pred = predict_chain(ai, w1, w2, base_assignment)
            if chain_pred is not None:
                r['chain_answered'] += 1
                if chain_pred == w3:
                    r['chain_correct'] += 1
                    if is_unseen:
                        r['chain_unseen_correct'] += 1
                if is_unseen:
                    r['chain_unseen_answered'] += 1
            lp = logprob_chain(ai, w1, w2, w3, base_assignment)
            if lp is not None:
                r['chain_logloss'] -= lp; r['chain_logloss_n'] += 1

            # --- E2 context-sensitive chain ---
            ctx_pred = predict_chain_ctx(ai, w1, w2, base_assignment, context_assignment)
            if ctx_pred is not None:
                r['ctx_answered'] += 1
                if ctx_pred == w3:
                    r['ctx_correct'] += 1
                    if is_unseen:
                        r['ctx_unseen_correct'] += 1
                if is_unseen:
                    r['ctx_unseen_answered'] += 1
            lp = logprob_chain_ctx(ai, w1, w2, w3, base_assignment, context_assignment)
            if lp is not None:
                r['ctx_logloss'] -= lp; r['ctx_logloss_n'] += 1

    if verbose:
        T  = r['total']
        UT = r['unseen_total']

        def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'
        def ppl(loss, n): return f'{2 ** (loss / n):.1f}' if n else 'N/A'

        print(f'\n  Test trigrams: {T:,}  (unseen pairs: {UT:,} = {pct(UT, T)})')
        print()
        print(f'  {"Metric":<38} {"Flat":>8} {"E1 chain":>10} {"E2 ctx":>10}')
        print(f'  {"-"*66}')
        print(f'  {"Coverage (answered/total)":<38} '
              f'{pct(r["flat_answered"],T):>8} '
              f'{pct(r["chain_answered"],T):>10} '
              f'{pct(r["ctx_answered"],T):>10}')
        print(f'  {"Top-1 accuracy (of answered)":<38} '
              f'{pct(r["flat_correct"],r["flat_answered"]):>8} '
              f'{pct(r["chain_correct"],r["chain_answered"]):>10} '
              f'{pct(r["ctx_correct"],r["ctx_answered"]):>10}')
        print(f'  {"Top-1 accuracy (of total)":<38} '
              f'{pct(r["flat_correct"],T):>8} '
              f'{pct(r["chain_correct"],T):>10} '
              f'{pct(r["ctx_correct"],T):>10}')
        print(f'  {"Perplexity (lower=better)":<38} '
              f'{ppl(r["flat_logloss"],r["flat_logloss_n"]):>8} '
              f'{ppl(r["chain_logloss"],r["chain_logloss_n"]):>10} '
              f'{ppl(r["ctx_logloss"],r["ctx_logloss_n"]):>10}')
        print()
        print(f'  === UNSEEN WORD PAIRS ===')
        print(f'  {"Coverage (answered/unseen)":<38} '
              f'{pct(r["flat_unseen_answered"],UT):>8} '
              f'{pct(r["chain_unseen_answered"],UT):>10} '
              f'{pct(r["ctx_unseen_answered"],UT):>10}')
        print(f'  {"Acc. of answered (unseen)":<38} '
              f'{pct(r["flat_unseen_correct"],r["flat_unseen_answered"]):>8} '
              f'{pct(r["chain_unseen_correct"],r["chain_unseen_answered"]):>10} '
              f'{pct(r["ctx_unseen_correct"],r["ctx_unseen_answered"]):>10}')
        print(f'  {"Acc. of total (unseen)":<38} '
              f'{pct(r["flat_unseen_correct"],UT):>8} '
              f'{pct(r["chain_unseen_correct"],UT):>10} '
              f'{pct(r["ctx_unseen_correct"],UT):>10}')
        print()

        # Verdict
        e1_unseen = r['chain_unseen_correct'] / max(r['chain_unseen_answered'], 1)
        e2_unseen = r['ctx_unseen_correct']   / max(r['ctx_unseen_answered'],   1)
        flat_unseen = r['flat_unseen_correct'] / max(r['flat_unseen_answered'],  1)
        best = max(e1_unseen, e2_unseen, flat_unseen)
        if best == e2_unseen and r['ctx_unseen_answered'] > 0 and e2_unseen > flat_unseen:
            print('  RESULT: E2 context-sensitive chain is BEST on unseen pairs.')
        elif best == e1_unseen and r['chain_unseen_answered'] > 0 and e1_unseen > flat_unseen:
            print('  RESULT: E1 category chain is BEST on unseen pairs.')
        else:
            print('  RESULT: Flat bigram matches or exceeds chains on unseen pairs.')

    return r


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Category-chain word prediction (language.ctkg E1+E2).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--corpus',       default='EarlyModernLatin',
                   help='Corpus name or path')
    p.add_argument('--n_train',      type=int, default=2000,
                   help='Max training lines (default: 2000)')
    p.add_argument('--n_test',       type=int, default=None,
                   help='Max test lines (default: 20%% of n_train)')
    p.add_argument('--split',        type=float, default=0.8,
                   help='Train fraction (default: 0.8)')
    p.add_argument('--n_clusters',   type=int, default=12,
                   help='Word cluster count for E1 category chain (default: 12)')
    p.add_argument('--n_ctx_clusters', type=int, default=None,
                   help='Context-sensitive cluster count for E2 (default: n_clusters)')
    p.add_argument('--ctx_min_count', type=int, default=3,
                   help='Min occurrences for a (c_prev,word) pair to get a context cluster (default: 3)')
    p.add_argument('--phase',        choices=['e1', 'e2', 'both'], default='both',
                   help='Which phase to run: e1, e2, or both (default: both)')
    p.add_argument('--save',         metavar='PATH',
                   help='Save AI checkpoint after training')
    p.add_argument('--load',         metavar='PATH',
                   help='Load existing checkpoint')
    p.add_argument('--seed',         type=int, default=42)
    args = p.parse_args()

    n_ctx = args.n_ctx_clusters or args.n_clusters

    _banner('Language Pipeline: Category-Chain Word Prediction (E1 + E2)')
    print(f'  Corpus:           {args.corpus}')
    print(f'  Train:            {args.n_train} lines  ({args.split*100:.0f}%)')
    print(f'  E1 clusters (K):  {args.n_clusters}')
    print(f'  E2 ctx clusters:  {n_ctx}')
    print(f'  Phase:            {args.phase}')

    # Locate corpus.
    d = (args.corpus if os.path.isdir(args.corpus)
         else os.path.join(_DATA_DIR, args.corpus))
    if not os.path.isdir(d):
        print(f'ERROR: corpus not found: {args.corpus!r}')
        sys.exit(1)

    # Load ALL pairs shuffled, then split.
    total_n = args.n_train + (args.n_test or max(args.n_train // 4, 100))
    pairs = find_pairs(d, max_n=total_n, shuffle=True, seed=args.seed)
    texts = _stream_texts(pairs)

    n_train = min(args.n_train, int(len(texts) * args.split))
    train_texts = texts[:n_train]
    test_texts  = texts[n_train:n_train + (args.n_test or max(n_train // 4, 50))]
    print(f'  Lines:            {len(train_texts)} train  {len(test_texts)} test')

    # ---- Step 1: Run multi-scale discovery on training corpus ----
    print('\n  Running multi-scale discovery on training corpus...')
    ai, chunk_maps = run_pipeline(
        texts=train_texts,
        n_levels=3,
        max_merges=500,
        save_path=None,
        load_path=args.load,
        verbose=False,
    )

    # ---- Step 2: Build E1 cluster assignment ----
    _banner('Building E1 word cluster assignment')
    assignment, clusters = build_assignment(ai, n_clusters=args.n_clusters)
    print(f'  Vocab size in assignment: {len(assignment):,}')
    print(f'  Clusters formed: {len(clusters)}')
    for cid in sorted(clusters.keys())[:6]:
        sample = clusters[cid][:8]
        print(f'    C{cid:02d} ({len(clusters[cid])} members): '
              f'{" ".join(repr(w) for w in sample)}')
    if len(clusters) > 6:
        print(f'    ... {len(clusters) - 6} more clusters')

    # ---- Step 3: Train E1 chain concepts ----
    train_chain(ai, train_texts, assignment, verbose=True)

    # ---- Step 4: Collect training pairs for seen/unseen split ----
    train_pairs: set[tuple[str, str]] = set()
    for text in train_texts:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            train_pairs.add((tokens[i], tokens[i + 1]))
    print(f'\n  Training word bigrams (for seen/unseen split): {len(train_pairs):,}')

    # ---- Phase selection ----
    context_assignment: dict = {}
    ctx_clusters: dict = {}

    if args.phase in ('e2', 'both'):
        # ---- Step 5: Build E2 context-sensitive cluster assignment ----
        context_assignment, ctx_clusters = build_trigram_assignment(
            train_texts, assignment, n_clusters=n_ctx,
            min_examples=args.ctx_min_count, verbose=True,
        )

        if context_assignment:
            # ---- Step 6: Train E2 context-sensitive chain ----
            train_chain_ctx(ai, train_texts, assignment, context_assignment, verbose=True)

    # ---- Evaluation ----
    if args.phase == 'e1':
        evaluate(ai, test_texts, assignment, train_pairs, verbose=True)
    else:
        # Full three-way comparison: flat vs E1 vs E2
        if context_assignment:
            evaluate_all(ai, test_texts, assignment, context_assignment,
                         train_pairs, verbose=True)
        else:
            print('  WARNING: E2 clustering failed; falling back to E1-only evaluation.')
            evaluate(ai, test_texts, assignment, train_pairs, verbose=True)

    # ---- Save ----
    if args.save:
        ai.save_checkpoint(args.save)
        print(f'  Checkpoint saved: {args.save}')

    # ---- Summary ----
    _banner('Summary')
    K  = len(clusters)
    KC = len(ctx_clusters)
    V  = len(assignment)
    print(f'  Word vocabulary:          {V:,}')
    print(f'  E1 base clusters (K):     {K}')
    print(f'  E2 context clusters (KC): {KC}')
    print(f'  E1 next_cat:              K^2 ~ {K**2:,}  (vs V^2 = {V**2:,})')
    print(f'  E2 next_cat_ctx:          K × KC ~ {K * KC:,}  entries')
    print(f'  E1 word_given_cat:        K^3 ~ {K**3:,}')
    print(f'  E2 word_given_cat_ctx:    K × KC^2 ~ {K * KC * KC:,}')
    ratio_e1 = V**2 / max(K**2, 1)
    ratio_e2 = V**2 / max(K * KC, 1)
    print(f'  E1 compression vs flat:   {ratio_e1:.0f}x')
    print(f'  E2 compression vs flat:   {ratio_e2:.0f}x')
    print()
    print('  Next step (Phase E3): soft retrieval for contextual representations.')


if __name__ == '__main__':
    main()
