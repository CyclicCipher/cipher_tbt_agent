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
    test_texts:         list,
    base_assignment:    dict,
    context_assignment: dict,
    train_pairs:        set,
    nc_soft:            dict | None = None,
    wgc_soft:           dict | None = None,
    verbose:            bool = True,
) -> dict:
    """Evaluate flat bigram, E1 chain, E2 context chain, and (optional) E3 soft chain.

    E3 is enabled when nc_soft and wgc_soft are provided (precomputed soft dists).
    """
    has_e3 = nc_soft is not None and wgc_soft is not None
    banner_suffix = 'E3 soft' if has_e3 else 'E2 ctx'
    _banner(f'Evaluation: flat bigram vs E1 chain vs E2 ctx vs {banner_suffix}')

    r = dict(
        total=0,
        flat_correct=0, flat_answered=0,
        flat_unseen_correct=0, flat_unseen_answered=0,
        flat_logloss=0.0, flat_logloss_n=0,
        chain_correct=0, chain_answered=0,
        chain_unseen_correct=0, chain_unseen_answered=0,
        chain_logloss=0.0, chain_logloss_n=0,
        ctx_correct=0, ctx_answered=0,
        ctx_unseen_correct=0, ctx_unseen_answered=0,
        ctx_logloss=0.0, ctx_logloss_n=0,
        e3_correct=0, e3_answered=0,
        e3_unseen_correct=0, e3_unseen_answered=0,
        e3_logloss=0.0, e3_logloss_n=0,
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

            # --- E3 soft retrieval (optional) ---
            if has_e3:
                e3_pred = predict_chain_e3(
                    ai, w1, w2, base_assignment, nc_soft, wgc_soft)
                if e3_pred is not None:
                    r['e3_answered'] += 1
                    if e3_pred == w3:
                        r['e3_correct'] += 1
                        if is_unseen:
                            r['e3_unseen_correct'] += 1
                    if is_unseen:
                        r['e3_unseen_answered'] += 1
                lp = logprob_chain_e3(
                    ai, w1, w2, w3, base_assignment, nc_soft, wgc_soft)
                if lp is not None:
                    r['e3_logloss'] -= lp; r['e3_logloss_n'] += 1

    if verbose:
        T  = r['total']
        UT = r['unseen_total']

        def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'
        def ppl(loss, n): return f'{2**(loss/n):.1f}' if n else 'N/A'

        if has_e3:
            hdr = f'  {"Metric":<36} {"Flat":>8} {"E1 chain":>9} {"E2 ctx":>8} {"E3 soft":>9}'
            sep = f'  {"-"*70}'
            def row(label, fv, cv, xv, ev):
                return f'  {label:<36} {fv:>8} {cv:>9} {xv:>8} {ev:>9}'
        else:
            hdr = f'  {"Metric":<38} {"Flat":>8} {"E1 chain":>10} {"E2 ctx":>10}'
            sep = f'  {"-"*66}'
            def row(label, fv, cv, xv, ev=None):
                return f'  {label:<38} {fv:>8} {cv:>10} {xv:>10}'

        print(f'\n  Test trigrams: {T:,}  (unseen pairs: {UT:,} = {pct(UT, T)})')
        print()
        print(hdr); print(sep)

        fa = r['flat_answered']; fc = r['flat_correct']
        ca = r['chain_answered']; cc = r['chain_correct']
        xa = r['ctx_answered'];   xc = r['ctx_correct']
        ea = r['e3_answered'];    ec = r['e3_correct']

        fua = r['flat_unseen_answered'];  fuc = r['flat_unseen_correct']
        cua = r['chain_unseen_answered']; cuc = r['chain_unseen_correct']
        xua = r['ctx_unseen_answered'];   xuc = r['ctx_unseen_correct']
        eua = r['e3_unseen_answered'];    euc = r['e3_unseen_correct']

        print(row('Coverage (answered/total)',
                  pct(fa,T), pct(ca,T), pct(xa,T), pct(ea,T)))
        print(row('Top-1 accuracy (of answered)',
                  pct(fc,fa), pct(cc,ca), pct(xc,xa), pct(ec,ea)))
        print(row('Top-1 accuracy (of total)',
                  pct(fc,T), pct(cc,T), pct(xc,T), pct(ec,T)))
        print(row('Perplexity (lower=better)',
                  ppl(r['flat_logloss'],r['flat_logloss_n']),
                  ppl(r['chain_logloss'],r['chain_logloss_n']),
                  ppl(r['ctx_logloss'],r['ctx_logloss_n']),
                  ppl(r['e3_logloss'],r['e3_logloss_n'])))
        print()
        print('  === UNSEEN WORD PAIRS ===')
        print(row('Coverage (answered/unseen)',
                  pct(fua,UT), pct(cua,UT), pct(xua,UT), pct(eua,UT)))
        print(row('Acc. of answered (unseen)',
                  pct(fuc,fua), pct(cuc,cua), pct(xuc,xua), pct(euc,eua)))
        print(row('Acc. of total (unseen)',
                  pct(fuc,UT), pct(cuc,UT), pct(xuc,UT), pct(euc,UT)))
        print()

        # Verdict: compare all enabled phases on unseen-pair accuracy.
        scores = {
            'Flat': fuc / max(fua, 1),
            'E1':   cuc / max(cua, 1),
            'E2':   xuc / max(xua, 1),
        }
        if has_e3:
            scores['E3'] = euc / max(eua, 1)
        best_name = max(scores, key=scores.get)
        if best_name != 'Flat' and scores[best_name] > scores['Flat']:
            print(f'  RESULT: {best_name} is BEST on unseen pairs '
                  f'({100*scores[best_name]:.1f}% vs Flat {100*scores["Flat"]:.1f}%).')
        else:
            print('  RESULT: Flat bigram matches or exceeds chains on unseen pairs.')

    return r


# ---------------------------------------------------------------------------
# Phase E3: Soft Retrieval (Attention-Equivalent)
# ---------------------------------------------------------------------------

def build_cluster_word_dists(
    assignment:  dict,
    clusters:    dict,
    train_texts: list,
    verbose:     bool = True,
) -> dict:
    """Frequency-weighted word distribution for each cluster (member-word similarity).

    Weights cluster members by corpus frequency.  This measures lexical overlap
    between clusters — which words ARE in each cluster.  Use
    build_cluster_succ_dists() for the task-relevant similarity instead.
    """
    import collections
    word_freq: collections.Counter = collections.Counter()
    for text in train_texts:
        word_freq.update(text.split())

    cluster_dists: dict = {}
    for cid, members in clusters.items():
        freq_total = sum(word_freq.get(w, 1) for w in members)
        cluster_dists[cid] = {w: word_freq.get(w, 1) / freq_total for w in members}

    if verbose:
        for cid in sorted(cluster_dists.keys())[:3]:
            top = sorted(cluster_dists[cid].items(), key=lambda kv: -kv[1])[:4]
            print(f'    C{cid:02d} members: {" ".join(f"{w}({p:.3f})" for w, p in top)}')

    return cluster_dists


def build_cluster_succ_dists(
    ai:       'SymbolicAI',
    clusters: dict,
    verbose:  bool = True,
) -> dict:
    """Successor-distribution for each cluster (task-relevant similarity).

    For next-word prediction, two clusters are similar if words in them tend to
    be followed by the same words — not if they contain the same words.

    succ_dist[c] = frequency-weighted average of ask_dist('next_word_hier', (w,))
                   over all words w in cluster c.

    This captures the SYNTACTIC ROLE of each cluster: preposition-clusters
    and verb-clusters have very different successor distributions even if they
    share no member words.  It is the correct similarity for next-word prediction
    because it measures interchangeability of contexts, not lexical overlap.
    """
    import collections
    _banner('Phase E3: Building cluster successor distributions')

    succ_dists: dict = {}
    n_words_used = 0

    for cid, members in clusters.items():
        merged: dict = collections.defaultdict(float)
        total_weight = 0.0

        for word in members:
            dist = ai.ask_dist('next_word_hier', (word,))
            if dist is None:
                continue
            # The dist keys are tuples like ('est',); sum probabilities per word.
            for out_tup, prob in dist.items():
                nw = out_tup[0] if isinstance(out_tup, tuple) else str(out_tup)
                merged[nw] += prob
            total_weight += 1.0
            n_words_used += 1

        if total_weight > 0:
            succ_dists[cid] = {w: p / total_weight for w, p in merged.items()}
        else:
            succ_dists[cid] = {}

    if verbose:
        print(f'  Words with successor distributions: {n_words_used}')
        for cid in sorted(succ_dists.keys())[:3]:
            top = sorted(succ_dists[cid].items(), key=lambda kv: -kv[1])[:4]
            print(f'    C{cid:02d} successors: {" ".join(f"{w}({p:.3f})" for w, p in top)}')

    return succ_dists


def _jsd(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence (base-2) between distributions p and q.
    Symmetric and bounded: 0.0 = identical, 1.0 = fully disjoint vocabularies.
    """
    result = 0.0
    for k in set(p) | set(q):
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        if mk < 1e-12:
            continue
        if pk > 1e-12:
            result += 0.5 * pk * math.log2(pk / mk)
        if qk > 1e-12:
            result += 0.5 * qk * math.log2(qk / mk)
    return result


def build_cluster_similarity_matrix(
    cluster_dists: dict,
    K:             int,
    temperature:   float = 2.0,
    verbose:       bool = True,
) -> list:
    """K×K matrix of cluster similarities: sim(c, c') = exp(-T × JSD(P_c, P_c')).

    Temperature controls selectivity:
        T high → only very similar clusters contribute (approaches E1 exact-match).
        T low  → even dissimilar clusters contribute (more diffuse smoothing).
    """
    _banner('Phase E3: Building cluster similarity matrix')

    matrix = [[0.0] * K for _ in range(K)]
    for i in range(K):
        di = cluster_dists.get(i, {})
        for j in range(K):
            dj = cluster_dists.get(j, {})
            matrix[i][j] = 1.0 if i == j else math.exp(-temperature * _jsd(di, dj))

    if verbose:
        pairs = [(matrix[i][j], i, j)
                 for i in range(K) for j in range(i + 1, K)]
        pairs.sort(reverse=True)
        print(f'  Similarity matrix ({K}×{K}), temperature={temperature}:')
        print(f'  Most similar off-diagonal cluster pairs:')
        for sim, i, j in pairs[:5]:
            print(f'    C{i:02d} ↔ C{j:02d}: {sim:.3f}')
        avg_off = sum(s for s, _, _ in pairs) / max(len(pairs), 1)
        print(f'  Average off-diagonal similarity: {avg_off:.3f}')

    return matrix


def precompute_dist_cache(
    ai:           'SymbolicAI',
    concept_name: str,
) -> dict:
    """Precompute output distribution for each unique stored input key.

    Returns {input_key: {output_key: probability}}.
    Avoids repeated iteration over store.examples in hot loops.
    """
    import collections
    store = ai.stores.get(concept_name)
    if store is None:
        return {}

    key_counts: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    key_totals: dict = collections.defaultdict(float)

    for stored_inputs, stored_outputs in store.examples:
        key_counts[stored_inputs][stored_outputs] += 1
        key_totals[stored_inputs] += 1

    return {
        key: {out: cnt / key_totals[key] for out, cnt in counts.items()}
        for key, counts in key_counts.items()
    }


def ask_weighted_soft(
    query_key:  tuple,
    dist_cache: dict,
    sim_matrix: list,
    K:          int,
) -> dict | None:
    """Soft-weighted retrieval — the distributional analogue of self-attention.

    Computes:
        P_soft(output | query) ∝ Σ_key w(query, key) × P(output | key)
    where:
        w(query, key) = Π_i sim_matrix[ int(query[i]) ][ int(key[i]) ]

    Replaces exact-match lookup with similarity-weighted aggregation over ALL
    stored keys. Keys with zero weight are skipped for efficiency.
    """
    import collections
    weighted_output: dict = collections.defaultdict(float)
    total_weight = 0.0

    for stored_key, out_dist in dist_cache.items():
        if len(stored_key) != len(query_key):
            continue

        weight = 1.0
        for qi_s, ki_s in zip(query_key, stored_key):
            try:
                qi, ki = int(qi_s), int(ki_s)
                w = (sim_matrix[qi][ki]
                     if 0 <= qi < K and 0 <= ki < K
                     else (1.0 if qi == ki else 0.0))
            except (ValueError, TypeError):
                w = 1.0 if qi_s == ki_s else 0.0
            weight *= w
            if weight < 1e-12:
                break

        if weight < 1e-12:
            continue

        for out, prob in out_dist.items():
            weighted_output[out] += weight * prob
        total_weight += weight

    if total_weight < 1e-12:
        return None

    return {k: v / total_weight for k, v in weighted_output.items()}


def precompute_all_soft_dists(
    dist_cache: dict,
    sim_matrix: list,
    K:          int,
    arity:      int,
    verbose:    bool = True,
) -> dict:
    """Precompute soft distributions for all K^arity possible query keys.

    Runs once before evaluation; then predict/logprob are O(1) lookups.
    For K=12, arity=2: 144 queries; arity=3: 1728 queries.
    """
    import itertools

    all_queries = list(itertools.product(range(K), repeat=arity))
    n = len(all_queries)
    if verbose:
        print(f'  Precomputing soft dists: {n} queries (K={K}, arity={arity})')

    soft_dists: dict = {}
    for query_ids in all_queries:
        qkey = tuple(str(q) for q in query_ids)
        soft_dists[qkey] = ask_weighted_soft(qkey, dist_cache, sim_matrix, K)

    n_pop = sum(1 for d in soft_dists.values() if d is not None)
    if verbose:
        print(f'  Populated: {n_pop}/{n}')

    return soft_dists


def predict_chain_e3(
    ai:         'SymbolicAI',
    w1:         str,
    w2:         str,
    assignment: dict,
    nc_soft:    dict,
    wgc_soft:   dict,
) -> str | None:
    """Phase E3: Predict w3 via similarity-weighted chain (with E1 fallback).

    Uses precomputed soft distributions for O(1) inference. Falls back to
    E1 exact-match if the word is OOV or soft lookup returns nothing.
    """
    c1 = assignment.get(w1)
    c2 = assignment.get(w2)
    if c1 is None or c2 is None:
        return None

    c3_dist = nc_soft.get((str(c1), str(c2)))
    if c3_dist is None:
        return predict_chain(ai, w1, w2, assignment)

    c3_tup = max(c3_dist, key=c3_dist.get)
    c3_str = c3_tup[0] if isinstance(c3_tup, tuple) else str(c3_tup)

    w3_dist = wgc_soft.get((str(c1), str(c2), c3_str))
    if w3_dist is None:
        return predict_chain(ai, w1, w2, assignment)

    best = max(w3_dist, key=w3_dist.get)
    return best[0] if isinstance(best, tuple) else str(best)


def logprob_chain_e3(
    ai:         'SymbolicAI',
    w1:         str,
    w2:         str,
    w3:         str,
    assignment: dict,
    nc_soft:    dict,
    wgc_soft:   dict,
) -> float | None:
    """Log₂ probability via E3 soft-retrieval chain (with E1 fallback)."""
    c1 = assignment.get(w1)
    c2 = assignment.get(w2)
    c3 = assignment.get(w3)
    if c1 is None or c2 is None or c3 is None:
        return None

    c3_dist = nc_soft.get((str(c1), str(c2)))
    if c3_dist is None:
        return logprob_chain(ai, w1, w2, w3, assignment)

    p_c3 = c3_dist.get((str(c3),), 0.0)
    if p_c3 < 1e-12:
        return logprob_chain(ai, w1, w2, w3, assignment)

    w3_dist = wgc_soft.get((str(c1), str(c2), str(c3)))
    if w3_dist is None:
        return None

    p_w3 = w3_dist.get((w3,), 0.0)
    if p_w3 < 1e-12:
        return None

    return math.log2(p_c3 * p_w3)


# ---------------------------------------------------------------------------
# Phase E3b: Mixture calibration (E1 + E3 interpolation)
# ---------------------------------------------------------------------------

def tune_mixture_alpha(
    ai:         'SymbolicAI',
    dev_texts:  list,
    assignment: dict,
    nc_soft:    dict,
    wgc_soft:   dict,
    n_grid:     int = 20,
    verbose:    bool = True,
) -> float:
    """Learn α that maximises log-likelihood of P_mix = (1-α)×P_E1 + α×P_E3.

    E1 assigns probability 1 to its prediction when it has an exact match and
    0 elsewhere.  E3 assigns a soft distribution.  The mixture gives:

        P_mix(w3) = (1-α) × [1 if w3 == E1_pred else 0]
                  + α     × P_E3(w3)

    When E1 has no match: P_mix(w3) = α × P_E3(w3), so log-prob = log(α) + log(P_E3).
    When E1 has a match: P_mix(correct) = (1-α) + α×P_E3(correct) ≥ (1-α).

    α is found by 1D grid search over [0.01, 0.99] maximising total log-likelihood
    on dev_texts.  No gradient descent required.
    """
    _banner('Phase E3b: Tuning mixture coefficient α')

    # Pre-collect (E1_prob, E3_prob) pairs for all dev trigrams.
    # E1_prob = P_E1(w3 | w1, w2): 1.0 if chain predicts correctly, else 0.0
    # E3_prob = P_E3(w3 | w1, w2): soft probability from wgc_soft
    pairs = []
    for text in dev_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            c1 = assignment.get(w1)
            c2 = assignment.get(w2)
            c3 = assignment.get(w3)
            if c1 is None or c2 is None or c3 is None:
                continue

            # E1 probability for w3.
            e1_pred = predict_chain(ai, w1, w2, assignment)
            p_e1 = 1.0 if e1_pred == w3 else 0.0

            # E3 probability for w3.
            c3_dist = nc_soft.get((str(c1), str(c2)))
            if c3_dist is None:
                continue
            p_c3 = c3_dist.get((str(c3),), 0.0)
            if p_c3 < 1e-12:
                continue
            w3_dist = wgc_soft.get((str(c1), str(c2), str(c3)))
            if w3_dist is None:
                continue
            p_e3 = p_c3 * w3_dist.get((w3,), 0.0)

            if p_e1 > 0 or p_e3 > 1e-12:
                pairs.append((p_e1, p_e3))

    if not pairs:
        if verbose:
            print('  No dev trigrams usable — α = 0.5 (default)')
        return 0.5

    # Grid search: α ∈ [0.01, 0.99]
    best_alpha = 0.5
    best_ll    = float('-inf')
    alphas = [i / n_grid for i in range(1, n_grid)]  # 0.05, 0.10, ..., 0.95

    for alpha in alphas:
        ll = 0.0
        n  = 0
        for p_e1, p_e3 in pairs:
            p_mix = (1 - alpha) * p_e1 + alpha * p_e3
            if p_mix > 1e-12:
                ll += math.log2(p_mix)
                n  += 1
        if n > 0 and ll > best_ll:
            best_ll    = ll
            best_alpha = alpha

    if verbose:
        print(f'  Dev trigrams with signal: {len(pairs):,}')
        print(f'  Best α = {best_alpha:.2f}  '
              f'(log-likelihood = {best_ll:.1f} over {len(pairs)} trigrams)')
        # Show how LL varies across α for context.
        sample_alphas = [0.05, 0.1, 0.2, 0.5, best_alpha]
        for a in sorted(set(sample_alphas)):
            ll = sum(math.log2(max((1-a)*p1 + a*p3, 1e-12)) for p1, p3 in pairs)
            print(f'    α={a:.2f}: LL={ll:.1f}')

    return best_alpha


def logprob_chain_mix(
    ai:         'SymbolicAI',
    w1:         str,
    w2:         str,
    w3:         str,
    assignment: dict,
    nc_soft:    dict,
    wgc_soft:   dict,
    alpha:      float,
) -> float | None:
    """Log₂ probability under E1+E3 mixture: P_mix = (1-α)P_E1 + αP_E3."""
    c1 = assignment.get(w1)
    c2 = assignment.get(w2)
    c3 = assignment.get(w3)
    if c1 is None or c2 is None or c3 is None:
        return None

    # E1 component.
    e1_pred = predict_chain(ai, w1, w2, assignment)
    p_e1 = 1.0 if e1_pred == w3 else 0.0

    # E3 component.
    c3_dist = nc_soft.get((str(c1), str(c2)))
    p_e3 = 0.0
    if c3_dist is not None:
        p_c3 = c3_dist.get((str(c3),), 0.0)
        if p_c3 >= 1e-12:
            w3_dist = wgc_soft.get((str(c1), str(c2), str(c3)))
            if w3_dist is not None:
                p_e3 = p_c3 * w3_dist.get((w3,), 0.0)

    p_mix = (1 - alpha) * p_e1 + alpha * p_e3
    if p_mix < 1e-12:
        return None
    return math.log2(p_mix)


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
    p.add_argument('--phase',        choices=['e1', 'e2', 'e3', 'both', 'all'],
                   default='all',
                   help='Phases to run: e1, e2, e3, both (E1+E2), all (E1+E2+E3) (default: all)')
    p.add_argument('--e3_temperature', type=float, default=2.0,
                   help='E3 similarity temperature: higher=more selective (default: 2.0)')
    p.add_argument('--save',         metavar='PATH',
                   help='Save AI checkpoint after training')
    p.add_argument('--load',         metavar='PATH',
                   help='Load existing checkpoint')
    p.add_argument('--seed',         type=int, default=42)
    args = p.parse_args()

    n_ctx = args.n_ctx_clusters or args.n_clusters
    run_e2 = args.phase in ('e2', 'both', 'e3', 'all')
    run_e3 = args.phase in ('e3', 'all')

    _banner('Language Pipeline: Category-Chain Word Prediction (E1+E2+E3)')
    print(f'  Corpus:           {args.corpus}')
    print(f'  Train:            {args.n_train} lines  ({args.split*100:.0f}%)')
    print(f'  E1 clusters (K):  {args.n_clusters}')
    print(f'  E2 ctx clusters:  {n_ctx}')
    print(f'  Phase:            {args.phase}')
    if run_e3:
        print(f'  E3 temperature:   {args.e3_temperature}')

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
    nc_soft: dict | None = None
    wgc_soft: dict | None = None

    if run_e2:
        # ---- Step 5: Build E2 context-sensitive cluster assignment ----
        context_assignment, ctx_clusters = build_trigram_assignment(
            train_texts, assignment, n_clusters=n_ctx,
            min_examples=args.ctx_min_count, verbose=True,
        )

        if context_assignment:
            # ---- Step 6: Train E2 context-sensitive chain ----
            train_chain_ctx(ai, train_texts, assignment, context_assignment, verbose=True)

    if run_e3:
        # ---- Step 7: Build E3 soft retrieval components ----
        # Use successor distributions (task-relevant similarity):
        # sim(c, c') = exp(-T × JSD(succ_dist[c], succ_dist[c']))
        # where succ_dist[c] = average next-word dist of words in cluster c.
        # This measures "do words in c and c' predict the same continuations?"
        # — directly relevant to next-word prediction (not just lexical overlap).
        _banner('Phase E3: Soft Retrieval (successor-distribution similarity)')
        K = len(clusters)
        succ_dists = build_cluster_succ_dists(ai, clusters, verbose=True)
        sim_matrix = build_cluster_similarity_matrix(
            succ_dists, K=K, temperature=args.e3_temperature, verbose=True)

        nc_cache  = precompute_dist_cache(ai, 'next_cat')
        wgc_cache = precompute_dist_cache(ai, 'word_given_cat')
        print(f'\n  Precomputed dist caches: next_cat={len(nc_cache)} keys, '
              f'word_given_cat={len(wgc_cache)} keys')

        nc_soft  = precompute_all_soft_dists(nc_cache,  sim_matrix, K, arity=2, verbose=True)
        wgc_soft = precompute_all_soft_dists(wgc_cache, sim_matrix, K, arity=3, verbose=True)

        # ---- Step 8: Tune mixture α on a held-out dev slice ----
        # Reserve 10% of test data for α tuning (rest goes to evaluation).
        n_dev = max(len(test_texts) // 10, 20)
        dev_texts  = test_texts[:n_dev]
        eval_texts = test_texts[n_dev:]
        alpha = tune_mixture_alpha(
            ai, dev_texts, assignment, nc_soft, wgc_soft, verbose=True)
        print(f'  Mixture α = {alpha:.2f}  '
              f'(blends E1 exact-match with E3 soft prediction)')
    else:
        eval_texts = test_texts
        alpha = 0.5  # unused

    # ---- Evaluation ----
    if args.phase == 'e1':
        evaluate(ai, eval_texts, assignment, train_pairs, verbose=True)
    elif not context_assignment and not run_e3:
        print('  WARNING: E2 clustering failed; falling back to E1-only evaluation.')
        evaluate(ai, eval_texts, assignment, train_pairs, verbose=True)
    else:
        results = evaluate_all(ai, eval_texts, assignment, context_assignment,
                               train_pairs,
                               nc_soft=nc_soft, wgc_soft=wgc_soft,
                               verbose=True)

        # Extra: report mixture perplexity alongside E3.
        if run_e3:
            _banner('Phase E3b: Mixture calibration perplexity')
            mix_ll = 0.0; mix_n = 0
            for text in eval_texts:
                toks = text.split()
                for i in range(len(toks) - 2):
                    lp = logprob_chain_mix(
                        ai, toks[i], toks[i+1], toks[i+2],
                        assignment, nc_soft, wgc_soft, alpha)
                    if lp is not None:
                        mix_ll -= lp; mix_n += 1
            mix_ppl = 2 ** (mix_ll / mix_n) if mix_n else float('inf')
            e3_n = results['e3_logloss_n']
            e3_ppl = 2 ** (results['e3_logloss'] / e3_n) if e3_n else float('inf')
            print(f'  E3 raw perplexity:      {e3_ppl:.1f}  '
                  f'(N={e3_n})')
            print(f'  Mixture (α={alpha:.2f}) PPL: {mix_ppl:.1f}  '
                  f'(N={mix_n})')
            print(f'  → mixture reduces perplexity by {e3_ppl/mix_ppl:.1f}×'
                  if mix_ppl > 0 else '')

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
    if run_e3:
        print('  E3 soft retrieval: ENABLED (similarity-weighted aggregation)')
        print(f'  E3 temperature: {args.e3_temperature}')
    else:
        print('  Next step (Phase E3): --phase e3 or --phase all for soft retrieval.')


if __name__ == '__main__':
    main()
