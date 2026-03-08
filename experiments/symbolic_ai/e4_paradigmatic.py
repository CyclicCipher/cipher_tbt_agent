"""e4_paradigmatic.py — Phase E4: Paradigmatic Axis (Frame Semantics)

The PARADIGMATIC axis captures SUBSTITUTABILITY, not co-occurrence:
    cat ≈ dog  because both appear in "The ___ sleeps", "I pet the ___", etc.

This is orthogonal to the SYNTAGMATIC axis (E1-E3), which captures what follows:
    dog → barks  (syntagmatic: what comes next)
    cat ≈ dog    (paradigmatic: what slots they share)

Architecture:
    slot_occupants: (c_prev, c_next) → w_middle
        For each training trigram (w1, w2, w3):
            c1 = assignment[w1],  c3 = assignment[w3]
            teach slot_occupants: (c1, c3) → w2

    Word paradigmatic profile:
        P_paradigm(w) = distribution over (c1, c3) frames where w appeared as middle

    Paradigmatic similarity:
        sim(w, w') = exp(-T × JSD(P_paradigm(w), P_paradigm(w')))
        → "can w and w' substitute for each other?"

    Evaluation task: fill-in-the-blank
        Given (w1, ___, w3), predict w2 via slot_occupants[(c1, c3)]
        Key test: (c1, c3) frame SEEN, but (w1, w3) pair UNSEEN
        → same generalisation logic as E1's next-word test

Usage:
    python e4_paradigmatic.py --corpus EarlyModernLatin --n_train 5000
    python e4_paradigmatic.py --corpus EarlyModernLatin --n_train 5000 --words est in de
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
from language_pipeline import build_assignment, train_chain, _jsd

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
# Training
# ---------------------------------------------------------------------------

def train_slot_occupants(
    ai,
    train_texts: list[str],
    assignment: dict,
    verbose: bool = True,
) -> tuple[int, int]:
    """Train slot_occupants: (c_prev, c_next) → w_middle.

    For each training trigram (w1, w2, w3):
        c1 = assignment[w1],  c3 = assignment[w3]
        teach slot_occupants: (c1, c3) → w2

    This records what words fill the MIDDLE position given surrounding
    category context — the paradigmatic (substitutability) axis.

    Contrast with word_given_cat: (c1, c2, c3) → w3  (next-word, syntagmatic).
    """
    _banner('Phase E4: Training slot_occupants (paradigmatic axis)')

    if 'slot_occupants' not in ai.stores:
        ai.add_concept(
            name='slot_occupants', domain='language',
            description='Paradigmatic fill-in-blank: (c_prev, c_next) → w_middle',
            input_type=['cat', 'cat'], output_type=['word'], tier='theorem',
        )

    n_used = n_skip = 0
    for text in train_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            c1 = assignment.get(w1)
            c3 = assignment.get(w3)
            if c1 is None or c3 is None:
                n_skip += 1
                continue
            ai.teach('slot_occupants', (str(c1), str(c3)), (w2,))
            n_used += 1

    if verbose:
        store = ai.stores.get('slot_occupants')
        n_ex     = len(store.examples) if store else 0
        n_frames = len({inp for inp, _ in store.examples}) if store else 0
        print(f'  Trigrams used:    {n_used:,}  skipped (OOV): {n_skip:,}')
        print(f'  slot_occupants:   {n_ex:,} examples,  {n_frames:,} unique (c1,c3) frames')
        print(f'  Total K² frames:  {len(set(assignment.values()))**2}  '
              f'(populated: {n_frames})')

        # Show sample frames
        frame_words: dict = collections.defaultdict(collections.Counter)
        for inp, out in store.examples:
            w = out[0] if isinstance(out, tuple) else str(out)
            frame_words[inp][w] += 1
        busiest = sorted(frame_words.items(), key=lambda kv: -sum(kv[1].values()))[:3]
        for frame, counter in busiest:
            top = counter.most_common(6)
            print(f'    Frame {frame}: ' + '  '.join(f'{w}({n})' for w, n in top))

    return n_used, n_skip


# ---------------------------------------------------------------------------
# Paradigmatic profiles & similarity
# ---------------------------------------------------------------------------

def build_word_frame_profiles(
    ai,
    verbose: bool = True,
) -> dict[str, dict]:
    """Build word → frame distribution from slot_occupants.

    P_paradigm(w)[(c1, c3)] ∝ count of frames where w appeared as w_middle.

    Returns {word: {(c1_str, c3_str): probability}}.
    """
    _banner('Phase E4: Building paradigmatic word profiles')

    store = ai.stores.get('slot_occupants')
    if store is None:
        print('  No slot_occupants store found — run train_slot_occupants first.')
        return {}

    word_frame_counts: dict = collections.defaultdict(collections.Counter)
    for inp, out in store.examples:
        w2  = out[0] if isinstance(out, tuple) else str(out)
        frame = inp      # tuple (c1_str, c3_str)
        word_frame_counts[w2][frame] += 1

    profiles: dict = {}
    for word, frame_counter in word_frame_counts.items():
        total = sum(frame_counter.values())
        profiles[word] = {f: cnt / total for f, cnt in frame_counter.items()}

    if verbose:
        n_words    = len(profiles)
        avg_frames = sum(len(p) for p in profiles.values()) / max(n_words, 1)
        print(f'  Words with frame profiles: {n_words:,}')
        print(f'  Avg frames per word:       {avg_frames:.1f}')
        # Show most "paradigmatically promiscuous" words (many frames)
        top_words = sorted(profiles, key=lambda w: -len(profiles[w]))[:5]
        for w in top_words:
            print(f'    {w!r}: {len(profiles[w])} distinct (c_prev, c_next) frames')

    return profiles


def find_paradigmatic_neighbors(
    word: str,
    profiles: dict,
    topn: int = 8,
    temperature: float = 2.0,
) -> list[tuple[str, float]]:
    """Find words most paradigmatically similar to `word`.

    Similarity = exp(-T × JSD(P_paradigm(word), P_paradigm(other))).
    Returns [(neighbor, similarity), ...] sorted descending.
    """
    if word not in profiles:
        return []
    p = profiles[word]
    scores = []
    for other, q in profiles.items():
        if other == word:
            continue
        sim = math.exp(-temperature * _jsd(p, q))
        scores.append((other, sim))
    return sorted(scores, key=lambda x: -x[1])[:topn]


def show_paradigmatic_neighbors(
    profiles: dict,
    words: list[str] | None = None,
    topn: int = 6,
) -> None:
    """Print paradigmatic neighbors for sample words."""
    _banner('Phase E4: Paradigmatic neighbors (substitutability)')

    if not profiles:
        print('  No profiles available.')
        return

    if words is None:
        # Auto-pick: highest frame-count words (most paradigmatically informative)
        words = sorted(profiles, key=lambda w: -len(profiles[w]))[:10]

    for word in words:
        neighbors = find_paradigmatic_neighbors(word, profiles, topn=topn)
        if not neighbors:
            print(f'  {word!r}: (no neighbors)')
            continue
        ns = '   '.join(f'{w}({s:.2f})' for w, s in neighbors)
        n_frames = len(profiles.get(word, {}))
        print(f'  {word!r} [{n_frames}f] ≈  {ns}')


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_slot(
    ai,
    w1: str,
    w3: str,
    assignment: dict,
) -> str | None:
    """Predict middle word w2 given surrounding words w1 and w3.

    Uses slot_occupants: (assignment[w1], assignment[w3]) → argmax w2.
    """
    c1 = assignment.get(w1)
    c3 = assignment.get(w3)
    if c1 is None or c3 is None:
        return None
    result = ai.ask('slot_occupants', (str(c1), str(c3)))
    if result is None:
        return None
    return result[0] if isinstance(result, tuple) else str(result)


def logprob_slot(
    ai,
    w1: str,
    w2: str,
    w3: str,
    assignment: dict,
) -> float | None:
    """Log₂ P(w2 | c1, c3) via slot_occupants. None if OOV or unseen frame."""
    c1 = assignment.get(w1)
    c3 = assignment.get(w3)
    if c1 is None or c3 is None:
        return None
    dist = ai.ask_dist('slot_occupants', (str(c1), str(c3)))
    if dist is None:
        return None
    p = dist.get((w2,), 0.0)
    if p < 1e-12:
        return None
    return math.log2(p)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_slot_filling(
    ai,
    test_texts:       list[str],
    assignment:       dict,
    train_slot_pairs: set,       # set of (w1, w3) pairs seen in training
    unigram_dist:     dict,      # {word: probability} for flat baseline
    verbose:          bool = True,
) -> dict:
    """Evaluate fill-in-the-blank: predict w2 from (w1, w3) context.

    Comparison:
        Flat unigram:  predict the most common word (context-free).
        Slot chain:    predict w2 from (c1, c3) frame via slot_occupants.

    Key metric: accuracy on trigrams where (w1, w3) pair was UNSEEN in training.
    If the (c1, c3) frame was seen, the slot chain can still predict — this is
    the paradigmatic analogue of E1's unseen syntagmatic pair test.
    """
    _banner('Phase E4: Evaluating slot filling (fill-in-the-blank)')

    # Flat unigram: always predict the most common word.
    flat_pred = max(unigram_dist, key=unigram_dist.get) if unigram_dist else None

    r = dict(
        total=0,
        slot_correct=0, slot_answered=0,
        flat_correct=0, flat_answered=0,
        unseen_total=0,
        slot_unseen_correct=0, slot_unseen_answered=0,
        flat_unseen_correct=0, flat_unseen_answered=0,
        slot_logloss=0.0, slot_logloss_n=0,
        flat_logloss=0.0, flat_logloss_n=0,
    )

    for text in test_texts:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            is_unseen = (w1, w3) not in train_slot_pairs

            r['total'] += 1
            if is_unseen:
                r['unseen_total'] += 1

            # Slot occupants prediction
            pred = predict_slot(ai, w1, w3, assignment)
            if pred is not None:
                r['slot_answered'] += 1
                if pred == w2:
                    r['slot_correct'] += 1
                    if is_unseen:
                        r['slot_unseen_correct'] += 1
                if is_unseen:
                    r['slot_unseen_answered'] += 1

            lp = logprob_slot(ai, w1, w2, w3, assignment)
            if lp is not None:
                r['slot_logloss'] -= lp
                r['slot_logloss_n'] += 1

            # Flat unigram baseline
            if flat_pred is not None:
                r['flat_answered'] += 1
                if flat_pred == w2:
                    r['flat_correct'] += 1
                    if is_unseen:
                        r['flat_unseen_correct'] += 1
                if is_unseen:
                    r['flat_unseen_answered'] += 1

            lp_flat = math.log2(unigram_dist[w2]) if w2 in unigram_dist else None
            if lp_flat is not None:
                r['flat_logloss'] -= lp_flat
                r['flat_logloss_n'] += 1

    if verbose:
        T  = r['total']
        UT = r['unseen_total']

        def pct(n, d): return f'{100*n/d:.1f}%' if d else 'N/A'
        def ppl(loss, n): return f'{2**(loss/n):.1f}' if n else 'N/A'

        fa = r['flat_answered']; fc = r['flat_correct']
        sa = r['slot_answered']; sc = r['slot_correct']
        fua = r['flat_unseen_answered']; fuc = r['flat_unseen_correct']
        sua = r['slot_unseen_answered']; suc = r['slot_unseen_correct']

        W = 14
        LW = 36

        def row(label, fv, sv):
            return f'  {label:<{LW}} {fv:>{W}} {sv:>{W}}'

        print(f'\n  Test trigrams: {T:,}   Unseen (w1,w3) pairs: {UT:,} ({pct(UT,T)})')
        print()
        print(row('Metric', 'Flat unigram', 'Slot chain'))
        print('  ' + '-' * (LW + 2 * W + 2))
        print(row('Coverage (answered/total)',      pct(fa, T),   pct(sa, T)))
        print(row('Top-1 accuracy (of answered)',   pct(fc, fa),  pct(sc, sa)))
        print(row('Top-1 accuracy (of total)',      pct(fc, T),   pct(sc, T)))
        print(row('Perplexity (lower=better)',
                  ppl(r['flat_logloss'], r['flat_logloss_n']),
                  ppl(r['slot_logloss'], r['slot_logloss_n'])))
        print()
        print(f'  {"=== UNSEEN (w1,w3) PAIRS (key test) ===":<{LW + 2*W + 2}}')
        print('  ' + '-' * (LW + 2 * W + 2))
        print(row('Coverage (answered/unseen)',     pct(fua, UT), pct(sua, UT)))
        print(row('Acc. of answered (unseen)',      pct(fuc, fua), pct(suc, sua)))
        print(row('Acc. of total (unseen)',         pct(fuc, UT), pct(suc, UT)))
        print()

        slot_acc_unseen = suc / max(sua, 1)
        flat_acc_unseen = fuc / max(fua, 1)
        if slot_acc_unseen > flat_acc_unseen and sua > 0:
            print(f'  RESULT: Slot chain BETTER on unseen (w1,w3) pairs '
                  f'({pct(suc, sua)} vs {pct(fuc, fua)}). Paradigmatic axis validated.')
        elif sua == 0:
            print('  RESULT: Slot chain answered no unseen pairs — corpus too small or K too high.')
        else:
            print(f'  RESULT: Flat unigram matches/exceeds slot chain on unseen pairs. '
                  f'Consider larger corpus or more clusters.')

    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Phase E4: Paradigmatic Axis (Frame Semantics)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--corpus',     default='EarlyModernLatin')
    p.add_argument('--n_train',    type=int,   default=5000)
    p.add_argument('--n_test',     type=int,   default=None)
    p.add_argument('--split',      type=float, default=0.8)
    p.add_argument('--n_clusters', type=int,   default=12)
    p.add_argument('--load',       metavar='PATH',
                   help='Load symbolic AI checkpoint')
    p.add_argument('--save',       metavar='PATH',
                   help='Save symbolic AI checkpoint')
    p.add_argument('--words',      nargs='+', metavar='WORD',
                   help='Show paradigmatic neighbors for these specific words')
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    _banner('Phase E4: Paradigmatic Axis (Frame Semantics)')
    print(f'  Corpus:      {args.corpus}')
    print(f'  Train lines: {args.n_train}')
    print(f'  K clusters:  {args.n_clusters}')

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

    # ---- Unigram baseline ----
    unigram_counts: collections.Counter = collections.Counter()
    for text in train_texts:
        unigram_counts.update(text.split())
    total_uni = sum(unigram_counts.values())
    unigram_dist = {w: c / total_uni for w, c in unigram_counts.items()}

    # ---- Seen/unseen split for slot task ----
    # The test for E4 is: (w1, w3) pair UNSEEN (not (w2, w3) as in E1).
    train_slot_pairs: set = set()
    for text in train_texts:
        toks = text.split()
        for i in range(len(toks) - 2):
            train_slot_pairs.add((toks[i], toks[i + 2]))   # (w1, w3)
    print(f'\n  Training (w1,w3) frame pairs: {len(train_slot_pairs):,}')

    # ---- Phase E4: train slot_occupants ----
    train_slot_occupants(ai, train_texts, assignment, verbose=True)

    # ---- Paradigmatic profiles ----
    profiles = build_word_frame_profiles(ai, verbose=True)

    # ---- Show paradigmatic neighbors ----
    show_paradigmatic_neighbors(profiles, words=args.words, topn=6)

    # ---- Evaluate slot filling ----
    evaluate_slot_filling(
        ai, test_texts, assignment, train_slot_pairs, unigram_dist, verbose=True)

    if args.save:
        ai.save_checkpoint(args.save)
        print(f'\n  Checkpoint saved: {args.save}')


if __name__ == '__main__':
    main()
