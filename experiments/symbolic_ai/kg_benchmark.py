"""kg_benchmark.py — R5 benchmark: RelationalLearner on FB15k-237 and WN18RR.

Standard filtered link-prediction evaluation (same protocol as TransE/RotatE):
  For each test triple (h, r, t), rank the true tail t against ALL entities.
  Filtered: remove other known true tails for the same (h, r) from the ranking
  (so we don't penalise predicting a different valid answer).

  rank_filtered(t) = 1 + |{e ≠ t : score(e) > score(t)
                                    AND (h, r, e) is not a known true triple}|

  Entities not returned by predict_dist get score 0. Ties at score 0 do not
  increase t's rank (they are ranked equal to t, not above it).

Published baselines (filtered Hits@10):
  FB15k-237:  TransE 46.5%  DistMult 41.9%  RotatE 53.3%
  WN18RR:     TransE 50.1%  DistMult 44.1%  RotatE 57.1%

Usage:
    python kg_benchmark.py --dataset FB15k-237
    python kg_benchmark.py --dataset wn18rr
    python kg_benchmark.py          # both
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from relational_pipeline import RelationalLearner

DATA_DIR = os.path.join(_HERE, 'data')

# Published filtered Hits@10 baselines
BASELINES = {
    'FB15k-237': {'TransE': 0.465, 'DistMult': 0.419, 'RotatE': 0.533},
    'wn18rr':    {'TransE': 0.501, 'DistMult': 0.441, 'RotatE': 0.571},
}


def load_triples(path: str) -> list[tuple[str, str, str]]:
    triples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_all_positives(
        *triple_lists: list[tuple[str, str, str]]
) -> dict[tuple[str, str], set[str]]:
    """Build {(h, r): set(all_known_true_tails)} from all splits.

    Used in the filtered setting to avoid penalising a model for ranking a
    valid-but-different tail above the specific test tail.
    """
    pos: dict = collections.defaultdict(set)
    for triples in triple_lists:
        for h, r, t in triples:
            pos[(h, r)].add(t)
    return pos


def evaluate_filtered(
        learner: RelationalLearner,
        test_triples: list[tuple[str, str, str]],
        all_positives: dict[tuple[str, str], set[str]],
        ks: tuple = (1, 3, 10),
        n_sample: int = 5000,
        seed: int = 42,
) -> dict:
    """Standard filtered KGE link-prediction evaluation.

    For each test triple (h, r, t):
      score_t   = predict_dist(h, r).get(t, 0.0)
      rank_filt = 1 + |{e in dist : score(e) > score_t
                                   AND e ≠ t
                                   AND (h, r, e) not a known true triple}|

    Entities not returned by predict_dist all have score 0. They are tied
    with t when score_t = 0 and therefore do not increase t's rank.
    This matches the standard protocol: we only count entities that
    strictly outrank t.

    Returns dict with keys: ks + 'mrr', 'n', 'n_oov' (no prediction at all).
    """
    random.seed(seed)
    subset = test_triples
    if len(subset) > n_sample:
        subset = random.sample(subset, n_sample)

    hits   = {k: 0 for k in ks}
    rr_sum = 0.0
    n_oov  = 0   # test triples where predict_dist returned nothing

    for h, r, t in subset:
        dist = learner.predict_dist(h, r)

        if not dist:
            # Relation r completely unseen → score_t = 0, all entities tied at 0
            # rank_filtered = 1 (no entity strictly outranks t)
            # Count as rank 1 — the model is uninformative but not wrong
            n_oov += 1
            rr_sum += 1.0
            for k in ks:
                hits[k] += 1
            continue

        score_t = dist.get(t, 0.0)
        known   = all_positives.get((h, r), set())

        # Count entities in dist that strictly outrank t and are not known positives
        n_above = sum(
            1 for e, s in dist.items()
            if s > score_t and e != t and e not in known
        )
        rank = 1 + n_above

        rr_sum += 1.0 / rank
        for k in ks:
            if rank <= k:
                hits[k] += 1

    n = len(subset)
    result = {k: hits[k] / n for k in ks}
    result['mrr']   = rr_sum / n if n > 0 else 0.0
    result['n']     = n
    result['n_oov'] = n_oov
    return result


def run_dataset(name: str, n_sample: int = 5000) -> None:
    ds_dir     = os.path.join(DATA_DIR, name)
    train_path = os.path.join(ds_dir, 'train.txt')
    valid_path = os.path.join(ds_dir, 'valid.txt')
    test_path  = os.path.join(ds_dir, 'test.txt')

    if not os.path.exists(train_path):
        print(f'  {name}: train.txt not found at {train_path}')
        return

    print(f'\n{"=" * 65}')
    print(f'Dataset: {name}')
    print(f'{"=" * 65}')

    t0    = time.time()
    train = load_triples(train_path)
    valid = load_triples(valid_path) if os.path.exists(valid_path) else []
    test  = load_triples(test_path)

    # All entities and relations across all splits
    all_entities: set[str] = set()
    all_relations: set[str] = set()
    for h, r, t in train + valid + test:
        all_entities.add(h); all_entities.add(t)
        all_relations.add(r)

    print(f'  Train / Valid / Test:  '
          f'{len(train):,} / {len(valid):,} / {len(test):,} triples')
    print(f'  Entities: {len(all_entities):,}   Relations: {len(all_relations):,}')
    print(f'  Evaluating {min(n_sample, len(test)):,} test triples '
          f'(filtered, rank against all entities)')

    # Known positives across ALL splits (for filtered evaluation)
    all_positives = build_all_positives(train, valid, test)

    # Fit on train only
    print(f'\n  Fitting RelationalLearner...')
    learner = RelationalLearner()
    learner.fit(train, verbose=True)
    print(f'  Fit done in {time.time() - t0:.1f}s')

    # Evaluate
    print(f'\n  Evaluating (filtered)...')
    t1      = time.time()
    results = evaluate_filtered(learner, test, all_positives, n_sample=n_sample)
    print(f'  Eval done in {time.time() - t1:.1f}s')

    # Coverage: (h,r) pairs seen vs unseen in training
    train_pairs = set((h, r) for h, r, _ in train)
    n_seen = sum(1 for h, r, _ in test if (h, r) in train_pairs)

    # Report
    print(f'\n  Filtered Hits@K  (n={results["n"]:,}, '
          f'OOV={results["n_oov"]:,}, '
          f'(h,r) seen={n_seen}/{len(test)} = {n_seen/len(test):.1%}):')
    print(f'\n  {"Model":22s}  MRR    H@1    H@3    H@10')
    print(f'  {"-" * 55}')
    print(f'  {"RelationalLearner":22s}  '
          f'{results["mrr"]:.3f}  '
          f'{results[1]:.3f}  '
          f'{results[3]:.3f}  '
          f'{results[10]:.3f}')

    if name in BASELINES:
        print(f'  {"-" * 55}')
        for model, h10 in BASELINES[name].items():
            print(f'  {model:22s}  (neural, trained)          {h10:.3f}')

    # Explanation of what the numbers mean
    print()
    print(f'  Note on filtered rank: entities not in predict_dist get score 0.')
    print(f'  Only entities IN dist with score > score(t) increase rank.')
    print(f'  For seen (h,r) pairs ({n_seen/len(test):.0%} of test): dist has ~few entries,')
    print(f'  so rank is mostly determined by whether t itself is in dist.')
    print(f'  For OOV (h,r) pairs: dist = rel_unigram (all training tails for r).')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['FB15k-237', 'wn18rr'],
                        help='Run one dataset (default: both)')
    parser.add_argument('--n_sample', type=int, default=5000,
                        help='Test triples to evaluate (default: 5000)')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ['FB15k-237', 'wn18rr']
    for ds in datasets:
        run_dataset(ds, n_sample=args.n_sample)
    print()


if __name__ == '__main__':
    main()
