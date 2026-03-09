"""kg_benchmark.py — R5 benchmark: RelationalLearner on FB15k-237 and WN18RR.

Standard link-prediction evaluation:
  For each test triple (h, r, t), rank t by predict_dist(h, r).
  Hits@K = fraction of test triples where true tail ranks ≤ K.
  MRR    = mean reciprocal rank of true tail.

Published baselines (Hits@10) for reference:
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

# Published Hits@10 baselines (for comparison only)
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
                h, r, t = parts
                triples.append((h, r, t))
    return triples


def hits_at_k(learner: RelationalLearner,
              test_triples: list[tuple[str, str, str]],
              ks: tuple = (1, 3, 10),
              n_sample: int = 5000,
              seed: int = 42) -> dict:
    """Compute Hits@K and MRR on test triples.

    For each (h, r, t): call predict_dist(h, r), rank t in output distribution.
    If t not in distribution: rank = len(dist)+1 (worst case within known atoms).
    """
    random.seed(seed)
    subset = test_triples
    if len(subset) > n_sample:
        subset = random.sample(subset, n_sample)

    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    n_hit_any = 0

    for h, r, t in subset:
        dist = learner.predict_dist(h, r)
        if not dist:
            # No prediction possible (h or r totally unseen)
            continue
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        rank = next((i + 1 for i, (tok, _) in enumerate(ranked) if tok == t),
                    len(ranked) + 1)
        rr_sum += 1.0 / rank
        n_hit_any += 1
        for k in ks:
            if rank <= k:
                hits[k] += 1

    n = len(subset)
    result = {k: hits[k] / n for k in ks}
    result['mrr'] = rr_sum / n if n > 0 else 0.0
    result['coverage'] = n_hit_any / n if n > 0 else 0.0
    result['n'] = n
    return result


def run_dataset(name: str, n_sample: int = 5000) -> None:
    ds_dir = os.path.join(DATA_DIR, name)
    train_path = os.path.join(ds_dir, 'train.txt')
    test_path  = os.path.join(ds_dir, 'test.txt')

    if not os.path.exists(train_path):
        print(f'  {name}: train.txt not found at {train_path}')
        return

    print(f'\n{"=" * 60}')
    print(f'Dataset: {name}')
    print(f'{"=" * 60}')

    t0 = time.time()
    train = load_triples(train_path)
    test  = load_triples(test_path)

    # Dataset stats
    entities  = set()
    relations = set()
    for h, r, t in train:
        entities.add(h); entities.add(t); relations.add(r)
    print(f'  Train triples: {len(train):,}')
    print(f'  Test  triples: {len(test):,}  (evaluating {n_sample:,} sampled)')
    print(f'  Entities:      {len(entities):,}')
    print(f'  Relations:     {len(relations):,}')

    # Fit
    print(f'\n  Fitting RelationalLearner...')
    learner = RelationalLearner()
    learner.fit(train, verbose=True)
    print(f'  Fit done in {time.time()-t0:.1f}s')

    # Evaluate
    print(f'\n  Evaluating Hits@K...')
    t1 = time.time()
    results = hits_at_k(learner, test, n_sample=n_sample)
    print(f'  Eval done in {time.time()-t1:.1f}s')

    # Report
    print(f'\n  Results (n={results["n"]:,}, coverage={results["coverage"]:.1%}):')
    print(f'  {"Model":20s}  MRR    H@1    H@3    H@10')
    print(f'  {"-"*52}')
    print(f'  {"RelationalLearner":20s}  '
          f'{results["mrr"]:.3f}  '
          f'{results[1]:.3f}  '
          f'{results[3]:.3f}  '
          f'{results[10]:.3f}')

    if name in BASELINES:
        print(f'  {"-"*52}')
        for model, h10 in BASELINES[name].items():
            print(f'  {model:20s}  (trained)          {h10:.3f}')

    # Coverage breakdown: how many test (h,r) pairs were seen in training?
    train_pairs = set((h, r) for h, r, t in train)
    seen = sum(1 for h, r, t in test if (h, r) in train_pairs)
    print(f'\n  Test (h,r) pairs seen in training: {seen:,}/{len(test):,} '
          f'({seen/len(test):.1%})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
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
