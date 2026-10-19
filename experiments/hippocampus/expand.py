"""Learned query expansion (H2b): which stored tokens should a question's tokens also cue?

From the reader's training worlds (labelled, as the reader's own training is), count, for every question, the tokens of
the episode that holds its answer which are NOT in the question. A question token whose counts are dominated by one
partner (share >= MIN_SHARE over >= MIN_COUNT questions) gets that partner's code added to its query.

Nothing pairs relations by hand. The expected outcome is that a converse token r° learns its relation r: a reverse
question (b, r°) is answered by an episode (a, r, b), whose tokens outside the question are a (which varies) and r
(which never does). Entity tokens and forward relation tokens should learn nothing: their partners vary.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from h1 import TRAIN_WORLDS, VOCAB, conv, make_facts, rel

MIN_SHARE, MIN_COUNT = 0.5, 20
CACHE = Path(__file__).parent / "runs" / "h2b" / "expansion.json"


def learn_expansion():
    count, total = defaultdict(Counter), Counter()
    for w in TRAIN_WORLDS:
        for a, r, b in make_facts(seed=w).facts:
            ep = (a, rel(r), b)
            for q in ((a, rel(r)), (b, conv(r))):                  # forward and reverse questions
                for t in q:
                    total[t] += 1
                    for u in set(ep) - set(q):
                        count[t][u] += 1
    partner, stats = {}, {}
    for t, c in count.items():
        u, n = c.most_common(1)[0]
        share = n / total[t]
        stats[t] = (u, round(share, 3), total[t])
        if share >= MIN_SHARE and total[t] >= MIN_COUNT:
            partner[t] = u
    return partner, stats


def expansion_table():
    """Partner per token as a (VOCAB,) tensor, -1 for none; cached with the learning statistics."""
    if CACHE.exists():
        partner = {int(k): v for k, v in json.load(open(CACHE))["partner"].items()}
    else:
        partner, stats = learn_expansion()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        top_shares = sorted(((s[1], t) for t, s in stats.items() if t not in partner), reverse=True)[:10]
        json.dump(dict(partner=partner, min_share=MIN_SHARE, min_count=MIN_COUNT,
                       best_unexpanded=[dict(token=t, partner=stats[t][0], share=sh, n=stats[t][2])
                                        for sh, t in top_shares]), open(CACHE, "w"), indent=1)
    table = torch.full((VOCAB,), -1, dtype=torch.long)
    for t, u in partner.items():
        table[t] = u
    return table
