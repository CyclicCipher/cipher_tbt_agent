"""Synthetic relational facts for the reversal test (Experiment 1).

Entities are single tokens 0..N-1. Each of the R base relations r has its own converse token r°. A fact (a, r, b) is
written forward as the two-token prompt [a, r] with answer b, and in reverse as [b, r°] with answer a.

Relation kinds, two of each (with R = 6):
  one-to-one   - every subject has one object and every object one subject
  many-to-one  - forward answer unique; the reverse question has 1-3 answers
  one-to-many  - forward question has 1-3 answers; the reverse answer unique

Reverse questions are grouped by their key (r, b). A group is PAIRED (its facts are trained in both directions) or
ONE-WAY (forward only). Splitting by group means no reverse prompt of a one-way group is ever trained, so a one-way
reverse question can only be answered from the forward facts.
"""
import random
from dataclasses import dataclass, field

KINDS = ("one-to-one", "many-to-one", "one-to-many")


@dataclass
class Facts:
    n_ent: int
    n_rel: int
    facts: list                               # (a, r, b) with r in 0..n_rel-1
    paired: set                               # reverse keys (r, b) trained in both directions
    kinds: list                               # kind of each relation
    train: list = field(default_factory=list)  # (prompt (t0, t1), answer)
    evals: dict = field(default_factory=dict)  # name -> list of (prompt, correct set, leak set)

    @property
    def vocab(self):
        return self.n_ent + 2 * self.n_rel

    def rel_tok(self, r):
        return self.n_ent + r

    def conv_tok(self, r):
        return self.n_ent + self.n_rel + r


def make_facts(n_ent=1000, n_rel=6, per_rel=500, paired_frac=0.5, seed=0):
    rng = random.Random(seed)
    kinds = [KINDS[r % 3] for r in range(n_rel)]
    facts = []
    for r, kind in enumerate(kinds):
        ents = list(range(n_ent))
        if kind == "one-to-one":
            subj = rng.sample(ents, per_rel)
            obj = rng.sample(ents, per_rel)
            facts += [(a, r, b) for a, b in zip(subj, obj) if a != b]
        else:                                  # hubs with 1-3 partners each; partners used once per relation
            n_hub = per_rel // 2
            pool = ents[:]
            rng.shuffle(pool)
            hubs, rest = pool[:n_hub], pool[n_hub:]
            rng.shuffle(rest)
            i = 0
            for h in hubs:
                for _ in range(rng.randint(1, 3)):
                    p = rest[i % len(rest)]
                    i += 1
                    facts.append((p, r, h) if kind == "many-to-one" else (h, r, p))
    facts = sorted(set(facts))
    rev_keys = sorted({(r, b) for a, r, b in facts})
    paired = set(k for k in rev_keys if rng.random() < paired_frac)
    F = Facts(n_ent, n_rel, facts, paired, kinds)

    fwd, rev = {}, {}
    links = {}                                 # entity -> entities linked to it by any fact, either direction
    n_link = {}                                # unordered entity pair -> number of facts linking them
    for a, r, b in facts:
        n_link[frozenset((a, b))] = n_link.get(frozenset((a, b)), 0) + 1
        fwd.setdefault((a, r), set()).add(b)
        rev.setdefault((r, b), set()).add(a)
        links.setdefault(a, set()).add(b)
        links.setdefault(b, set()).add(a)

    for a, r, b in facts:
        F.train.append(((a, F.rel_tok(r)), b))
        if (r, b) in paired:
            F.train.append(((b, F.conv_tok(r)), a))

    def ev(keys, table, conv):
        out = []
        for key in keys:
            if conv:
                r, b = key
                prompt, cue = (b, F.conv_tok(r)), b
            else:
                a, r = key
                prompt, cue = (a, F.rel_tok(r)), a
            correct = table[key]
            out.append((prompt, sorted(correct), sorted(links[cue] - correct)))
        return out

    F.evals["fwd"] = ev(sorted(fwd), fwd, False)
    F.evals["rev_paired"] = ev(sorted(k for k in rev if k in paired), rev, True)
    F.evals["rev_oneway"] = ev(sorted(k for k in rev if k not in paired), rev, True)
    # the PRIMARY set: one-way questions none of whose answers is linked to b by any other fact, so "name something
    # linked to b" cannot score by coincidence
    F.evals["rev_oneway_clean"] = ev(sorted(k for k in rev if k not in paired and
                                            all(n_link[frozenset((a, k[1]))] == 1 for a in rev[k])), rev, True)
    return F


if __name__ == "__main__":
    F = make_facts()
    print(f"{len(F.facts)} facts, {len(F.train)} training prompts, vocab {F.vocab}")
    for k, v in F.evals.items():
        chance = sum(len(c) for _, c, _ in v) / len(v) / F.n_ent
        leak = sum(len(l) for _, _, l in v) / len(v)
        print(f"{k:11s} {len(v):5d} questions  chance hits@1 {chance:.4f}  mean leak-set size {leak:.1f}")
