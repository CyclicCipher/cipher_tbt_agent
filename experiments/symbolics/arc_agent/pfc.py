"""The prefrontal cortex (docs/AGENT.md part 24): a column that learns which column to trust.

When several assoc columns, each in a different frame, explain the same change, they form a group of competing models.
The responsible one has the largest posterior, log P(kind | context) + the sum of its log predictive probabilities
since the group formed: a prior times the prequential evidence (Bayesian model selection; MOSAIC's responsibilities).

The prior is a column's prediction. The PFC column is an ordinary Column (column.py), the same learning as every other
column, over a choice frame (frames.py): its state is the kind of model that predicted best last; its operator is a
switch to a kind, or staying. Each step, for each group, it observes which candidate predicted best and learns that,
by attention over context experts by track record, from game-agnostic context: the region's size, colours and shape,
the action, the level. One PFC column serves every group, as a reusable model over many instances of its frame. Its
experience carries across regions, levels and games, so the higher column learns the lower competition's prior
(hierarchical Bayes): the right model wins after fewer observations as experience grows. Learning to learn, from
ordinary column learning one level up; nothing here targets it.

Only the responsible column of a group is planned with (Miller & Cohen: PFC biases which pathway carries the task);
the others keep learning, latent, and a switch is an ordinary operator, so a changed rule is followed by evidence.
"""
import math
import random

from arc_agent.column import Column
from arc_agent.frames import Choice

KINDS = ("translation", "recolour", "bar")
LOST = math.log(1e-4)                   # a candidate that could not find itself predicted its observation this badly


class PFC:
    def __init__(self, seed=0):
        self.col = Column(-1, Choice(KINDS), None, 0)
        self.groups = {}                                # gid -> dict(members, ev, state, ctx)
        self.member = {}                                # cid -> gid
        self.next, self.rng = 0, random.Random(seed)

    # ---- groups of competing models ---------------------------------------------------------------------------------
    def group(self, cids, kinds, context):
        gid = self.next
        self.next += 1
        self.groups[gid] = dict(members=list(cids), kinds=dict(zip(cids, kinds)), ev={c: 0.0 for c in cids},
                                state=None, ctx=dict(context), tie={c: self.rng.random() * 1e-6 for c in cids})
        for c in cids:
            self.member[c] = gid
        return gid

    def ctx(self, g, act):
        out = dict(g["ctx"])
        out[("act",)] = act
        out[("self",)] = g["state"]
        return out

    def prior(self, g, act):
        """P(the next responsible kind) from the PFC column's prediction of its own next state."""
        p, rest = self.col.dist(("pfc",), self.ctx(g, act))
        out = {}
        for k in KINDS:
            q = p.get(("to", k), rest)
            if k == g["state"]:
                q += p.get(("stay",), 0.0)
            out[k] = q
        z = sum(out.values()) or 1.0
        return {k: v / z for k, v in out.items()}

    def responsible(self, cid, act=None):
        """True if this column may be planned with: it belongs to no group, or it is its group's responsible model."""
        gid = self.member.get(cid)
        if gid is None:
            return True
        g = self.groups[gid]
        if not hasattr(self, "_cache") or self._cache[0] != (gid, g["state"], tuple(g["ev"].values())):
            pri = self.prior(g, act or "key")
            best = max(g["members"], key=lambda c: math.log(max(pri[g["kinds"][c]], 1e-12)) + g["ev"][c] + g["tie"][c])
            self._cache = ((gid, g["state"], tuple(g["ev"].values())), best)
        return self._cache[1] == cid

    # ---- learning: the evidence, and the PFC column's own observation ---------------------------------------------
    def observe(self, evidence, act):
        """After a step: each candidate's log predictive probability of the group's cells (brain._group_evidence:
        every rival scored on the same cells) is evidence; the kind that predicted best is the PFC column's
        observation, learned like any column's. A candidate that could not predict (lost) gets LOST."""
        for gid, g in self.groups.items():
            lps = {}
            for cid in g["members"]:
                lps[cid] = evidence.get(cid, LOST)
                g["ev"][cid] += lps[cid]
            if all(lp == LOST for lp in lps.values()):
                continue                                # the region is gone: nothing to learn here
            top = max(lps.values())
            best = {g["kinds"][c] for c, lp in lps.items() if lp >= top - 1e-9}
            if len(best) != 1:                          # every candidate predicted as well: no information
                continue
            kind = best.pop()
            op = self.col.frame.op_between(g["state"], kind)
            self.col.learn(("pfc",), self.ctx(g, act), op)
            g["state"] = kind

    def attended(self):
        b = self.col.keys.get(("pfc",))
        return None if b is None else b.attended()
