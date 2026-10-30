"""L2/3's objects: hidden causes of what is at each state, learned at sleep (docs/LAB.md part 12; docs/JOINT.md §3.3).

**What an object is.** What came next at a state can depend on which object is there: a "7" and a "1" put different
strokes at one position. The object is never seen. It is learned from what occurs together within an episode (IT's
temporal contiguity, Li & DiCarlo 2008), with the frame shared by every object: an object is its own content at each
state, a set of cells.

**The input** is a region's episodes, each a list of observations (state, sensor, input): its content at the states of
the region's frame.

**The criterion: the code length of the episodes** (MDL), with
- **each episode's content,** cross-validated: each observation coded by its object's other episodes (leave one out),
  with information.py's content model;
- **which object each episode belongs to,** coded by a Chinese restaurant process with α = 1, the same prior
  strength as content's: P(partition) = α^K Π Γ(n_k) Γ(α) / Γ(N + α).

**The search: generalize, then split** (part 6's rule for P4, here for objects; amendment 1).
1. **Generalize:** every episode is one object.
2. **Split where refuted:** an object is split in two if that shortens the code.
   - **The proposal** (bisecting 2-means, seeded farthest-first): two seeds, the episode the object codes best and the
     one farthest from it (coded worst by it alone). Each episode joins the seed that codes it better, then the half
     whose content codes it better, until none moves (at most 40 passes).
     - **A half's prior is its parent** (one pseudo-observation of the whole's content), as a part's is in a hierarchy:
       a half of one episode is then that episode and the whole, not that episode alone, which would code everything
       else as never seen.
     - **The proposal leaves out the partition's prior:** it only proposes. The test is the code length, with the flat
       prior the column uses.
   - **The test:** the split stands if the code length falls. It is the halves' content, against the whole's, plus the
     partition's change, log α + log Γ(n_a) + log Γ(n_b) − log Γ(n).
   - **Then each half** is tried again, until no split shortens the code.
- **Why not one episode at a time** (the pre-registration's sequential seating): a lone episode compared with small
  early objects is coded better by a new object whenever any of its content is stochastic (an equation's first digit),
  so arithmetic would fall into one object per first digit. A split compared over whole groups is not fooled: one first
  digit shortens one cell of each episode but costs the label and every other cell's evidence.

**Each object gets a random SDR,** its identity, in the order of its first episode.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from arc_agent_v2.architecture.HDClibrary import SDR

ALPHA = 1.0
PASSES = 40                                             # the proposal's 2-means, until no episode moves


@dataclass
class Obj:
    sdr: object
    n: int                                              # the episodes it explains (its prior weight)
    cells: dict                                         # (state, sensor) -> {input: times}
    members: list = field(default_factory=list)         # the hippocampus's indices of its episodes


class _Data:
    """The observations as arrays: per observation, its episode, cell and input; per input, its bits."""

    def __init__(self, episodes, d, w):
        self.d, self.a = d, w / d
        self.w = w
        cells, inputs = {}, {}
        ep, cell, inp = [], [], []
        for j, (_, obs) in enumerate(episodes):
            for state, sensor, x in obs:
                ep.append(j)
                cell.append(cells.setdefault((state, sensor), len(cells)))
                inp.append(inputs.setdefault(x, len(inputs)))
        self.cells, self.inputs = list(cells), list(inputs)
        self.ep, self.cell, self.inp = np.array(ep, np.int64), np.array(cell, np.int64), np.array(inp, np.int64)
        width = max((x.idx.size for x in self.inputs), default=0)
        self.B = np.full((len(self.inputs), width), -1, np.int64)       # each input's bits, padded with -1
        for i, x in enumerate(self.inputs):
            self.B[i, :x.idx.size] = x.idx
        self.nC = len(self.cells)

    def stats(self, rows):
        """Counts over the observations `rows`: (per cell and bit, n_cells x d; per cell, the observations)."""
        bits = self.B[self.inp[rows]]
        cell = np.repeat(self.cell[rows], bits.shape[1]).reshape(bits.shape)
        ok = bits >= 0
        key = cell[ok] * self.d + bits[ok]
        C = np.bincount(key, minlength=self.nC * self.d).reshape(self.nC, self.d).astype(np.float64)
        N = np.bincount(self.cell[rows], minlength=self.nC).astype(np.float64)
        return C, N

    def loglik(self, rows, C, N, member, A=None):
        """Each observation's log-likelihood (tempered, information.Content's) under counts (C, N): leave-one-out where
        `member` (the observation is in them), else plain. A: the prior's mean per cell and bit (default w/d), of
        strength one pseudo-observation."""
        A = self.a if A is None else A
        bits = self.B[self.inp[rows]]
        ok = bits >= 0
        c = self.cell[rows]
        safe_bits = np.where(ok, bits, 0)
        Cx = np.where(ok, C[c[:, None], safe_bits], 0.0)
        Ax = A[c[:, None], safe_bits] if np.ndim(A) else np.full(bits.shape, A)
        Nc = N[c]
        out = np.empty(rows.size)
        plain = ~member
        top = 1 - 1e-12
        if plain.any():                                 # q = (C + A) / (N + 1)
            N1 = N + 1.0
            Tc = np.log1p(-np.minimum((C + A) / N1[:, None], top)).sum(1)
            q = np.minimum((Cx[plain] + Ax[plain]) / N1[c[plain]][:, None], top)
            out[plain] = Tc[c[plain]] + np.where(ok[plain], np.log(q) - np.log1p(-q), 0.0).sum(1)
        if member.any():                                # leave one out: q' = (C - [i in x] + A) / N
            safe = np.maximum(N, 1.0)
            full = C >= N[:, None]                      # bits every observation has: only an observation's own
            U = np.where(full, 0.0, np.log1p(-np.minimum((C + A) / safe[:, None], top))).sum(1)
            m, cm = member, c[member]
            Nm = Nc[m][:, None]
            Cm, Am = Cx[m], Ax[m]
            own_full = Cm >= Nm
            sub = np.where(ok[m] & ~own_full, np.log1p(-np.minimum((Cm + Am) / Nm, top)), 0.0).sum(1)
            add = np.where(ok[m], np.log(np.minimum((Cm - 1.0 + Am) / Nm, top)), 0.0).sum(1)
            out[m] = U[cm] - sub + add
        return out / (2 * self.w)


def _split(data, eps, obs_rows):
    """Try to split the object made of episodes `eps` (local indices). Returns (half a, half b) or None."""
    n = len(eps)
    if n < 2:
        return None
    rows = obs_rows(eps)
    if rows.size == 0:
        return None
    local = {e: k for k, e in enumerate(eps)}
    ep_of = np.array([local[e] for e in data.ep[rows].tolist()], np.int64)

    def per_episode(ll):
        return np.bincount(ep_of, weights=ll, minlength=n)

    C, N = data.stats(rows)
    whole = per_episode(data.loglik(rows, C, N, np.ones(rows.size, bool)))
    P = (C + data.a) / (N + 1.0)[:, None]               # the proposal's prior for a half: its parent, the whole
    one = lambda e: data.stats(rows[ep_of == e])        # one episode's content, as a half of one
    half = lambda s: per_episode(data.loglik(rows, *one(s), ep_of == s, P))
    sa = int(np.argmax(whole))                          # the proposal's seeds: the most typical episode,
    sb = int(np.argmin(half(sa)))                       # and the one farthest from it
    if sa == sb:
        return None
    scores = np.stack([half(s) for s in (sa, sb)])
    g = np.argmax(scores, 0)                            # each episode to the seed that codes it better
    g[sa], g[sb] = 0, 1
    for _ in range(PASSES):                             # then 2-means on content (the test, below, adds the
        if g.min() == g.max():                          # partition's prior)
            return None
        scores = np.empty((2, n))
        for h in (0, 1):
            in_h = g == h
            Ch, Nh = data.stats(rows[in_h[ep_of]])
            scores[h] = per_episode(data.loglik(rows, Ch, Nh, in_h[ep_of], P))
        new = np.argmax(scores, 0)
        if (new == g).all():
            break
        g = new
    na, nb = int((g == 0).sum()), int((g == 1).sum())
    if na == 0 or nb == 0:
        return None
    halves = 0.0
    for h in (0, 1):
        in_h = g == h
        Ch, Nh = data.stats(rows[in_h[ep_of]])
        halves += per_episode(data.loglik(rows, Ch, Nh, in_h[ep_of]))[in_h].sum()
    gain = halves - whole.sum() + math.log(ALPHA) + math.lgamma(na) + math.lgamma(nb) - math.lgamma(n)
    if gain <= 0:
        return None
    return [e for e, h in zip(eps, g) if h == 0], [e for e, h in zip(eps, g) if h == 1]


def learn(episodes, d, w, seed=0):
    """Objects for a region: episodes = [(the hippocampus's index, [(state, sensor, input)])]. Returns [Obj], ordered
    by first episode."""
    episodes = [(i, obs) for i, obs in episodes if obs]
    if not episodes:
        return []
    data = _Data(episodes, d, w)
    order = np.argsort(data.ep, kind="stable")
    start = np.searchsorted(data.ep[order], np.arange(len(episodes) + 1))
    obs_rows = lambda eps: np.concatenate([order[start[e]:start[e + 1]] for e in eps]) if eps else \
        np.zeros(0, np.int64)
    done, queue = [], [list(range(len(episodes)))]
    while queue:
        eps = queue.pop(0)
        halves = _split(data, eps, obs_rows)
        if halves is None:
            done.append(eps)
        else:
            queue.extend(halves)
    done.sort(key=min)
    out = []
    for k, eps in enumerate(done):
        cells = {}
        for e in eps:
            for state, sensor, x in episodes[e][1]:
                c = cells.setdefault((state, sensor), {})
                c[x] = c.get(x, 0) + 1
        sdr = SDR.random(d, w, np.random.default_rng((seed, 1212, k)))
        out.append(Obj(sdr, len(eps), cells, [episodes[e][0] for e in eps]))
    return out
