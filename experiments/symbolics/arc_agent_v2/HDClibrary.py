"""The one datatype of arc agent v2, the SDR, and every operation on it (ARCHITECTURE.md section 2).

Nothing in v2 is a token. A symbol, a place, a displacement, a feature, a memory key: each is an SDR, a sparse binary
vector of dimension d (config.BrainConfig.d, 2048 by default) with few active bits (w = 40 by default, 2%). Similarity
is overlap, the number of shared active bits. Two random SDRs share about w*w/d bits (0.8 at the defaults), so an
overlap of a few bits already means "related".

Contents:
- **the datatype:** `SDR`, an immutable set of active bits, with its set algebra (& | - ^);
- **similarity:** overlap, jaccard, match;
- **making sparse codes:**
  - kwta: the k most excited bits;
  - thin: a random subset;
  - bundle: a sparse superposition that keeps what many inputs share;
  - cdt: context-dependent thinning, a similarity-preserving binding of free SDRs (Rachkovskij & Kussul 2001);
- **permutations:** shift and Permutation (roles, sequence positions);
- **block codes (`Blocks`):** an SDR with exactly one active bit per block, each block a small torus (a module).
  - Binding adds the blocks' offsets and unbinding subtracts them. This is an invertible binding that keeps the code
    sparse (sparse block codes: Laiho et al. 2015; Frady, Kleyko & Sommer 2023).
  - So a block code is a coordinate system. power(base, x) puts x on a ring in every block (the number line in polar
    coordinates), and binding two places rotates the first by the second.
- **encoders (the boundary where data becomes SDRs):**
  - CategoryEncoder: a random SDR per symbol, so no structure is given;
  - ScalarEncoder: a run of bits, optionally on a ring;
  - GridEncoder: 2-D places as block codes, i.e. grid-cell modules;
- **memories:**
  - ItemMemory: cleanup, the stored SDR nearest a query, by an inverted index;
  - Synapses: HTM permanences with Hebbian learning;
  - SpatialPooler: HTM's learned encoder (Cui, Ahmad & Hawkins 2017).
"""
import hashlib
from functools import lru_cache

import numpy as np


# ---- the datatype ---------------------------------------------------------------------------------------------------
class SDR:
    """d bits, of which those in `idx` (sorted, unique, int32) are active. Immutable and hashable."""
    __slots__ = ("d", "idx", "_h")

    def __init__(self, d, idx=()):
        a = np.unique(np.asarray(idx, dtype=np.int64))
        if a.size and (a[0] < 0 or a[-1] >= d):
            raise ValueError("an active bit outside 0..d-1")
        self.d = int(d)
        self.idx = a.astype(np.int32)
        self.idx.flags.writeable = False
        self._h = None

    @classmethod
    def _sorted(cls, d, idx):
        """From bits already sorted, unique and in range (the set operations' results)."""
        s = cls.__new__(cls)
        s.d, s.idx, s._h = int(d), np.asarray(idx, dtype=np.int32), None
        s.idx.flags.writeable = False
        return s

    @classmethod
    def random(cls, d, w, rng):
        return cls._sorted(d, np.sort(rng.choice(d, size=w, replace=False)))

    @classmethod
    def from_dense(cls, x):
        x = np.asarray(x).ravel()
        return cls._sorted(x.size, np.flatnonzero(x))

    @classmethod
    def empty(cls, d):
        return cls._sorted(d, np.zeros(0, np.int32))

    @property
    def w(self):
        """The number of active bits."""
        return int(self.idx.size)

    @property
    def sparsity(self):
        return self.w / self.d

    def dense(self, dtype=np.uint8):
        x = np.zeros(self.d, dtype)
        x[self.idx] = 1
        return x

    def _same(self, o):
        if not isinstance(o, SDR) or o.d != self.d:
            raise ValueError("SDRs of different dimensions")

    def __and__(self, o):
        self._same(o)
        return SDR._sorted(self.d, np.intersect1d(self.idx, o.idx, assume_unique=True))

    def __or__(self, o):
        self._same(o)
        return SDR._sorted(self.d, np.union1d(self.idx, o.idx))

    def __sub__(self, o):
        self._same(o)
        return SDR._sorted(self.d, np.setdiff1d(self.idx, o.idx, assume_unique=True))

    def __xor__(self, o):
        self._same(o)
        return SDR._sorted(self.d, np.setxor1d(self.idx, o.idx, assume_unique=True))

    def __contains__(self, bit):
        i = np.searchsorted(self.idx, bit)
        return bool(i < self.idx.size and self.idx[i] == bit)

    def __eq__(self, o):
        return isinstance(o, SDR) and o.d == self.d and np.array_equal(o.idx, self.idx)

    def __hash__(self):
        if self._h is None:
            self._h = hash((self.d, self.idx.tobytes()))
        return self._h

    def __repr__(self):
        head = ", ".join(map(str, self.idx[:8].tolist())) + (", ..." if self.w > 8 else "")
        return "SDR(d=%d, w=%d, [%s])" % (self.d, self.w, head)


# ---- similarity ------------------------------------------------------------------------------------------------------
def overlap(a, b):
    """Shared active bits: HTM's similarity."""
    a._same(b)
    return int(np.intersect1d(a.idx, b.idx, assume_unique=True).size)


def jaccard(a, b):
    u = a.w + b.w - overlap(a, b)
    return overlap(a, b) / u if u else 1.0


def match(a, b, theta):
    """Does a match b: at least theta shared bits (a dendritic segment's threshold)."""
    return overlap(a, b) >= theta


def chance_overlap(d, w1, w2=None):
    """The expected overlap of two random SDRs of w1 and w2 active bits."""
    return w1 * (w1 if w2 is None else w2) / d


# ---- making sparse codes --------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _priority(d):
    """A fixed random order of the bits, for breaking ties without favouring low indices (deterministic)."""
    p = np.random.default_rng(1234567).permutation(d)
    p.flags.writeable = False
    return p


def kwta(scores, k, floor=None):
    """k-winners-take-all: the k bits with the highest scores (ties broken by a fixed random order). With floor, only
    bits scoring above it may win, so fewer than k can."""
    s = np.asarray(scores, dtype=np.float64)
    d = s.size
    order = np.lexsort((_priority(d), -s))[:k]
    if floor is not None:
        order = order[s[order] > floor]
    return SDR(d, order)


def thin(a, w, rng):
    """A random subset of w of a's bits (a itself if it has no more than w)."""
    if a.w <= w:
        return a
    return SDR._sorted(a.d, np.sort(rng.choice(a.idx, size=w, replace=False)))


def union(sdrs):
    sdrs = list(sdrs)
    d = sdrs[0].d
    return SDR._sorted(d, np.unique(np.concatenate([s.idx for s in sdrs]))) if sdrs else None


def intersection(sdrs):
    sdrs = list(sdrs)
    out = sdrs[0]
    for s in sdrs[1:]:
        out = out & s
    return out


def counts(sdrs, weights=None, d=None):
    """How many of the SDRs (weighted) have each bit active."""
    sdrs = list(sdrs)
    d = d if d is not None else sdrs[0].d
    c = np.zeros(d)
    for i, s in enumerate(sdrs):
        c[s.idx] += 1.0 if weights is None else weights[i]
    return c


def bundle(sdrs, w, weights=None):
    """A sparse superposition: the w bits active in the most inputs (weighted). Each input overlaps the bundle well
    above chance, so one SDR holds a set; what the inputs share survives, what is particular to one is dropped first."""
    return kwta(counts(sdrs, weights), w, floor=0)


def cdt(sdrs, w, seed=0, max_perms=64):
    """Context-dependent thinning (Rachkovskij & Kussul 2001): a binding of free SDRs that keeps similarity. The union
    z of the inputs is ANDed with fixed random permutations of itself until w bits survive. The result depends on
    every input jointly (a bit survives because of the others), and similar input sets give similar results. It is not
    invertible; block codes (Blocks.bind) are."""
    z = union(sdrs)
    out = SDR.empty(z.d)
    for k in range(max_perms):
        out = out | (z & _cdt_perm(z.d, seed, k)(z))
        if out.w >= w:
            break
    if out.w > w:                                        # keep w bits, by the fixed order
        keep = np.argsort(_priority(z.d)[out.idx])[:w]
        out = SDR(z.d, out.idx[keep])
    return out


@lru_cache(maxsize=None)
def _cdt_perm(d, seed, k):
    return Permutation(d, np.random.default_rng((seed, k, 77)))


# ---- permutations ----------------------------------------------------------------------------------------------------
def shift(a, k):
    """a's bits shifted cyclically by k: a fixed permutation, for marking a role or a sequence position."""
    return SDR(a.d, (a.idx.astype(np.int64) + k) % a.d)


class Permutation:
    """A fixed random permutation of the d bits: bit i goes to p[i]."""

    def __init__(self, d, rng=None, p=None):
        self.p = np.asarray(p if p is not None else rng.permutation(d), dtype=np.int64)
        self.d = self.p.size

    def __call__(self, a):
        return SDR(self.d, self.p[a.idx])

    def inverse(self):
        return Permutation(self.d, p=np.argsort(self.p))

    def power(self, k):
        p = np.arange(self.d)
        base = self.p if k >= 0 else np.argsort(self.p)
        for _ in range(abs(k)):
            p = base[p]
        return Permutation(self.d, p=p)


# ---- block codes: SDRs that are coordinates -----------------------------------------------------------------------------
class Blocks:
    """Sparse block codes. The d bits form k blocks of L = d / k bits; a block code has exactly one active bit per block.
    Each block is a module, a torus of the given shape (prod(shape) = L), and the place of its active bit on that
    torus is the block's offset. Binding adds offsets (componentwise, mod the shape) and unbinding subtracts them, an
    invertible binding that keeps the code sparse. So a block code is a point on a product of tori: a coordinate
    system. power(base, x) multiplies the offsets by x, putting x on a ring in every block (fractional power encoding:
    the number line in polar coordinates), and bind(power(g, a), power(g, b)) = power(g, a + b): the first argument
    turns the hand to itself, the second turns it by itself.

    This is innate algebra, for innate frames (V1's retinotopy). Never use power(g, x) with x the value a symbol stands
    for: that hands an abstract domain its number line instead of letting it be learned (ARCHITECTURE.md section 0)."""

    def __init__(self, d, k, shape=None):
        if d % k:
            raise ValueError("d must be a multiple of the number of blocks")
        self.d, self.k, self.L = d, k, d // k
        self.shape = tuple(shape) if shape is not None else (self.L,)
        if int(np.prod(self.shape)) != self.L:
            raise ValueError("the block shape must hold L = d / k bits")
        self.dims = np.array(self.shape, np.int64)

    def is_code(self, a):
        return a.d == self.d and a.w == self.k and np.array_equal(a.idx // self.L, np.arange(self.k))

    def offsets(self, a):
        """(k, ndim) offsets of a block code."""
        if not self.is_code(a):
            raise ValueError("not a block code: one active bit per block is required")
        flat = a.idx.astype(np.int64) % self.L
        return np.stack(np.unravel_index(flat, self.shape), -1)

    def code(self, offs):
        """The block code with these (k, ndim) offsets (taken mod the shape)."""
        o = np.asarray(offs, np.int64).reshape(self.k, len(self.shape)) % self.dims
        flat = np.ravel_multi_index(tuple(o.T), self.shape)
        return SDR._sorted(self.d, np.arange(self.k) * self.L + flat)

    def random(self, rng):
        return self.code(np.stack([rng.integers(0, n, self.k) for n in self.shape], -1))

    def identity(self):
        return self.code(np.zeros((self.k, len(self.shape)), np.int64))

    def bind(self, a, b):
        return self.code(self.offsets(a) + self.offsets(b))

    def unbind(self, c, b):
        """The a with bind(a, b) = c."""
        return self.code(self.offsets(c) - self.offsets(b))

    def inverse(self, a):
        return self.code(-self.offsets(a))

    def power(self, a, x):
        """a's offsets times x (rounded, mod the shape): exact for integer x, a quantized phase for real x."""
        return self.code(np.rint(self.offsets(a) * x).astype(np.int64))

    def bundle(self, sdrs, weights=None):
        """A block code from several: in each block, the bit with the most votes (ties by the fixed order)."""
        c = counts(sdrs, weights, self.d).reshape(self.k, self.L)
        pr = _priority(self.d).reshape(self.k, self.L)
        best = np.array([np.lexsort((pr[i], -c[i]))[0] for i in range(self.k)])
        return SDR._sorted(self.d, np.arange(self.k) * self.L + best)

    def similarity(self, a, b):
        """The fraction of blocks that agree."""
        return overlap(a, b) / self.k


# ---- encoders: where data becomes SDRs -------------------------------------------------------------------------------
class CategoryEncoder:
    """A random SDR per category, the same every run (seeded by the category's text). Categories share nothing: the
    encoder gives no structure, so any structure among them must be learned. decode returns the seen category whose
    SDR overlaps most."""

    def __init__(self, d, w, seed=0):
        self.d, self.w, self.seed = d, w, seed
        self.memory, self.labels = ItemMemory(d), []

    def encode(self, label):
        h = int.from_bytes(hashlib.sha256(("%d|%r" % (self.seed, label)).encode()).digest()[:8], "little")
        s = SDR.random(self.d, self.w, np.random.default_rng(h))
        if label not in self.labels:
            self.labels.append(label)
            self.memory.add(s)
        return s

    def decode(self, sdr):
        i, _ = self.memory.nearest(sdr)
        return None if i is None else self.labels[i]


class ScalarEncoder:
    """A run of w bits whose start follows the value: near values overlap. periodic=True puts the run on a ring of d
    bits (hi wraps to lo): an angle, a phase."""

    def __init__(self, d, w, lo, hi, periodic=False):
        self.d, self.w, self.lo, self.hi, self.periodic = d, w, float(lo), float(hi), periodic

    def encode(self, x):
        f = (float(x) - self.lo) / (self.hi - self.lo)
        if self.periodic:
            start = int(np.floor(f * self.d)) % self.d
            return SDR(self.d, (start + np.arange(self.w)) % self.d)
        start = int(round(np.clip(f, 0, 1) * (self.d - self.w)))
        return SDR._sorted(self.d, start + np.arange(self.w))

    def decode(self, sdr):
        if sdr.w == 0:
            return None
        if self.periodic:
            ang = 2 * np.pi * sdr.idx / self.d
            c = np.angle(np.exp(1j * ang).mean()) / (2 * np.pi) * self.d - (self.w - 1) / 2
            return self.lo + (c % self.d) / self.d * (self.hi - self.lo)
        return self.lo + (sdr.idx.mean() - (self.w - 1) / 2) / (self.d - self.w) * (self.hi - self.lo)


class GridEncoder:
    """2-D places as block codes: grid-cell modules. Block i is an m x m torus (L = m * m); a place p's offset there is
    floor(m * (R_i p / s_i)) mod m, R_i a rotation and s_i the module's period. Periods grow geometrically (ratio
    about 1.4, as between grid modules), so near places agree in the coarse modules and differ in the fine ones:
    overlap falls with distance. This is a METRIC code.

    A quantized code can't be both metric and exactly algebraic. Binding the codes of two places (Blocks.bind) agrees
    with the code of their sum in only about a quarter of the blocks (the floors' carries are lost). For exact algebra
    over an integer lattice, use block codes of powers instead: bind(power(gx, x), power(gy, y)) is an exact group code
    of (x, y), but its distinct places share almost nothing. Which one L6 should use is an open question
    (ARCHITECTURE.md section 5)."""

    def __init__(self, d, k, s_min=2.0, ratio=1.4, seed=0):
        m = int(round(np.sqrt(d // k)))
        if m * m * k != d:
            raise ValueError("d / k must be a square number of bits")
        self.blocks = Blocks(d, k, (m, m))
        self.m = m
        rng = np.random.default_rng(seed)
        self.periods = s_min * ratio ** (np.arange(k) % 12)          # 12 scales, repeated with new orientations
        th = rng.uniform(0, np.pi / 3, k)
        self.R = np.stack([np.stack([np.cos(th), -np.sin(th)], -1), np.stack([np.sin(th), np.cos(th)], -1)], -2)

    def phases(self, p):
        """(k, 2) phases in [0, 1): where p falls in each module's period."""
        q = np.einsum("kij,j->ki", self.R, np.asarray(p, float)) / self.periods[:, None]
        return q % 1.0

    def encode(self, p):
        return self.blocks.code(np.floor(self.phases(p) * self.m).astype(np.int64))


# ---- memories --------------------------------------------------------------------------------------------------------
class ItemMemory:
    """Cleanup memory: stored SDRs, recalled by overlap. An inverted index (bit -> the items that have it) makes a query
    cost the postings of its active bits, not a scan over items."""

    def __init__(self, d):
        self.d, self.items, self.post = d, [], [[] for _ in range(d)]

    def __len__(self):
        return len(self.items)

    def add(self, sdr):
        i = len(self.items)
        self.items.append(sdr)
        for b in sdr.idx.tolist():
            self.post[b].append(i)
        return i

    def overlaps(self, q):
        """Every stored item's overlap with q."""
        if not self.items:
            return np.zeros(0, np.int64)
        hits = [self.post[b] for b in q.idx.tolist() if self.post[b]]
        if not hits:
            return np.zeros(len(self.items), np.int64)
        return np.bincount(np.concatenate([np.asarray(h, np.int64) for h in hits]), minlength=len(self.items))

    def nearest(self, q, k=1):
        """The index and overlap of the stored item nearest q (k > 1: arrays of the k nearest, best first)."""
        o = self.overlaps(q)
        if o.size == 0:
            return (None, 0) if k == 1 else (np.zeros(0, int), np.zeros(0, int))
        if k == 1:
            i = int(np.argmax(o))
            return i, int(o[i])
        top = np.argsort(-o, kind="stable")[:k]
        return top, o[top]


class Synapses:
    """HTM permanences: n_post cells (or segments), each with potential synapses to n_pre input bits. A synapse is
    connected when its permanence is at least `connected`. Hebbian learning on a cell: +inc on synapses from active
    inputs, -dec on synapses from inactive ones, clipped to [0, 1]."""

    def __init__(self, n_post, n_pre, potential=1.0, connected=0.5, inc=0.03, dec=0.015, spread=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.connected, self.inc, self.dec = connected, inc, dec
        self.pool = rng.random((n_post, n_pre)) < potential
        self.perm = np.clip(connected + rng.uniform(-spread, spread, (n_post, n_pre)), 0, 1).astype(np.float32)
        self.perm *= self.pool

    def overlaps(self, pre):
        """For each post cell, its connected synapses from pre's active bits."""
        sub = self.perm[:, pre.idx]
        return ((sub >= self.connected) & self.pool[:, pre.idx]).sum(1)

    def learn(self, pre, post):
        """Hebbian update of the post cells (indices) toward the pattern pre."""
        post = np.asarray(post, np.int64)
        if post.size == 0:
            return
        x = pre.dense(bool)
        rows = self.perm[post]
        rows += np.where(x, self.inc, -self.dec).astype(np.float32)[None, :]
        np.clip(rows, 0, 1, out=rows)
        rows *= self.pool[post]
        self.perm[post] = rows

    def bump(self, post, amount):
        """Raise every potential permanence of the post cells (reviving a cell that never wins)."""
        post = np.asarray(post, np.int64)
        self.perm[post] = np.clip(self.perm[post] + amount * self.pool[post], 0, 1)


class SpatialPooler:
    """HTM's learned encoder (Cui, Ahmad & Hawkins 2017): an input SDR to an output SDR of d bits with w active, which
    keeps the inputs' overlaps and spreads use over all output bits. Each step: overlaps of connected synapses; a
    stimulus threshold; boosting by how under-used each bit is (homeostasis); k-winners-take-all; Hebbian learning on
    the winners; duty cycles, and a bump for bits that are never even stimulated."""

    def __init__(self, n_in, d, w, potential=0.5, connected=0.5, inc=0.03, dec=0.015, stimulus=1, boost=2.0,
                 duty=0.01, min_duty=0.001, seed=0):
        self.syn = Synapses(d, n_in, potential, connected, inc, dec, seed=seed)
        self.d, self.w, self.stimulus, self.boost, self.duty = d, w, stimulus, boost, duty
        self.target, self.min_duty = w / d, min_duty
        self.active_duty = np.full(d, w / d)
        self.overlap_duty = np.ones(d)

    def compute(self, x, learn=True):
        o = self.syn.overlaps(x).astype(np.float64)
        o[o < self.stimulus] = 0
        o = o * np.exp(self.boost * (self.target - self.active_duty))   # frozen when not learning (duty is)
        out = kwta(o, self.w, floor=0)
        if learn:
            self.syn.learn(x, out.idx)
            a = np.zeros(self.d)
            a[out.idx] = 1
            self.active_duty += self.duty * (a - self.active_duty)
            self.overlap_duty += self.duty * ((o > 0) - self.overlap_duty)
            dead = np.flatnonzero(self.overlap_duty < self.min_duty)
            if dead.size:
                self.syn.bump(dead, 0.1 * self.syn.connected)
        return out
