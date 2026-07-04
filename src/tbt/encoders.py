"""SDR encoders/decoders — the PERIPHERAL's transduction library (ARCHITECTURE.md §4 peripherals, §P5).

The ONE place a raw value becomes the model's currency, the **SDR**, and the ONE place a motor "thought" (an SDR
emitted by L5) becomes an effector command. Everything in the cortex is an SDR (HTM); the boundary to the world is
where SDRs are made from, and turned back into, raw quantities. This module is that boundary — nothing here is
cognition (rules 4–5), only transduction and effection.

**The anti-parallel-system principle (why encode and decode live together).** A decoder is NOT a separate system — it
is the encoder's INVERSE. Each `Encoder` is BIDIRECTIONAL: `encode(value) -> SDR` and `decode(SDR) -> value` are the two
directions of one object. So a motor command is interpreted by the SAME encoder that would sense that quantity:

  * a discrete action  — `CategoryEncoder(actions).decode(motor_sdr) -> action`   (ARC ACTION1-4)
  * a click / coordinate — `GridEncoder.decode(target_sdr) -> (x, y)`             (ARC ACTION6)
  * a continuous effector — `ScalarEncoder.decode(command_sdr) -> scalar`         (a joint angle / force)

There is one representation across sense and act, so no separate motor-decoder can fester (rule 1). L5 emits the SDR
"thought"; the motor ORGAN calls `encoder.decode` on it; the loop is closed through one code.

**The three SDR rules** (Ahmad & Hawkins, *Properties of SDRs*; Purdy, *Encoding Data for HTM Systems*):
  1. semantic **OVERLAP = similarity** — near inputs share active bits (the property the localist `state_node` and the
     exact-match `L4.encode` / `retina.view_signature` LACK — they retire onto this, P5);
  2. **DETERMINISM** — the same input always gives the same SDR;
  3. fixed **LENGTH** + **SPARSITY** — every SDR is `n` bits with a small active fraction `w`.

Analytic encoders (Scalar/Periodic/Category/Grid/Multi) are invertible -> usable for MOTOR decode. The learned
`SpatialPooler` is the general SENSORY encoder when no analytic one fits; its inverse is a learned classifier, so it is
not used for motor decode. Pure numpy + stdlib (torch-free, runs in the test venv).
"""

from __future__ import annotations

import numpy as np


class SDR:
    """A sparse distributed representation: `n` bits with a set of `active` indices. Overlap = shared active bits."""

    __slots__ = ("n", "active")

    def __init__(self, n: int, active) -> None:
        self.n = int(n)
        self.active = frozenset(int(i) for i in active)

    @property
    def w(self) -> int:
        """The number of active bits (the sparsity)."""
        return len(self.active)

    def dense(self) -> np.ndarray:
        """The `n`-length binary vector."""
        a = np.zeros(self.n, dtype=np.uint8)
        if self.active:
            a[list(self.active)] = 1
        return a

    def overlap(self, other: "SDR") -> int:
        """Shared active bits with another SDR — the HTM similarity measure."""
        return len(self.active & other.active)

    @staticmethod
    def concat(sdrs) -> "SDR":
        """Lay SDRs side by side into one (the union of shifted bit-fields) — how composite codes are built."""
        active, off = set(), 0
        for s in sdrs:
            active |= {i + off for i in s.active}
            off += s.n
        return SDR(off, active)

    def __eq__(self, o) -> bool:
        return isinstance(o, SDR) and self.n == o.n and self.active == o.active

    def __hash__(self) -> int:
        return hash((self.n, self.active))

    def __repr__(self) -> str:
        return f"SDR(n={self.n}, w={self.w}, active={sorted(self.active)})"


class Encoder:
    """Bidirectional transducer: `encode(value) -> SDR` and its inverse `decode(SDR) -> value`. Subclasses set `n`."""

    n: int

    def encode(self, value) -> SDR:
        raise NotImplementedError

    def decode(self, sdr: SDR):
        raise NotImplementedError


class ScalarEncoder(Encoder):
    """A bounded scalar -> a contiguous window of `w` active bits, positioned by the value. Nearby values overlap
    (rule 1). `periodic=True` wraps the window (for a cyclic quantity — an angle/heading; 0 and max overlap).
    `decode` inverts to the value (the window's centre; a circular mean when periodic)."""

    def __init__(self, minval: float, maxval: float, n: int = 100, w: int = 11, periodic: bool = False,
                 clip: bool = True) -> None:
        assert 0 < w < n and maxval > minval
        self.minv, self.maxv, self.n, self.w = float(minval), float(maxval), int(n), int(w)
        self.periodic, self.clip = periodic, clip
        self.nbuckets = n if periodic else n - w + 1

    def _frac(self, v: float) -> float:
        f = (v - self.minv) / (self.maxv - self.minv)
        return f % 1.0 if self.periodic else (min(max(f, 0.0), 1.0) if self.clip else f)

    def encode(self, value) -> SDR:
        f = self._frac(float(value))
        if self.periodic:
            c, half = int(round(f * self.n)) % self.n, self.w // 2       # window CENTRED on the phase (decode reads its centre)
            return SDR(self.n, {(c - half + k) % self.n for k in range(self.w)})
        start = int(round(f * (self.nbuckets - 1)))
        return SDR(self.n, range(start, start + self.w))

    def decode(self, sdr: SDR):
        if not sdr.active:
            return None
        if self.periodic:
            ang = np.array([2 * np.pi * i / self.n for i in sdr.active])
            mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)
            return self.minv + (mean / (2 * np.pi)) * (self.maxv - self.minv)
        start = float(np.mean(list(sdr.active))) - (self.w - 1) / 2.0     # window centre -> its start bucket
        return self.minv + (start / (self.nbuckets - 1)) * (self.maxv - self.minv)


class CategoryEncoder(Encoder):
    """A discrete category -> a DISJOINT block of `w` bits (zero overlap between categories: a category has no
    "nearness", rule 1's degenerate case). Grows online up to `capacity` (fixed length). `decode` = the category
    whose block the SDR most overlaps. The natural encoder for colour (ARC's 16) and the discrete action set."""

    def __init__(self, categories=(), w: int = 11, capacity: int = 32) -> None:
        self.w, self.capacity = int(w), int(capacity)
        self.cats: dict = {}
        for c in categories:
            self.add(c)

    @property
    def n(self) -> int:
        return self.w * self.capacity

    def add(self, category) -> int:
        if category not in self.cats:
            if len(self.cats) >= self.capacity:
                raise ValueError("CategoryEncoder capacity exceeded")
            self.cats[category] = len(self.cats)
        return self.cats[category]

    def _block(self, i: int):
        return set(range(i * self.w, (i + 1) * self.w))

    def encode(self, category) -> SDR:
        return SDR(self.n, self._block(self.add(category)))

    def decode(self, sdr: SDR):
        if not sdr.active or not self.cats:
            return None
        best, best_ov = None, 0
        for c, i in self.cats.items():
            ov = len(sdr.active & self._block(i))
            if ov > best_ov:
                best, best_ov = c, ov
        return best


class GridEncoder(Encoder):
    """The LOCATION encoder (ARCHITECTURE §2 L6): an N-D coordinate -> a grid-cell SDR. Per (scale, axis) a ring of
    `scale` cells with a WINDOW of `mw` cells (a bump) active around the coordinate's PHASE; the SDR is the union
    (w = mw x scales x dims active bits). The bump is what gives GRADED metric overlap (nearby phases share window
    cells; a single active cell would give only a step function). It obeys all three SDR rules AND two more the
    location needs: PATH-INTEGRABLE (a translation shifts each phase, additive mod scale) and DECODABLE (co-prime
    scales -> a unique position within lcm(scales), read per axis by argmax overlap). The sparse, decodable form of
    the `hex_code` prior; the a-priori metric overlap the localist `state_node` lacks (§10 P4a, the SDR test)."""

    def __init__(self, scales=(7, 11, 13, 17), dims: int = 2, mw: int = 3, bounds=None) -> None:
        self.scales = tuple(int(s) for s in scales)
        assert mw < min(self.scales), "the window must be smaller than the smallest ring"
        self.dims, self.mw = int(dims), int(mw)
        self.bounds = bounds                                    # per-dim (lo, hi) for decode; default 0..lcm-1
        self._off = np.cumsum([0] + [s for s in self.scales for _ in range(self.dims)])   # bit offset per module
        self.n = int(self._off[-1])

    def _module(self, k: int, axis: int) -> int:
        """Bit offset of the module for the k-th scale, given axis."""
        return int(self._off[k * self.dims + axis])

    def _window(self, base: int, coord_axis: float, s: int):
        """The active ring cells (the bump) for a coordinate on one (scale, axis) module."""
        cell, half = int(round(coord_axis)) % s, self.mw // 2
        return {base + (cell - half + j) % s for j in range(self.mw)}

    def encode(self, coord) -> SDR:
        c = np.atleast_1d(np.asarray(coord, dtype=float))
        active = set()
        for k, s in enumerate(self.scales):
            for axis in range(self.dims):
                active |= self._window(self._module(k, axis), c[axis], s)
        return SDR(self.n, active)

    def decode(self, sdr: SDR, bounds=None):
        bounds = bounds or self.bounds or [(0, int(np.lcm.reduce(self.scales)) - 1)] * self.dims
        out = []
        for axis in range(self.dims):
            lo, hi = bounds[axis]
            best, best_ov = int(lo), -1
            for x in range(int(lo), int(hi) + 1):
                ov = sum(len(self._window(self._module(k, axis), x, s) & sdr.active)
                         for k, s in enumerate(self.scales))
                if ov > best_ov:
                    best, best_ov = x, ov
            out.append(best)
        return tuple(out)


class MultiEncoder(Encoder):
    """Concatenate sub-encoders into one composite SDR (feature-at-location = e.g. colour (category) x location
    (grid)). `encode` takes a `{name: value}` dict and lays the fields side by side; `decode` splits the SDR back
    into `{name: value}` by field. The one way to build/read a conjunctive SDR."""

    def __init__(self, fields) -> None:
        self.fields = list(fields)                             # [(name, encoder), ...]
        self.n = sum(e.n for _, e in self.fields)

    def encode(self, values: dict) -> SDR:
        active, off = set(), 0
        for name, e in self.fields:
            active |= {i + off for i in e.encode(values[name]).active}
            off += e.n
        return SDR(self.n, active)

    def decode(self, sdr: SDR) -> dict:
        out, off = {}, 0
        for name, e in self.fields:
            sub = SDR(e.n, {i - off for i in sdr.active if off <= i < off + e.n})
            out[name] = e.decode(sub)
            off += e.n
        return out


class SpatialPooler:
    """The LEARNED general encoder (HTM spatial pooler): a raw input (binary vector / SDR) -> a stable output SDR with
    semantic overlap, learned online. k-winners-take-all over random potential synapses + Hebbian permanence + boosting
    of under-used columns. Use when no analytic encoder fits (ARCHITECTURE §P5). Deterministic with `learn=False`.
    NB its inverse is a learned classifier, not an analytic decode -> it is a SENSORY encoder; MOTOR decoding uses the
    analytic (invertible) encoders above."""

    def __init__(self, n_inputs: int, n_cols: int = 1024, w: int = 40, potential: float = 0.5, lr: float = 0.05,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.n, self.w, self.lr = int(n_cols), int(w), float(lr)
        self.conn = rng.random((n_cols, n_inputs)) < potential           # each column's potential synapse mask
        self.perm = rng.random((n_cols, n_inputs)) * self.conn           # permanences (connected when > 0.5)
        self.duty = np.zeros(n_cols)                                      # active-duty cycle (drives boosting)

    def encode(self, x, learn: bool = True) -> SDR:
        x = (np.asarray(x, dtype=float) > 0).astype(float)
        boost = np.exp(0.1 * (self.duty.mean() - self.duty))
        overlap = ((self.perm > 0.5) & self.conn) @ x * boost
        winners = np.argpartition(overlap, -self.w)[-self.w:]
        winners = winners[overlap[winners] > 0]
        if learn and len(winners):
            self.perm[winners] += self.lr * (2 * x - 1) * self.conn[winners]     # Hebbian: reward connected active inputs
            np.clip(self.perm, 0.0, 1.0, out=self.perm)
            self.duty *= 0.99
            self.duty[winners] += 0.01
        return SDR(self.n, winners)
