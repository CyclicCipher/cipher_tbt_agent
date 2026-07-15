"""SDR encoders/decoders — the PERIPHERAL's transduction library. Salvaged 2026-07-09, unchanged; see `ARCHITECTURE.md`
(the boundary where data becomes SDRs) / `STATUS.md`. (Docstrings below reference the legacy plan doc-sections — historical.)

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
    """A discrete, UNORDERED category -> a DISJOINT block of `w` bits (zero overlap between categories: the labels have
    no "nearness"). Grows online up to `capacity` (fixed length). `decode` = the category whose block the SDR most
    overlaps. For ARBITRARY symbols: ARC's 16 palette INDICES (symbolic — a task cares about which cells SHARE a colour,
    not colour similarity; assuming colour-3 ≈ colour-4 would inject a false prior, the bitter lesson) and the discrete
    action set. NB a CONTINUOUS colour space (RGB) is NOT categorical — its points have metric relations; encode it with
    per-channel `ScalarEncoder`s combined by a `MultiEncoder` so similar colours OVERLAP. Category is a deliberate
    per-datatype choice (ARC's palette is unordered), not colour's universal encoding."""

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

    def modules(self):
        """The index groups of the independent MODULES (each (scale, axis) ring) — the block structure a block-diagonal
        operator learns over (each small ring is a continuous-attractor; Burak & Fiete). A partition of `0..n`."""
        return [list(range(int(self._off[k]), int(self._off[k + 1]))) for k in range(len(self._off) - 1)]

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


class ConjunctiveEncoder(Encoder):
    """The TENSOR-PRODUCT (conjunctive) of sub-encoders — e.g. grid × head-direction cells, the SE(2) representation
    space (`reference_operator_as_group_representation`). `encode` = the OUTER PRODUCT of the sub-SDRs, flattened: a cell
    is active iff EVERY field is active there (mixed-radix index). This is the code a heading-DEPENDENT action (FORWARD)
    acts on LINEARLY (an orthogonal permutation), so a learned operator `M(action)` can represent the non-abelian group —
    which a `MultiEncoder` concatenation CANNOT. `n` = ∏ fields' n; `w` = ∏ fields' w. `decode` projects back per field."""

    def __init__(self, fields) -> None:
        self.fields = list(fields)                             # [(name, encoder), ...]
        self.n = 1
        for _, e in self.fields:
            self.n *= e.n

    def encode(self, values: dict) -> SDR:
        combos = [0]                                           # flatten (a0,a1,...) -> a0·(n1·n2…) + a1·(n2…) + …
        for name, e in self.fields:
            acts = e.encode(values[name]).active
            combos = [c * e.n + a for c in combos for a in acts]
        return SDR(self.n, combos)

    def modules(self):
        """The block structure for a block-diagonal operator on the tensor code: each module of the FIRST (base) field
        crossed with ALL of the coupled factors — so a base-shift CONDITIONED on the others (e.g. heading-dependent
        FORWARD) stays within one block. A partition of the code indices."""
        _name0, e0 = self.fields[0]
        rest_n = self.n // e0.n
        base = e0.modules() if hasattr(e0, "modules") else [list(range(e0.n))]
        return [[p * rest_n + r for p in mod for r in range(rest_n)] for mod in base]

    def module_grids(self):
        """Each base module as a 2-TORUS index grid `(s_base × rest_n)`: `grid[i, j]` = the flat conjunctive index of the
        i-th base-ring cell crossed with rest-index j. Axis 0 = the base (location) phase (cyclic within the ring); axis
        1 = the coupled rest phase (e.g. heading, cyclic). This is the structure the `ModularOperator` learns a
        heading-conditioned shift over — a base shift that MAY DEPEND on the rest coordinate (the semidirect / non-abelian
        coupling), which the flat `modules()` block cannot express. A partition of the code indices (as 2-D grids)."""
        _name0, e0 = self.fields[0]
        rest_n = self.n // e0.n
        base = e0.modules() if hasattr(e0, "modules") else [list(range(e0.n))]
        return [np.array([[p * rest_n + r for r in range(rest_n)] for p in mod]) for mod in base]

    def decode(self, sdr: SDR) -> dict:
        ns = [e.n for _, e in self.fields]
        per_field = [set() for _ in self.fields]              # project the flat indices back to each field's active set
        for flat in sdr.active:
            rem, coords = flat, []
            for n in reversed(ns):
                coords.append(rem % n)
                rem //= n
            for k, c in enumerate(reversed(coords)):
                per_field[k].add(c)
        return {name: e.decode(SDR(e.n, acts)) for (name, e), acts in zip(self.fields, per_field)}


class SpatialPooler:
    """The LEARNED general encoder (HTM spatial pooler): a raw input (binary vector / SDR) -> a fixed-sparsity output SDR
    of active MINICOLUMNS that preserves semantic overlap, learned online. Use when no analytic encoder fits
    (ARCHITECTURE §P5); it is also the FEED-FORWARD PROXIMAL stage the L2/3 column pooler reuses (MICROCIRCUIT §7 —
    ONE spatial pooler, rule 1). Deterministic with `learn=False` (duty cycles frozen → boosting frozen). Its inverse is
    a learned classifier, not an analytic decode → it is a SENSORY encoder; MOTOR decoding uses the analytic encoders above.

    **The canonical HTM SP (Cui, Ahmad & Hawkins 2017, *The HTM Spatial Pooler*; `notes/htm notes.txt` 97–119). Per step,
    the order is BOOST → INHIBIT → LEARN:**
      1. **overlap** — each column's connected synapses to the active input: `overlap = ((perm ≥ connected) & potential) @ x`;
      2. **stimulus threshold** — a column needs `overlap ≥ stimulus_threshold` connected active inputs to be eligible
         (below it contributes nothing — noise rejection);
      3. **boosting** (homeostasis, BEFORE inhibition) — scale the eligible overlaps by `exp(boost·(target_density −
         active_duty))`, so UNDER-used columns are favoured and no clique hogs the code (notes 108/112);
      4. **inhibition** — k-winners-take-all: the top `w` boosted-overlap columns fire (global inhibition = one
         neighbourhood; topological receptive fields are a deferred refinement);
      5. **learning** (Hebbian, winners only) — on each winner's POTENTIAL synapses: `+syn_inc` where the input bit is 1,
         `−syn_dec` where it is 0 (SEPARATE rates), clipped to [0,1]; the pattern is gradually carved into the permanences;
      6. **duty cycles + dead-column revival** — track `active_duty` (drives boosting) and `overlap_duty` (fraction of
         steps a column cleared the stimulus threshold); any column whose `overlap_duty` falls below `min_overlap_duty`
         has ALL its potential permanences bumped up, so a permanently-silent column re-connects and joins the code
         (notes 112 — required, else only a few columns ever contribute).

    **Data types:** `potential` bool `[n_cols × n_inputs]` (the fixed potential pool — which inputs a column MAY see);
    `perm` float `[n_cols × n_inputs]` ∈ [0,1] (a synapse is CONNECTED iff `perm ≥ connected` AND in the pool);
    `active_duty`/`overlap_duty` float `[n_cols]` (EMA homeostatic traces); output = an `SDR(n_cols, winners)` of fixed
    sparsity `w`. It pools over SPACE (input bits at ONE instant) → the complement of the column pooler's pooling over
    TIME (MICROCIRCUIT §7)."""

    def __init__(self, n_inputs: int, n_cols: int = 1024, w: int = 40, potential: float = 0.5,
                 connected: float = 0.5, syn_inc: float = 0.03, syn_dec: float = 0.015,
                 stimulus_threshold: int = 1, boost: float = 2.0, duty_alpha: float = 0.01,
                 min_overlap_duty: float = 0.01, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.n, self.w = int(n_cols), int(w)
        self.connected, self.syn_inc, self.syn_dec = float(connected), float(syn_inc), float(syn_dec)
        self.stimulus_threshold, self.boost, self.duty_alpha = int(stimulus_threshold), float(boost), float(duty_alpha)
        self.min_overlap_duty = float(min_overlap_duty)
        self.target = self.w / self.n                                    # target active-duty (fair share of the code)
        self.pool = rng.random((n_cols, n_inputs)) < potential           # POTENTIAL pool (fixed): inputs a column may see
        self.perm = rng.random((n_cols, n_inputs)) * self.pool           # permanences (connected when ≥ `connected`)
        self.active_duty = np.full(n_cols, self.target)                  # start fair → no initial boost bias
        self.overlap_duty = np.ones(n_cols)                             # start "alive" (revival only fires on real silence)

    def encode(self, x, learn: bool = True) -> SDR:
        x = (np.asarray(x, dtype=float) > 0).astype(float)
        overlap = ((self.perm >= self.connected) & self.pool) @ x        # (1) connected synapses to the active input
        eligible = overlap >= self.stimulus_threshold                   # (2) stimulus threshold
        score = overlap * np.exp(self.boost * (self.target - self.active_duty)) * eligible   # (3) boosting, before inhibition
        winners = np.argpartition(score, -self.w)[-self.w:]             # (4) k-winners-take-all inhibition
        winners = winners[score[winners] > 0]
        if learn:
            if len(winners):                                            # (5) Hebbian on winners' potential synapses
                self.perm[winners] += (self.syn_inc * x - self.syn_dec * (1 - x)) * self.pool[winners]
                np.clip(self.perm, 0.0, 1.0, out=self.perm)
            a = self.duty_alpha                                         # (6) duty cycles (EMA) + dead-column revival
            fired = np.zeros(self.n); fired[winners] = 1.0
            self.active_duty = (1 - a) * self.active_duty + a * fired
            self.overlap_duty = (1 - a) * self.overlap_duty + a * eligible.astype(float)
            weak = self.overlap_duty < self.min_overlap_duty            # columns that never clear the stimulus threshold
            if weak.any():
                self.perm[weak] += 0.1 * self.connected * self.pool[weak]
                np.clip(self.perm, 0.0, 1.0, out=self.perm)
        return SDR(self.n, winners)
