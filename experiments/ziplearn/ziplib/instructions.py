"""ziplib.instructions — the instruction set: weight TEMPLATES one head, one MLP row, one route query or the boundary
operator executes exactly over one-hot codes (DESIGN §21.3; the module spec's `instructions.py`).

An instruction has three methods, all classmethods (an instruction is a kind, a circuit is an instance = `params`):

  needs(params, layout, arch)              -> Needs(heads, hd_min, content_dims, position_pairs, rows)   what the compiler must place
  emit(params, layout, arch, slot, M)      -> [Write]   the weights, as (tensor name, index, value) triples at the addresses
                                                        e28.py lines 96-132 and e18.py lines 54-77 write today:
                                                          blocks.{L}.attn.qkv.weight[Q|K|V + h*hd + c, dim]   Q, K, V = 0, d, 2d
                                                          blocks.{L}.attn.proj.weight[dim, h*hd + c]
                                                          blocks.{L}.mlp.0.weight[r, dim] / mlp.0.bias[r] / blocks.{L}.mlp.2.weight[dim, r]
                                                          blocks.{L}.res_attn.w[dim] | res_mlp.w[dim]      (Branch)
                                                          head.weight[v, dim]                              (Readout)
  test(params, layout, brain, n=256, slot) -> Report    the unit test `brainbuilder.verify` runs: argmax on the intended target for
                                                        100% of the queries, the smallest target-minus-runner-up logit gap >= M, the
                                                        written subspace equal to the expected value to `tol`.

The set (DESIGN §21.3): Gather, Match, Pool, Broadcast (heads); Row, Compare (MLP units); Branch (a route query); Readout
(the unembedding); and the boundary PARAMETERS Keep / Clear / Quantise / Commit / Anchor / Halt, which own no weights and fill
`h1_lid.BoundaryOp` (`boundary_op`). `INSTRUCTIONS` is the whitelist `Blueprint.validate` checks; nothing outside it compiles.
The E29 rule: an instruction names operations and subspaces, never a colour, a character or an offset picked for a game.

What MOVES here (the old location is deleted at the cutover): `WrittenSim.__init__` lines 97-115 (Gather, onehot), 117-132
(Match), e18.py `write` lines 56-59 (Gather, rope: the bias-channel q/k with key phases), 66-75 (Match in the lowest rotary
pairs), 77 (Readout). e5.py's gates are Python lambdas (no tensors), so `Compare` is written from §21.3 item 6's formula.

The `layout` argument is a `Layout`, or the `Codec` (its layout PLUS the token classes): Gather, Match, Pool and Broadcast
need every class flag to sink the tokens of the other classes (E28: entries and the two nulls attend to the zero token, so
nothing is written into them). With a bare `Layout` the classes come from `params["classes"]` (class name -> flag name);
absent both, no sink is written. `Slot` is the placement the compiler resolved (layer, head, hd, d, MLP row, rotary channels,
register slot ids). `M` is the sharpness `compile` derives (`ln((max_len - 1) p* / (1 - p*))`); `params["M"]` overrides it
per circuit -- an operation constant like `Readout.M_out`, needed because e18.write used M = 3 (position), 2 (content), 10 (read-out).

Sharpness conventions, exactly E28's: a query dim carries `weight * M`, a key dim carries `M`, so one matched dim scores M^2
(the substrate then scales by 1/sqrt(hd)); a value/proj copy carries 1.0.
"""
from __future__ import annotations

import math
import numbers
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from .layout import Layout, LayoutError, ChannelPlan, channels as _channels

POS_CODES = ("onehot", "rope", "rope2d")          # "learned" (the Attn name for E28's coordinates in the residual) = "onehot"
GATHER_GAP_MIN = 1.0                              # rope Gather: the compiler raises the pairs until M * gap >= this (§21.3 item 1)
MATCH_DRIFT_MAX = 0.1                             # rope Match: max_len * theta_c on a content pair should stay below this (§21.3 item 2)


class InstructionError(ValueError):
    """A template cannot be written: an unknown subspace, a head too narrow, a profile that cannot reach the gap, ..."""


def _h1():
    """The substrate, imported lazily (only the rotary frequencies `_freqs` and `BoundaryOp` are needed here)."""
    try:
        import h1_lid  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers"))
        import h1_lid  # noqa: F401
    return sys.modules["h1_lid"]


# =====================================================================================================================
# the data
# =====================================================================================================================

@dataclass(frozen=True)
class Write:
    """One written number: `tensor` is a state_dict name, `index` the element's index tuple, `value` the number."""
    tensor: str
    index: tuple
    value: float


@dataclass
class Needs:
    """What one circuit asks the compiler for: `heads` (0 or 1), `hd_min` (the narrowest head that holds it), `content_dims`
    (q/k channels that carry content -- one rotary pair each under RoPE), `position_pairs` (rotary pairs per axis that carry
    position), `rows` (MLP hidden units). `whole_window`: the Match spans several subspaces, so the compiler gives it a
    one-head layer with hd = d (E28's `lookup = Attn(d, 1)`; §21.4 step 4)."""
    heads: int = 0
    hd_min: int = 0
    content_dims: int = 0
    position_pairs: int = 0
    rows: int = 0
    whole_window: bool = False


@dataclass
class Slot:
    """Where the compiler put a circuit. `layer`, `head`; `hd` (head width; None = a one-head layer, hd = d); `d` (None = the
    layout's d); `row` (the first MLP hidden unit, Row/Compare); `channels` (the rotary pairs of a RoPE head, from
    `Layout.channels`; None = the instruction's own default); `registers` (register name -> slot id in the `slot` subspace,
    Pool/Broadcast); `n_zero` (rope2d: the layer's position-free tail pairs, `h1_lid.rope_theta`); `name` (the circuit's
    name, for errors)."""
    layer: int = 0
    head: int = 0
    hd: int | None = None
    d: int | None = None
    row: int | None = None
    channels: ChannelPlan | None = None
    registers: dict[str, int] | None = None
    n_zero: int = 0
    name: str | None = None


@dataclass
class Report:
    """A unit test's result. `passed` = argmax_acc == 1.0 and min_gap >= M and subspace_err <= tol."""
    name: str
    passed: bool
    n: int = 0
    argmax_acc: float = float("nan")
    min_gap: float = float("nan")
    M: float = float("nan")
    subspace_err: float = float("nan")
    tol: float = 1e-3
    note: str = ""
    extra: dict = field(default_factory=dict)


# =====================================================================================================================
# helpers: the layout / codec / arch views, the addresses
# =====================================================================================================================

def _split(layout_or_codec, params):
    """(Layout, {class: flag} | None, codec | None) from a Layout or a Codec."""
    if isinstance(layout_or_codec, Layout):
        classes = params.get("classes") if isinstance(params, dict) else None
        return layout_or_codec, (dict(classes) if classes else None), None
    codec = layout_or_codec
    layout = getattr(codec, "layout", None)
    if not isinstance(layout, Layout):
        raise InstructionError(f"layout must be a Layout or a Codec, got {type(layout_or_codec).__name__}")
    classes = {name: c.flag for name, c in codec.classes.items()}
    if isinstance(params, dict) and params.get("classes"):
        classes = dict(params["classes"])
    return layout, classes, codec


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pos(arch, codec):
    pos = _get(arch, "pos") or (getattr(codec, "pos", None) if codec is not None else None) or "onehot"
    if pos == "learned":
        pos = "onehot"
    if pos not in POS_CODES:
        raise InstructionError(f"pos must be one of {POS_CODES}, got {pos!r}")
    return pos


def _dims(arch):
    return dict(_get(arch, "dims", None) or {})


def _int_or_symbol(value, arch, what):
    """An int, or a symbol bound from `arch.dims` (`"V"`, `"nA"`)."""
    if isinstance(value, bool) or value is None:
        raise InstructionError(f"{what}: expected an int or a dims symbol, got {value!r}")
    if isinstance(value, int):
        return value
    dims = _dims(arch)
    if isinstance(value, str) and value in dims:
        return int(dims[value])
    raise InstructionError(f"{what}: {value!r} is neither an int nor a symbol of arch.dims {sorted(dims)}")


def derived_M(arch):
    """`M = ln((max_len - 1) p* / (1 - p*))` (§21.4 step 5) from the arch; None when the arch has no max_len."""
    T = _get(arch, "max_len")
    if T is None:
        return None
    p = float(_get(arch, "p_star", 0.99) or 0.99)
    return math.log((int(T) - 1) * p / (1.0 - p))


def _M(params, arch, M=None):
    """The sharpness a circuit is written at: `params["M"]` > the `M` argument > the arch's derived M."""
    if isinstance(params, dict) and params.get("M") is not None:
        return float(params["M"])
    if M is not None:
        return float(M)
    m = derived_M(arch)
    if m is None:
        raise InstructionError("no sharpness: pass M, or params['M'], or an arch with max_len")
    return m


def _test_M(params, brain):
    """The M a unit test compares the logit gap with: params["M"], else brain.M, else the brain's arch's derived M."""
    if params.get("M") is not None:
        return float(params["M"])
    if getattr(brain, "M", None) is not None:
        return float(brain.M)
    return _M(params, getattr(brain, "arch", None), None)


def _slot(slot, layout):
    """Resolve the slot's d and hd against the layout (hd None = a one-head layer)."""
    slot = slot if slot is not None else Slot()
    d = slot.d if slot.d is not None else layout.d
    hd = slot.hd if slot.hd is not None else d
    if hd <= 0 or hd > d:
        raise InstructionError(f"slot {slot.name or ''}: hd = {hd} must be in 1..d = {d}")
    return slot, d, hd


def _who(slot, cls_name):
    return f"{cls_name}{'' if slot is None or slot.name is None else ' ' + repr(slot.name)}"


class _Addr:
    """The tensor addresses of one head / one Block (the names `h1_lid.Block` gives its parameters)."""

    def __init__(self, layer, head, hd, d):
        self.L, self.h0, self.hd, self.d = int(layer), int(head) * int(hd), int(hd), int(d)
        self.qkv = f"blocks.{self.L}.attn.qkv.weight"
        self.proj = f"blocks.{self.L}.attn.proj.weight"
        self.Q, self.K, self.V = 0, self.d, 2 * self.d

    def _chk(self, c):
        if not 0 <= c < self.hd:
            raise InstructionError(f"blocks.{self.L}: head channel {c} is outside the head width hd = {self.hd}")
        return c

    def q(self, c, dim, value):
        return Write(self.qkv, (self.Q + self.h0 + self._chk(c), int(dim)), float(value))

    def k(self, c, dim, value):
        return Write(self.qkv, (self.K + self.h0 + self._chk(c), int(dim)), float(value))

    def v(self, c, dim, value):
        return Write(self.qkv, (self.V + self.h0 + self._chk(c), int(dim)), float(value))

    def p(self, dim, c, value):
        return Write(self.proj, (int(dim), self.h0 + self._chk(c)), float(value))


def _mlp_addr(layer):
    L = int(layer)
    return f"blocks.{L}.mlp.0.weight", f"blocks.{L}.mlp.0.bias", f"blocks.{L}.mlp.2.weight"


def _content_channel(layout, slot, hd, pos, key_sub, c, packed_index, plan=None):
    """The head channel of content dim `c` of key subspace `key_sub`. A whole-window head (hd == d) uses the residual dim
    itself (E28: `Wl[Q + sl.start + c, sl.start + c]`, `Wl[K + F, F]`); otherwise the dims are PACKED: under RoPE into dim 2c
    of the content pairs (E18 line 69: `W2[2 * c, ...]` over the lowest-frequency pairs), under onehot into 0, 1, 2, ..."""
    if hd == layout.d and pos == "onehot":
        return (layout[key_sub].start if isinstance(key_sub, str) else layout.flag(key_sub[0])) + c
    if pos == "onehot":
        return packed_index
    dims = plan.content_dims if plan is not None else _channels(hd, dict(content_dims=packed_index + 1)).content_dims
    if packed_index >= len(dims):
        raise InstructionError(f"{_who(slot, 'head')}: content channel {packed_index} exceeds the {len(dims)} content pairs planned")
    return dims[packed_index]


def _theta(hd, pos="rope", n_zero=0):
    """Pair c's rotary frequency, as the substrate computes it: `h1_lid.rope_theta(hd)` under 1-D `rope`, the per-axis
    ladder `rope_theta(hd, True, n_zero)` under `rope2d` (row pairs 0 .. P/2-1, column pairs P/2 .. P-1, each axis its own
    `_freqs`; the last `n_zero` pairs position-free)."""
    return _h1().rope_theta(hd, pos == "rope2d", n_zero, "cpu")


def scatter(writes: Iterable[Write], tensors: dict[str, torch.Tensor], strict: bool = True) -> dict[str, int]:
    """Write every `Write` into `tensors[name]` with ONE `index_put_` per tensor (§21.4 step 6). Two writes to one address
    with different values raise when `strict`. Returns the count written per tensor."""
    by_tensor: dict[str, dict[tuple, float]] = {}
    for w in writes:
        seen = by_tensor.setdefault(w.tensor, {})
        if w.index in seen and strict and abs(seen[w.index] - w.value) > 1e-12:
            raise InstructionError(f"conflicting writes to {w.tensor}{list(w.index)}: {seen[w.index]} and {w.value}")
        seen[w.index] = w.value
    counts = {}
    for name, cells in by_tensor.items():
        if name not in tensors:
            raise InstructionError(f"no tensor {name!r} to write into (have {sorted(tensors)})")
        t = tensors[name]
        idx = tuple(torch.as_tensor([k[i] for k in cells], dtype=torch.long, device=t.device) for i in range(t.dim()))
        vals = torch.as_tensor(list(cells.values()), dtype=t.dtype, device=t.device)
        with torch.no_grad():
            t.index_put_(idx, vals)
        counts[name] = len(cells)
    return counts


# =====================================================================================================================
# the instruction base
# =====================================================================================================================

class Instruction:
    """A weight template. Subclasses implement `needs`, `emit`, `test` as classmethods; `PARAMS` lists the parameter names
    (`validate` checks the keys, `emit` the values)."""
    PARAMS: tuple[str, ...] = ()

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        raise NotImplementedError

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        raise NotImplementedError

    @classmethod
    def test(cls, params, layout, brain, n=256, slot=None, tol=1e-3) -> Report:
        return Report(cls.__name__, False, note=f"{cls.__name__} has no unit test yet")

    @classmethod
    def gap_min(cls, params, layout, arch, slot=None) -> float:
        """The head's smallest target-minus-rival score gap in matched-dim units (one matched dim = M^2 before the 1/sqrt(hd)):
        1.0 for every head written in whole matches (Match, Pool, Broadcast); Gather overrides (its null sits at a half match)."""
        return 1.0

    @classmethod
    def check_params(cls, params):
        unknown = set(params) - set(cls.PARAMS) - {"M", "classes", "name"}
        if unknown:
            raise InstructionError(f"{cls.__name__}: unknown params {sorted(unknown)}; it takes {list(cls.PARAMS)}")


def _classes_other_than(classes, cls_name, slot, who):
    if classes is None:
        return []
    if cls_name is not None and cls_name not in classes:
        raise InstructionError(f"{_who(slot, who)}: token class {cls_name!r} is not one of {sorted(classes)}")
    return [c for c in classes if c != cls_name]


def _default_sink(params, classes, key):
    """The class the tokens of OTHER classes attend to (E28: the zero token for gather, the border token for lookup):
    `params[key]` when given, else "zero" when such a class exists, else None."""
    sink = params.get(key, "__default__")
    if sink != "__default__":
        return sink
    return "zero" if classes and "zero" in classes else None


# =====================================================================================================================
# Gather -- one head: copy `src` of the token at `offset` into my `dst`
# =====================================================================================================================

class Gather(Instruction):
    """`Gather(offset, src, dst, null, cls)` (§21.3 item 1). `onehot`: MOVES e28.py lines 97-115 -- the query is my row/col
    code PERMUTED by the offset (`W[Q + h0 + r + di, row.start + r] = M`), the key my own code; the `null` token's key scores
    1.5 M against the `cls` flag (between a full match of 2 and a partial of 1) and catches an off-grid neighbour; the tokens of
    every other class query the `sink` (E28: zero) whose key scores 1.5 M, so nothing is written into them. `rope` / `rope2d`:
    MOVES e18.py lines 56-59 per axis -- q = (M, 0) from the `bias` flag in each position pair, k = M (cos phi_c, sin phi_c)
    with phi_c = -o * theta_c (E18: o = -1 gives (cos theta, sin theta), the score peaks one position back); the row offset on
    the row-axis pairs (0 .. P/2-1 under rope2d), the column offset on the column-axis pairs (P/2 .. P-1); under 1-D `rope`
    every pair is the sequence axis and the offset must be (0, dj). The class gate and sink live in the lowest-frequency pair.
    `profile` gives the score gap over every offset within the extent; `pairs_needed` raises the pairs per axis until
    M * gap >= GATHER_GAP_MIN, an error naming the head when the axis cannot reach it. Value/proj copy `src` -> `dst`.
    Extra params: `sink` (class), `row`/`col` (the coordinate subspaces, default from the codec's class fields or "row"/"col"),
    `max_extent` ((rows, cols) the profile spans; default (H - 1, W - 1) from arch.dims, else max_len - 1), `bias` (the flag
    the rope q/k read; default "bias")."""
    PARAMS = ("offset", "src", "dst", "null", "cls", "sink", "row", "col", "max_extent", "bias")

    @staticmethod
    def _offset(params):
        o = params.get("offset")
        if not (isinstance(o, (list, tuple)) and len(o) == 2 and all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in o)):
            raise InstructionError(f"Gather: offset must be (di, dj) ints, got {o!r} (a template offset 'o' must be bound by expand)")
        return int(o[0]), int(o[1])

    @staticmethod
    def _coord_subs(params, codec, cls_name, layout):
        row = params.get("row")
        col = params.get("col")
        if codec is not None and cls_name in codec.classes:
            c = codec.classes[cls_name]
            row = row or (c.with_source("coord.row") or [None])[0]
            col = col or (c.with_source("coord.col") or [None])[0]
        row, col = row or "row", col or "col"
        if row not in layout or col not in layout:
            raise InstructionError(f"Gather (onehot): coordinate subspaces {row!r}/{col!r} are not in the layout {layout.names}")
        return row, col

    # -- the rope profile ------------------------------------------------------------------------------------------------
    @staticmethod
    def _axis_gap(theta_pairs, o, extent):
        """Score gap (in units of M^2, i.e. of matched pairs) between the target offset `o` and the best rival within
        [-extent, extent] on one axis: sum_c cos((D - o) theta_c) at D = o minus its max over D != o."""
        if len(theta_pairs) == 0:
            return 0.0
        D = torch.arange(-int(extent), int(extent) + 1, dtype=torch.float32)
        th = torch.as_tensor([float(t) for t in theta_pairs], dtype=torch.float32)
        s = torch.cos((D[:, None] - o) * th[None, :]).sum(1)
        target = float(len(theta_pairs))
        rivals = s[D != o]
        return target - float(rivals.max()) if rivals.numel() else target

    @classmethod
    def _extents(cls, params, arch):
        ext = params.get("max_extent")
        if ext is not None:
            return int(ext[0]), int(ext[1])
        dims = _dims(arch)
        T = _get(arch, "max_len", None)
        fallback = int(T) - 1 if T is not None else 256
        return int(dims.get("H", fallback)) - (1 if "H" in dims else 0), int(dims.get("W", fallback)) - (1 if "W" in dims else 0)

    @classmethod
    def _axis_pairs(cls, hd, pos, axis, n):
        """The pair indices of `n` position pairs on `axis` (0 = row, 1 = column): the HIGHEST-frequency pairs of that axis's
        half (rope2d) or of the whole head (1-D rope, where every pair is the sequence axis)."""
        P = hd // 2
        if pos == "rope":
            return list(range(0, n))
        half = P // 2
        return list(range(0, n)) if axis == 0 else list(range(half, half + n))

    @classmethod
    def profile(cls, params, arch, pairs, hd, pos=None, max_extent=None):
        """The minimum score gap (units of M^2) of a rope Gather with `pairs` = (n_row, n_col) position pairs in a head of width
        `hd`, over every (Di, Dj) within `max_extent`; the worst rival is off on one axis, so the gap is the min over axes."""
        pos = pos or _pos(arch, None)
        di, dj = cls._offset(params)
        theta = _theta(hd, pos)
        ext_i, ext_j = max_extent if max_extent is not None else cls._extents(params, arch)
        n_i, n_j = pairs
        gaps = []
        if pos == "rope":
            if di != 0:
                raise InstructionError(f"Gather: under 1-D rope the offset must be (0, dj), got {(di, dj)}")
            gaps.append(cls._axis_gap([theta[c] for c in cls._axis_pairs(hd, pos, 1, n_j)], dj, ext_j))
        else:
            if ext_i > 0:
                gaps.append(cls._axis_gap([theta[c] for c in cls._axis_pairs(hd, pos, 0, n_i)], di, ext_i))
            if ext_j > 0:
                gaps.append(cls._axis_gap([theta[c] for c in cls._axis_pairs(hd, pos, 1, n_j)], dj, ext_j))
        return min(gaps) if gaps else float("inf")

    @classmethod
    def pairs_needed(cls, params, arch, hd, M, pos=None, max_extent=None, slot=None):
        """(n_row, n_col): the fewest position pairs per axis with M * gap >= GATHER_GAP_MIN (E18 at M = 3, hd = 32,
        max_len 128: 4 pairs, e18.py line 56). Raises naming the head when an axis cannot reach the gap."""
        pos = pos or _pos(arch, None)
        di, dj = cls._offset(params)
        ext_i, ext_j = max_extent if max_extent is not None else cls._extents(params, arch)
        P = hd // 2
        limit = P if pos == "rope" else P // 2
        theta = _theta(hd, pos)
        out = []
        for axis, (o, ext) in enumerate(((di, ext_i), (dj, ext_j))):
            if pos == "rope" and axis == 0:
                if di != 0:
                    raise InstructionError(f"{_who(slot, 'Gather')}: under 1-D rope the offset must be (0, dj), got {(di, dj)}")
                out.append(0)
                continue
            if ext <= 0:
                out.append(0)
                continue
            found = None
            for n in range(1, limit + 1):
                gap = cls._axis_gap([theta[c] for c in cls._axis_pairs(hd, pos, axis, n)], o, ext)
                if M * gap >= GATHER_GAP_MIN:
                    found = n
                    break
            if found is None:
                best = max(cls._axis_gap([theta[c] for c in cls._axis_pairs(hd, pos, axis, n)], o, ext) for n in range(1, limit + 1))
                raise InstructionError(f"{_who(slot, 'Gather')}: the {'row' if axis == 0 else 'column'} axis cannot reach M * gap >= "
                                       f"{GATHER_GAP_MIN} for offset {(di, dj)} over extent {ext} with hd = {hd} (best M * gap = "
                                       f"{M * best:.3f} at M = {M:.2f}; the axis's pairs have theta <= "
                                       f"{float(theta[cls._axis_pairs(hd, pos, axis, 1)[0]]):.4g})")
            out.append(found)
        return tuple(out)

    # -- needs ---------------------------------------------------------------------------------------------------------
    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        src, dst = params["src"], params["dst"]
        w_src = layout.width(src)
        if layout.width(dst) < w_src:
            raise InstructionError(f"Gather: dst {dst!r} ({layout.width(dst)}) is narrower than src {src!r} ({w_src})")
        cls_name = params.get("cls")
        others = _classes_other_than(classes, cls_name, None, "Gather")
        gate = params.get("null") is not None or (bool(others) and _default_sink(params, classes, "sink") is not None)
        if pos == "onehot":
            row, col = cls._coord_subs(params, codec, cls_name, layout)
            return Needs(heads=1, hd_min=max(layout.width(row) + layout.width(col) + 2, w_src), content_dims=0, position_pairs=0)
        hd = _get(arch, "head_dim", None)
        M = float(params["M"]) if params.get("M") is not None else (derived_M(arch) or 3.0)
        hd = hd if isinstance(hd, int) and hd > 0 else 32                   # hd unknown ("auto"): profile at E18's head width
        n_i, n_j = cls.pairs_needed(params, arch, hd, M, pos)
        pairs = n_i + n_j
        return Needs(heads=1, hd_min=max(2 * pairs + (2 if gate else 0), w_src), content_dims=1 if gate else 0,
                     position_pairs=max(n_i, n_j))

    @classmethod
    def gap_min(cls, params, layout, arch, slot=None) -> float:
        """The head's smallest score gap between its target and any rival, in matched-dim units (M^2 each): `onehot` with a
        `null` -- the null sits at 1.5 matches against the target's 2, so 0.5 (E28 lines 107-112); rope -- half the profile
        gap when a null/sink is written at (full - gap/2), else the gap. The compiler sets the head's written M from it
        (`brainbuilder.compile`, step 5) so that the softmax gap in nats is the derived sharpness whatever the unit."""
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        cls_name = params.get("cls")
        others = _classes_other_than(classes, cls_name, slot, "Gather")
        gated = params.get("null") is not None or (bool(others) and _default_sink(params, classes, "sink") is not None)
        if pos == "onehot":
            return 0.5 if gated else 1.0
        slot, d, hd = _slot(slot, layout)
        M = float(params["M"]) if params.get("M") is not None else (derived_M(arch) or 3.0)
        n_i, n_j = cls.pairs_needed(params, arch, hd, M, pos, slot=slot)
        gap = cls.profile(params, arch, (n_i, n_j), hd, pos)
        return gap / 2 if gated else gap

    # -- emit ----------------------------------------------------------------------------------------------------------
    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        M = _M(params, arch, M)
        slot, d, hd = _slot(slot, layout)
        A = _Addr(slot.layer, slot.head, hd, d)
        di, dj = cls._offset(params)
        src, dst, cls_name, null = params["src"], params["dst"], params.get("cls"), params.get("null")
        w_src = layout.width(src)
        if layout.width(dst) < w_src:
            raise InstructionError(f"{_who(slot, 'Gather')}: dst {dst!r} is narrower than src {src!r}")
        others = _classes_other_than(classes, cls_name, slot, "Gather")
        sink = _default_sink(params, classes, "sink") if others else None
        if sink is not None and classes is not None and sink not in classes:
            raise InstructionError(f"{_who(slot, 'Gather')}: sink class {sink!r} is not one of {sorted(classes)}")
        flag_of = (lambda name: layout.flag(classes[name])) if classes else layout.flag
        W = []
        if pos == "onehot":
            row, col = cls._coord_subs(params, codec, cls_name, layout)
            R, Cw = layout[row], layout[col]
            H, Wd = R.stop - R.start, Cw.stop - Cw.start
            if hd < H + Wd + 2:
                raise InstructionError(f"{_who(slot, 'Gather')}: hd = {hd} < H + W + 2 = {H + Wd + 2}")
            for r in range(H):                                              # query: my row shifted by di; key: your row
                if 0 <= r + di < H:
                    W.append(A.q(r + di, R.start + r, M))
                W.append(A.k(r, R.start + r, M))
            for c in range(Wd):
                if 0 <= c + dj < Wd:
                    W.append(A.q(H + c + dj, Cw.start + c, M))
                W.append(A.k(H + c, Cw.start + c, M))
            b = H + Wd                                                      # the class gate and the null (1.5 matches)
            if cls_name is not None:
                W.append(A.q(b, flag_of(cls_name), M))
            if null is not None:
                W.append(A.k(b, flag_of(null), 1.5 * M))
            if sink is not None:                                            # the other classes: the sink (value 0)
                for other in others:
                    W.append(A.q(b + 1, flag_of(other), M))
                W.append(A.k(b + 1, flag_of(sink), 1.5 * M))
        else:
            bias = layout.flag(params.get("bias") or "bias")
            n_i, n_j = cls.pairs_needed(params, arch, hd, M, pos, slot=slot)
            if slot.channels is not None and slot.channels.position_pairs:  # the compiler pinned the pairs per axis
                n = len(slot.channels.position_pairs)
                n_i, n_j = (n if n_i else 0), (n if n_j else 0)
            theta = _theta(hd, pos, slot.n_zero)
            used = []
            for axis, (o, n) in enumerate(((di, n_i), (dj, n_j))):
                for c in cls._axis_pairs(hd, pos, axis, n):
                    phi = -o * float(theta[c])                              # E18: o = -1 -> k = M (cos theta, sin theta)
                    W.append(A.q(2 * c, bias, M))
                    W.append(A.k(2 * c, bias, M * math.cos(phi)))
                    W.append(A.k(2 * c + 1, bias, M * math.sin(phi)))
                    used.append(c)
            full = len(used)
            if null is not None or sink is not None:
                gap = cls.profile(params, arch, (n_i, n_j), hd, pos)
                gate_pair = hd // 2 - 1                                     # the gate: the lowest-frequency pair (E18's lesson)
                if gate_pair in used or gate_pair < 0:
                    raise InstructionError(f"{_who(slot, 'Gather')}: hd = {hd} has no spare pair for the class gate")
                g = 2 * gate_pair
                if cls_name is not None:
                    W.append(A.q(g, flag_of(cls_name), M))
                if null is not None:                                        # between the full match and its best rival
                    W.append(A.k(g, flag_of(null), (full - gap / 2) * M))
                if sink is not None:
                    for other in others:
                        W.append(A.q(g + 1, flag_of(other), M))
                    W.append(A.k(g + 1, flag_of(sink), (full - gap / 2) * M))
        S, Dd = layout[src], layout[dst]
        for c in range(w_src):                                              # value: the attended token's src -> my dst
            W.append(A.v(c, S.start + c, 1.0))
            W.append(A.p(Dd.start + c, c, 1.0))
        return W

    # -- test ----------------------------------------------------------------------------------------------------------
    @classmethod
    def test(cls, params, layout, brain, n=256, slot=None, tol=1e-3, seed=0, H=None, W=None) -> Report:
        """Random H x W frames through the codec (+ the nulls); every cell must attend to the token at its offset (the `null`
        when off-grid), every null to the sink; the written `dst` must equal the neighbour's `src` to `tol`. `H`/`W` default
        to the coordinate subspaces' widths (onehot) or arch.dims H/W, else 6 x 7."""
        layout, classes, codec = _split(layout, params)
        codec = codec or getattr(brain, "codec", None)
        if codec is None or slot is None:
            return Report("Gather", False, note="Gather.test needs a Codec (layout=codec or brain.codec) and slot=")
        attn = brain.model.blocks[slot.layer].attn
        pos = _pos(getattr(brain, "arch", None), codec)
        M = _test_M(params, brain)
        di, dj = cls._offset(params)
        g = torch.Generator().manual_seed(seed)
        cell = codec.cell_class
        dims = _dims(getattr(brain, "arch", None))
        row_w = H or (layout.width(cell.with_source("coord.row")[0]) if pos == "onehot" and cell.with_source("coord.row") else int(dims.get("H", 6)))
        col_w = W or (layout.width(cell.with_source("coord.col")[0]) if pos == "onehot" and cell.with_source("coord.col") else int(dims.get("W", 7)))
        src, dst = layout[params["src"]], layout[params["dst"]]
        w_src = src.stop - src.start
        frames = max(1, n // (row_w * col_w))
        accs, gaps, errs = [], [], []
        null_names = [c for c in ("border", "zero") if c in codec.classes]
        for _ in range(frames):
            frame = torch.randint(0, w_src, (row_w, col_w), generator=g)
            x_cells, coords = codec.encode_frame(frame.numpy())
            nulls = codec.nulls(null_names) if null_names else torch.zeros(0, layout.d)
            x = torch.cat([nulls, x_cells], 0)
            coords_all = torch.cat([torch.zeros(nulls.shape[0], 2, dtype=torch.long), coords], 0)
            scores = _head_scores(attn, x, slot.head, pos, coords_all)
            n0 = nulls.shape[0]
            target = torch.full((x.shape[0],), -1, dtype=torch.long)
            for i in range(row_w):
                for j in range(col_w):
                    t = n0 + i * col_w + j
                    ii, jj = i + di, j + dj
                    if 0 <= ii < row_w and 0 <= jj < col_w:
                        target[t] = n0 + ii * col_w + jj
                    elif params.get("null") is not None:
                        target[t] = null_names.index(params["null"])
            sink = _default_sink(params, classes, "sink")
            for t in range(n0):
                if sink is not None and sink in null_names:
                    target[t] = null_names.index(sink)
            q = target >= 0
            acc, gap = _argmax_stats(scores[q], target[q])
            accs.append(acc)
            gaps.append(gap)
            y = x[None] + attn(x[None], None, 0, coords_all if pos != "onehot" else None)
            want = x[target[q], src.start:src.stop]
            got = y[0, q, dst.start:dst.start + w_src]
            errs.append(float((got - want).detach().abs().max()) if got.numel() else 0.0)
        acc, gap, err = min(accs), min(gaps), max(errs)
        return Report("Gather", acc == 1.0 and gap >= M and err <= tol, n=frames * row_w * col_w, argmax_acc=acc, min_gap=gap, M=M,
                      subspace_err=err, tol=tol)


def _rot(z, coords, theta, pos):
    """The substrate's rotation (Attn.forward), for reading a head's scores in a test. `z`: (T, hd)."""
    if pos == "onehot":
        return z
    P = theta.shape[0]
    c = coords.to(torch.float32)
    if pos == "rope":
        ang = c[:, 1:2] * theta[None, :]
    else:
        ang = torch.cat([c[:, 0:1] * theta[:P // 2], c[:, 1:2] * theta[P // 2:]], -1)
    cos, sin = ang.cos(), ang.sin()
    a, b = z[:, 0::2], z[:, 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


def _head_scores(attn, x, head, pos, coords):
    """The (T, T) logits of head `head` of `attn` on tokens `x` (T, d): the softmax's input, scaled as SDPA scales it."""
    with torch.no_grad():
        q, k, _ = attn.qkv(x).chunk(3, -1)
        hd = attn.hd
        q, k = q[:, head * hd:(head + 1) * hd], k[:, head * hd:(head + 1) * hd]
        if pos != "onehot":
            theta = attn.theta2d if pos == "rope2d" else attn.theta                # the ladder the substrate rotates with
            q, k = _rot(q, coords, theta, pos), _rot(k, coords, theta, pos)
        return (q @ k.T) / math.sqrt(hd)


def _argmax_stats(scores, target):
    """(fraction of rows whose argmax is the target, the smallest target-minus-runner-up gap)."""
    if scores.shape[0] == 0:
        return 1.0, float("inf")
    am = scores.argmax(1)
    acc = float((am == target).float().mean())
    tgt = scores.gather(1, target[:, None])[:, 0]
    others = scores.clone()
    others[torch.arange(scores.shape[0]), target] = -float("inf")
    gap = float((tgt - others.max(1).values).min())
    return acc, gap


# =====================================================================================================================
# Match -- one head: the token whose `k` equals my `q` writes its `v` into my `dst`
# =====================================================================================================================

class Match(Instruction):
    """`Match(q, k, v, dst, key_class, null, weights, mode)` (§21.3 item 2). MOVES e28.py lines 117-132: query and key are the
    named subspaces at weight M per dim (`q[i]` pairs with `k[i]`; `k=None` means `k = q`); `weights[sub]` multiplies that
    subspace's QUERY (E28's `g * M` on the action); `"auto"` = (sum of the other key subspaces' weights) + 1, so one match on
    it outweighs a full match on all the others (E28's `g = nO + 2`, derived: nO + 1 window subspaces of weight 1, plus 1).
    The class gate: a `cls` (default: the codec's cell class) token's query carries M on the gate channel where a `key_class`
    token's key carries (S + 1) M, S = the sum of the weights, so only keys of that class can win; the tokens of every other
    class query the `null` whose key carries (S + 2) M (E28: `Wl[K + F, F] = (2 + nO + g) M`, `Wl[K + S, S] = (3 + nO + g) M`).
    Value: the attended token's `v`, minus its `dst` under `mode="replace"` (E28's `OUT - T`: on a full match the entry's
    colour equals the cell's, so the sum replaces it -- it REPLACES only when the attended token's dst equals the query's),
    plain under `mode="add"` (E18). `min(width(v), width(dst))` dims are copied (E28: range(V) = 16 into colour(17); E18:
    range(V) = 8 of tok(9)). A whole-window head (hd = d) uses the residual dims as channels; a RoPE head packs the content
    into the LOWEST-frequency pairs (E18 lines 66-70); `drift` = max_len * theta_c on those pairs is the number the compiler checks
    against MATCH_DRIFT_MAX (E18 at Arch(max_len 128) sits at 2.28 and scores 1.000 on its 17-token sequences, so `emit` reports,
    never refuses)."""
    PARAMS = ("q", "k", "v", "dst", "key_class", "null", "weights", "mode", "cls")

    @staticmethod
    def _subs(params):
        q = params.get("q")
        q = [q] if isinstance(q, str) else list(q or [])
        k = params.get("k")
        k = q if k is None else ([k] if isinstance(k, str) else list(k))
        if not q or len(q) != len(k):
            raise InstructionError(f"Match: q {q} and k {k} must be non-empty lists of equal length")
        mode = params.get("mode", "replace")
        if mode not in ("replace", "add"):
            raise InstructionError(f"Match: mode must be 'replace' or 'add', got {mode!r}")
        return q, k, mode

    @classmethod
    def weights(cls, params, layout) -> list[float]:
        """The per-subspace query multipliers, `"auto"` resolved from the layout."""
        q, _, _ = cls._subs(params)
        spec = dict(params.get("weights") or {})
        unknown = set(spec) - set(q)
        if unknown:
            raise InstructionError(f"Match: weights name subspaces {sorted(unknown)} that are not in q {q}")
        fixed = {s: float(w) for s, w in spec.items() if w != "auto"}
        base = [fixed.get(s, 1.0) for s in q]
        out = []
        for s, b in zip(q, base):
            if spec.get(s) == "auto":
                out.append(sum(bb for ss, bb in zip(q, base) if ss != s and spec.get(ss) != "auto") + 1.0)
            else:
                out.append(b)
        return out

    @classmethod
    def _content_dims(cls, params, layout):
        q, k, _ = cls._subs(params)
        total = 0
        for qs, ks in zip(q, k):
            if layout.width(qs) != layout.width(ks):
                raise InstructionError(f"Match: q {qs!r} ({layout.width(qs)}) and k {ks!r} ({layout.width(ks)}) differ in width")
            total += layout.width(ks)
        return total

    @classmethod
    def drift(cls, params, layout, arch, slot=None) -> float:
        """max over the content pairs of `max_len * theta_c` (§21.3 item 2's check); 0 under onehot."""
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        if pos == "onehot":
            return 0.0
        slot, d, hd = _slot(slot, layout)
        n = cls._content_dims(params, layout) + (1 if params.get("key_class") is not None else 0) + (1 if params.get("null") is not None else 0)
        plan = slot.channels if slot.channels is not None and slot.channels.content_pairs else _channels(hd, dict(content_dims=n), slot.name)
        theta = _theta(hd, pos, slot.n_zero)
        T = int(_get(arch, "max_len", 0) or 0)
        return max(T * float(theta[c]) for c in plan.content_pairs[:n]) if plan.content_pairs else 0.0

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        q, k, mode = cls._subs(params)
        n = cls._content_dims(params, layout) + (1 if params.get("key_class") is not None else 0) + (1 if params.get("null") is not None else 0)
        w_val = min(layout.width(params["v"]), layout.width(params["dst"]))
        whole = len(q) > 1
        if pos == "onehot":
            return Needs(heads=1, hd_min=layout.d if whole else max(n, w_val), content_dims=n, whole_window=whole)
        return Needs(heads=1, hd_min=max(2 * n, w_val), content_dims=n, whole_window=False)

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        M = _M(params, arch, M)
        slot, d, hd = _slot(slot, layout)
        A = _Addr(slot.layer, slot.head, hd, d)
        q, k, mode = cls._subs(params)
        ws = cls.weights(params, layout)
        key_class, null = params.get("key_class"), params.get("null")
        v_sub, dst = params["v"], params["dst"]
        n_content = cls._content_dims(params, layout)
        n_ch = n_content + (1 if key_class is not None else 0) + (1 if null is not None else 0)
        plan = None
        if pos != "onehot":
            plan = slot.channels if slot.channels is not None and slot.channels.content_pairs else _channels(hd, dict(content_dims=n_ch), slot.name)
            if len(plan.content_pairs) < n_ch:
                raise InstructionError(f"{_who(slot, 'Match')}: {n_ch} content channels but only {len(plan.content_pairs)} content pairs")
            # the content drift `max_len * theta_c` is the compiler's check (§21.3 item 2): `Match.drift` gives the number, `emit`
            # writes regardless -- E18 at Arch(max_len 128) sits at 2.28 on pair 7 and scores 1.000 on its 17-token sequences
        elif hd != layout.d and hd < max(n_ch, 1):
            raise InstructionError(f"{_who(slot, 'Match')}: hd = {hd} < {n_ch} channels")
        W = []
        packed = 0
        S = float(sum(ws))
        for (qs, ks), w in zip(zip(q, k), ws):                              # window dims: query at w * M, key at M
            Qs, Ks = layout[qs], layout[ks]
            for c in range(Ks.stop - Ks.start):
                ch = _content_channel(layout, slot, hd, pos, ks, c, packed, plan)
                W.append(A.q(ch, Qs.start + c, w * M))
                W.append(A.k(ch, Ks.start + c, M))
                packed += 1
        flag_of = (lambda name: layout.flag(classes[name])) if classes else layout.flag
        query_cls = params.get("cls")
        if query_cls is None and codec is not None and (key_class is not None or null is not None):
            try:
                query_cls = codec.cell_class.name
            except Exception:                                               # no positioned class: no gate (E18)
                query_cls = None
        if key_class is not None:
            ch = _content_channel(layout, slot, hd, pos, (classes[key_class] if classes else key_class,), 0, packed, plan)
            packed += 1
            if query_cls is not None:
                W.append(A.q(ch, flag_of(query_cls), M))                    # a cell's query: keys of key_class only
            W.append(A.k(ch, flag_of(key_class), (S + 1) * M))
        if null is not None:
            ch = _content_channel(layout, slot, hd, pos, (classes[null] if classes else null,), 0, packed, plan)
            packed += 1
            for other in _classes_other_than(classes, query_cls, slot, "Match"):
                W.append(A.q(ch, flag_of(other), M))                        # every other class: the null (value 0)
            W.append(A.k(ch, flag_of(null), (S + 2) * M))
        Vs, Ds = layout[v_sub], layout[dst]
        for c in range(min(Vs.stop - Vs.start, Ds.stop - Ds.start)):        # value: v (- dst) of the attended token -> my dst
            W.append(A.v(c, Vs.start + c, 1.0))
            if mode == "replace":
                W.append(A.v(c, Ds.start + c, -1.0))
            W.append(A.p(Ds.start + c, c, 1.0))
        return W

    @classmethod
    def test(cls, params, layout, brain, n=256, slot=None, tol=1e-3, seed=0) -> Report:
        """`n` key tokens (class `key_class`, or the query class) with distinct random one-hot keys; one query token per key
        carrying the same window; the query must attend to its key with gap >= M, and its `dst` must become
        `dst + v - dst_key` (replace) / `dst + v` (add) to `tol`. The nulls, when present, must be what the keys attend to."""
        layout, classes, codec = _split(layout, params)
        codec = codec or getattr(brain, "codec", None)
        if codec is None or slot is None:
            return Report("Match", False, note="Match.test needs a Codec and slot=")
        attn = brain.model.blocks[slot.layer].attn
        pos = _pos(getattr(brain, "arch", None), codec)
        M = _test_M(params, brain)
        q, k, mode = cls._subs(params)
        key_class, null = params.get("key_class"), params.get("null")
        g = torch.Generator().manual_seed(seed)
        try:
            query_cls = params.get("cls") or codec.cell_class.name
        except Exception:
            query_cls = next(iter(codec.classes))
        key_cls = key_class or query_cls
        v_sub, dst = layout[params["v"]], layout[params["dst"]]
        # distinct random windows
        widths = [layout.width(s) for s in k]
        seen, keys = set(), []
        while len(keys) < n:
            win = tuple(int(torch.randint(0, w, (1,), generator=g)) for w in widths)
            if win not in seen:
                seen.add(win)
                keys.append(win)
        kf = codec.classes[key_cls]
        qf = codec.classes[query_cls]
        x_keys, x_q = [], []
        for win in keys:
            kv = torch.zeros(layout.d)
            kv[layout.flag(kf.flag)] = 1.0
            for s, c in zip(k, win):
                kv[layout[s].start + c] = 1.0
            kv[v_sub.start + int(torch.randint(0, v_sub.stop - v_sub.start, (1,), generator=g))] = 1.0
            x_keys.append(kv)
            qv = torch.zeros(layout.d)
            qv[layout.flag(qf.flag)] = 1.0
            for s, c in zip(q, win):
                qv[layout[s].start + c] = 1.0
            if "bias" in layout:
                qv[layout.flag("bias")] = 1.0
            x_q.append(qv)
        null_names = [c for c in ("border", "zero") if c in codec.classes]
        nulls = codec.nulls(null_names) if null_names else torch.zeros(0, layout.d)
        x = torch.cat([nulls, torch.stack(x_keys), torch.stack(x_q)], 0)
        T = x.shape[0]
        coords = torch.stack([torch.zeros(T, dtype=torch.long), torch.arange(T)], 1)
        scores = _head_scores(attn, x, slot.head, pos, coords)
        n0 = nulls.shape[0]
        qi = torch.arange(n0 + n, n0 + 2 * n)
        target = torch.arange(n0, n0 + n)
        acc, gap = _argmax_stats(scores[qi], target)
        with torch.no_grad():
            y = x[None] + attn(x[None], None, 0, coords if pos != "onehot" else None)
        w = min(v_sub.stop - v_sub.start, dst.stop - dst.start)
        want = x[qi, dst.start:dst.start + w] + x[target, v_sub.start:v_sub.start + w]
        if mode == "replace":
            want = want - x[target, dst.start:dst.start + w]
        err = float((y[0, qi, dst.start:dst.start + w] - want).abs().max())
        extra = {}
        if n0 and null is not None and null in null_names:                  # the keys (and nulls) must sit on the null
            acc_k, gap_k = _argmax_stats(scores[n0:n0 + n], torch.full((n,), null_names.index(null)))
            extra = dict(keys_on_null_acc=acc_k, keys_on_null_gap=gap_k)
            acc, gap = min(acc, acc_k), min(gap, gap_k)
        return Report("Match", acc == 1.0 and gap >= M and err <= tol, n=n, argmax_acc=acc, min_gap=gap, M=M, subspace_err=err, tol=tol,
                      extra=extra)


# =====================================================================================================================
# Pool -- one head: a register reads the mean of W * src over the tokens of a class
# =====================================================================================================================

class Pool(Instruction):
    """`Pool(reg, key_class, src, W, dst, gate=None, sign=+1)` (§21.3 item 3; NEW). The register `reg`'s token is the query:
    M on its slot id (the `slot` subspace, `slot.registers[reg]`; without a `slot` subspace, M on the register class flag, so
    every register answers) at the gate channel, where a `key_class` token's key carries M on its class flag and, when a
    `gate` flag is named, 2M on that flag (3M on the class flag without a gate). A sink channel: every class queries the sink
    (default "zero") at M, whose key is 2M. Scores in matched dims: the target tokens 3, the sink 2, an ungated token of the
    class 1, the floor 0 -- the register lands on the gated tokens of the class when there are any and on the sink otherwise
    (a no-op); every other token lands on the sink, so nothing is written into it. The value is `sign * W * src` -- `src` a subspace or a
    difference `"a - b"` (§21.2.2's `goal - colour`); `W=None` is the identity (widths must agree), a NAMED W is STATE in
    `Brain.W` that ZipLearn counts (GCML's inverse model): `emit` writes the q/k/proj plumbing and `value_writes(...,
    W_tensor)` gives the addresses of W's numbers (rows = dst dims, columns = src dims) for `ziplearn.write`. The softmax over
    equal scores is the MEAN. Extra params: `sink`, `slot_sub` (default "slot")."""
    PARAMS = ("reg", "key_class", "src", "W", "dst", "gate", "sign", "sink", "slot_sub")

    @staticmethod
    def _src(params):
        src = params["src"]
        if isinstance(src, str) and " - " in src:
            a, b = [s.strip() for s in src.split(" - ", 1)]
            return [(a, 1.0), (b, -1.0)]
        return [(src, 1.0)]

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        w_dst = layout.width(params["dst"])
        return Needs(heads=1, hd_min=max(w_dst, 2 if pos == "onehot" else 4), content_dims=2)

    @classmethod
    def _register_query(cls, A, layout, classes, slot, params, ch, M, who):
        """M on the register's slot id when the layout has a `slot` subspace and the slot id is known, else on the class flag."""
        reg = params["reg"]
        slot_sub = params.get("slot_sub") or "slot"
        if slot_sub in layout and slot is not None and slot.registers and reg in slot.registers:
            return [A.q(ch, layout[slot_sub].start + int(slot.registers[reg]), M)]
        if classes and "register" in classes:
            return [A.q(ch, layout.flag(classes["register"]), M)]
        raise InstructionError(f"{who}: no 'register' token class and no '{slot_sub}' subspace to address register {reg!r}")

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        M = _M(params, arch, M)
        slot, d, hd = _slot(slot, layout)
        A = _Addr(slot.layer, slot.head, hd, d)
        who = _who(slot, "Pool")
        key_class, gate, dst = params["key_class"], params.get("gate"), params["dst"]
        sign = float(params.get("sign", 1) or 1)
        flag_of = (lambda name: layout.flag(classes[name])) if classes else layout.flag
        plan = None if pos == "onehot" else (slot.channels if slot.channels is not None and slot.channels.content_pairs else _channels(hd, dict(content_dims=2), slot.name))
        g_ch = _content_channel(layout, slot, hd, pos, (classes[key_class] if classes else key_class,), 0, 0, plan)
        sink = _default_sink(params, classes, "sink")
        s_ch = _content_channel(layout, slot, hd, pos, (classes[sink] if classes else sink,), 0, 1, plan) if sink else None
        W = cls._register_query(A, layout, classes, slot, params, g_ch, M, who)
        # scores in matched dims (M^2 each): the target tokens 3, the sink 2, an ungated token of the class 1, everything else 0
        # -- every gap >= 1 match and the sink 2 above the floor, so the register's mass lands where it should at the derived M
        if gate is not None:
            W.append(A.k(g_ch, flag_of(key_class), M))
            W.append(A.k(g_ch, layout.flag(gate), 2.0 * M))
        else:
            W.append(A.k(g_ch, flag_of(key_class), 3.0 * M))
        if sink is not None:
            for name in (classes or {}):
                W.append(A.q(s_ch, flag_of(name), M))
            W.append(A.k(s_ch, flag_of(sink), 2.0 * M))
        Ds = layout[dst]
        w_dst = Ds.stop - Ds.start
        if hd < w_dst:
            raise InstructionError(f"{who}: hd = {hd} < width(dst) = {w_dst}")
        if params.get("W") is None:                                         # identity value: src (- b) -> dst
            for sub, s in cls._src(params):
                if layout.width(sub) != w_dst:
                    raise InstructionError(f"{who}: W=None needs width(src {sub!r}) = width(dst) = {w_dst}, got {layout.width(sub)}")
                for c in range(w_dst):
                    W.append(A.v(c, layout[sub].start + c, sign * s))
        for c in range(w_dst):
            W.append(A.p(Ds.start + c, c, 1.0))
        return W

    @classmethod
    def value_writes(cls, params, layout, slot, W_tensor, sign=None) -> list[Write]:
        """The addresses of a NAMED W (width(dst) x width(src)): `qkv.weight[V + h0 + r, src.start + c] = sign * s * W[r, c]`
        for each term (sub, s) of `src` (`"a - b"` gives +W on a and -W on b)."""
        layout, classes, codec = _split(layout, params)
        slot, d, hd = _slot(slot, layout)
        A = _Addr(slot.layer, slot.head, hd, d)
        sign = float(params.get("sign", 1) or 1) if sign is None else float(sign)
        Wt = torch.as_tensor(W_tensor, dtype=torch.float32)
        w_dst = layout.width(params["dst"])
        out = []
        for sub, s in cls._src(params):
            if Wt.shape != (w_dst, layout.width(sub)):
                raise InstructionError(f"Pool W must be {(w_dst, layout.width(sub))}, got {tuple(Wt.shape)}")
            S = layout[sub]
            nz = Wt.nonzero()
            for r, c in nz.tolist():
                out.append(A.v(r, S.start + c, sign * s * float(Wt[r, c])))
        return out


# =====================================================================================================================
# Broadcast -- one head: every token reads a register's `sub` into its `dst`
# =====================================================================================================================

class Broadcast(Instruction):
    """`Broadcast(reg, sub, dst)` (§21.3 item 4; NEW: E28's per-pass `x[0, n:, self.A] = 1` as a head). Key: M on the register
    class flag plus M on the register's slot id when the layout has a `slot` subspace (so `reg` outscores the other registers by
    one match); query: M on the class flag of every token class (or of `cls` only), so every token attends to the register and
    ADDS its `sub` into `dst`: a copy onto a `dst` the boundary cleared or the anchor set (a "v - dst" value would read the
    REGISTER's dst, a no-op when dst = sub, so there is no replace mode here). Scores in matched dims: the named register 3
    (class flag 1 + slot id 2), another register 1, the floor 0; a token of a non-reading class queries the `sink` (default
    "zero") at 2, so nothing is written into it. Extra params: `cls` (the reading class; default every class), `sink`,
    `slot_sub` (default "slot")."""
    PARAMS = ("reg", "sub", "dst", "cls", "sink", "slot_sub")

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        return Needs(heads=1, hd_min=max(layout.width(params["sub"]), 3 if pos == "onehot" else 6), content_dims=3)

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        pos = _pos(arch, codec)
        M = _M(params, arch, M)
        slot, d, hd = _slot(slot, layout)
        A = _Addr(slot.layer, slot.head, hd, d)
        who = _who(slot, "Broadcast")
        reg, sub, dst = params["reg"], params["sub"], params["dst"]
        if classes is None or "register" not in classes:
            raise InstructionError(f"{who}: needs a 'register' token class (pass the codec, or params['classes'])")
        flag_of = lambda name: layout.flag(classes[name])                   # noqa: E731
        plan = None if pos == "onehot" else (slot.channels if slot.channels is not None and slot.channels.content_pairs else _channels(hd, dict(content_dims=3), slot.name))
        f_ch = _content_channel(layout, slot, hd, pos, (classes["register"],), 0, 0, plan)
        slot_sub = params.get("slot_sub") or "slot"
        W = []
        readers = [params["cls"]] if params.get("cls") else list(classes)
        for name in readers:                                                # a reader: the register class (1) + the slot id (2)
            W.append(A.q(f_ch, flag_of(name), M))
        W.append(A.k(f_ch, flag_of("register"), M))
        if slot_sub in layout and slot.registers and reg in slot.registers:
            s_ch = _content_channel(layout, slot, hd, pos, slot_sub, int(slot.registers[reg]), 1, plan)
            for name in readers:
                W.append(A.q(s_ch, flag_of(name), M))
            W.append(A.k(s_ch, layout[slot_sub].start + int(slot.registers[reg]), 2.0 * M))
        sink = _default_sink(params, classes, "sink")
        non_readers = [c for c in classes if c not in readers]
        if sink is not None and non_readers:                                # a non-reader: the sink at 2, so nothing is written into it
            z_ch = _content_channel(layout, slot, hd, pos, (classes[sink],), 0, 2, plan)
            for name in non_readers:
                W.append(A.q(z_ch, flag_of(name), M))
            W.append(A.k(z_ch, flag_of(sink), 2.0 * M))
        Ss, Ds = layout[sub], layout[dst]
        w = Ss.stop - Ss.start
        if Ds.stop - Ds.start < w or hd < w:
            raise InstructionError(f"{who}: dst {dst!r} or hd = {hd} narrower than sub {sub!r} ({w})")
        for c in range(w):
            W.append(A.v(c, Ss.start + c, 1.0))
            W.append(A.p(Ds.start + c, c, 1.0))
        return W


# =====================================================================================================================
# Row / Compare -- MLP hidden units
# =====================================================================================================================

def _key_vector(key, layout, who):
    """A d-vector from `key`: a dict {subspace: weight (a width-1 flag / scalar) | a sequence of the subspace's width | a
    dict {label: weight}}, or a full d-vector."""
    if isinstance(key, (torch.Tensor, list, tuple)) and not isinstance(key, dict):
        v = torch.as_tensor(key, dtype=torch.float32).reshape(-1)
        if v.numel() != layout.d:
            raise InstructionError(f"{who}: a key vector must have d = {layout.d} entries, got {v.numel()}")
        return v
    v = torch.zeros(layout.d)
    for sub, val in dict(key).items():
        if isinstance(val, (list, tuple, torch.Tensor)):
            s = layout[sub]
            t = torch.as_tensor(val, dtype=torch.float32).reshape(-1)
            if t.numel() != s.stop - s.start:
                raise InstructionError(f"{who}: key[{sub!r}] has {t.numel()} entries for a subspace of {s.stop - s.start}")
            v[s] = t
        elif isinstance(val, dict):
            for lab, w in val.items():
                v[layout.flag(f"{sub}.{lab}")] = float(w)
        else:
            try:
                v[layout.flag(sub)] = float(val)                              # a flag or a labelled dim
            except LayoutError:
                s = layout[sub]
                if s.stop - s.start != 1:
                    raise InstructionError(f"{who}: key[{sub!r}] = {val!r} on a {s.stop - s.start}-wide subspace: give a vector of "
                                           f"that width (a symbol index would be a content value)") from None
                v[s.start] = float(val)
    return v


class Row(Instruction):
    """`Row(key, threshold, value)` (§21.3 item 5; NEW -- Geva et al. 2021's key-value row; the halting row; the slot a
    consolidated entry is written into). One hidden unit r: `mlp.0.weight[r] = M * key`, `mlp.0.bias[r] = -M * threshold`,
    `mlp.2.weight[:, r] = value / (M * margin)` with `margin = (full match = sum of the key's positive weights) - threshold`, so a
    full match (pre-activation M * margin, where GELU is the identity) contributes exactly `value` and anything at or below the
    threshold contributes ~0 (GELU(-M/2) ~ 1e-6 at M = 10)."""
    PARAMS = ("key", "threshold", "value")

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        return Needs(rows=1)

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        M = _M(params, arch, M)
        slot = slot if slot is not None else Slot(row=0)
        if slot.row is None:
            raise InstructionError(f"{_who(slot, 'Row')}: slot.row (the MLP hidden unit) is required")
        who = _who(slot, "Row")
        key = _key_vector(params["key"], layout, who)
        value = _key_vector(params["value"], layout, who)
        thr = float(params.get("threshold", 0.5))
        full = float(key.clamp(min=0).sum())
        margin = full - thr
        if margin <= 0:
            raise InstructionError(f"{who}: threshold {thr} >= the full match {full}; the row could never fire")
        w0, b0, w2 = _mlp_addr(slot.layer)
        r = int(slot.row)
        W = [Write(w0, (r, int(i)), M * float(key[i])) for i in key.nonzero().flatten().tolist()]
        W.append(Write(b0, (r,), -M * thr))
        W += [Write(w2, (int(i), r), float(value[i]) / (M * margin)) for i in value.nonzero().flatten().tolist()]
        return W

    @classmethod
    def test(cls, params, layout, brain, n=256, slot=None, tol=1e-3, seed=0) -> Report:
        """Random tokens with the key present or absent: the value must appear iff the key is present, to `tol`."""
        layout, classes, codec = _split(layout, params)
        if slot is None:
            return Report("Row", False, note="Row.test needs slot=")
        mlp = brain.model.blocks[slot.layer].mlp
        key = _key_vector(params["key"], layout, "Row")
        value = _key_vector(params["value"], layout, "Row")
        g = torch.Generator().manual_seed(seed)
        present = torch.rand(n, generator=g) < 0.5
        x = torch.zeros(n, layout.d)
        x[present] = key
        with torch.no_grad():
            y = mlp(x)
        want = present[:, None].float() * value[None]
        err = float((y - want).abs().max())
        return Report("Row", err <= tol, n=n, subspace_err=err, tol=tol, argmax_acc=1.0, min_gap=float("inf"), M=float("nan"))


class Compare(Instruction):
    """`Compare(a, b, flag)` (§21.3 item 6): C = min(width(a), width(b)) hidden units, unit c = GELU(M (a_c + b_c) - 1.5 M),
    which fires (at M/2) only when BOTH codes have dim c, each weighted -2/M into `flag`; plus one unit with key = the `bias`
    flag (threshold 0, value +1): `flag = 1` iff the two one-hot codes differ (a code with a dim the other lacks, e.g.
    colour = BORDER against pred(V), differs). e5.py's gates were Python lambdas; this is their tensor form. Rows: C + 1 from
    `slot.row`. Extra param: `bias` (the constant flag; default "bias")."""
    PARAMS = ("a", "b", "flag", "bias")

    @classmethod
    def _C(cls, params, layout):
        return min(layout.width(params["a"]), layout.width(params["b"]))

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        layout, _, _ = _split(layout, params)
        return Needs(rows=cls._C(params, layout) + 1)

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        M = _M(params, arch, M)
        slot = slot if slot is not None else Slot(row=0)
        if slot.row is None:
            raise InstructionError(f"{_who(slot, 'Compare')}: slot.row is required")
        a, b = layout[params["a"]], layout[params["b"]]
        flag = layout.flag(params["flag"])
        bias = layout.flag(params.get("bias") or "bias")
        C = cls._C(params, layout)
        w0, b0, w2 = _mlp_addr(slot.layer)
        r0 = int(slot.row)
        W = []
        for c in range(C):
            r = r0 + c
            W += [Write(w0, (r, a.start + c), M), Write(w0, (r, b.start + c), M), Write(b0, (r,), -1.5 * M), Write(w2, (flag, r), -2.0 / M)]
        r = r0 + C
        W += [Write(w0, (r, bias), M), Write(b0, (r,), 0.0), Write(w2, (flag, r), 1.0 / M)]
        return W

    @classmethod
    def test(cls, params, layout, brain, n=256, slot=None, tol=1e-3, seed=0) -> Report:
        layout, classes, codec = _split(layout, params)
        if slot is None:
            return Report("Compare", False, note="Compare.test needs slot=")
        mlp = brain.model.blocks[slot.layer].mlp
        a, b = layout[params["a"]], layout[params["b"]]
        flag = layout.flag(params["flag"])
        bias = layout.flag(params.get("bias") or "bias")
        g = torch.Generator().manual_seed(seed)
        x = torch.zeros(n, layout.d)
        ia = torch.randint(0, a.stop - a.start, (n,), generator=g)
        same = torch.rand(n, generator=g) < 0.5
        ib = torch.where(same & (ia < b.stop - b.start), ia, torch.randint(0, b.stop - b.start, (n,), generator=g))
        x[torch.arange(n), a.start + ia] = 1.0
        x[torch.arange(n), b.start + ib] = 1.0
        x[:, bias] = 1.0
        with torch.no_grad():
            y = mlp(x)
        want = (ia != ib).float()
        err = float((y[:, flag] - want).abs().max())
        return Report("Compare", err <= tol, n=n, subspace_err=err, tol=tol, argmax_acc=1.0, min_gap=float("inf"))


# =====================================================================================================================
# Branch -- a route query; Readout -- the unembedding
# =====================================================================================================================

class Branch(Instruction):
    """`Branch(mixer, table)` (§21.3 item 7; NEW): an `AttnRes` mixer (`res="attnres"`: `blocks.{L}.res_attn` before the
    attention, `res_mlp` before the MLP) whose pseudo-query `w` carries M on the flag dims named in `table` (flag -> the source
    expected to carry it), so the source whose flag is set takes the softmax mass (a hard per-token route, DESIGN §15). Absent,
    the residual is the plain sum. `mixer` is "attn" or "mlp"."""
    PARAMS = ("mixer", "table")

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        return Needs()

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        M = _M(params, arch, M)
        slot = slot if slot is not None else Slot()
        mixer = params.get("mixer", "attn")
        if mixer not in ("attn", "mlp"):
            raise InstructionError(f"{_who(slot, 'Branch')}: mixer must be 'attn' or 'mlp', got {mixer!r}")
        name = f"blocks.{int(slot.layer)}.res_{mixer}.w"
        return [Write(name, (layout.flag(flag),), M) for flag in dict(params["table"])]


class Readout(Instruction):
    """`Readout(sub, vocab, M_out)` (§21.3 item 8): `head.weight[v, sub.start + v] = M_out` for v < vocab (MOVES e18.py line 77).
    `vocab` is an int or a dims symbol ("V", "nA"); it must not exceed `sub`'s width or the head's rows."""
    PARAMS = ("sub", "vocab", "M_out")

    @classmethod
    def needs(cls, params, layout, arch) -> Needs:
        cls.check_params(params)
        return Needs()

    @classmethod
    def emit(cls, params, layout, arch, slot, M) -> list[Write]:
        cls.check_params(params)
        layout, classes, codec = _split(layout, params)
        s = layout[params["sub"]]
        vocab = _int_or_symbol(params.get("vocab", s.stop - s.start), arch, "Readout.vocab")
        if vocab > s.stop - s.start:
            raise InstructionError(f"Readout: vocab {vocab} exceeds the width of {params['sub']!r} ({s.stop - s.start})")
        M_out = float(params.get("M_out", 10.0))
        return [Write("head.weight", (v, s.start + v), M_out) for v in range(vocab)]

    @classmethod
    def test(cls, params, layout, brain, n=256, slot=None, tol=1e-3, seed=0) -> Report:
        layout, classes, codec = _split(layout, params)
        s = layout[params["sub"]]
        vocab = _int_or_symbol(params.get("vocab", s.stop - s.start), getattr(brain, "arch", None), "Readout.vocab")
        g = torch.Generator().manual_seed(seed)
        sym = torch.randint(0, vocab, (n,), generator=g)
        x = torch.zeros(n, layout.d)
        x[torch.arange(n), s.start + sym] = 1.0
        with torch.no_grad():
            logits = brain.model.head(x)
        acc, gap = _argmax_stats(logits, sym)
        M_out = float(params.get("M_out", 10.0))
        return Report("Readout", acc == 1.0 and gap >= M_out - tol, n=n, argmax_acc=acc, min_gap=gap, M=M_out, subspace_err=0.0, tol=tol)


# =====================================================================================================================
# the boundary parameters -- no weights of their own; they fill BoundaryOp
# =====================================================================================================================

def _subs_list(subs, layout):
    out = []
    for s in ([subs] if isinstance(subs, str) else list(subs or [])):
        if s.endswith("[*]"):
            stem = s[:-3]
            hits = [n for n in layout.names if n.startswith(stem + "[")]
            if not hits:
                raise InstructionError(f"boundary: {s!r} names no subspace of the layout")
            out += hits
        else:
            layout[s]
            out.append(s)
    return out


@dataclass
class Keep:
    """The subspaces that survive the boundary on the operated tokens (a diagonal mask; §21.3 item 9)."""
    subs: list[str]

    def mask(self, layout) -> torch.Tensor:
        m = torch.zeros(layout.d)
        for s in _subs_list(self.subs, layout):
            m[layout[s]] = 1.0
        return m


@dataclass
class Clear:
    """The complement of Keep on the named subspaces: the scratch the boundary zeroes (E28: the gathered neighbours)."""
    subs: list[str]

    def mask(self, layout) -> torch.Tensor:
        m = torch.ones(layout.d)
        for s in _subs_list(self.subs, layout):
            m[layout[s]] = 0.0
        return m


@dataclass
class Quantise:
    """`Quantise(read, write)`: a read matrix (d x g) summing the `read` subspaces into one group of g = width(write) pre-
    activations, argmax, a one-hot into `write` (E28 lines 184-187: the colour re-quantised; the executive: util + noise -> action)."""
    read: list[str]
    write: str

    def matrix(self, layout) -> tuple[torch.Tensor, slice]:
        w = layout[self.write]
        g = w.stop - w.start
        R = torch.zeros(layout.d, g)
        for s in _subs_list(self.read, layout):
            sl = layout[s]
            if sl.stop - sl.start < g:
                raise InstructionError(f"Quantise: read {s!r} ({sl.stop - sl.start}) is narrower than write {self.write!r} ({g})")
            R[torch.arange(sl.start, sl.start + g), torch.arange(g)] = 1.0     # E28: colour(17) read over V = 16 -> the first g dims
        return R, w


@dataclass
class Commit:
    """`Commit(from, to)`: copy after quantise (E28's `pred -> colour` = the simulator advancing one step)."""
    src: str
    dst: str

    def slices(self, layout) -> tuple[slice, slice]:
        f, t = layout[self.src], layout[self.dst]
        if f.stop - f.start > t.stop - t.start:
            raise InstructionError(f"Commit: {self.src!r} is wider than {self.dst!r}")
        return f, slice(t.start, t.start + (f.stop - f.start))


@dataclass
class Anchor:
    """`Anchor(sub, from)`: the slice the register's or runner's anchor vector is copied into on every cell token (E28: the action)."""
    sub: str
    source: str = "runner"

    def slice(self, layout) -> slice:
        return layout[self.sub]


@dataclass
class Halt:
    """`Halt(flag)`: the flag dim the runner reads (the loop also stops on register convergence)."""
    flag: str

    def dim(self, layout) -> int:
        return layout.flag(self.flag)


def boundary_op(layout, keep=None, clear=None, quantise=(), commit=(), anchor=None, halt=None, register_dims=(), cls_flags=None,
                cell_flag=None, register_flag=None, anchor_token=None, tie_tol=1e-3):
    """Build `h1_lid.BoundaryOp` from the boundary parameters (names or the dataclasses above). `keep` and `clear` partition the
    layout: given both, they must agree (a subspace in both is an error); given one, the other is its complement. `register_flag`,
    `anchor_token` and `tie_tol` pass through to the operator (the register tokens the convergence test compares; the token whose
    anchor slice is the anchor when the runner gives none; the quantise's tie tolerance = 1 - p*, `brainbuilder.compile`)."""
    keep_m = Keep(keep).mask(layout) if keep is not None else None
    clear_m = Clear(clear).mask(layout) if clear is not None else None
    if keep_m is not None and clear_m is not None:
        both = [n for n in layout.names if float(keep_m[layout[n]].sum()) and not float(clear_m[layout[n]].sum())]
        if both:
            raise InstructionError(f"boundary: subspaces {both} are in both keep and clear")
        mask = keep_m
        gaps = [n for n in layout.names if not float(keep_m[layout[n]].sum()) and float(clear_m[layout[n]].sum())]
        if gaps:
            raise InstructionError(f"boundary: subspaces {gaps} are in neither keep nor clear")
    else:
        mask = keep_m if keep_m is not None else (clear_m if clear_m is not None else torch.ones(layout.d))
    quant = []
    for q in quantise:
        q = q if isinstance(q, Quantise) else Quantise(q["read"], q["write"])
        quant.append(q.matrix(layout))
    comm = []
    for c in commit:
        c = c if isinstance(c, Commit) else Commit(*c)
        comm.append(c.slices(layout))
    anc = None
    if anchor is not None:
        a = anchor if isinstance(anchor, Anchor) else (Anchor(anchor["sub"], anchor.get("from", "runner")) if isinstance(anchor, dict) else Anchor(anchor))
        anc = a.slice(layout)
    hflag = None
    if halt is not None:
        hflag = (halt if isinstance(halt, Halt) else Halt(halt)).dim(layout)
    return _h1().BoundaryOp(mask, quant, comm, anc, hflag, list(register_dims), cls_flags, cell_flag, register_flag, anchor_token, tie_tol)


# =====================================================================================================================
# the whitelist
# =====================================================================================================================

INSTRUCTIONS: dict[str, type[Instruction]] = {
    "Gather": Gather, "Match": Match, "Pool": Pool, "Broadcast": Broadcast,
    "Row": Row, "Compare": Compare, "Branch": Branch, "Readout": Readout,
}

BOUNDARY_PARAMETERS = {"Keep": Keep, "Clear": Clear, "Quantise": Quantise, "Commit": Commit, "Anchor": Anchor, "Halt": Halt}

__all__ = ["Write", "Needs", "Slot", "Report", "Instruction", "InstructionError", "scatter", "derived_M",
           "Gather", "Match", "Pool", "Broadcast", "Row", "Compare", "Branch", "Readout",
           "Keep", "Clear", "Quantise", "Commit", "Anchor", "Halt", "boundary_op",
           "INSTRUCTIONS", "BOUNDARY_PARAMETERS", "GATHER_GAP_MIN", "MATCH_DRIFT_MAX", "POS_CODES"]
