"""ziplib.layout — the residual layout (DESIGN §21.2; the module spec's `layout.py`).

A brain's residual vector is a sequence of named SUBSPACES, each a slice of dims. `Layout.allocate` lays them out
sequentially in declaration order -- exactly the `base += C` of `WrittenSim.__init__` (e28.py lines 75-84): colour,
one neighbour subspace per offset, the action, the four class flags, ROW, COL, OUT. Symbolic widths ("C", "nA", "H",
...) are bound from `dims` (the blueprint's `Arch.dims`). Dims left over when `d > d_layout` sit on a free-list that a
later store's subspace may claim (`free(n)`); a layout wider than `d` is an ERROR naming the dims needed --
superposition is refused (DESIGN §21.4 step 2), a narrower brain is a different blueprint.

`channels(head_dim, needs)` is E18's lesson as a rule (e18.py lines 56 and 67): inside one RoPE head, CONTENT dims
go in the LOWEST-frequency rotary pairs (the highest pair indices, where rotation over the sequence is negligible)
and POSITION goes in the HIGHEST-frequency pairs (the lowest indices, which discriminate a distance of one from
zero). `h1_lid._freqs` gives pair c the frequency base^(-c / n_pairs): pair 0 is the fastest.

No model code here; nothing is learned. Everything is a slice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

KINDS = ("onehot", "flag", "scalar", "dist")


class LayoutError(ValueError):
    """The layout does not fit, a name is unknown, or the free-list is exhausted."""


@dataclass
class Subspace:
    """One named slice of the residual. `width` is an int or a symbol bound from `dims`. `kind`: "onehot" (exactly one
    dim at 1), "flag" (0/1 dims; a width-1 flag is a token-class or halt flag), "scalar" (real values), "dist" (a
    distribution over the dims, one-hot after `quantise`). `labels`, optional, names the dims of a wider flag or
    scalar subspace (the worked example's `flags(9)` = cell, entry, register, ...), so `Layout.flag("halt")` resolves."""
    name: str
    width: int | str
    kind: str = "onehot"
    labels: list[str] | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise LayoutError(f"subspace {self.name!r}: kind {self.kind!r} is not one of {KINDS}")


def _bind(width, dims, name):
    if isinstance(width, bool) or not isinstance(width, (int, str)):
        raise LayoutError(f"subspace {name!r}: width must be an int or a symbol, got {width!r}")
    if isinstance(width, str):
        if dims is None or width not in dims:
            raise LayoutError(f"subspace {name!r}: symbolic width {width!r} is not bound; dims = {sorted(dims or {})}")
        width = int(dims[width])
    if width <= 0:
        raise LayoutError(f"subspace {name!r}: width must be > 0, got {width}")
    return width


class Layout:
    """The resolved layout: name -> slice, in allocation order, over a residual of `d` dims.

    Build with `Layout.allocate(subspaces, d=None, dims=None)`; read with `layout[name]` (a slice), `layout.flag(name)`
    (one dim), `layout.d` / `layout.d_layout`; claim spare dims with `layout.free(n)`; round-trip with `to_json` /
    `from_json`."""

    def __init__(self, slices, kinds, d, widths=None, labels=None, free_start=None):
        self.slices: dict[str, slice] = dict(slices)                # name -> slice, in declaration order
        self.kinds: dict[str, str] = dict(kinds)                    # name -> kind
        self.widths: dict[str, int | str] = dict(widths or {name: s.stop - s.start for name, s in self.slices.items()})
        self.labels: dict[str, list[str]] = dict(labels or {})       # name -> dim labels (wider flag/scalar subspaces)
        self.d = int(d)
        self._free_start = self.d_layout if free_start is None else int(free_start)   # the free-list is [free_start, d)
        self._claimed: dict[str, slice] = {}

    # -- construction ------------------------------------------------------------------------------------------------
    @classmethod
    def allocate(cls, subspaces: Iterable[Subspace | dict], d: int | None = None, dims: dict | None = None) -> "Layout":
        """Sequential slices in declaration order (e28.py `base += C`), symbolic widths bound from `dims`. `d=None`
        makes the residual exactly as wide as the layout; `d > d_layout` leaves the spare dims on the free-list;
        `d < d_layout` raises naming the dims needed."""
        subs = [s if isinstance(s, Subspace) else Subspace(**s) for s in subspaces]
        slices, kinds, widths, labels = {}, {}, {}, {}
        base = 0
        for s in subs:
            if s.name in slices:
                raise LayoutError(f"subspace {s.name!r} declared twice")
            w = _bind(s.width, dims, s.name)
            if s.labels is not None and len(s.labels) != w:
                raise LayoutError(f"subspace {s.name!r}: {len(s.labels)} labels for width {w}")
            slices[s.name] = slice(base, base + w)
            kinds[s.name] = s.kind
            widths[s.name] = s.width
            if s.labels is not None:
                labels[s.name] = list(s.labels)
            base += w
        if d is None:
            d = base
        if base > d:
            raise LayoutError(f"layout needs {base} dims but d = {d}: {base - d} more dims are needed "
                              f"(superposition refused; widen d or drop a subspace)")
        return cls(slices, kinds, d, widths, labels)

    # -- reading -----------------------------------------------------------------------------------------------------
    @property
    def d_layout(self) -> int:
        """The dims the declared subspaces occupy (the final `base` of WrittenSim.__init__)."""
        return max((s.stop for s in self.slices.values()), default=0)

    @property
    def names(self) -> list[str]:
        return list(self.slices)

    def __getitem__(self, name: str) -> slice:
        try:
            return self.slices[name]
        except KeyError:
            raise LayoutError(f"no subspace {name!r}; the layout has {self.names}") from None

    def __contains__(self, name) -> bool:
        return name in self.slices

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def width(self, name: str) -> int:
        s = self[name]
        return s.stop - s.start

    def kind(self, name: str) -> str:
        return self.kinds[name]

    def flag(self, name: str) -> int:
        """The one dim of a flag: a width-1 subspace by its name; a dim of a wider labelled subspace by its label
        ("halt" inside `flags(9)`) or by "sub.label" / "sub.<k>"."""
        if name in self.slices:
            s = self.slices[name]
            if s.stop - s.start != 1:
                raise LayoutError(f"subspace {name!r} is {s.stop - s.start} wide; a flag is one dim (use a label or 'name.k')")
            return s.start
        if "." in name:
            sub, key = name.split(".", 1)
            s = self[sub]
            if key.isdigit():
                k = int(key)
            elif sub in self.labels and key in self.labels[sub]:
                k = self.labels[sub].index(key)
            else:
                raise LayoutError(f"no dim {key!r} in subspace {sub!r} (labels: {self.labels.get(sub)})")
            if not 0 <= k < s.stop - s.start:
                raise LayoutError(f"dim {k} is outside subspace {sub!r} of width {s.stop - s.start}")
            return s.start + k
        for sub, labs in self.labels.items():
            if name in labs:
                return self.slices[sub].start + labs.index(name)
        raise LayoutError(f"no flag {name!r}; flags are width-1 subspaces or labelled dims: "
                          f"{[n for n, s in self.slices.items() if s.stop - s.start == 1] + [l for ls in self.labels.values() for l in ls]}")

    # -- the free-list -----------------------------------------------------------------------------------------------
    @property
    def n_free(self) -> int:
        return self.d - self._free_start

    def free(self, n: int, name: str | None = None) -> slice:
        """Claim `n` spare dims (the dims past the layout, below d), as a slice; registered under `name` when given
        so `layout[name]` finds it afterwards. Raises when the free-list is exhausted."""
        if n <= 0:
            raise LayoutError(f"free({n}): n must be > 0")
        if self._free_start + n > self.d:
            raise LayoutError(f"free({n}): only {self.n_free} spare dims left of d = {self.d} (layout {self.d_layout}); "
                              f"{self._free_start + n - self.d} more are needed")
        s = slice(self._free_start, self._free_start + n)
        self._free_start += n
        if name is not None:
            if name in self.slices:
                raise LayoutError(f"free({n}, {name!r}): subspace {name!r} already exists")
            self.slices[name] = s
            self.kinds[name] = "scalar"
            self.widths[name] = n
            self._claimed[name] = s
        return s

    # -- (de)serialisation -------------------------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(dict(
            d=self.d,
            free_start=self._free_start,
            subspaces=[dict(name=n, start=s.start, stop=s.stop, kind=self.kinds[n], width=self.widths.get(n, s.stop - s.start),
                            labels=self.labels.get(n), claimed=n in self._claimed) for n, s in self.slices.items()],
        ), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Layout":
        obj = json.loads(text) if isinstance(text, str) else text
        slices = {e["name"]: slice(e["start"], e["stop"]) for e in obj["subspaces"]}
        kinds = {e["name"]: e["kind"] for e in obj["subspaces"]}
        widths = {e["name"]: e.get("width", e["stop"] - e["start"]) for e in obj["subspaces"]}
        labels = {e["name"]: e["labels"] for e in obj["subspaces"] if e.get("labels") is not None}
        lay = cls(slices, kinds, obj["d"], widths, labels, free_start=obj.get("free_start"))
        lay._claimed = {e["name"]: slices[e["name"]] for e in obj["subspaces"] if e.get("claimed")}
        return lay

    def __repr__(self):
        body = ", ".join(f"{n}[{s.start}:{s.stop}]" for n, s in self.slices.items())
        return f"Layout(d={self.d}, d_layout={self.d_layout}, free={self.n_free}; {body})"

    # -- head channels (E18's rule) ----------------------------------------------------------------------------------
    @staticmethod
    def channels(head_dim: int, needs, name: str | None = None) -> "ChannelPlan":
        return channels(head_dim, needs, name)


@dataclass
class ChannelPlan:
    """Which rotary pairs of one head carry content and which carry position. Pair c is dims (2c, 2c+1); a content
    dim is written into dim 2c of its pair, as e18.py line 66-67 do (`W2[2 * c, T_SUB + v]`)."""
    content_pairs: list[int] = field(default_factory=list)
    position_pairs: list[int] = field(default_factory=list)

    @property
    def content_dims(self) -> list[int]:
        return [2 * c for c in self.content_pairs]

    @property
    def position_dims(self) -> list[int]:
        return [dim for c in self.position_pairs for dim in (2 * c, 2 * c + 1)]


def _need(needs, key, default=0):
    if needs is None:
        return default
    if isinstance(needs, dict):
        return int(needs.get(key, default) or 0)
    return int(getattr(needs, key, default) or 0)


def channels(head_dim: int, needs, name: str | None = None) -> ChannelPlan:
    """Allocate one head's rotary pairs: `needs.content_dims` content dims (one per pair) in the LOWEST-frequency pairs
    (the highest indices, `range(n_pairs - content, n_pairs)`: e18.py line 67, `lowest = range(HD // 2 - NT, HD // 2)`)
    and `needs.position_pairs` pairs in the HIGHEST-frequency pairs (`range(0, position)`: e18.py line 56, the four
    highest-frequency pairs). Raises naming the head when the two would overlap or the head is too narrow."""
    who = f"head {name!r}" if name is not None else "head"
    if head_dim <= 0 or head_dim % 2:
        raise LayoutError(f"{who}: head_dim must be a positive even number of dims (rotary pairs), got {head_dim}")
    n_pairs = head_dim // 2
    content = _need(needs, "content_dims")
    position = _need(needs, "position_pairs")
    if content < 0 or position < 0:
        raise LayoutError(f"{who}: negative need (content_dims={content}, position_pairs={position})")
    if content + position > n_pairs:
        raise LayoutError(f"{who}: {content} content pairs + {position} position pairs do not fit in {n_pairs} rotary pairs "
                          f"(head_dim {head_dim}); the head needs head_dim >= {2 * (content + position)}")
    return ChannelPlan(content_pairs=list(range(n_pairs - content, n_pairs)), position_pairs=list(range(0, position)))


__all__ = ["Subspace", "Layout", "LayoutError", "ChannelPlan", "channels", "KINDS"]
