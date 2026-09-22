"""ziplib.codec — data <-> tokens (DESIGN §21.2; the module spec's `codec.py`).

A TOKEN CLASS says what a token is: which class flag it carries and which subspaces it fills from where ("input" = the
caller's data, "coord.row"/"coord.col" = its grid coordinates, "const:<v>" = a fixed symbol, "store" = a store entry's
key/value, "runner" = filled by the runner between passes). The CODEC turns data into residual vectors under the layout:

  encode(cls, **fields)     one token of a class
  encode_frame(frame)       MOVES `WrittenSim.encode` (e28.py): one cell token per grid cell -- colour one-hot, ROW/COL
                            one-hot (the `onehot` coordinate code), the cell flag; plus the (row, col) coordinate list
  encode_seq(seq)           a text as the row-0 case: token id as the input, col = position (E18's RoPE is `rope2d` at row 0)
  nulls()                   MOVES e28.py lines 151-155: the border token (its flag; colour = BORDER) and the zero token (its flag)
  entry(key, value, conf)   MOVES e28.py lines 134-150: a store entry as a memory token -- one-hot per key subspace, the class
                            flag, the value one-hot; a key subspace the mask dropped is left all-zero = a WILDCARD
  register(name, slot)      a register token: the register class flag (+ the slot id, when a `slot` subspace exists)
  decode(v, tol)            argmax per one-hot subspace, None where the slice is not one-hot to `tol` (the decompiler's primitive)
  coords_for(tokens)        the T x 2 coordinate list `Attn.forward(coords=)` takes; non-cell tokens get (0, 0)

Coordinate codes. `onehot` writes the row and column as one-hots into the `row`/`col` subspaces (E28; exact, small
grids). `rope2d` writes only the `bias` flag (E18's constant channel every embedding carries) and leaves the coordinates
to the `coords` argument of `Attn.forward`, which rotates the row-axis rotary pairs by `row * theta` and the column-axis
pairs by `col * theta`. Symbols: a NEGATIVE symbol indexes a one-hot subspace from its end, so the data's BORDER = -1
lands on the last colour dim, which is `WrittenSim.BORDER = V` (index C - 1 of `colour(C = V + 1)`): no domain
constant lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from .layout import Layout, LayoutError

FIELD_SOURCES = ("input", "coord.row", "coord.col", "store", "runner")   # plus "const:<v>"
POS_CODES = ("onehot", "rope2d", "rope")                       # "rope": the 1-D sequence case, encoded as rope2d (bias flag; position in coords)


class CodecError(ValueError):
    pass


@dataclass
class TokenClass:
    """`name`: cell, entry, register, border, zero, ...; `flag`: the class-flag dim's name in the layout;
    `fields`: subspace -> "input" | "coord.row" | "coord.col" | "const:<v>" | "store" | "runner"."""
    name: str
    flag: str
    fields: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for sub, src in self.fields.items():
            if not (src in FIELD_SOURCES or src.startswith("const:")):
                raise CodecError(f"token class {self.name!r}: field {sub!r} has source {src!r}; "
                                 f"allowed: {FIELD_SOURCES} or 'const:<v>'")

    def with_source(self, source):
        return [sub for sub, src in self.fields.items() if src == source]

    @property
    def consts(self):
        return {sub: src[len("const:"):] for sub, src in self.fields.items() if src.startswith("const:")}


def _parse_const(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


class Codec:
    """Data <-> tokens under a `Layout` and a list of `TokenClass`es (module docstring)."""

    def __init__(self, layout: Layout, token_classes: Iterable[TokenClass | dict], pos: str = "onehot"):
        if pos not in POS_CODES:
            raise CodecError(f"pos must be one of {POS_CODES}, got {pos!r}")
        self.layout, self.pos = layout, pos
        self.classes: dict[str, TokenClass] = {}
        for c in token_classes:
            c = c if isinstance(c, TokenClass) else TokenClass(c.get("name") or c["class"], c["flag"], dict(c.get("fields", {})))
            if c.name in self.classes:
                raise CodecError(f"token class {c.name!r} declared twice")
            for sub, src in c.fields.items():
                if sub not in layout and not (pos != "onehot" and src.startswith("coord.")):
                    raise CodecError(f"token class {c.name!r}: field {sub!r} is not a subspace of the layout {layout.names}")
            layout.flag(c.flag)                                        # must resolve
            self.classes[c.name] = c
        self.d = layout.d
        try:                                                           # rope2d: the constant channel q/k read (E18's BIAS dim)
            self._bias = layout.flag("bias")
        except LayoutError:
            self._bias = None
        if pos != "onehot" and self._bias is None:
            raise CodecError("pos='rope2d' needs a 'bias' flag in the layout (E18's constant channel the position heads read)")

    # -- resolving classes -------------------------------------------------------------------------------------------
    def _class(self, cls):
        if isinstance(cls, TokenClass):
            return cls
        try:
            return self.classes[cls]
        except KeyError:
            raise CodecError(f"no token class {cls!r}; the codec has {list(self.classes)}") from None

    def _class_with(self, source, fallback_name):
        """The unique class carrying a field of `source`, else the class named `fallback_name`."""
        hits = [c for c in self.classes.values() if source in c.fields.values()]
        if len(hits) == 1:
            return hits[0]
        if fallback_name in self.classes:
            return self.classes[fallback_name]
        raise CodecError(f"{'no' if not hits else len(hits)} token classes carry a {source!r} field and none is named "
                         f"{fallback_name!r}; pass cls= explicitly")

    @property
    def cell_class(self) -> TokenClass:
        return self._class_with("coord.row", "cell")

    @property
    def entry_class(self) -> TokenClass:
        return self._class_with("store", "entry")

    @property
    def register_class(self) -> TokenClass:
        return self._class_with("runner", "register")

    def flag_of(self, cls) -> int:
        return self.layout.flag(self._class(cls).flag)

    # -- writing one subspace ----------------------------------------------------------------------------------------
    def _put(self, vec, sub, value):
        """Write `value` into subspace `sub` of `vec`: an int -> a one-hot (negative = from the end); a sequence of the
        subspace's width -> copied (a distribution, a scalar field); a bool/number into a width-1 flag or scalar."""
        s = self.layout[sub]
        w = s.stop - s.start
        kind = self.layout.kind(sub)
        if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
            t = torch.as_tensor(np.asarray(value), dtype=vec.dtype).reshape(-1)
            if t.numel() != w:
                raise CodecError(f"subspace {sub!r} is {w} wide; got a vector of {t.numel()}")
            vec[s] = t
            return
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        if kind in ("onehot", "dist") or (kind == "flag" and w > 1):
            k = int(value)
            if not -w <= k < w:
                raise CodecError(f"symbol {k} is outside subspace {sub!r} of width {w}")
            vec[s.start + (k % w)] = 1.0
        else:                                                          # a width-1 flag, or a scalar
            if w == 1:
                vec[s.start] = float(value)
            else:
                vec[s] = float(value)

    # -- one token ---------------------------------------------------------------------------------------------------
    def encode(self, cls, **fields) -> torch.Tensor:
        """One token of class `cls`. Keyword `fields` give the values of its "input"/"store"/"runner" subspaces (by
        subspace name) and `row=`/`col=` its coordinates; constants come from the class; a field not given stays zero
        (a wildcard on a store entry, an empty register slot)."""
        c = self._class(cls)
        vec = torch.zeros(self.d)
        vec[self.layout.flag(c.flag)] = 1.0
        unknown = set(fields) - set(c.fields) - {"row", "col"}
        if unknown:
            raise CodecError(f"token class {c.name!r} has no fields {sorted(unknown)}; it has {list(c.fields)}")
        for sub, src in c.fields.items():
            if src.startswith("const:"):
                self._put(vec, sub, _parse_const(src[len("const:"):]))
            elif src in ("coord.row", "coord.col"):
                if self.pos == "onehot":
                    key = "row" if src == "coord.row" else "col"
                    if key in fields and fields[key] is not None:
                        self._put(vec, sub, int(fields[key]))
                else:
                    vec[self._bias] = 1.0                              # rope2d: the constant channel; coords carry position
            elif sub in fields and fields[sub] is not None:            # input | store | runner
                self._put(vec, sub, fields[sub])
        return vec

    # -- a frame (MOVES WrittenSim.encode) ---------------------------------------------------------------------------
    def encode_frame(self, frame, cls=None, **consts):
        """One cell token per cell, raster order t = i * W + j: the cell's symbol one-hot in the class's "input"
        subspace, ROW/COL one-hots under `onehot` (the bias flag under `rope2d`), the cell flag; and coords (H*W, 2) =
        (i, j). Extra "input" fields of the class take the constants in `consts`."""
        c = self.cell_class if cls is None else self._class(cls)
        frame = np.asarray(frame)
        if frame.ndim != 2:
            raise CodecError(f"encode_frame wants an H x W frame, got shape {frame.shape}")
        H, W = frame.shape
        rows, cols = np.divmod(np.arange(H * W), W)
        x, coords = self._encode_grid(c, frame.reshape(-1), rows, cols, consts)
        return x, coords

    def encode_seq(self, seq, cls=None, **consts):
        """A sequence as the row-0 case of a frame: token t at (row 0, col t). Under `rope2d` this is E18's RoPE."""
        c = self.cell_class if cls is None else self._class(cls)
        seq = np.asarray(seq).reshape(-1)
        L = len(seq)
        return self._encode_grid(c, seq, np.zeros(L, dtype=np.int64), np.arange(L), consts)

    def _encode_grid(self, c, symbols, rows, cols, consts):
        T = len(symbols)
        x = torch.zeros(T, self.d)
        t = torch.arange(T)
        inputs = c.with_source("input")
        if not inputs:
            raise CodecError(f"token class {c.name!r} has no 'input' field to put the data in")
        data_sub, others = inputs[0], inputs[1:]
        missing = [o for o in others if o not in consts]
        if missing:
            raise CodecError(f"token class {c.name!r}: 'input' fields {missing} need values (pass them as keywords)")
        s = self.layout[data_sub]
        w = s.stop - s.start
        sym = torch.as_tensor(np.asarray(symbols, dtype=np.int64))
        if bool(((sym < -w) | (sym >= w)).any()):
            raise CodecError(f"a symbol is outside subspace {data_sub!r} of width {w}: range {int(sym.min())}..{int(sym.max())}")
        x[t, s.start + (sym % w)] = 1.0
        x[:, self.layout.flag(c.flag)] = 1.0
        rows_t, cols_t = torch.as_tensor(np.asarray(rows, dtype=np.int64)), torch.as_tensor(np.asarray(cols, dtype=np.int64))
        for sub, src in c.fields.items():
            if src.startswith("const:"):
                v = torch.zeros(self.d)
                self._put(v, sub, _parse_const(src[len("const:"):]))
                x[:, self.layout[sub]] = v[self.layout[sub]]
            elif src in ("coord.row", "coord.col") and self.pos == "onehot":
                ss = self.layout[sub]
                idx = rows_t if src == "coord.row" else cols_t
                if bool((idx >= ss.stop - ss.start).any()):
                    raise CodecError(f"coordinate {int(idx.max())} does not fit subspace {sub!r} of width {ss.stop - ss.start}")
                x[t, ss.start + idx] = 1.0
            elif sub in others:
                v = torch.zeros(self.d)
                self._put(v, sub, consts[sub])
                x[:, self.layout[sub]] = v[self.layout[sub]]
        if self.pos != "onehot":                                       # rope/rope2d: a grid token's position lives in coords; the bias
            x[:, self._bias] = 1.0                                     # channel is what the position heads' q/k read
        coords = torch.stack([rows_t, cols_t], 1)
        return x, coords

    # -- the null tokens (MOVES e28.py lines 151-155) ----------------------------------------------------------------
    def nulls(self, names=("border", "zero")) -> torch.Tensor:
        """The sink tokens, stacked in the given order: each is its class flag plus the class's constants (E28: the
        border token's colour is BORDER; the zero token is only its flag)."""
        return torch.stack([self.encode(n) for n in names])

    # -- a store entry as a memory token (MOVES e28.py lines 134-150) -------------------------------------------------
    def entry(self, key: dict, value: dict, conf=None, cls=None) -> torch.Tensor:
        """`key`: subspace -> symbol for the cells the mask keeps (a subspace absent from `key` stays all-zero: a
        wildcard); `value`: subspace -> symbol (E28's OUT = the entry's majority colour); `conf`: written into the
        class's `conf` field when it has one. The class flag is set."""
        c = self.entry_class if cls is None else self._class(cls)
        fields = {}
        for sub, sym in {**key, **value}.items():
            if sub not in c.fields:
                raise CodecError(f"entry: {sub!r} is not a field of token class {c.name!r} ({list(c.fields)})")
            fields[sub] = sym
        if conf is not None and "conf" in c.fields:
            fields["conf"] = conf
        return self.encode(c, **fields)

    # -- a register token ----------------------------------------------------------------------------------------------
    def register(self, name: str, slot: int, cls=None) -> torch.Tensor:
        """The register class flag, its constants, and the slot id as a one-hot in the `slot` subspace when the layout
        declares one (otherwise the slot is the token's position among the registers). Its `holds` start empty."""
        c = self.register_class if cls is None else self._class(cls)
        vec = self.encode(c)
        if "slot" in self.layout:
            self._put(vec, "slot", int(slot))
        return vec

    # -- reading a token back -----------------------------------------------------------------------------------------
    def decode(self, v, tol: float = 1e-3) -> dict:
        """Per subspace: onehot/dist -> the argmax, or None when the slice is not one-hot to `tol` (a `dist` that is
        not one-hot returns its raw slice instead, for the lifter's count distribution); flag -> 0/1/None per dim
        (labelled dims by label); scalar -> the value (width 1) or the slice."""
        v = torch.as_tensor(v).reshape(-1)
        if v.numel() != self.d:
            raise CodecError(f"decode wants a vector of d = {self.d}, got {v.numel()}")
        out = {}
        for name in self.layout:
            s = self.layout[name]
            z = v[s]
            kind = self.layout.kind(name)
            if kind in ("onehot", "dist"):
                k = int(z.argmax())
                onehot = torch.zeros_like(z)
                onehot[k] = 1.0
                if float((z - onehot).abs().max()) <= tol:
                    out[name] = k
                else:
                    out[name] = None if kind == "onehot" else z.clone()
            elif kind == "flag":
                bits = [1 if abs(float(b) - 1.0) <= tol else 0 if abs(float(b)) <= tol else None for b in z]
                if z.numel() == 1:
                    out[name] = bits[0]
                elif name in self.layout.labels:
                    out[name] = dict(zip(self.layout.labels[name], bits))
                else:
                    out[name] = bits
            else:
                out[name] = float(z[0]) if z.numel() == 1 else z.clone()
        return out

    def token_class_of(self, v, tol: float = 1e-3):
        """The name of the class whose flag is set in `v` (None if none or several)."""
        v = torch.as_tensor(v).reshape(-1)
        hits = [c.name for c in self.classes.values() if abs(float(v[self.layout.flag(c.flag)]) - 1.0) <= tol]
        return hits[0] if len(hits) == 1 else None

    # -- coordinates for Attn.forward(coords=) ------------------------------------------------------------------------
    def coords_for(self, tokens, cell_coords=None) -> torch.Tensor:
        """(T, 2) coordinates for a token sequence: cell tokens carry their (row, col) -- read from their ROW/COL
        one-hots under `onehot`, taken in order from `cell_coords` under `rope2d` -- and every other token (nulls,
        entries, registers) gets (0, 0); they are excluded from gathers by their class flags, as in E28."""
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 3:
            tokens = tokens[0]
        T = tokens.shape[0]
        coords = torch.zeros(T, 2, dtype=torch.long)
        c = self.cell_class
        is_cell = tokens[:, self.layout.flag(c.flag)] > 0.5
        rows_sub, cols_sub = c.with_source("coord.row"), c.with_source("coord.col")
        if self.pos == "onehot" and rows_sub and cols_sub and cell_coords is None:
            coords[is_cell, 0] = tokens[is_cell][:, self.layout[rows_sub[0]]].argmax(1)
            coords[is_cell, 1] = tokens[is_cell][:, self.layout[cols_sub[0]]].argmax(1)
        else:
            if cell_coords is None:
                raise CodecError("coords_for: under rope2d the cell coordinates are not in the tokens; pass cell_coords")
            cc = torch.as_tensor(cell_coords, dtype=torch.long).reshape(-1, 2)
            n = int(is_cell.sum())
            if cc.shape[0] != n:
                raise CodecError(f"coords_for: {n} cell tokens but {cc.shape[0]} coordinates")
            coords[is_cell] = cc
        return coords


def coords_for(tokens, codec: Codec, cell_coords=None) -> torch.Tensor:
    """Module-level form of `Codec.coords_for` (the spec's `coords_for(tokens)`; the codec says which flag is the cell's)."""
    return codec.coords_for(tokens, cell_coords)


__all__ = ["TokenClass", "Codec", "CodecError", "coords_for", "FIELD_SOURCES", "POS_CODES"]
