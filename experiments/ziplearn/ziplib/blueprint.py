"""ziplib.blueprint — the blueprint schema (DESIGN §21.2.1; the module spec's `blueprint.py`).

A BLUEPRINT is one JSON document: the repeatable top-down instructions for building a brain -- which subspaces the
residual has, which token classes exist, which circuits (instruction instances) sit where and read/write what, which
stores and registers exist, how the loop's boundary operator behaves, the routes, and the tests -- never a weight that
encodes a fact. `brainbuilder.compile(Blueprint.load(path), arch)` turns it into a `Brain`. This module holds the
dataclasses (`Blueprint`, `Circuit`, `StoreSpec`, `Register`, `Loop`, `Boundary`, `Route`, `Test`, and `Arch`, the size
numbers), the JSON round trip (`load` / `save`), the template expansion (`expand`) and the mechanical refusals
(`validate`). Two nested pieces are NOT redefined here: a subspace is `ziplib.layout.Subspace` and a token class is
`ziplib.codec.TokenClass` (one canonical component per concept).

Templates (the conventions `blueprints/README.md` states). A subspace or circuit name containing `[o]` is a template
over the born receptive FIELD's offsets: `expand()` instantiates one per `lattice(r)` offset, named `nbr[di,dj]`
(`nbr[-1,-1]`, ..., `nbr[1,1]`), and a `per: "offset"` circuit's params value `"o"` becomes the bound offset `[di, dj]`.
`x[*]` names EVERY instance of the template `x[o]` and may stand in any list of names (a store key, a `Match` q/k list,
`reads`/`writes`, the boundary's `keep`/`clear`, a register's `holds`) or as a mapping key (a token class's fields);
`expand()` splices it. `validate()` reads the DECLARED form (so a message names `gather[o]`, not eight circuits) and
then expands as a check.

`validate() -> [str]` refuses, mechanically (the E29 guard, DESIGN §21.2.1): an `instr` outside `INSTRUCTIONS`
(imported lazily from `ziplib.instructions`; when that module is not importable ONE message says the whitelist was not
checked); a `reads`/`writes`/key/hold/boundary name that is not a declared subspace; fewer than one test; a `params`
value that is a CONTENT SYMBOL; a register with an `init`. What "content symbol" means here, operationally: every
STRING in `params` must resolve to something the blueprint declares -- a subspace (or `a - b` of two, for `Pool.src`),
a token class, a register, a width symbol (`vocab: "nA"`), an operation word (`OPERATION_WORDS`), the bound offset
`"o"` inside a `per: "offset"` circuit, or, under the key `W`, a value-matrix slot name -- anything else is a name the
brain has no referent for (a colour name, a character, a word); a bare INTEGER is a symbol index (a colour, a character
code) unless its key is an operation constant (`OPERATION_CONSTANTS`: `threshold`, `M_out`, `sign` in {-1, 1}, `vocab`)
or it weights a flag/scalar subspace inside a mapping (`Row.key = {goal_met: 1}`; `weights`); a list of numbers is an
offset LIST (a field, declared once under `field`) except a single `offset: [di, dj]` inside the field on a `per: null`
circuit; floats are gains and thresholds and pass.

`Arch(d_model, n_head, head_dim, n_layer, pos, max_len, p_star, dims)` holds the size numbers; `Arch.auto(dims)` leaves
the sizes to the compiler and `Arch.load(path)` reads them from JSON. `Blueprint.max_len(dims)` gives the sequence
length the worked example states (nulls + store capacities + H*W cells + register slots = 350 for gridworld) and
`Arch.sharpness(max_len)` the derived M = ln((max_len - 1) p* / (1 - p*)) of DESIGN §21.4 step 5 (10.45 at 350).
"""
from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from .layout import Layout, LayoutError, Subspace, KINDS

STANDARD_SYMBOLS = ("H", "W", "C", "V", "nA", "L")            # the width symbols DESIGN §21.2.1 names
FORMS = ("tokens", "bank", "rows")                            # a store's tensor form
FILLERS = ("core", "store", "runner")                         # what fills a register
PERS = (None, "offset")                                       # a circuit's `per`
POS_CODES = ("onehot", "rope2d", "rope")                      # Arch.pos ("rope" = the 1-D sequence case of rope2d, E18/B0)
FIELD_KINDS = ("lattice",)                                    # the only receptive-field kind designed
OPERATION_WORDS = ("auto", "replace", "add")                  # params strings that name an operation, not a thing
OPERATION_CONSTANTS = ("threshold", "M_out", "sign", "vocab", "M")  # params keys whose value is a bare number by design (M: a typed sharpness, e18's constants)
TEMPLATE, EVERY = "[o]", "[*]"


class BlueprintError(ValueError):
    """The document does not have the schema's shape (raised by `load`, `from_dict`, `expand`); the semantic refusals
    are the list `validate` returns."""


# ---------------------------------------------------------------------------------------------------------------------
# the receptive field
# ---------------------------------------------------------------------------------------------------------------------

def lattice(r: int) -> list[tuple[int, int]]:
    """The born receptive field: `ziplib.store.Field.lattice(r)` -- every offset (di, dj) with |di|, |dj| <= r, the centre
    excluded, in raster order (di outer, dj inner: the order of `LocalRule`'s window cells): 8 offsets at r = 1, 24 at
    r = 2, 80 at r = 4. The definition lives in `store.py` (the module spec's `Field`); this is the schema's call to it,
    imported lazily so that `import ziplib.blueprint` stays free of the tensor modules."""
    r = int(r)
    if r < 0:
        raise BlueprintError(f"field r must be >= 0, got {r}")
    from .store import Field
    return [(int(o[0]), int(o[1])) for o in Field.lattice(r)]


def _offset_tag(o) -> str:
    return f"[{int(o[0])},{int(o[1])}]"


# ---------------------------------------------------------------------------------------------------------------------
# the nested dataclasses (DESIGN §21.2.1, one per schema object)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Circuit:
    """One instruction INSTANCE placed in the brain. `instr` is a key of `INSTRUCTIONS`; `params` its parameters (names
    only: subspaces, token classes, registers, `W` slots, operation words and constants); `per` is None or "offset"
    (one instance per field offset, the name and params templated by `[o]`); `place` = {"layer": int|"auto", "head":
    int|"auto"}; `reads`/`writes` the subspaces the compiler orders layers by; `prunable` marks a head the sleep pass
    may zero. `extra` keeps keys the schema does not name (`validate` refuses them)."""
    name: str
    instr: str
    params: dict = dc_field(default_factory=dict)
    per: str | None = None
    place: dict = dc_field(default_factory=lambda: {"layer": "auto", "head": "auto"})
    reads: list = dc_field(default_factory=list)
    writes: list = dc_field(default_factory=list)
    prunable: bool = False
    extra: dict = dc_field(default_factory=dict)


@dataclass
class StoreSpec:
    """A store: its token class (`cls`; the JSON key is "class"), the key subspaces, the value subspace, the tensor form
    ("tokens" | "bank" | "rows"), the reserved `capacity` (rows of the sequence / bank) and the null token classes its
    readers land on."""
    name: str
    cls: str
    key: list = dc_field(default_factory=list)
    value: str = ""
    form: str = "tokens"
    capacity: int = 0
    nulls: list = dc_field(default_factory=list)
    extra: dict = dc_field(default_factory=dict)


@dataclass
class Register:
    """A register token: `slots` copies, the subspaces it `holds` across passes, and what fills it ("core" | "store" |
    "runner"). A register never has an `init` -- goals are filled by the goal model (DESIGN §21.1 item 7); an `init`
    key lands in `extra` and `validate` refuses it."""
    name: str
    slots: int = 1
    holds: list = dc_field(default_factory=list)
    filled_by: str = "core"
    extra: dict = dc_field(default_factory=dict)


@dataclass
class Boundary:
    """The boundary operator's parameters (DESIGN §21.3 item 9): `keep` and `clear` partition the subspaces; `quantise` =
    [{"read": [subspaces], "write": subspace}] (argmax over the read group, one-hot into write); `commit` = [[from, to]]
    copies after quantise; `anchor` = {"sub": subspace, "from": register | "runner"} or None; `halt` = a flag or None."""
    keep: list = dc_field(default_factory=list)
    clear: list = dc_field(default_factory=list)
    quantise: list = dc_field(default_factory=list)
    commit: list = dc_field(default_factory=list)
    anchor: dict | None = None
    halt: str | None = None
    extra: dict = dc_field(default_factory=dict)


@dataclass
class Loop:
    """`core`: "auto" (the compiler orders the layers) or an explicit layer list; `tied`: one core reused per pass;
    `boundary`: the `Boundary`; `sequence`: {"feedback": [[from, to]]} for the sequence axis (`Brain.think`) or None."""
    core: Any = "auto"
    tied: bool = True
    boundary: Boundary | None = None
    sequence: dict | None = None
    extra: dict = dc_field(default_factory=dict)


@dataclass
class Route:
    """A `Branch` route: the `AttnRes` mixer layer and its table flag -> source (DESIGN §15)."""
    mixer: Any = None
    table: dict = dc_field(default_factory=dict)
    extra: dict = dc_field(default_factory=dict)


@dataclass
class Test:
    """A blueprint test: `generator` and `oracle` are dotted names of callables, optionally with a parenthesised list of
    literal keyword arguments (`parse()` gives (dotted_name, kwargs)); `criterion` is {"exact": true} or {"min": x}."""
    name: str
    generator: str
    oracle: str
    criterion: dict = dc_field(default_factory=dict)
    extra: dict = dc_field(default_factory=dict)

    def parse(self, which: str = "generator") -> tuple[str, dict]:
        return parse_callable(getattr(self, which))


_CALL = re.compile(r"^\s*([A-Za-z_][\w\.]*)\s*(?:\((.*)\))?\s*$", re.S)


def parse_callable(text: str) -> tuple[str, dict]:
    """`"pkg.mod.fn(a=1, b=[0, 1])"` -> ("pkg.mod.fn", {"a": 1, "b": [0, 1]}); `"pkg.fn"` -> ("pkg.fn", {}). Keyword
    arguments only, each a Python literal (`ast.literal_eval`); nothing is evaluated."""
    m = _CALL.match(text or "")
    if not m:
        raise BlueprintError(f"not a callable reference: {text!r} (want 'dotted.name' or 'dotted.name(k=literal, ...)')")
    name, args = m.group(1), m.group(2)
    kwargs = {}
    if args is not None and args.strip():
        try:
            call = ast.parse(f"f({args})", mode="eval").body
        except SyntaxError as e:
            raise BlueprintError(f"cannot parse the arguments of {text!r}: {e}") from None
        if call.args:
            raise BlueprintError(f"{text!r}: positional arguments are not allowed; name every argument")
        for kw in call.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                raise BlueprintError(f"{text!r}: argument {kw.arg!r} is not a literal") from None
    return name, kwargs


# ---------------------------------------------------------------------------------------------------------------------
# Arch -- the size numbers
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Arch:
    """The architecture numbers (DESIGN §21.2.1): `d_model`, `n_head` (an int, "auto", or one int per layer), `head_dim`,
    `n_layer` -- each an int or "auto" (the compiler picks the smallest that fits); `pos` "onehot" | "rope2d"; `max_len`
    (an int, or "auto" = `Blueprint.max_len(dims)`); `p_star` the softmax mass the sharpness M is derived for; `dims`
    the width symbols' values ({"H": 8, "W": 11, "C": 17, "V": 16, "nA": 4} for gridworld)."""
    d_model: Any = "auto"
    n_head: Any = "auto"
    head_dim: Any = "auto"
    n_layer: Any = "auto"
    pos: str = "onehot"
    max_len: Any = "auto"
    p_star: float = 0.99
    dims: dict = dc_field(default_factory=dict)
    causal: Any = "auto"                                      # True | False | "auto" (= True for a 1-D `rope` sequence brain, E18; else False, E28)
    norm: str = "none"                                        # "none" (written one-hot codes, E28) | "layer" (e18.write's constants presuppose LayerNorm's rescaling)

    @classmethod
    def auto(cls, dims: dict, pos: str = "onehot", max_len: Any = "auto", p_star: float = 0.99) -> "Arch":
        """Every size "auto"; the compiler derives them from the blueprint (DESIGN §21.2.2's `Arch.auto`)."""
        return cls("auto", "auto", "auto", "auto", pos, max_len, p_star, {k: int(v) for k, v in dict(dims).items()})

    @classmethod
    def from_dict(cls, obj: dict) -> "Arch":
        if not isinstance(obj, dict):
            raise BlueprintError(f"an Arch is a JSON object, got {type(obj).__name__}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(obj) - known)
        if unknown:
            raise BlueprintError(f"Arch: unknown keys {unknown}; the keys are {sorted(known)}")
        a = cls(**{k: obj[k] for k in obj})
        a.dims = {k: int(v) for k, v in dict(a.dims or {}).items()}
        return a

    @classmethod
    def load(cls, path) -> "Arch":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return dict(d_model=self.d_model, n_head=self.n_head, head_dim=self.head_dim, n_layer=self.n_layer,
                    pos=self.pos, max_len=self.max_len, p_star=self.p_star, dims=dict(self.dims), causal=self.causal, norm=self.norm)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def sharpness(self, max_len: int | None = None) -> float:
        """M = ln((max_len - 1) p* / (1 - p*)) -- DESIGN §21.4 step 5: with the target one match ahead of every rival
        its softmax mass over max_len tokens is >= p*. 10.45 at max_len 350, p* 0.99."""
        T = self.max_len if max_len is None else max_len
        if not isinstance(T, int) or isinstance(T, bool) or T < 2:
            raise BlueprintError(f"sharpness needs an integer max_len >= 2, got {T!r} (resolve 'auto' with Blueprint.max_len(dims))")
        if not (0.0 < float(self.p_star) < 1.0):
            raise BlueprintError(f"p_star must lie in (0, 1), got {self.p_star!r}")
        return math.log((T - 1) * self.p_star / (1.0 - self.p_star))

    def validate(self) -> list[str]:
        msgs = []

        def size(name, v, allow_list=False):
            if v == "auto":
                return
            if isinstance(v, bool) or (isinstance(v, int) and v <= 0):
                msgs.append(f"arch.{name}: must be a positive int or 'auto', got {v!r}")
            elif isinstance(v, list):
                if not allow_list:
                    msgs.append(f"arch.{name}: a list is not allowed here, got {v!r}")
                elif not v or any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in v):
                    msgs.append(f"arch.{name}: a per-layer list needs one positive int per layer, got {v!r}")
            elif not isinstance(v, int):
                msgs.append(f"arch.{name}: must be a positive int or 'auto', got {v!r}")

        size("d_model", self.d_model)
        size("n_head", self.n_head, allow_list=True)
        size("head_dim", self.head_dim)
        size("n_layer", self.n_layer)
        size("max_len", self.max_len)
        if self.pos not in POS_CODES:
            msgs.append(f"arch.pos: must be one of {POS_CODES}, got {self.pos!r}")
        if self.causal not in (True, False, "auto"):
            msgs.append(f"arch.causal: must be true, false or 'auto', got {self.causal!r}")
        if self.norm not in ("none", "layer"):
            msgs.append(f"arch.norm: must be 'none' or 'layer', got {self.norm!r}")
        if isinstance(self.p_star, bool) or not isinstance(self.p_star, (int, float)) or not (0.0 < float(self.p_star) < 1.0):
            msgs.append(f"arch.p_star: must lie in (0, 1), got {self.p_star!r}")
        if not isinstance(self.dims, dict):
            msgs.append(f"arch.dims: must be a mapping symbol -> int, got {type(self.dims).__name__}")
        else:
            for k, v in self.dims.items():
                if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                    msgs.append(f"arch.dims[{k!r}]: must be a positive int, got {v!r}")
        return msgs


# ---------------------------------------------------------------------------------------------------------------------
# JSON -> dataclasses (the loader; shape errors raise BlueprintError naming the location)
# ---------------------------------------------------------------------------------------------------------------------

def _need(obj: dict, key: str, where: str, types=None, default=..., allow_none=False):
    if key not in obj:
        if default is ...:
            raise BlueprintError(f"{where}: missing key {key!r}")
        return default
    v = obj[key]
    if v is None and allow_none:
        return None
    if types is not None and (not isinstance(v, types) or (isinstance(v, bool) and bool not in _as_tuple(types))):
        want = " | ".join(t.__name__ for t in _as_tuple(types))
        raise BlueprintError(f"{where}.{key}: expected {want}, got {type(v).__name__} {v!r}")
    return v


def _as_tuple(types):
    return types if isinstance(types, tuple) else (types,)


def _extra(obj: dict, known: tuple) -> dict:
    return {k: copy.deepcopy(v) for k, v in obj.items() if k not in known}


def _object(obj, where):
    if not isinstance(obj, dict):
        raise BlueprintError(f"{where}: expected a JSON object, got {type(obj).__name__}")
    return obj


def _list(obj, where):
    if obj is None:
        return []
    if not isinstance(obj, list):
        raise BlueprintError(f"{where}: expected a list, got {type(obj).__name__}")
    return obj


def _subspace_from(obj, where) -> Subspace:
    _object(obj, where)
    known = ("name", "width", "kind", "labels")
    unknown = sorted(set(obj) - set(known))
    if unknown:
        raise BlueprintError(f"{where}: unknown keys {unknown}; a subspace has {list(known)}")
    try:
        return Subspace(_need(obj, "name", where, str), _need(obj, "width", where, (int, str)),
                        _need(obj, "kind", where, str, "onehot"), _need(obj, "labels", where, list, None, allow_none=True))
    except LayoutError as e:
        raise BlueprintError(f"{where}: {e}") from None


def _token_class():
    from .codec import TokenClass, CodecError                     # lazy: codec imports torch; the schema does not need it
    return TokenClass, CodecError


def _token_from(obj, where):
    _object(obj, where)
    TokenClass, CodecError = _token_class()
    known = ("class", "name", "flag", "fields")
    unknown = sorted(set(obj) - set(known))
    if unknown:
        raise BlueprintError(f"{where}: unknown keys {unknown}; a token class has ['class', 'flag', 'fields']")
    name = obj.get("class", obj.get("name"))
    if not isinstance(name, str):
        raise BlueprintError(f"{where}: missing key 'class' (the token class's name)")
    fields = _need(obj, "fields", where, dict, {})
    for k, v in fields.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise BlueprintError(f"{where}.fields: entries are subspace -> source strings, got {k!r}: {v!r}")
    try:
        return TokenClass(name, _need(obj, "flag", where, str), dict(fields))
    except CodecError as e:
        raise BlueprintError(f"{where}: {e}") from None


def _circuit_from(obj, where) -> Circuit:
    _object(obj, where)
    known = ("name", "instr", "params", "per", "place", "reads", "writes", "prunable")
    place = _need(obj, "place", where, dict, {"layer": "auto", "head": "auto"})
    return Circuit(name=_need(obj, "name", where, str), instr=_need(obj, "instr", where, str),
                   params=copy.deepcopy(_need(obj, "params", where, dict, {})),
                   per=_need(obj, "per", where, str, None, allow_none=True),
                   place=dict(place), reads=list(_list(obj.get("reads"), f"{where}.reads")),
                   writes=list(_list(obj.get("writes"), f"{where}.writes")),
                   prunable=_need(obj, "prunable", where, bool, False), extra=_extra(obj, known))


def _store_from(obj, where) -> StoreSpec:
    _object(obj, where)
    known = ("name", "class", "key", "value", "form", "capacity", "nulls")
    return StoreSpec(name=_need(obj, "name", where, str), cls=_need(obj, "class", where, str),
                     key=list(_list(obj.get("key"), f"{where}.key")), value=_need(obj, "value", where, str),
                     form=_need(obj, "form", where, str, "tokens"), capacity=_need(obj, "capacity", where, int, 0),
                     nulls=list(_list(obj.get("nulls"), f"{where}.nulls")), extra=_extra(obj, known))


def _register_from(obj, where) -> Register:
    _object(obj, where)
    known = ("name", "slots", "holds", "filled_by")
    return Register(name=_need(obj, "name", where, str), slots=_need(obj, "slots", where, int, 1),
                    holds=list(_list(obj.get("holds"), f"{where}.holds")),
                    filled_by=_need(obj, "filled_by", where, str, "core"), extra=_extra(obj, known))


def _boundary_from(obj, where) -> Boundary | None:
    if obj is None:
        return None
    _object(obj, where)
    known = ("keep", "clear", "quantise", "commit", "anchor", "halt")
    return Boundary(keep=list(_list(obj.get("keep"), f"{where}.keep")), clear=list(_list(obj.get("clear"), f"{where}.clear")),
                    quantise=copy.deepcopy(_list(obj.get("quantise"), f"{where}.quantise")),
                    commit=copy.deepcopy(_list(obj.get("commit"), f"{where}.commit")),
                    anchor=copy.deepcopy(_need(obj, "anchor", where, dict, None, allow_none=True)),
                    halt=_need(obj, "halt", where, str, None, allow_none=True), extra=_extra(obj, known))


def _loop_from(obj, where) -> Loop:
    _object(obj, where)
    known = ("core", "tied", "boundary", "sequence")
    return Loop(core=copy.deepcopy(obj.get("core", "auto")), tied=_need(obj, "tied", where, bool, True),
                boundary=_boundary_from(obj.get("boundary"), f"{where}.boundary"),
                sequence=copy.deepcopy(_need(obj, "sequence", where, dict, None, allow_none=True)), extra=_extra(obj, known))


def _route_from(obj, where) -> Route:
    _object(obj, where)
    known = ("mixer", "table")
    return Route(mixer=copy.deepcopy(obj.get("mixer")), table=dict(_need(obj, "table", where, dict, {})), extra=_extra(obj, known))


def _test_from(obj, where) -> Test:
    _object(obj, where)
    known = ("name", "generator", "oracle", "criterion")
    return Test(name=_need(obj, "name", where, str), generator=_need(obj, "generator", where, str),
                oracle=_need(obj, "oracle", where, str), criterion=dict(_need(obj, "criterion", where, dict, {})),
                extra=_extra(obj, known))


def _with_extra(d: dict, extra: dict) -> dict:
    d.update(copy.deepcopy(extra))
    return d


# ---------------------------------------------------------------------------------------------------------------------
# template expansion helpers ([o] and [*])
# ---------------------------------------------------------------------------------------------------------------------

def _bind_offset(obj, o):
    """Inside a `per: "offset"` circuit: the string "o" -> [di, dj]; `[o]` inside any string -> `[di,dj]`; recursive."""
    if isinstance(obj, str):
        if obj == "o":
            return [int(o[0]), int(o[1])]
        return obj.replace(TEMPLATE, _offset_tag(o))
    if isinstance(obj, list):
        return [_bind_offset(e, o) for e in obj]
    if isinstance(obj, dict):
        return {(_bind_offset(k, o) if isinstance(k, str) else k): _bind_offset(v, o) for k, v in obj.items()}
    return obj


def _every(name: str, instances: dict, where: str) -> list[str]:
    template = name.replace(EVERY, TEMPLATE)
    if template not in instances:
        raise BlueprintError(f"{where}: {name!r} names no template subspace {template!r}")
    return list(instances[template])


def _splice(obj, instances: dict, where: str):
    """Replace every `x[*]` by the instances of `x[o]`: spliced into lists, one key per instance in mappings; an `x[*]`
    standing alone as a single name is an error (it is many names)."""
    if isinstance(obj, str):
        if EVERY in obj:
            raise BlueprintError(f"{where}: {obj!r} stands for every instance and cannot be a single name here")
        return obj
    if isinstance(obj, list):
        out = []
        for i, e in enumerate(obj):
            if isinstance(e, str) and EVERY in e:
                out.extend(_every(e, instances, f"{where}[{i}]"))
            else:
                out.append(_splice(e, instances, f"{where}[{i}]"))
        return out
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v2 = _splice(v, instances, f"{where}.{k}")
            if isinstance(k, str) and EVERY in k:
                for name in _every(k, instances, where):
                    out[name] = copy.deepcopy(v2)
            else:
                out[k] = v2
        return out
    return obj


def _has_template(obj) -> bool:
    if isinstance(obj, str):
        return TEMPLATE in obj or EVERY in obj
    if isinstance(obj, list):
        return any(_has_template(e) for e in obj)
    if isinstance(obj, dict):
        return any(_has_template(k) or _has_template(v) for k, v in obj.items())
    return False


# ---------------------------------------------------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Blueprint:
    """The genome (module docstring). `subspaces` are `ziplib.layout.Subspace`, `tokens` are `ziplib.codec.TokenClass`;
    everything else is a dataclass of this module. `field` is {"kind": "lattice", "r": int}."""
    name: str
    field: dict = dc_field(default_factory=lambda: {"kind": "lattice", "r": 1})
    subspaces: list = dc_field(default_factory=list)
    tokens: list = dc_field(default_factory=list)
    circuits: list = dc_field(default_factory=list)
    stores: list = dc_field(default_factory=list)
    registers: list = dc_field(default_factory=list)
    loop: Loop = dc_field(default_factory=Loop)
    routes: list = dc_field(default_factory=list)
    tests: list = dc_field(default_factory=list)
    extra: dict = dc_field(default_factory=dict)

    SECTIONS = ("name", "field", "subspaces", "tokens", "circuits", "stores", "registers", "loop", "routes", "tests")

    # -- JSON round trip ---------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, obj: dict) -> "Blueprint":
        _object(obj, "blueprint")
        w = "blueprint"
        bp = cls(
            name=_need(obj, "name", w, str),
            field=dict(_need(obj, "field", w, dict, {"kind": "lattice", "r": 1})),
            subspaces=[_subspace_from(s, f"subspaces[{i}]") for i, s in enumerate(_list(obj.get("subspaces"), "subspaces"))],
            tokens=[_token_from(t, f"tokens[{i}]") for i, t in enumerate(_list(obj.get("tokens"), "tokens"))],
            circuits=[_circuit_from(c, f"circuits[{i}]") for i, c in enumerate(_list(obj.get("circuits"), "circuits"))],
            stores=[_store_from(s, f"stores[{i}]") for i, s in enumerate(_list(obj.get("stores"), "stores"))],
            registers=[_register_from(r, f"registers[{i}]") for i, r in enumerate(_list(obj.get("registers"), "registers"))],
            loop=_loop_from(obj.get("loop", {}), "loop"),
            routes=[_route_from(r, f"routes[{i}]") for i, r in enumerate(_list(obj.get("routes"), "routes"))],
            tests=[_test_from(t, f"tests[{i}]") for i, t in enumerate(_list(obj.get("tests"), "tests"))],
            extra=_extra(obj, cls.SECTIONS),
        )
        return bp

    @classmethod
    def load(cls, path) -> "Blueprint":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError as e:
                raise BlueprintError(f"{path}: not JSON ({e})") from None
        return cls.from_dict(obj)

    def to_dict(self) -> dict:
        b = self.loop.boundary
        boundary = None if b is None else _with_extra(dict(keep=list(b.keep), clear=list(b.clear), quantise=copy.deepcopy(b.quantise),
                                                           commit=copy.deepcopy(b.commit), anchor=copy.deepcopy(b.anchor), halt=b.halt), b.extra)
        d = dict(
            name=self.name,
            field=dict(self.field),
            subspaces=[_strip_none(dict(name=s.name, width=s.width, kind=s.kind, labels=s.labels)) for s in self.subspaces],
            tokens=[{"class": t.name, "flag": t.flag, "fields": dict(t.fields)} for t in self.tokens],
            circuits=[_with_extra(dict(name=c.name, instr=c.instr, params=copy.deepcopy(c.params), per=c.per, place=dict(c.place),
                                       reads=list(c.reads), writes=list(c.writes), prunable=c.prunable), c.extra) for c in self.circuits],
            stores=[_with_extra({"name": s.name, "class": s.cls, "key": list(s.key), "value": s.value, "form": s.form,
                                 "capacity": s.capacity, "nulls": list(s.nulls)}, s.extra) for s in self.stores],
            registers=[_with_extra(dict(name=r.name, slots=r.slots, holds=list(r.holds), filled_by=r.filled_by), r.extra) for r in self.registers],
            loop=_with_extra(dict(core=copy.deepcopy(self.loop.core), tied=self.loop.tied, boundary=boundary,
                                  sequence=copy.deepcopy(self.loop.sequence)), self.loop.extra),
            routes=[_with_extra(dict(mixer=copy.deepcopy(r.mixer), table=dict(r.table)), r.extra) for r in self.routes],
            tests=[_with_extra(dict(name=t.name, generator=t.generator, oracle=t.oracle, criterion=dict(t.criterion)), t.extra) for t in self.tests],
        )
        return _with_extra(d, self.extra)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    # -- lookups -----------------------------------------------------------------------------------------------------
    def subspace(self, name: str) -> Subspace:
        return _find(self.subspaces, name, "subspace")

    def token(self, cls: str):
        return _find(self.tokens, cls, "token class")

    def circuit(self, name: str) -> Circuit:
        return _find(self.circuits, name, "circuit")

    def store(self, name: str) -> StoreSpec:
        return _find(self.stores, name, "store")

    def register(self, name: str) -> Register:
        return _find(self.registers, name, "register")

    def test(self, name: str) -> Test:
        return _find(self.tests, name, "test")

    @property
    def r(self) -> int:
        return int(self.field.get("r", 0))

    def offsets(self) -> list[tuple[int, int]]:
        """The born receptive field: `lattice(field.r)`."""
        return lattice(self.r)

    def symbols(self) -> list[str]:
        """The width symbols the blueprint uses (what `Arch.dims` must bind)."""
        return sorted({s.width for s in self.subspaces if isinstance(s.width, str)})

    def templates(self) -> dict[str, list[str]]:
        """template subspace name (`nbr[o]`) -> its instance names over the field, in lattice order."""
        offs = self.offsets()
        return {s.name: [s.name.replace(TEMPLATE, _offset_tag(o)) for o in offs] for s in self.subspaces if TEMPLATE in s.name}

    @property
    def expanded(self) -> bool:
        """True when no `[o]` / `[*]` template and no `per` remain."""
        return not any(c.per is not None for c in self.circuits) and not _has_template(self.to_dict())

    # -- expand ------------------------------------------------------------------------------------------------------
    def expand(self) -> "Blueprint":
        """A NEW blueprint with every template instantiated over `lattice(field.r)`: template subspaces (`nbr[o]`) become
        one subspace per offset in place; `per: "offset"` circuits become one circuit per offset (`"o"` -> [di, dj],
        `[o]` -> `[di,dj]` in name, params, reads, writes; `per` -> None); every `x[*]` is spliced into its instances.
        Idempotent: an expanded blueprint expands to an equal copy. The tests are not touched."""
        offs = self.offsets()
        instances = self.templates()
        subs = []
        for s in self.subspaces:
            if s.name in instances:
                subs += [Subspace(n, s.width, s.kind, None if s.labels is None else list(s.labels)) for n in instances[s.name]]
            else:
                subs.append(Subspace(s.name, s.width, s.kind, None if s.labels is None else list(s.labels)))
        circuits = []
        for c in self.circuits:
            if c.per is None:
                circuits.append(copy.deepcopy(c))
            elif c.per == "offset":
                if TEMPLATE not in c.name:
                    raise BlueprintError(f"circuits[{c.name}]: per = 'offset' needs '[o]' in the name (one circuit per offset must have its own name)")
                for o in offs:
                    circuits.append(Circuit(name=_bind_offset(c.name, o), instr=c.instr, params=_bind_offset(c.params, o), per=None,
                                            place=copy.deepcopy(c.place), reads=_bind_offset(c.reads, o), writes=_bind_offset(c.writes, o),
                                            prunable=c.prunable, extra=copy.deepcopy(c.extra)))
            else:
                raise BlueprintError(f"circuits[{c.name}]: per must be null or 'offset', got {c.per!r}")
        stage = Blueprint(self.name, dict(self.field), subs, copy.deepcopy(self.tokens), circuits, copy.deepcopy(self.stores),
                          copy.deepcopy(self.registers), copy.deepcopy(self.loop), copy.deepcopy(self.routes), copy.deepcopy(self.tests),
                          copy.deepcopy(self.extra))
        d = stage.to_dict()
        d["tokens"] = [_splice(t, instances, f"tokens[{t['class']}]") for t in d["tokens"]]
        d["circuits"] = [_splice(c, instances, f"circuits[{c['name']}]") for c in d["circuits"]]
        d["stores"] = [_splice(s, instances, f"stores[{s['name']}]") for s in d["stores"]]
        d["registers"] = [_splice(r, instances, f"registers[{r['name']}]") for r in d["registers"]]
        d["loop"] = _splice(d["loop"], instances, "loop")
        d["routes"] = _splice(d["routes"], instances, "routes")
        for i, s in enumerate(d["subspaces"]):
            if EVERY in s["name"]:
                raise BlueprintError(f"subspaces[{i}]: a subspace cannot be named with '[*]' ({s['name']!r})")
        return Blueprint.from_dict(d)

    # -- derived numbers ---------------------------------------------------------------------------------------------
    def max_len(self, dims: dict) -> int:
        """The sequence length the compiler sizes M for: the null token classes (every class a store's `nulls` or a
        circuit's `null` names) + the stores' capacities + the grid's cells (H*W from `dims`, or L for a sequence) + the
        registers' slots. Gridworld at H 8, W 11: 2 + 192 + 64 + 88 + 4 = 350 (DESIGN §21.2.2)."""
        bp = self.expand()
        classes = {t.name for t in bp.tokens}
        nulls = set()
        for s in bp.stores:
            nulls |= {n for n in s.nulls if n in classes}
        for c in bp.circuits:
            n = c.params.get("null")
            if isinstance(n, str) and n in classes:
                nulls.add(n)
        if "H" in dims and "W" in dims:
            cells = int(dims["H"]) * int(dims["W"])
        elif "L" in dims:
            cells = int(dims["L"])
        else:
            raise BlueprintError("max_len needs H and W (a frame) or L (a sequence) in dims")
        return len(nulls) + sum(int(s.capacity) for s in bp.stores) + cells + sum(int(r.slots) for r in bp.registers)

    def layout(self, dims: dict, d: int | None = None) -> Layout:
        """`Layout.allocate` over the EXPANDED subspaces (the compiler's step 2); `d_layout` 251 for gridworld at r 1."""
        return Layout.allocate(self.expand().subspaces, d=d, dims=dims)

    # -- validate ----------------------------------------------------------------------------------------------------
    def validate(self, dims: dict | None = None, instructions: dict | None = None) -> list[str]:
        """The mechanical refusals (module docstring), as a list of messages; empty = valid. `instructions` overrides the
        lazy import of `ziplib.instructions.INSTRUCTIONS`; `dims`, when given, also binds the width symbols and runs
        `Layout.allocate` on the expanded subspaces."""
        return _Validator(self, dims, instructions).run()


def _find(items, name, what):
    for it in items:
        if it.name == name:
            return it
    raise BlueprintError(f"no {what} {name!r}; the blueprint has {[it.name for it in items]}")


def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------------------------------------------------
# the validator
# ---------------------------------------------------------------------------------------------------------------------

class _Validator:
    def __init__(self, bp: Blueprint, dims, instructions):
        self.bp = bp
        self.dims = dims
        self.instructions_arg = instructions
        self.msgs: list[str] = []
        self.subs = {s.name: s for s in bp.subspaces}                 # declared, templates included
        self.templates = {s.name for s in bp.subspaces if TEMPLATE in s.name}
        self.classes = {t.name for t in bp.tokens}
        self.registers = {r.name for r in bp.registers}
        self.symbols = set(STANDARD_SYMBOLS) | set(bp.symbols())

    def say(self, msg: str):
        if msg not in self.msgs:
            self.msgs.append(msg)

    # -- names -------------------------------------------------------------------------------------------------------
    def resolve(self, name, where: str, per=None) -> bool:
        """A declared subspace; `x[*]` of a declared template; `x[o]` of a declared template inside a per-offset circuit."""
        if not isinstance(name, str):
            self.say(f"{where}: a subspace name must be a string, got {name!r}")
            return False
        if EVERY in name:
            if name.replace(EVERY, TEMPLATE) in self.templates:
                return True
            self.say(f"{where}: {name!r} names no template subspace {name.replace(EVERY, TEMPLATE)!r}")
            return False
        if TEMPLATE in name:
            if name in self.templates and per == "offset":
                return True
            if name in self.templates:
                self.say(f"{where}: {name!r} is a template; outside a per: 'offset' circuit write {name.replace(TEMPLATE, EVERY)!r} for every instance")
            else:
                self.say(f"{where}: {name!r} is not a declared subspace")
            return False
        if name in self.subs:
            return True
        self.say(f"{where}: {name!r} is not a declared subspace (declared: {list(self.subs)})")
        return False

    def is_flag(self, name, where: str) -> bool:
        if not isinstance(name, str):
            self.say(f"{where}: a flag is named by a string, got {name!r}")
            return False
        if name in self.subs:
            if self.subs[name].kind != "flag":
                self.say(f"{where}: {name!r} is a {self.subs[name].kind} subspace, not a flag")
                return False
            return True
        if "." in name and name.split(".", 1)[0] in self.subs:      # sub.label / sub.<k> (Layout.flag's forms)
            return True
        self.say(f"{where}: {name!r} is not a declared flag subspace (flags: {[n for n, s in self.subs.items() if s.kind == 'flag']})")
        return False

    def names(self, seq, where: str, per=None):
        if not isinstance(seq, list):
            self.say(f"{where}: expected a list of subspace names, got {seq!r}")
            return
        for i, n in enumerate(seq):
            self.resolve(n, f"{where}[{i}]", per)

    def unique(self, items, where: str):
        seen = set()
        for it in items:
            if it.name in seen:
                self.say(f"{where}: {it.name!r} is declared twice")
            seen.add(it.name)

    def unknown(self, extra: dict, where: str, known):
        for k in extra:
            self.say(f"{where}: unknown key {k!r} (the schema's keys are {list(known)})")

    # -- params: the content-symbol guard ----------------------------------------------------------------------------
    def string_param(self, key, v: str, where: str, c: Circuit) -> None:
        if v in OPERATION_WORDS:
            return
        if v == "o":
            if c.per != "offset":
                self.say(f"{where}: 'o' (the bound offset) only means something in a per: 'offset' circuit")
            return
        if key == "W":                                                  # a value-matrix slot ZipLearn counts (Brain.W)
            if not v.isidentifier():
                self.say(f"{where}: a W slot is an identifier, got {v!r}")
            return
        if " - " in v:                                                  # Pool.src "a - b"
            a, b = (p.strip() for p in v.split(" - ", 1))
            self.resolve(a, where, c.per)
            self.resolve(b, where, c.per)
            return
        if v in self.classes or v in self.registers or v in self.symbols:
            return
        if v in self.subs or (EVERY in v and v.replace(EVERY, TEMPLATE) in self.templates) or (TEMPLATE in v and v in self.templates):
            if TEMPLATE in v and c.per != "offset":
                self.say(f"{where}: {v!r} is a template; only a per: 'offset' circuit may name it")
            return
        what = "a character" if len(v) == 1 else "a name the brain has no referent for"
        self.say(f"{where}: {v!r} is {what} -- a content symbol; params may only name subspaces, token classes, registers, "
                 f"width symbols, W slots and the operation words {OPERATION_WORDS}")

    def number_param(self, key, v, where: str, sub=None, mapping_key=None) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, float) and not float(v).is_integer():
            return                                                      # a gain or a threshold
        if isinstance(v, float):
            v_int = int(v)
        else:
            v_int = v
        if key in OPERATION_CONSTANTS:
            if key == "sign" and v_int not in (-1, 1):
                self.say(f"{where}: sign must be -1 or 1, got {v!r}")
            if key == "vocab" and v_int <= 0:
                self.say(f"{where}: vocab must be a positive size, got {v!r}")
            return
        if mapping_key == "weights":
            return                                                      # a per-subspace multiplier
        if sub is not None:                                             # inside a mapping subspace -> number
            kind = self.subs[sub].kind if sub in self.subs else None
            if kind in ("flag", "scalar"):
                return                                                  # a weight on a flag/scalar dim (Row.key = {goal_met: 1})
            self.say(f"{where}: {v!r} on the {kind or 'undeclared'} subspace {sub!r} is a symbol index -- a content symbol "
                     f"(a colour, a character); a Row/Compare parameter weights flags and scalars, it never picks a symbol")
            return
        self.say(f"{where}: the bare number {v!r} is a content symbol (a colour index, a character code); the operation "
                 f"constants are {OPERATION_CONSTANTS}")

    def list_param(self, key, v: list, where: str, c: Circuit) -> None:
        numeric = [e for e in v if isinstance(e, (int, float)) and not isinstance(e, bool)]
        if v and len(numeric) == len(v):
            if key == "offset" and len(v) == 2 and all(isinstance(e, int) for e in v):
                if c.per == "offset":
                    self.say(f"{where}: a per: 'offset' circuit takes its offset from the field; write \"o\", not {v!r}")
                    return
                r = self.bp.r
                if max(abs(v[0]), abs(v[1])) > r or tuple(v) == (0, 0):
                    self.say(f"{where}: offset {tuple(v)} lies outside the field (r = {r}) or is the centre; an offset chosen "
                             f"for a game is content -- widen field.r instead")
                return
            self.say(f"{where}: a list of numbers {v!r} is an offset list / a symbol list -- content; an offset list is a FIELD, "
                     f"declared once under 'field'")
            return
        for i, e in enumerate(v):
            self.param(key, e, f"{where}[{i}]", c)

    def dict_param(self, key, v: dict, where: str, c: Circuit) -> None:
        for k, e in v.items():
            w = f"{where}.{k}"
            if not isinstance(k, str):
                self.say(f"{w}: a mapping key in params is a subspace name, got {k!r}")
                continue
            self.resolve(k, w, c.per)
            sub = k if k in self.subs else None
            if isinstance(e, bool) or isinstance(e, (int, float)):
                self.number_param(key, e, w, sub=sub, mapping_key=key)
            elif isinstance(e, str):
                self.string_param(key, e, w, c)
            elif e is None:
                pass
            else:
                self.param(key, e, w, c)

    def param(self, key, v, where: str, c: Circuit) -> None:
        if v is None or isinstance(v, bool):
            return
        if isinstance(v, str):
            self.string_param(key, v, where, c)
        elif isinstance(v, (int, float)):
            self.number_param(key, v, where)
        elif isinstance(v, list):
            self.list_param(key, v, where, c)
        elif isinstance(v, dict):
            self.dict_param(key, v, where, c)
        else:
            self.say(f"{where}: unsupported params value {v!r}")

    # -- sections ----------------------------------------------------------------------------------------------------
    def run(self) -> list[str]:
        bp = self.bp
        if not isinstance(bp.name, str) or not bp.name:
            self.say("name: a blueprint needs a non-empty name")
        self.check_field()
        self.check_subspaces()
        self.check_tokens()
        self.check_circuits()
        self.check_stores()
        self.check_registers()
        self.check_loop()
        self.check_routes()
        self.check_tests()
        self.unknown(bp.extra, "blueprint", Blueprint.SECTIONS)
        self.check_expansion()
        return list(self.msgs)

    def check_field(self):
        f = self.bp.field
        if not isinstance(f, dict):
            self.say(f"field: expected {{'kind': 'lattice', 'r': int}}, got {f!r}")
            return
        if f.get("kind") not in FIELD_KINDS:
            self.say(f"field.kind: must be one of {FIELD_KINDS}, got {f.get('kind')!r}")
        r = f.get("r")
        if isinstance(r, bool) or not isinstance(r, int) or r < 0:
            self.say(f"field.r: must be an int >= 0, got {r!r}")
        for k in set(f) - {"kind", "r"}:
            self.say(f"field: unknown key {k!r} (a field is {{'kind', 'r'}})")

    def check_subspaces(self):
        self.unique(self.bp.subspaces, "subspaces")
        for s in self.bp.subspaces:
            w = f"subspaces[{s.name}]"
            if s.kind not in KINDS:
                self.say(f"{w}: kind {s.kind!r} is not one of {KINDS}")
            if isinstance(s.width, bool) or (isinstance(s.width, int) and s.width <= 0):
                self.say(f"{w}: width must be > 0 or a symbol, got {s.width!r}")
            elif isinstance(s.width, str) and not s.width.isidentifier():
                self.say(f"{w}: width symbol {s.width!r} is not an identifier")
            if EVERY in s.name:
                self.say(f"{w}: a subspace cannot be named with '[*]'; a template is named with '[o]'")
            if self.dims is not None and isinstance(s.width, str) and s.width not in self.dims:
                self.say(f"{w}: width symbol {s.width!r} is not bound by dims {sorted(self.dims)}")

    def check_tokens(self):
        self.unique(self.bp.tokens, "tokens")
        for t in self.bp.tokens:
            w = f"tokens[{t.name}]"
            self.is_flag(t.flag, f"{w}.flag")
            for sub, src in t.fields.items():
                if src in ("coord.row", "coord.col"):
                    continue                                            # names no subspace under rope2d (the codec's rule)
                self.resolve(sub, f"{w}.fields[{sub}]")

    def check_circuits(self):
        bp = self.bp
        self.unique(bp.circuits, "circuits")
        whitelist, note = self.whitelist()
        if note:
            self.say(note)
        for c in bp.circuits:
            w = f"circuits[{c.name}]"
            if whitelist is not None and c.instr not in whitelist:
                self.say(f"{w}.instr: {c.instr!r} is not in INSTRUCTIONS {sorted(whitelist)}")
            if c.per not in PERS:
                self.say(f"{w}.per: must be null or 'offset', got {c.per!r}")
            templated = TEMPLATE in c.name or _has_template_o(c.params) or _has_template_o(c.reads) or _has_template_o(c.writes)
            if c.per == "offset" and TEMPLATE not in c.name:
                self.say(f"{w}: per = 'offset' needs '[o]' in the name (each instance needs its own name)")
            if c.per is None and templated:
                self.say(f"{w}: '[o]' appears in a circuit without per = 'offset'")
            if not isinstance(c.place, dict):
                self.say(f"{w}.place: expected {{'layer': int|'auto', 'head': int|'auto'}}, got {c.place!r}")
            else:
                for k in ("layer", "head"):
                    v = c.place.get(k, "auto")
                    if v != "auto" and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                        self.say(f"{w}.place.{k}: must be an int >= 0 or 'auto', got {v!r}")
                for k in set(c.place) - {"layer", "head"}:
                    self.say(f"{w}.place: unknown key {k!r}")
            if not isinstance(c.prunable, bool):
                self.say(f"{w}.prunable: must be a bool, got {c.prunable!r}")
            self.names(c.reads, f"{w}.reads", c.per)
            self.names(c.writes, f"{w}.writes", c.per)
            if not isinstance(c.params, dict):
                self.say(f"{w}.params: expected a mapping, got {c.params!r}")
            else:
                for key, v in c.params.items():
                    self.param(key, v, f"{w}.params.{key}", c)
            self.unknown(c.extra, w, ("name", "instr", "params", "per", "place", "reads", "writes", "prunable"))

    def whitelist(self):
        if self.instructions_arg is not None:
            return set(self.instructions_arg), None
        try:
            from .instructions import INSTRUCTIONS                      # lazy: the whitelist lives with the instructions
        except Exception as e:                                          # noqa: BLE001 -- report, do not hide, whatever stops the import
            return None, f"instructions: ziplib.instructions is not importable ({type(e).__name__}: {e}); the instr whitelist was not checked"
        return set(INSTRUCTIONS), None

    def check_stores(self):
        bp = self.bp
        self.unique(bp.stores, "stores")
        fields_of = {t.name: t.fields for t in bp.tokens}
        for s in bp.stores:
            w = f"stores[{s.name}]"
            if s.cls not in self.classes:
                self.say(f"{w}.class: {s.cls!r} is not a declared token class {sorted(self.classes)}")
            self.names(s.key, f"{w}.key")
            self.resolve(s.value, f"{w}.value")
            if s.form not in FORMS:
                self.say(f"{w}.form: must be one of {FORMS}, got {s.form!r}")
            if isinstance(s.capacity, bool) or not isinstance(s.capacity, int) or s.capacity < 0:
                self.say(f"{w}.capacity: must be an int >= 0, got {s.capacity!r}")
            for n in s.nulls:
                if n not in self.classes:
                    self.say(f"{w}.nulls: {n!r} is not a declared token class")
            if s.cls in fields_of:                                      # the codec's entry() needs every key/value as a store field
                stored = {k for k, src in fields_of[s.cls].items() if src == "store"}
                for n in list(s.key) + [s.value]:
                    if isinstance(n, str) and n not in stored:
                        self.say(f"{w}: {n!r} is not a 'store' field of token class {s.cls!r} (its store fields: {sorted(stored)})")
            self.unknown(s.extra, w, ("name", "class", "key", "value", "form", "capacity", "nulls"))

    def check_registers(self):
        bp = self.bp
        self.unique(bp.registers, "registers")
        for r in bp.registers:
            w = f"registers[{r.name}]"
            if isinstance(r.slots, bool) or not isinstance(r.slots, int) or r.slots < 1:
                self.say(f"{w}.slots: must be an int >= 1, got {r.slots!r}")
            self.names(r.holds, f"{w}.holds")
            if r.filled_by not in FILLERS:
                self.say(f"{w}.filled_by: must be one of {FILLERS}, got {r.filled_by!r}")
            if "init" in r.extra:
                self.say(f"{w}: has 'init' -- a blueprint never initialises a register; GOAL is filled by the goal model's counted "
                         f"keys or by the harness (DESIGN §21.1 item 7)")
            self.unknown({k: v for k, v in r.extra.items() if k != "init"}, w, ("name", "slots", "holds", "filled_by"))

    def check_loop(self):
        loop = self.bp.loop
        w = "loop"
        if loop.core != "auto" and not (isinstance(loop.core, list) and all(isinstance(l, int) and not isinstance(l, bool) and l >= 0 for l in loop.core)):
            self.say(f"{w}.core: must be 'auto' or a list of layer indices, got {loop.core!r}")
        if not isinstance(loop.tied, bool):
            self.say(f"{w}.tied: must be a bool, got {loop.tied!r}")
        if loop.sequence is not None:
            if not isinstance(loop.sequence, dict) or set(loop.sequence) - {"feedback"}:
                self.say(f"{w}.sequence: expected {{'feedback': [[from, to]]}} or null, got {loop.sequence!r}")
            else:
                for i, pair in enumerate(loop.sequence.get("feedback") or []):
                    self.pair(pair, f"{w}.sequence.feedback[{i}]")
        self.unknown(loop.extra, w, ("core", "tied", "boundary", "sequence"))
        b = loop.boundary
        if b is None:
            self.say(f"{w}.boundary: missing (write a no-op boundary -- keep every subspace, no anchor, no halt -- for a one-pass brain)")
            return
        w = "loop.boundary"
        self.names(b.keep, f"{w}.keep")
        self.names(b.clear, f"{w}.clear")
        keep = {n.replace(EVERY, TEMPLATE) for n in b.keep if isinstance(n, str)}
        clear = {n.replace(EVERY, TEMPLATE) for n in b.clear if isinstance(n, str)}
        for n in sorted(keep & clear):
            self.say(f"{w}: {n!r} is both kept and cleared")
        for n in self.subs:
            if n not in keep and n not in clear:
                self.say(f"{w}: {n!r} is neither kept nor cleared (keep and clear must partition the subspaces)")
        if not isinstance(b.quantise, list):
            self.say(f"{w}.quantise: expected a list of {{'read': [...], 'write': sub}}, got {b.quantise!r}")
        else:
            for i, q in enumerate(b.quantise):
                wq = f"{w}.quantise[{i}]"
                if not isinstance(q, dict) or set(q) != {"read", "write"}:
                    self.say(f"{wq}: expected {{'read': [subspaces], 'write': subspace}}, got {q!r}")
                    continue
                self.names(q["read"], f"{wq}.read")
                self.resolve(q["write"], f"{wq}.write")
        if not isinstance(b.commit, list):
            self.say(f"{w}.commit: expected a list of [from, to], got {b.commit!r}")
        else:
            for i, pair in enumerate(b.commit):
                self.pair(pair, f"{w}.commit[{i}]")
        if b.anchor is not None:
            if not isinstance(b.anchor, dict) or set(b.anchor) != {"sub", "from"}:
                self.say(f"{w}.anchor: expected {{'sub': subspace, 'from': register | 'runner'}} or null, got {b.anchor!r}")
            else:
                self.resolve(b.anchor["sub"], f"{w}.anchor.sub")
                src = b.anchor["from"]
                if src != "runner" and src not in self.registers:
                    self.say(f"{w}.anchor.from: {src!r} is neither a declared register {sorted(self.registers)} nor 'runner'")
        if b.halt is not None:
            self.is_flag(b.halt, f"{w}.halt")
        self.unknown(b.extra, w, ("keep", "clear", "quantise", "commit", "anchor", "halt"))

    def pair(self, pair, where: str):
        if not isinstance(pair, list) or len(pair) != 2:
            self.say(f"{where}: expected [from, to], got {pair!r}")
            return
        self.resolve(pair[0], f"{where}[0]")
        self.resolve(pair[1], f"{where}[1]")

    def check_routes(self):
        for i, r in enumerate(self.bp.routes):
            w = f"routes[{i}]"
            if isinstance(r.mixer, bool) or not isinstance(r.mixer, int) or r.mixer < 0:
                self.say(f"{w}.mixer: must be a layer index (int >= 0), got {r.mixer!r}")
            if not isinstance(r.table, dict) or not r.table:
                self.say(f"{w}.table: expected a non-empty mapping flag -> source, got {r.table!r}")
            else:
                for flag in r.table:
                    self.is_flag(flag, f"{w}.table[{flag}]")
            self.unknown(r.extra, w, ("mixer", "table"))

    def check_tests(self):
        bp = self.bp
        if not bp.tests:
            self.say("tests: a blueprint needs at least one test (DESIGN §21.2.1)")
        self.unique(bp.tests, "tests")
        for t in bp.tests:
            w = f"tests[{t.name}]"
            for which in ("generator", "oracle"):
                try:
                    parse_callable(getattr(t, which))
                except BlueprintError as e:
                    self.say(f"{w}.{which}: {e}")
            crit = t.criterion
            ok = isinstance(crit, dict) and (
                (set(crit) == {"exact"} and crit["exact"] is True)
                or (set(crit) == {"min"} and isinstance(crit["min"], (int, float)) and not isinstance(crit["min"], bool)))
            if not ok:
                self.say(f"{w}.criterion: expected {{'exact': true}} or {{'min': number}}, got {crit!r}")
            self.unknown(t.extra, w, ("name", "generator", "oracle", "criterion"))

    def check_expansion(self):
        try:
            ex = self.bp.expand()
        except BlueprintError as e:
            self.say(f"expand: {e}")
            return
        seen = set()
        for c in ex.circuits:
            if c.name in seen:
                self.say(f"expand: two circuits are named {c.name!r} after expansion")
            seen.add(c.name)
        seen = set()
        for s in ex.subspaces:
            if s.name in seen:
                self.say(f"expand: two subspaces are named {s.name!r} after expansion")
            seen.add(s.name)
        if self.dims is not None:
            try:
                Layout.allocate(ex.subspaces, dims=self.dims)
            except LayoutError as e:
                self.say(f"layout: {e}")


def _has_template_o(obj) -> bool:
    if isinstance(obj, str):
        return TEMPLATE in obj
    if isinstance(obj, list):
        return any(_has_template_o(e) for e in obj)
    if isinstance(obj, dict):
        return any(_has_template_o(k) or _has_template_o(v) for k, v in obj.items())
    return False


__all__ = ["Blueprint", "Circuit", "StoreSpec", "Register", "Loop", "Boundary", "Route", "Test", "Arch", "BlueprintError",
           "lattice", "parse_callable", "STANDARD_SYMBOLS", "FORMS", "FILLERS", "PERS", "POS_CODES", "OPERATION_WORDS",
           "OPERATION_CONSTANTS", "TEMPLATE", "EVERY"]
