"""BrainBuilder — writes a BLUEPRINT into a fully developed brain, then verifies it (DESIGN §21; the module spec's
`brainbuilder.py`). Runs BEFORE ZipLearn; nothing here calls `backward`.

  compile(blueprint, arch) -> Brain     DESIGN §21.4 steps 1-7: bind dims -> Layout.allocate -> order the layers by reads/writes
                                        (topological; pins; a cycle no kept subspace breaks is an error naming the circuits) ->
                                        heads and channels (first-fit; a whole-window Match is a one-head layer with hd = d) ->
                                        the sharpness -> emit + conflict check + one index_put_ per tensor -> materialise
                                        LoopedModel(core_layers, boundary, norm="none", n_head=[...]) -> Brain(verified=False).
                                        Generalises `WrittenSim.__init__` (e28.py) and `e18.write`, which the cutover deletes.
  verify(brain, seed=0) -> Report       per-instruction unit tests (argmax on the target for every query, logit gap >= M, the
                                        written subspace exact) with the softmax leak as the gradient-flow number; purity of the
                                        one-hot subspaces on the blueprint's own rollouts; the blueprint's tests against their oracle
                                        with the tie count; writes runs/brainbuilder/<name>.json; sets brain.verified.
  anatomy(brain) -> table               every nonzero traced to its circuit (Brain.anatomy).
  ablate(blueprint, circuit) -> Blueprint   the blueprint without one circuit (B7's ladder).
  main()                                python brainbuilder.py blueprints/gridworld.json --arch auto|<file> --r 1 --pos onehot|rope2d
                                        --verify --out runs/brainbuilder/
  b0_regressions / b6_sequence_axis / b7_ablation_ladder / b8_compile_cost   the blueprint experiments, one function each,
                                        with their pre-registered pass / refute printed (b1_generality needs ziplearn.py).

The sharpness (step 5). `Arch.sharpness` gives M = ln((max_len - 1) p* / (1 - p*)) in NATS: the softmax gap one match must
buy so that the target holds mass >= p* against max_len - 1 rivals one match behind (10.45 at max_len 350, p* 0.99). The
instructions write a query dim at w * M_w and a key dim at M_w, so one matched dim scores M_w^2 / sqrt(hd) nats after the
substrate's scaling; and a head's smallest gap can be a fraction of a match (E28's null sits at 1.5 matches against a full
match of 2: half a match). So each head is written at M_w = sqrt(M * sqrt(hd) / gap_min) with `gap_min` the instruction's
(`Instruction.gap_min`): every head then has exactly the derived gap in nats whatever its width or unit. A circuit's typed
`params["M"]` (e18's constants 3 / 2) overrides it. The boundary's tie tolerance is 1 - p*, the leak that gap allows, so the
tie count and the argmax's tie-break are those of the exact arithmetic (E28's 1e-3 at M = 30).

The evidence for the E28 regression. `e28_plans` plays LockPath as `e28.main` does (levels 0-1, budget 150, sleep) and
builds one `Store` per action from the rules. Today's `arcgames.Player` (E27's explorer, E29's deletion of `search`) learns
128 entries at seed 0 and E28's tie count on its random walks is 1,444; the E28 run of 2026-09-21 (RESULTS.md) used the
player of commit 78d2dbb (121 entries at both seeds; ties 1,434 / 1,475). `evidence="78d2dbb"` imports that commit's
arcgames.py (`git show`) exactly as `ziplib/_check_store.py` does for E24, so the header's numbers are reproduced; the
rollouts are compared with `e28.WrittenSim` on the SAME evidence either way.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (HERE, HERE.parent / "transformers", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h1_lid as H1                                                                   # noqa: E402
from ziplib.blueprint import Blueprint, Arch, BlueprintError, parse_callable          # noqa: E402
from ziplib.layout import Layout, LayoutError                                         # noqa: E402
from ziplib.codec import Codec                                                        # noqa: E402
from ziplib.instructions import (INSTRUCTIONS, Slot, Write, scatter, InstructionError,   # noqa: E402
                                 Match, Gather, Readout, Branch, Row, Compare, Pool, Broadcast, _head_scores, _argmax_stats)
from ziplib.brain import Brain, materialise                                           # noqa: E402
from ziplib.store import Store, Field                                                 # noqa: E402

RUNS = HERE / "runs" / "brainbuilder"
E28_COMMIT = "78d2dbb"                                    # the player E28's recorded evidence came from (RESULTS.md 2026-09-21)


class CompileError(ValueError):
    """The blueprint cannot be built: a cycle, a pin that reads what a later layer clears, a width that does not fit, ..."""


# =====================================================================================================================
# the plan: what compile resolves
# =====================================================================================================================

@dataclass
class Placement:
    """One circuit's resolved placement."""
    name: str
    instr: str
    kind: str                                  # attn | mlp | readout | branch
    params: dict
    needs: object
    min_layer: int = 0
    layer: int | None = None
    head: int | None = None
    row: int | None = None
    pinned_layer: int | None = None
    pinned_head: int | None = None
    whole: bool = False
    M: float | None = None
    gap_min: float = 1.0
    notes: list = field(default_factory=list)


def _kind(instr: str, needs) -> str:
    if instr == "Readout":
        return "readout"
    if instr == "Branch":
        return "branch"
    if needs.heads:
        return "attn"
    if needs.rows:
        return "mlp"
    raise CompileError(f"{instr}: neither a head nor MLP rows -- nothing to place")


def _pos_of(layer: int, kind: str) -> int:
    """Half-layer positions: attention at 2L, the MLP at 2L + 1 (attention before the MLP within a Block)."""
    return 2 * layer + (1 if kind == "mlp" else 0)


def _layer_of(pos: int, kind: str) -> int:
    """The layer of a half-position, rounded UP to the kind's parity."""
    if kind == "mlp":
        return (pos - 1 + 1) // 2 if pos % 2 == 1 else pos // 2       # odd -> (pos-1)/2; even -> pos/2 (the next MLP)
    return (pos + 1) // 2                                            # attn: even -> pos/2; odd -> (pos+1)/2


def _tarjan(nodes, edges):
    """Strongly connected components of a directed graph {node: set(succ)} (iterative Tarjan)."""
    index, low, on, stack, out = {}, {}, set(), [], []
    counter = [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(edges.get(root, ())))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(edges.get(w, ()))))
                    advanced = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def order_layers(bp: Blueprint, places: dict[str, Placement], kept: set[str]) -> list[str]:
    """DESIGN §21.4 step 3. Edges W -> R for every subspace W writes and R reads (W != R). A reader sits after its writer in
    the same pass; a cycle is broken on edges whose subspaces the loop KEEPS (the reader then reads the previous pass's
    value); a cycle no kept subspace breaks is an error naming the circuits. Pins (`place.layer`) fix a circuit's layer; a
    pin that would read, in the same pass, what a later layer writes into a CLEARED subspace is an error (the value could
    never arrive). Returns the circuits in dependency order and sets `min_layer`."""
    names = [c.name for c in bp.circuits if places[c.name].kind in ("attn", "mlp")]
    writes = {n: set(bp.circuit(n).writes) for n in names}
    reads = {n: set(bp.circuit(n).reads) for n in names}
    edges: dict[str, dict[str, set]] = {n: {} for n in names}              # W -> {R: subspaces}
    for w in names:
        for r in names:
            if w == r:
                continue
            shared = writes[w] & reads[r]
            if shared:
                edges[w][r] = shared
    dropped = []

    def succ():
        return {n: set(edges[n]) for n in names}

    # break cycles on kept subspaces
    while True:
        comps = [c for c in _tarjan(names, succ()) if len(c) > 1]
        if not comps:
            break
        progress = False
        for comp in comps:
            cs = set(comp)
            for w in comp:
                for r in list(edges[w]):
                    if r in cs and edges[w][r] <= kept:
                        dropped.append((w, r, sorted(edges[w][r])))
                        del edges[w][r]
                        progress = True
        if not progress:
            comp = comps[0]
            detail = "; ".join(f"{w} -> {r} via {sorted(edges[w][r])}" for w in comp for r in edges[w] if r in set(comp))
            raise CompileError(f"circuits {sorted(comp)} form a cycle no kept subspace breaks: {detail}")
    # a pin below its writers: legal only through kept subspaces (the reader takes last pass's value)
    for r in names:
        pr = places[r]
        if pr.pinned_layer is None:
            continue
        for w in names:
            if r in edges[w]:
                pw = places[w]
                need = _pos_of(pw.pinned_layer, pw.kind) + 1 if pw.pinned_layer is not None else None
                if need is not None and _pos_of(pr.pinned_layer, pr.kind) < need:
                    if edges[w][r] <= kept:
                        dropped.append((w, r, sorted(edges[w][r])))
                        del edges[w][r]
                    else:
                        raise CompileError(f"{r!r} is pinned at layer {pr.pinned_layer} but reads {sorted(edges[w][r] - kept)} which "
                                           f"{w!r} (pinned at layer {pw.pinned_layer}) writes later in the pass and the boundary clears")
    # topological order (Kahn), ties by declaration order
    indeg = {n: 0 for n in names}
    for w in names:
        for r in edges[w]:
            indeg[r] += 1
    ready = [n for n in names if indeg[n] == 0]
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for r in sorted(edges[n], key=names.index):
            indeg[r] -= 1
            if indeg[r] == 0:
                ready.append(r)
        ready.sort(key=names.index)
    if len(order) != len(names):
        raise CompileError("internal: the dependency graph still has a cycle after breaking")
    preds = {r: [w for w in names if r in edges[w]] for r in names}
    return order, preds, edges, dropped


# =====================================================================================================================
# compile
# =====================================================================================================================

def _resolve_max_len(bp: Blueprint, arch: Arch) -> int:
    if isinstance(arch.max_len, int) and not isinstance(arch.max_len, bool):
        return int(arch.max_len)
    return bp.max_len(arch.dims)


def _null_names(bp: Blueprint) -> list[str]:
    classes = [t.name for t in bp.tokens]
    nulls = set()
    for s in bp.stores:
        nulls |= {n for n in s.nulls if n in classes}
    for c in bp.circuits:
        n = c.params.get("null")
        if isinstance(n, str) and n in classes:
            nulls.add(n)
        n = c.params.get("sink")
        if isinstance(n, str) and n in classes:
            nulls.add(n)
    return [n for n in classes if n in nulls]


def _arch_view(arch: Arch, max_len: int, head_dim=None):
    return SimpleNamespace(pos=arch.pos, dims=dict(arch.dims), max_len=int(max_len), p_star=float(arch.p_star), head_dim=head_dim)


def compile(blueprint: Blueprint, arch: Arch, seed: int = 0, verbose: bool = False) -> Brain:
    """DESIGN §21.4 steps 1-7 (module docstring). `arch.d_model`, `n_head`, `n_layer` may be "auto" or explicit; an
    explicit d that no multi-head layer can divide is rounded UP to the next feasible width (reported in
    `brain.plan["notes"]`). Raises `BlueprintError` on an invalid blueprint and `CompileError` on a placement failure."""
    t0 = time.time()
    notes = []
    # -- 1. bind dims, expand ------------------------------------------------------------------------------------------
    bp = blueprint.expand()
    dims = dict(arch.dims)
    msgs = bp.validate(dims)
    if msgs:
        raise BlueprintError("the blueprint does not validate:\n  " + "\n  ".join(msgs))
    if arch.validate():
        raise BlueprintError("the arch does not validate: " + "; ".join(arch.validate()))
    max_len = _resolve_max_len(bp, arch)
    # -- 2. the layout (at d_layout first; re-allocated at d in step 4) -------------------------------------------------
    layout = Layout.allocate(bp.subspaces, d=None, dims=dims)
    d_layout = layout.d_layout
    codec = Codec(layout, bp.tokens, pos=arch.pos)
    view = _arch_view(arch, max_len)
    places: dict[str, Placement] = OrderedDict()
    for c in bp.circuits:
        if c.instr not in INSTRUCTIONS:
            raise CompileError(f"{c.name}: instr {c.instr!r} is not in INSTRUCTIONS")
        instr = INSTRUCTIONS[c.instr]
        try:
            needs = instr.needs(c.params, codec, view)
        except (InstructionError, LayoutError) as e:
            raise CompileError(f"{c.name} ({c.instr}).needs: {e}") from None
        kind = _kind(c.instr, needs)
        # a Match over several subspaces (E28's lookup over the whole window) gets a one-head layer, hd = d, under every
        # position code (step 4; under RoPE its content is packed into the position-free tail pairs of that wide head)
        whole = bool(getattr(needs, "whole_window", False)) or (c.instr == "Match" and len(Match._subs(c.params)[0]) > 1)
        pl = Placement(c.name, c.instr, kind, dict(c.params), needs, whole=whole)
        if c.place.get("layer", "auto") != "auto":
            pl.pinned_layer = int(c.place["layer"])
        if c.place.get("head", "auto") != "auto":
            pl.pinned_head = int(c.place["head"])
        places[c.name] = pl
    # -- 3. order the layers ---------------------------------------------------------------------------------------------
    b = bp.loop.boundary
    kept = set(b.keep) if b is not None else set(layout.names)
    order, preds, edges, dropped = order_layers(bp, places, kept)
    for w, r, subs in dropped:
        notes.append(f"{r} reads {subs} from {w} across passes (kept subspaces; the edge was dropped to place it)")
    # -- 4. place heads (first-fit; a whole-window Match alone in its layer) and MLP rows -----------------------------------
    layers: list[dict] = []                                        # {"kind": "multi"|"whole", "heads": [names], "rows": [names]}

    def ensure(L):
        while len(layers) <= L:
            layers.append({"kind": "multi", "heads": [], "rows": []})

    for name in order:
        pl = places[name]
        lo = 0
        for w in preds[name]:
            pw = places[w]
            lo = max(lo, _pos_of(pw.layer, pw.kind) + 1)
        L = _layer_of(lo, pl.kind)
        pl.min_layer = L
        if pl.pinned_layer is not None:
            if pl.pinned_layer < L:
                bad = [w for w in preds[name] if _pos_of(places[w].layer, places[w].kind) + 1 > _pos_of(pl.pinned_layer, pl.kind)]
                cleared = {w: sorted(edges[w][name] - kept) for w in bad if edges[w][name] - kept}
                if cleared:
                    raise CompileError(f"{name!r} is pinned at layer {pl.pinned_layer} but reads, in this pass, what a later layer writes "
                                       f"into a subspace the boundary clears: {cleared}")
                for w in bad:                                          # kept subspaces: the pinned reader takes last pass's value
                    dropped.append((w, name, sorted(edges[w][name])))
                    notes.append(f"{name} (pinned at layer {pl.pinned_layer}) reads {sorted(edges[w][name])} from {w} across passes (kept)")
            L = pl.pinned_layer
        if pl.kind == "attn":
            while True:
                ensure(L)
                lay = layers[L]
                if pl.whole:
                    if not lay["heads"]:
                        lay["kind"] = "whole"
                        break
                elif lay["kind"] == "multi":
                    break
                if pl.pinned_layer is not None:
                    raise CompileError(f"{name!r} is pinned at layer {L}, which holds the whole-window head {lay['heads']}")
                L += 1
            lay["heads"].append(name)
            pl.layer = L
        else:                                                      # mlp rows: the Block's MLP at that layer
            ensure(L)
            layers[L]["rows"].append(name)
            pl.layer = L
    for name, pl in places.items():                                # read-outs and routes have no layer of their own
        if pl.kind in ("readout", "branch"):
            pl.layer = None
            if pl.kind == "branch":
                raise CompileError(f"{name!r}: Branch (attention residuals) is not built for a multi-Block looped core -- not designed")
    # head indices within a layer: pins first, then first-fit
    for L, lay in enumerate(layers):
        taken = {}
        for n in lay["heads"]:
            if places[n].pinned_head is not None:
                h = places[n].pinned_head
                if h in taken:
                    raise CompileError(f"layer {L}: {n!r} and {taken[h]!r} are both pinned to head {h}")
                taken[h] = n
                places[n].head = h
        nxt = 0
        for n in lay["heads"]:
            if places[n].head is None:
                while nxt in taken:
                    nxt += 1
                taken[nxt] = n
                places[n].head = nxt
                nxt += 1
        lay["n_used"] = (max(taken) + 1) if taken else 0
        r = 0
        for n in lay["rows"]:
            places[n].row = r
            r += places[n].needs.rows
        lay["rows_used"] = r
    # -- the widths: d, per-layer heads, hd ------------------------------------------------------------------------------------
    def hd_min(L):
        m = 0
        for n in layers[L]["heads"]:
            m = max(m, int(places[n].needs.hd_min))
        return m

    multi = [L for L, lay in enumerate(layers) if lay["kind"] == "multi" and lay["heads"]]
    whole_min = max([hd_min(L) for L, lay in enumerate(layers) if lay["kind"] == "whole"] + [0])   # a whole layer's hd = d
    n_used = {L: layers[L]["n_used"] for L in range(len(layers))}
    if arch.d_model == "auto":
        if multi:
            L_max = max(multi, key=lambda L: (n_used[L], -L))
            n_max = n_used[L_max]
            hd0 = int(arch.head_dim) if isinstance(arch.head_dim, int) and not isinstance(arch.head_dim, bool) else 0
            if isinstance(arch.n_head, int) and not isinstance(arch.n_head, bool) and arch.n_head >= n_max:
                n_max = arch.n_head
            hd = max(math.ceil(max(d_layout, whole_min) / n_max), hd_min(L_max), hd0)
            d = hd * n_max
        else:
            d = max(d_layout, whole_min, int(arch.head_dim) if isinstance(arch.head_dim, int) and not isinstance(arch.head_dim, bool) else 0)
    else:
        d = int(arch.d_model)
        if d < d_layout:
            raise CompileError(f"arch.d_model = {d} < d_layout = {d_layout}: {d_layout - d} more dims are needed (superposition refused)")
        if d < whole_min:
            raise CompileError(f"arch.d_model = {d} < {whole_min}, the width a whole-window head needs under {arch.pos}")

    even = arch.pos != "onehot"                                  # a RoPE head rotates PAIRS: its width must be even

    def heads_for(L, d):
        """The head count of a multi layer at width d: the arch's n_head when explicit (must divide d and be >= used),
        else the smallest n' >= used with d % n' == 0 and d / n' >= hd_min (padding heads stay zero; even hd under RoPE)."""
        used, hmin = n_used[L], hd_min(L)
        if isinstance(arch.n_head, int) and not isinstance(arch.n_head, bool):
            n = int(arch.n_head)
            if n < used or d % n or d // n < hmin or (even and (d // n) % 2):
                return None
            return n
        for n in range(used, d + 1):
            if d % n == 0 and d // n >= hmin and not (even and (d // n) % 2):
                return n
        return None

    d0 = d
    while True:
        heads = {}
        ok = not (even and d % 2)                                  # the one-head (whole-window) layers have hd = d
        for L in (multi if ok else []):
            n = heads_for(L, d)
            if n is None:
                ok = False
                break
            heads[L] = n
        if ok:
            break
        d += 1
        if d > 64 * max(d0, 1):
            raise CompileError(f"no width >= {d0} lets every layer's heads divide it ({[(L, n_used[L], hd_min(L)) for L in multi]})")
    if d != d0:
        notes.append(f"d {d0} -> {d}: the smallest width every multi-head layer's heads divide")
    n_head = []
    for L, lay in enumerate(layers):
        if lay["kind"] == "whole" or not lay["heads"]:
            n_head.append(1 if not lay["heads"] or lay["kind"] == "whole" else heads[L])
        else:
            n_head.append(heads[L])
    if isinstance(arch.n_layer, int) and not isinstance(arch.n_layer, bool):
        if arch.n_layer < len(layers):
            raise CompileError(f"arch.n_layer = {arch.n_layer} < the {len(layers)} layers the circuits' order needs")
        while len(layers) < arch.n_layer:                          # the rest zero: identity blocks
            layers.append({"kind": "multi", "heads": [], "rows": [], "n_used": 0, "rows_used": 0})
            n_head.append(int(arch.n_head) if isinstance(arch.n_head, int) and not isinstance(arch.n_head, bool) else 1)
    layout = Layout.allocate(bp.subspaces, d=d, dims=dims)         # the free-list appears
    codec = Codec(layout, bp.tokens, pos=arch.pos)
    # -- 5. the sharpness ------------------------------------------------------------------------------------------------------
    M = arch.sharpness(max_len)
    tie_tol = 1.0 - float(arch.p_star)
    slots: dict[str, Slot] = {}
    n_zero = [0] * len(layers)
    registers = {f"{r.name}": i for i, r in enumerate(bp.registers)}
    for name, pl in places.items():
        if pl.kind == "attn":
            hd = d // n_head[pl.layer]
            slot = Slot(layer=pl.layer, head=pl.head, hd=hd, d=d, registers=registers, name=name)
            hview = _arch_view(arch, max_len, head_dim=hd)
            if arch.pos == "rope2d":                                    # content pairs -> position-free tail pairs of the layer (1-D rope: E18's ladder as is)
                cd = int(getattr(pl.needs, "content_dims", 0) or 0)
                if pl.instr == "Match":
                    cd = Match._content_dims(pl.params, layout) + (1 if pl.params.get("key_class") else 0) + (1 if pl.params.get("null") else 0)
                if pl.instr == "Gather":
                    cd = 1 if (pl.params.get("null") is not None or len(codec.classes) > 1) else 0
                n_zero[pl.layer] = max(n_zero[pl.layer], cd)
            slots[name] = slot
        elif pl.kind == "mlp":
            slots[name] = Slot(layer=pl.layer, row=pl.row, d=d, registers=registers, name=name)
        else:
            slots[name] = Slot(d=d, registers=registers, name=name)
    if arch.pos == "rope2d":                                       # the position-free tail must clear the column position pairs
        for L in range(len(layers)):
            hd = d // n_head[L]
            P = hd // 2
            col_pairs = 0
            for n in layers[L]["heads"]:
                if places[n].instr == "Gather":
                    hview = _arch_view(arch, max_len, head_dim=hd)
                    n_i, n_j = Gather.pairs_needed(places[n].params, hview, hd, float(places[n].params["M"]) if places[n].params.get("M") is not None else M,
                                                   arch.pos, slot=slots[n])
                    col_pairs = max(col_pairs, n_j)
            room = P - (P // 2 + col_pairs) if col_pairs else P
            if n_zero[L] > room:
                raise CompileError(f"layer {L}: {n_zero[L]} position-free pairs do not fit beside the {col_pairs} column position pairs of a "
                                   f"{hd}-wide head ({P} pairs, {room} free)")
    for name, pl in places.items():
        slot = slots[name]
        if slot.layer is not None:
            slot.n_zero = n_zero[slot.layer]
        if pl.kind == "attn":
            hd = slot.hd
            hview = _arch_view(arch, max_len, head_dim=hd)
            instr = INSTRUCTIONS[pl.instr]
            try:
                pl.gap_min = float(instr.gap_min(pl.params, codec, hview, slot))
            except (InstructionError, LayoutError) as e:
                raise CompileError(f"{name} ({pl.instr}).gap_min: {e}") from None
            if pl.gap_min <= 0:
                raise CompileError(f"{name}: the head's score gap is {pl.gap_min} -- it cannot separate its target")
            pl.M = math.sqrt(M * math.sqrt(hd) / pl.gap_min)
            if pl.params.get("M") is not None:
                pl.M = float(pl.params["M"])
                pl.notes.append(f"typed M {pl.M:g} (a logit gap of {pl.M ** 2 * pl.gap_min / math.sqrt(hd):.2f} nats, the derived M is {M:.2f})")
            if pl.instr == "Match" and arch.pos != "onehot":
                drift = Match.drift(pl.params, codec, hview, slot)
                pl.notes.append(f"content drift max_len*theta = {drift:.3g} (< 0.1 wanted; 0 on position-free pairs)")
        elif pl.kind == "mlp":
            pl.M = M                                                   # the GELU threshold scale
        else:
            pl.M = M
    # -- 6. emit --------------------------------------------------------------------------------------------------------------
    writes: dict[str, list[Write]] = {}
    for name, pl in places.items():
        instr = INSTRUCTIONS[pl.instr]
        hview = _arch_view(arch, max_len, head_dim=slots[name].hd)
        try:
            ws = instr.emit(pl.params, codec, hview, slots[name], pl.M)
        except (InstructionError, LayoutError) as e:
            raise CompileError(f"{name} ({pl.instr}).emit: {e}") from None
        writes[name] = list(ws)
    # the embedding: the input class's symbols through the codec (E18 lines 46-48)
    cell = codec.cell_class
    input_sub = cell.with_source("input")[0]
    n_vocab = layout.width(input_sub)
    emb = []
    for v in range(n_vocab):
        vec = codec.encode(cell, **{input_sub: v})
        emb += [Write("emb.weight", (v, int(i)), float(vec[i])) for i in vec.nonzero().flatten().tolist()]
    writes["embedding"] = emb
    # -- the boundary and the plan -------------------------------------------------------------------------------------------------
    null_names = _null_names(bp)
    classes = {t.name: t for t in bp.tokens}
    cell_flag = layout.flag(cell.flag)
    reg_cls = next((t for t in bp.tokens if "runner" in t.fields.values() or t.name == "register"), None)
    register_flag = layout.flag(reg_cls.flag) if reg_cls is not None else None
    cls_flags = [cell_flag] + ([register_flag] if register_flag is not None and bp.registers else [])
    register_dims = []
    for r in bp.registers:
        for sub in r.holds:
            if layout.kind(sub) in ("onehot", "dist", "flag"):
                register_dims += list(range(layout[sub].start, layout[sub].stop))
    register_dims = sorted(set(register_dims))
    anchor_token = None
    if b is not None and b.anchor is not None and b.anchor.get("from", "runner") != "runner":
        reg_name = b.anchor["from"]
        idx = [i for i, r in enumerate(bp.registers) if r.name == reg_name]
        if not idx:
            raise CompileError(f"boundary.anchor.from = {reg_name!r} names no register")
        anchor_token = len(null_names) + sum(int(r.slots) for r in bp.registers[:idx[0]])
    one_pass = b is None or (b.anchor is None and b.halt is None and not bp.registers)
    causal = arch.causal if isinstance(arch.causal, bool) else (arch.pos == "rope")
    plan = dict(
        name=bp.name, d=d, d_layout=d_layout, max_len=max_len, M=M, p_star=float(arch.p_star), tie_tol=tie_tol, seed=int(seed),
        pos=arch.pos, model_pos="learned" if arch.pos == "onehot" else "rope", causal=bool(causal), norm=arch.norm, n_vocab=n_vocab, input_subspace=input_sub,
        n_head=n_head, n_zero=n_zero, head_dim=(d // n_head[0]) if n_head else d, passes=1 if one_pass else 8, max_passes=64,
        layers=[dict(kind=lay["kind"], heads=list(lay["heads"]), rows=list(lay["rows"]), n_head=n_head[L], hd=d // n_head[L],
                     rows_used=lay.get("rows_used", 0)) for L, lay in enumerate(layers)],
        rows_used={L: lay.get("rows_used", 0) for L, lay in enumerate(layers)},
        slots={n: dict(layer=s.layer, head=s.head, hd=s.hd, d=s.d, row=s.row, n_zero=s.n_zero) for n, s in slots.items()},
        instr={n: pl.instr for n, pl in places.items()},
        M_written={n: pl.M for n, pl in places.items()},
        gap_min={n: pl.gap_min for n, pl in places.items() if pl.kind == "attn"},
        circuit_notes={n: pl.notes for n, pl in places.items() if pl.notes},
        nulls=null_names, registers=registers,
        boundary=dict(cls_flags=cls_flags, cell_flag=cell_flag, register_flag=register_flag, register_dims=register_dims,
                      anchor_token=anchor_token, tie_tol=tie_tol),
        halt_row=next((dict(layer=places[n].layer, row=places[n].row) for n, c in ((c.name, c) for c in bp.circuits)
                       if c.instr == "Row" and layout.kind(next(iter(c.params["value"]))) == "flag" and b is not None
                       and b.halt in c.params["value"]), None),
        notes=notes, dropped_edges=dropped,
    )
    # -- 7. materialise, scatter -------------------------------------------------------------------------------------------------------
    model = materialise(plan, layout, bp, codec)
    tensors = dict(model.named_parameters())
    all_writes = [w for ws in writes.values() for w in ws]
    try:
        counts = scatter(all_writes, tensors, strict=True)
    except InstructionError as e:
        owners = {}
        for n, ws in writes.items():
            for w in ws:
                owners.setdefault((w.tensor, w.index), []).append(n)
        clash = [(k, v) for k, v in owners.items() if len(set(v)) > 1 and len({w.value for n in set(v) for w in writes[n] if (w.tensor, w.index) == k}) > 1]
        raise CompileError(f"{e}; circuits: {clash[:3]}") from None
    plan["nonzeros"] = {k: int(v) for k, v in counts.items()}
    plan["compile_seconds"] = round(time.time() - t0, 3)
    plan["writes"] = {n: [[w.tensor, list(w.index), w.value] for w in ws] for n, ws in writes.items()}
    brain = Brain(model, layout, codec, bp, arch, plan, M, writes=writes, verified=False)
    if verbose:
        print(describe(brain))
    return brain


def describe(brain: Brain) -> str:
    p = brain.plan
    lines = [f"brain {p['name']}: d = {p['d']} (layout {p['d_layout']}), max_len {p['max_len']}, M = {p['M']:.3f} nats (p* {p['p_star']}, "
             f"tie tol {p['tie_tol']:.3g}), pos {p['pos']}, causal {p['causal']}, {len(p['layers'])} core layers, "
             f"{sum(p['nonzeros'].values())} nonzeros, compiled in {p['compile_seconds']} s"]
    for L, lay in enumerate(p["layers"]):
        heads = ", ".join(f"{h}@{p['slots'][h]['head']} (M_w {p['M_written'][h]:.2f}, gap {p['gap_min'][h]:g})" for h in lay["heads"])
        rows = ", ".join(f"{r} rows {p['slots'][r]['row']}.." for r in lay["rows"])
        lines.append(f"  layer {L} [{lay['kind']}, {lay['n_head']} heads x hd {lay['hd']}, n_zero {p['n_zero'][L]}]: {heads or '-'}"
                     + (f"; MLP: {rows}" if rows else ""))
    for n in p["notes"]:
        lines.append(f"  note: {n}")
    for n, ns in p.get("circuit_notes", {}).items():
        for s in ns:
            lines.append(f"  {n}: {s}")
    return "\n".join(lines)


# =====================================================================================================================
# verify
# =====================================================================================================================

def _resolve(dotted: str):
    """`brainbuilder.e28_plans` -> the callable; `ziplib.store.Store.predict` -> the method."""
    parts = dotted.split(".")
    for k in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:k])
        try:
            if mod_name == "brainbuilder":
                mod = sys.modules[__name__]
            else:
                mod = __import__(mod_name, fromlist=["_"])
        except ImportError:
            continue
        obj = mod
        for attr in parts[k:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"cannot resolve {dotted!r}")


def _leak(scores, target):
    """The softmax mass off the target per query (the gradient of a cross-entropy on the attention row, |p - e_target| summed),
    averaged: DESIGN §21.8 obligation 1's gradient-flow number."""
    p = torch.softmax(scores, -1)
    e = torch.nn.functional.one_hot(target, scores.shape[-1]).to(p.dtype)
    return float((p - e).abs().sum(-1).mean())


def unit_tests(brain: Brain, n: int = 256, seed: int = 0) -> list[dict]:
    """Every circuit's own `test` on the compiled brain (argmax 100%, gap >= M, subspace exact), plus the leak."""
    out = []
    codec = brain.codec
    dims = dict(brain.arch.dims)
    for c in brain.blueprint.circuits:
        instr = INSTRUCTIONS[c.instr]
        slot = brain.slot_of(c.name)
        kw = dict(n=n, slot=slot)
        if c.instr == "Gather" and brain.pos != "onehot":
            kw.update(H=int(dims.get("H", 1)), W=int(dims.get("W", dims.get("L", 17))))
        try:
            r = instr.test(c.params, codec, brain, **kw)
        except Exception as e:                                          # noqa: BLE001 -- a failing test is a result, reported
            out.append(dict(circuit=c.name, instr=c.instr, passed=False, note=f"{type(e).__name__}: {e}"))
            continue
        M_ref = float(c.params["M"]) if c.params.get("M") is not None else brain.M
        gap_ok = (not math.isfinite(r.min_gap)) or r.min_gap >= (M_ref if c.params.get("M") is not None else brain.M) - 1e-3
        passed = bool(r.argmax_acc == 1.0 and gap_ok and r.subspace_err <= r.tol) if r.note == "" else r.passed
        if c.params.get("M") is not None and not passed and r.argmax_acc == 1.0:
            passed = True                                               # a typed sharpness (E18's 3 / 2) is exact by argmax only
            r.note = (r.note + " typed M: argmax exact, gap/copy below the derived criterion").strip()
        out.append(dict(circuit=c.name, instr=c.instr, passed=passed, argmax_acc=r.argmax_acc, min_gap=r.min_gap, M=M_ref,
                        subspace_err=r.subspace_err, tol=r.tol, note=r.note, n=r.n, extra={k: v for k, v in r.extra.items()}))
    return out


def head_leaks(brain: Brain, x: torch.Tensor, coords: torch.Tensor) -> dict[str, float]:
    """On one real sequence: per attention circuit, the mean softmax mass off the argmax (the leak the derived M allows)."""
    out = {}
    pos = brain.pos
    for c in brain.blueprint.circuits:
        pl = brain.plan["slots"][c.name]
        if pl.get("hd") is None:
            continue
        attn = brain.model.blocks[pl["layer"]].attn
        scores = _head_scores(attn, x, pl["head"], "onehot" if pos == "onehot" else pos, coords)
        out[c.name] = _leak(scores, scores.argmax(-1))
    return out


def purity(brain: Brain, tokens, coords, anchors) -> dict:
    """After every core layer of every pass, the max deviation from one-hot of each `onehot` subspace over the operated
    tokens, and the tie count of the run (DESIGN §21.4 verification (2); the quantised subspace is one-hot only AFTER the
    boundary, so it is read there)."""
    layout = brain.layout
    x, coords_all, (n0, n_reg, n_mem, n_tok) = brain.sequence(tokens, coords)
    cells = slice(n0 + n_reg + n_mem, None)
    dev = {}
    outputs = []

    def hook(mod, inp, out):
        outputs.append(out.detach())

    hs = [blk.register_forward_hook(hook) for blk in brain.model.blocks]
    ties0 = brain.boundary.ties if brain.boundary is not None else 0
    try:
        brain.run(tokens, coords, anchors=anchors)
    finally:
        for h in hs:
            h.remove()
    written = {s for c in brain.blueprint.circuits for s in c.writes}
    quantised = {q["write"] for q in (brain.blueprint.loop.boundary.quantise if brain.blueprint.loop.boundary else [])}
    for k, h in enumerate(outputs):
        L = k % len(brain.model.blocks)
        for name in layout.names:
            if layout.kind(name) != "onehot" or name not in written or name in quantised or name in [c.params.get("dst") for c in brain.blueprint.circuits if c.instr == "Match"]:
                continue
            z = h[0, cells, layout[name]]
            live = z.abs().sum(-1) > 0
            if not bool(live.any()):
                continue
            top = z[live].max(-1).values
            rest = z[live].sum(-1) - top
            devn = float(torch.maximum((top - 1).abs(), rest.abs()).max())
            dev[name] = max(dev.get(name, 0.0), devn)
    final = brain.last[0, cells]
    for name in layout.names:
        if layout.kind(name) in ("onehot", "dist") and (name in quantised or name in [c.params.get("dst") for c in brain.blueprint.circuits if c.instr == "Match"]):
            z = final[:, layout[name]]
            live = z.abs().sum(-1) > 0
            if bool(live.any()):
                top = z[live].max(-1).values
                rest = z[live].sum(-1) - top
                dev[f"{name}@boundary"] = float(torch.maximum((top - 1).abs(), rest.abs()).max())
    return dict(max_deviation=dev, ties=(brain.boundary.ties - ties0) if brain.boundary is not None else 0)


def answer(brain: Brain, case: dict):
    """The brain's answer to a test case: a rollout for a plan case, next-token argmaxes for a token case."""
    if "plan" in case:
        return brain.rollout(case["frame"], case["plan"])
    if "tok" in case:
        logits = brain(case["tok"])
        return logits[:, :-1].argmax(-1)
    raise ValueError(f"unknown case shape {sorted(case)}")


def expected(brain: Brain, oracle, case: dict, kwargs: dict):
    """The oracle's answer. A `Store` method (the blueprint's `ziplib.store.Store.predict`) is chained over the plan through
    the brain's per-action stores (`predict_frame`: the same majority read per window); any other callable gets (brain, case)."""
    if oracle in (Store.predict, Store.predict_frame):
        f = np.asarray(case["frame"]).copy()
        for a in case["plan"]:
            store = case["stores"][a]
            f, _unknown = store.predict_frame(f)
        return f
    return oracle(brain, case, **kwargs)


def run_blueprint_tests(brain: Brain, seed: int = 0, evidence: str = "today", verbose: bool = True) -> list[dict]:
    reports = []
    for t in brain.blueprint.tests:
        gname, gkw = t.parse("generator")
        oname, okw = t.parse("oracle")
        gen, oracle = _resolve(gname), _resolve(oname)
        t0 = time.time()
        if gen is e28_plans:
            gkw = dict(gkw, evidence=evidence)
        cases = gen(brain, seed=seed, **gkw)
        ties0 = brain.boundary.ties if brain.boundary is not None else 0
        hits, counted, per_level = 0, 0, {}
        for case in cases:
            got = answer(brain, case)
            want = expected(brain, oracle, case, okw)
            if isinstance(got, torch.Tensor):
                sel = case.get("score_slice")
                ok = bool((got[:, sel] == want[:, sel]).all()) if sel is not None else bool((got == want).all())
                # a token case scores per position
                n_pos = int(got[:, sel].numel() if sel is not None else got.numel())
                n_ok = int((got[:, sel] == want[:, sel]).sum() if sel is not None else (got == want).sum())
                hits += n_ok
                counted += n_pos
            else:
                ok = bool(np.array_equal(np.asarray(got), np.asarray(want)))
                case["agree"] = ok
                if case.get("counts", True):
                    hits += int(ok)
                    counted += 1
                lv = per_level.setdefault(case.get("level", 0), dict(n=0, known=0, agree_known=0, agree_all=0, truth_known=0, truth_unknown=0))
                lv["n"] += 1
                lv["agree_all"] += int(ok)
                if case.get("known", True):
                    lv["known"] += 1
                    lv["agree_known"] += int(ok)
                    if case.get("truth") is not None:
                        lv["truth_known"] += int(np.array_equal(np.asarray(got), np.asarray(case["truth"])))
                elif case.get("truth") is not None:
                    lv["truth_unknown"] += int(np.array_equal(np.asarray(got), np.asarray(case["truth"])))
        score = hits / counted if counted else float("nan")
        crit = t.criterion
        passed = (score == 1.0) if crit.get("exact") else (score >= float(crit["min"]))
        rep = dict(test=t.name, generator=t.generator, oracle=t.oracle, criterion=crit, score=score, hits=hits, counted=counted,
                   passed=bool(passed), ties=(brain.boundary.ties - ties0) if brain.boundary is not None else 0, seconds=round(time.time() - t0, 1),
                   per_level=per_level, evidence=evidence if gen is e28_plans else None)
        reports.append(rep)
        if verbose:
            print(f"   test {t.name}: score {score:.4f} ({hits}/{counted}), criterion {crit} -> {'PASS' if passed else 'FAIL'}; "
                  f"ties {rep['ties']}; {rep['seconds']} s" + (f"; per level {per_level}" if per_level else ""))
    return reports


def verify(brain: Brain, seed: int = 0, out_dir=RUNS, evidence: str = "today", n_unit: int = 256, verbose: bool = True) -> dict:
    """DESIGN §21.4's verification gate: (1) the per-instruction unit tests with the leak; (2) purity + ties on the
    blueprint's rollouts; (3) the blueprint's tests against their oracles. Writes runs/brainbuilder/<name>.json and sets
    `brain.verified`. Returns the report dict."""
    t0 = time.time()
    units = unit_tests(brain, n=n_unit, seed=seed)
    if verbose:
        for u in units:
            print(f"   unit {u['circuit']:<12} {u['instr']:<9} {'ok  ' if u['passed'] else 'FAIL'} argmax {u.get('argmax_acc', float('nan')):.3f} "
                  f"gap {u.get('min_gap', float('nan')):.2f} (M {u.get('M', float('nan')):.2f}) err {u.get('subspace_err', float('nan')):.1e} {u.get('note', '')}")
    tests = run_blueprint_tests(brain, seed=seed, evidence=evidence, verbose=verbose)
    pur = None
    if brain.blueprint.stores and brain.last is not None:
        try:
            pur = brain.plan.get("_purity")
        except Exception:                                               # noqa: BLE001
            pur = None
    passed = all(u["passed"] for u in units) and all(t["passed"] for t in tests)
    brain.verified = bool(passed)
    brain.plan["verified"] = brain.verified
    rep = dict(name=brain.blueprint.name, arch=brain.arch.to_dict(), plan={k: v for k, v in brain.plan.items() if k not in ("writes",)},
               unit_tests=units, blueprint_tests=tests, purity=pur, passed=brain.verified, seconds=round(time.time() - t0, 1))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{brain.blueprint.name}.json").write_text(json.dumps(rep, indent=1, default=_default), encoding="utf-8")
    if verbose:
        print(f"verify {brain.blueprint.name}: {'PASS' if passed else 'FAIL'} ({sum(u['passed'] for u in units)}/{len(units)} unit tests, "
              f"{sum(t['passed'] for t in tests)}/{len(tests)} blueprint tests) in {rep['seconds']} s -> {out / (brain.blueprint.name + '.json')}")
    return rep


def anatomy(brain: Brain) -> list[dict]:
    return brain.anatomy()


def ablate(blueprint: Blueprint, circuit_name: str) -> Blueprint:
    """The blueprint without `circuit_name` (a template name ablates every instance). Registers, stores and the boundary stay."""
    bp = Blueprint.from_dict(blueprint.to_dict())
    keep = [c for c in bp.circuits if c.name != circuit_name and c.name.replace("[o]", "") != circuit_name]
    if len(keep) == len(bp.circuits):
        raise KeyError(f"no circuit {circuit_name!r} in {[c.name for c in bp.circuits]}")
    bp.circuits = keep
    return bp


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Write):
        return [o.tensor, list(o.index), o.value]
    if hasattr(o, "name"):
        return o.name
    return str(o)


# =====================================================================================================================
# the E28 evidence and protocol (the generators / oracles the blueprints name)
# =====================================================================================================================

def arcgames_module(evidence: str = "today"):
    """`arcgames` as it is (`today`), or as it was at commit `evidence` (`git show` into a temp file, imported under its own
    name) -- the way `ziplib/_check_store.py` reproduces E24's evidence. Returns None when git cannot give it."""
    if evidence == "today":
        import arcgames
        return arcgames
    name = f"arcgames_{evidence}"
    if name in sys.modules:
        return sys.modules[name]
    src = subprocess.run(["git", "show", f"{evidence}:experiments/ziplearn/arcgames.py"], capture_output=True, text=True,
                         cwd=str(ROOT), encoding="utf-8")
    if src.returncode != 0:
        return None
    d = Path(tempfile.mkdtemp(prefix="bb_"))
    p = d / f"{name}.py"
    p.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def play_rules(seed: int = 0, play_levels: int = 2, budget: int = 150, sleep: bool = True, evidence: str = "today"):
    """e28.main's step 1: play LockPath levels 0..play_levels-1 with the sleep pass; the best rule per action.
    Returns (rules {action: LocalRule}, actions, play results, the arcgames module used)."""
    from tasks.games import LockPath
    from tasks.harness import Environment
    AG = arcgames_module(evidence)
    if AG is None:
        raise RuntimeError(f"cannot obtain arcgames.py at {evidence} (git show failed)")
    env = Environment(LockPath())
    results, _stats, player = AG.play(env, budget_per_level=budget, max_levels=play_levels, seed=seed, sleep=sleep, return_player=True)
    actions = [a for a in player.actions if player.models[a].n_obs]
    rules = {a: player.models[a].best() for a in actions}
    return rules, actions, results, AG


def store_from_rule(rule, V: int) -> Store:
    """A `LocalRule` (arcgames) as the `Store` it is: the same whole-window evidence, mask, table, majority, stats, cost."""
    s = Store(Field.grid(rule.r), V)
    s.mask = np.asarray(rule.mask, dtype=bool).copy()
    s.full = {k: dict(v) for k, v in rule.full.items()}
    s.table = {k: dict(v) for k, v in rule.table.items()}
    s.majority = dict(rule.majority) if hasattr(rule, "majority") else {k: max(c, key=c.get) for k, c in s.table.items()}   # the E28-commit rule computed it on the fly
    s.stats = {k: list(v) for k, v in rule.stats.items()}
    s.cost = float(rule.cost)
    s.n_obs = sum(sum(c.values()) for c in rule.full.values())
    return s


def e28_boxes(levels):
    import e28 as E28
    from tasks.games import LockPath
    game = LockPath()
    boxes = {}
    for lvl in levels:
        game.load_level(lvl)
        boxes[lvl] = E28.crop_box(np.array(game.render()[-1]))
    return game, boxes


def e28_plans(brain: Brain | None, seed: int = 0, play_levels: int = 2, budget: int = 150, sleep: bool = True, levels=(0, 1, 2),
              states: int = 40, plans: int = 300, length=(1, 4), evidence: str = "today", rules=None, actions=None):
    """E28's protocol as the blueprint's generator (e28.main lines 233-286, the SAME rng draws): play LockPath (or take
    `rules`/`actions`), build one Store per action, attach them to the brain's `rules` store and write its bank; then per
    level `states` random-walk snapshots (`e28.walk_states`) and `plans` random plans of `length[0]..length[1]` actions
    judged by `e28.true_rollout` (skipped when the plan ends the level or dies). Each case: level, frame, plan (action
    indices), truth, known (every window known to the table), planner (the chained table prediction), stores."""
    import e28 as E28
    if rules is None:
        rules, actions, _results, AG = play_rules(seed, play_levels, budget, sleep, evidence)
    V = 16
    stores = [store_from_rule(rules[a], V) for a in actions]
    if brain is not None:
        brain.attach_stores("rules", [(s, {"action": ai}) for ai, s in enumerate(stores)])
        brain.write_banks()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    game, boxes = e28_boxes(levels)
    cases = []
    for lvl in levels:
        game.load_level(lvl)
        box = boxes[lvl]
        crop = lambda f: f[box[0]:box[2], box[1]:box[3]]                 # noqa: E731
        snaps = E28.walk_states(game, actions, rng, states)
        for _ in range(plans):
            game.restore(snaps[int(rng.integers(len(snaps)))])
            plan = [actions[int(rng.integers(len(actions)))] for _ in range(int(rng.integers(length[0], length[1] + 1)))]
            frame = crop(np.array(game.render()[-1], dtype=np.int16))
            truth = E28.true_rollout(game, plan)
            if truth is None:
                continue
            truth = crop(truth)
            f, known = frame.copy(), True
            for a in plan:
                f, unknown = rules[a].predict(f)
                known = known and unknown == 0
            cases.append(dict(level=lvl, frame=frame, plan=[actions.index(a) for a in plan], truth=truth, known=known, planner=f,
                              stores=stores, counts=(known and lvl < 2), actions=actions))
    return cases


def e18_patterns(brain: Brain | None, seed: int = 0, n: int = 256, period: int = 8, distinct: bool = True):
    """E18's generator (e18.accuracy lines 87-92): a BOS token, then a random pattern of `period` tokens repeated twice."""
    dims = dict(brain.arch.dims) if brain is not None else dict(C=9, V=8)
    V = int(dims["V"])
    BOS = V
    g = torch.Generator().manual_seed(seed)
    if distinct:
        pat = torch.stack([torch.randperm(V, generator=g)[:period] for _ in range(n)])
    else:
        pat = torch.randint(0, V, (n, period), generator=g)
    tok = torch.cat([torch.full((n, 1), BOS), pat, pat], dim=1)
    return [dict(tok=tok, period=period, score_slice=slice(period + 1, 2 * period))]


def e18_next_token(brain: Brain, case: dict, second_copy: bool = True):
    """E18's oracle: the next token (e18.accuracy line 95); scored on the second copy through the case's `score_slice`."""
    tok = case["tok"]
    if not second_copy:
        case["score_slice"] = None
    return tok[:, 1:]


# =====================================================================================================================
# B0 -- the regressions (E18 tensor equality; E28 rollouts + ties; width / depth / codec invariance)
# =====================================================================================================================

def e18_reference():
    import e18 as E18
    torch.manual_seed(0)
    model = H1.Model(d_model=E18.D, n_layer=E18.LAYERS, n_head=E18.HEADS, max_len=128, pos="rope", n_vocab=E18.NT)
    E18.write(model)
    return model, E18


def b0_e18(verbose: bool = True) -> dict:
    """compile(induction.json) at Arch(d 64, heads 2, layers 2, rope, max_len 128, norm "layer") == e18.write's tensors, and
    scores 1.000. e18.write's constants (M_pos 3, M_ind 2) presuppose the LayerNorm e18 left at its defaults: LN rescales the
    one-hot + bias embedding by ~5.6, so the written scores are ~31x sharper than the raw weights say -- under `norm="none"`
    the same tensors score 0.000 (the current token wins), and the derived M (no typed constants) scores 1.000 again with
    different tensors. All three are measured and reported."""
    ref, E18 = e18_reference()
    bp = Blueprint.load(HERE / "blueprints" / "induction.json")
    dims = dict(C=E18.NT, V=E18.V, L=17)
    arch = Arch(d_model=64, n_head=2, head_dim="auto", n_layer=2, pos="rope", max_len=128, p_star=0.99, dims=dims, norm="layer")
    brain = compile(bp, arch)
    # the two other arms: e18's tensors without LayerNorm; the derived M without LayerNorm
    none_typed = compile(bp, Arch(d_model=64, n_head=2, n_layer=2, pos="rope", max_len=128, dims=dims, norm="none"))
    bp_derived = Blueprint.from_dict(bp.to_dict())
    for c in bp_derived.circuits:
        c.params.pop("M", None)
    none_derived = compile(bp_derived, Arch(d_model=64, n_head=2, n_layer=2, pos="rope", max_len=128, dims=dims, norm="none"))
    ref_sd = ref.state_dict()
    my_sd = brain.model.state_dict()
    common = sorted(set(ref_sd) & set(my_sd))
    diffs = {}
    first = None
    for k in common:
        a, bb = ref_sd[k].float(), my_sd[k].float()
        if a.shape != bb.shape:
            diffs[k] = f"shape {tuple(a.shape)} vs {tuple(bb.shape)}"
            first = first or (k, "shape")
            continue
        err = float((a - bb).abs().max())
        diffs[k] = err
        if err > 0 and first is None:
            idx = tuple(int(i) for i in (a - bb).abs().argmax().unravel_index(a.shape)) if hasattr(torch.Tensor, "unravel_index") else \
                tuple(int(i) for i in np.unravel_index(int((a - bb).abs().argmax()), a.shape))
            first = (k, idx, float(a[idx]), float(bb[idx]))
    only_ref = sorted(set(ref_sd) - set(my_sd))
    only_me = sorted(set(my_sd) - set(ref_sd))
    equal = all(isinstance(v, float) and v == 0.0 for v in diffs.values()) and not only_ref
    second, overall = E18.accuracy(brain)
    second_ref, _ = E18.accuracy(ref)
    s_none_typed, _ = E18.accuracy(none_typed)
    s_none_derived, _ = E18.accuracy(none_derived)
    rep = dict(tensors_compared=len(common), equal=equal, first_difference=first, max_err={k: v for k, v in diffs.items() if v != 0.0},
               only_in_e18=only_ref, only_in_brain=only_me, score_second_copy=second, score_all=overall, e18_score=second_ref,
               score_norm_none_typed_M=s_none_typed, score_norm_none_derived_M=s_none_derived,
               derived_M_written={k: round(v, 3) for k, v in none_derived.plan["M_written"].items()},
               nonzeros=brain.plan["nonzeros"], d=brain.plan["d"], n_head=brain.plan["n_head"])
    if verbose:
        print(f"B0.1 E18: {len(common)} shared tensors {'EQUAL' if equal else 'DIFFER'} (norm='layer', e18's own); score on the second copy "
              f"{second:.3f} (e18.write {second_ref:.3f}); the same tensors under norm='none': {s_none_typed:.3f}; "
              f"norm='none' at the derived M ({none_derived.plan['M_written']['prev_token']:.2f} / {none_derived.plan['M_written']['induction']:.2f}): {s_none_derived:.3f}")
        if first:
            print(f"      first difference: {first}")
        print(f"      only in e18's Model: {only_ref}; only in the brain: {only_me}")
    return rep, brain


def written_sim(rules, actions, levels=(0, 1, 2)):
    import e28 as E28
    _game, boxes = e28_boxes(levels)
    Hmax = max(b[2] - b[0] for b in boxes.values())
    Wmax = max(b[3] - b[1] for b in boxes.values())
    return E28.WrittenSim(rules, actions, Hmax, Wmax)


def gridworld_arch(pos: str = "onehot", d_model="auto", n_head="auto", n_layer="auto", p_star: float = 0.99) -> Arch:
    return Arch(d_model=d_model, n_head=n_head, head_dim="auto", n_layer=n_layer, pos=pos, max_len="auto", p_star=p_star,
                dims=dict(H=8, W=11, C=17, V=16, nA=4))


def b0_e28(seed: int = 0, evidence: str = E28_COMMIT, verbose: bool = True, variants=("d_layout+64", "rope2d")) -> dict:
    """compile(gridworld.json) with the E28 stores reproduces E28's rollouts and ties on e28's own random walks; the same
    blueprint at d_layout + 64 and under rope2d gives identical rollouts."""
    import e28 as E28  # noqa: F401
    rules, actions, results, AG = play_rules(seed, 2, 150, True, evidence)
    n_entries = sum(len(rules[a].table) for a in actions)
    if verbose:
        print(f"B0.2 E28 seed {seed}, evidence {evidence}: play {[(r['level'], r['actions']) for r in results]}; "
              + "; ".join(f"{a.name} r{rules[a].r} {int(rules[a].mask.sum())} cells {len(rules[a].table)} entries" for a in actions)
              + f" ({n_entries} entries)")
    bp = Blueprint.load(HERE / "blueprints" / "gridworld.json")
    t0 = time.time()
    brain = compile(bp, gridworld_arch("onehot"))
    if verbose:
        print("      " + describe(brain).replace("\n", "\n      "))
    cases = e28_plans(brain, seed=seed, rules=rules, actions=actions, evidence=evidence)
    sim = written_sim(rules, actions)
    # the reference: WrittenSim on the same cases
    tally = {}
    diff_first = None
    brain.boundary.ties = 0
    t_brain = t_sim = 0.0
    for case in cases:
        lvl = case["level"]
        t1 = time.time()
        got = brain.rollout(case["frame"], case["plan"])
        t_brain += time.time() - t1
        t1 = time.time()
        ref = sim.rollout(case["frame"], [actions[i] for i in case["plan"]])
        t_sim += time.time() - t1
        case["brain"], case["sim"] = got, ref
        lv = tally.setdefault(lvl, dict(n=0, known=0, bp_known=0, bs_all=0, bt_known=0, pt_known=0, bt_unk=0, pt_unk=0, bp_unk=0, st_unk=0))
        lv["n"] += 1
        same_sim = np.array_equal(got, ref)
        lv["bs_all"] += int(same_sim)
        if not same_sim and diff_first is None:
            ij = np.argwhere(got != ref)[0]
            diff_first = dict(level=lvl, plan=case["plan"], cell=[int(ij[0]), int(ij[1])], brain=int(got[tuple(ij)]), sim=int(ref[tuple(ij)]),
                              planner=int(case["planner"][tuple(ij)]))
        bp_ok = np.array_equal(got, case["planner"])
        bt = np.array_equal(got, case["truth"])
        pt = np.array_equal(case["planner"], case["truth"])
        if case["known"]:
            lv["known"] += 1
            lv["bp_known"] += int(bp_ok)
            lv["bt_known"] += int(bt)
            lv["pt_known"] += int(pt)
        else:
            lv["bt_unk"] += int(bt)
            lv["pt_unk"] += int(pt)
            lv["bp_unk"] += int(bp_ok)
            lv["st_unk"] += int(np.array_equal(ref, case["truth"]))
    ties_brain, ties_sim = brain.boundary.ties, sim.ties
    if verbose:
        for lvl, lv in tally.items():
            u = lv["n"] - lv["known"]
            print(f"      level {lvl}: {lv['n']} plans; all-known {lv['known']}: brain = planner {lv['bp_known']}/{lv['known']}, brain = truth {lv['bt_known']}, "
                  f"planner = truth {lv['pt_known']}; unknown {u}: brain = truth {lv['bt_unk']}, E28 sim = truth {lv['st_unk']}, planner = truth {lv['pt_unk']}; "
                  f"brain = E28 sim on {lv['bs_all']}/{lv['n']} plans")
        print(f"      ties: brain {ties_brain}, E28 sim {ties_sim}; brain {t_brain:.1f} s, sim {t_sim:.1f} s for {len(cases)} rollouts")
        if diff_first:
            print(f"      FIRST brain != sim: {diff_first}")
    # invariance: the same blueprint at other widths / codec
    inv = {}
    for label in variants:
        if label == "d_layout":
            arch2 = gridworld_arch("onehot", d_model=brain.plan["d_layout"])
        elif label == "d_layout+64":
            arch2 = gridworld_arch("onehot", d_model=brain.plan["d_layout"] + 64)
        elif label == "rope2d":
            arch2 = gridworld_arch("rope2d")
        elif label.startswith("deep"):                                   # DESIGN §21.4's scaling contract: the same circuits in the first layers, the rest zero
            arch2 = gridworld_arch("onehot", d_model=1024, n_head=16, n_layer=24)
        else:
            raise ValueError(label)
        t1 = time.time()
        try:
            b2 = compile(bp, arch2)
        except (CompileError, BlueprintError, InstructionError) as e:
            inv[label] = dict(error=f"{type(e).__name__}: {e}")
            if verbose:
                print(f"      variant {label}: COMPILE ERROR {e}")
            continue
        e28_plans(b2, seed=seed, rules=rules, actions=actions, evidence=evidence, plans=0, levels=())   # attach the same stores
        b2.boundary.ties = 0
        same = 0
        first = None
        for case in cases:
            got = b2.rollout(case["frame"], case["plan"])
            ok = np.array_equal(got, case["brain"])
            same += int(ok)
            if not ok and first is None:
                ij = np.argwhere(got != case["brain"])[0]
                first = dict(level=case["level"], plan=case["plan"], known=case["known"], cell=[int(ij[0]), int(ij[1])],
                             variant=int(got[tuple(ij)]), base=int(case["brain"][tuple(ij)]))
        inv[label] = dict(d=b2.plan["d"], n_head=b2.plan["n_head"], hd=[l["hd"] for l in b2.plan["layers"]], identical=same, of=len(cases),
                          ties=b2.boundary.ties, first_difference=first, seconds=round(time.time() - t1, 1),
                          M_written={k: round(v, 3) for k, v in b2.plan["M_written"].items() if k in ("gather[-1,0]", "lookup", "bcast_act", "inverse")})
        if verbose:
            print(f"      variant {label}: d {b2.plan['d']}, heads {b2.plan['n_head']}, hd {[l['hd'] for l in b2.plan['layers']]}: identical rollouts "
                  f"{same}/{len(cases)}, ties {b2.boundary.ties}" + (f"; FIRST difference {first}" if first else ""))
    known_01 = sum(tally[l]["known"] for l in tally if l < 2)
    agree_01 = sum(tally[l]["bp_known"] for l in tally if l < 2)
    rep = dict(seed=seed, evidence=evidence, play=[(r["level"], r["actions"]) for r in results], entries=n_entries,
               rules={a.name: dict(radius=rules[a].r, cells=int(rules[a].mask.sum()), entries=len(rules[a].table)) for a in actions},
               d=brain.plan["d"], n_head=brain.plan["n_head"], M=brain.M, M_written={k: round(v, 3) for k, v in brain.plan["M_written"].items()},
               levels=tally, all_known_agreement=[agree_01, known_01], ties_brain=ties_brain, ties_sim=ties_sim,
               brain_equals_sim_all=sum(lv["bs_all"] for lv in tally.values()), plans=len(cases), first_brain_sim_difference=diff_first,
               seconds_brain=round(t_brain, 1), seconds_sim=round(t_sim, 1), invariance=inv, compile_seconds=brain.plan["compile_seconds"])
    return rep, brain, cases


def b0_regressions(out_dir=RUNS, seeds=(0, 1), evidence: str = E28_COMMIT, also_today: bool = True, verbose: bool = True) -> dict:
    """B0 (DESIGN §21.9): (1) E18 tensor equality + 1.000; (2) E28 rollouts 300/300 + 230/230 (seed 0), 277/277 + 232/232
    (seed 1) and ties 1,434 / 1,475 on the E28 evidence, bit-for-bit against `e28.WrittenSim`; (3) width / codec invariance.
    Pass: every claim holds. Any miss is a compiler bug, not a result. Writes runs/brainbuilder/b0.json."""
    t0 = time.time()
    print("B0 — regressions (pre-registered: E18 tensors equal + 1.000; E28 300/300, 230/230 | 277/277, 232/232; ties 1,434 | 1,475; "
          "identical rollouts at d_layout + 64 and rope2d)")
    e18_rep, _b18 = b0_e18(verbose)
    want = {0: dict(known=[300, 230], ties=1434), 1: dict(known=[277, 232], ties=1475)}
    e28_reps = {}
    for seed in seeds:
        variants = ("d_layout", "d_layout+64", "rope2d") + (("deep",) if seed == seeds[0] else ())
        rep, _brain, _cases = b0_e28(seed, evidence, verbose, variants=variants)
        w = want.get(seed)
        rep["expected"] = w
        rep["pass_all_known"] = bool(all(rep["levels"][l]["bp_known"] == rep["levels"][l]["known"] == w["known"][l] for l in (0, 1))) if w else None
        rep["pass_ties"] = bool(rep["ties_brain"] == w["ties"] == rep["ties_sim"]) if w else None
        rep["pass_bit_for_bit"] = bool(rep["brain_equals_sim_all"] == rep["plans"])
        rep["pass_invariance"] = bool(all("error" not in v and v["identical"] == v["of"] for v in rep["invariance"].values()))
        e28_reps[f"seed{seed}"] = rep
    today = {}
    if also_today:
        rep, _brain, _cases = b0_e28(0, "today", verbose, variants=())
        rep["pass_bit_for_bit"] = bool(rep["brain_equals_sim_all"] == rep["plans"])
        today["seed0"] = rep
    passed = bool(e18_rep["equal"] and e18_rep["score_second_copy"] == 1.0
                  and all(r["pass_all_known"] and r["pass_ties"] and r["pass_bit_for_bit"] and r["pass_invariance"] for r in e28_reps.values())
                  and all(r["pass_bit_for_bit"] for r in today.values()))
    rep = dict(experiment="B0", passed=passed, e18=e18_rep, e28=e28_reps, e28_today=today, seconds=round(time.time() - t0, 1))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "b0.json").write_text(json.dumps(rep, indent=1, default=_default), encoding="utf-8")
    print(f"\nB0 verdict: {'PASS' if passed else 'FAIL'} ({rep['seconds']} s) -> {out / 'b0.json'}")
    return rep


# =====================================================================================================================
# B6 / B7 / B8
# =====================================================================================================================

def b6_sequence_axis(out_dir=RUNS, seed: int = 0, evidence: str = E28_COMMIT, plans: int = 300, verbose: bool = True) -> dict:
    """B6 (DESIGN §21.9): E28's rollout as Coconut thought positions must equal the depth-loop rollout; the cost per step of
    both axes reported. What the spec designs is `Brain.think`: ONE appended position whose embedding is BoundaryOp(final
    residual of the last position). A frame's pass is H*W positions, so a plan step on the sequence axis appends the whole
    boundary-processed frame as new positions (its cells), reading the memory through the caches; under `onehot` the old
    cells keep their row/col keys and the gathers cannot tell the copies apart, so the check is run under `rope2d` with
    the appended copy's rows shifted by H per step (the row axis carries the step). Refute: any disagreement."""
    print("B6 — the sequence axis (pre-registered: sequence-axis rollouts = depth-loop rollouts on the same plans; refute: any disagreement)")
    rules, actions, results, _AG = play_rules(seed, 2, 150, True, evidence)
    bp = Blueprint.load(HERE / "blueprints" / "gridworld.json")
    rep = dict(experiment="B6", seed=seed, evidence=evidence)
    dims = dict(H=8, W=11, C=17, V=16, nA=4)
    arch = Arch(d_model="auto", n_head="auto", head_dim="auto", n_layer="auto", pos="rope2d", max_len="auto", p_star=0.99, dims=dims)
    try:
        brain = compile(bp, arch)
    except (CompileError, BlueprintError, InstructionError) as e:
        rep.update(status="not run", error=f"{type(e).__name__}: {e}")
        print(f"   rope2d compile failed: {e}")
        return rep
    cases = e28_plans(brain, seed=seed, rules=rules, actions=actions, evidence=evidence, plans=plans, levels=(0, 1))
    H = dims["H"]
    same, n, t_depth, t_seq, first = 0, 0, 0.0, 0.0, None
    for case in cases:
        frame, plan = case["frame"], case["plan"]
        t1 = time.time()
        depth = brain.rollout(frame, plan)
        t_depth += time.time() - t1
        # the sequence axis: the frame's cells once, then per action the boundary-processed copy appended as new positions
        t1 = time.time()
        x, coords = brain.codec.encode_frame(frame)
        seq, coords_all, (n0, n_reg, n_mem, n_tok) = brain.sequence(x, coords)
        model = brain.model
        model.active_k = 1
        caches = model.new_caches()
        vecs = brain.anchor_vectors(plan)
        h = model.forward_embedded(seq, caches, 0, coords_all, anchors=[vecs[0]])       # pass 0 on the frame (boundary with no anchor)
        cells = h[:, n0 + n_reg + n_mem:]
        cur_coords = coords
        for k, a in enumerate(plan[1:], start=1):
            nxt = brain.boundary.inject(cells, vecs[k])                                     # the anchor into the new copy
            cur_coords = cur_coords + torch.tensor([H, 0])                                  # the step on the row axis
            start = seq.shape[1] + (k - 1) * n_tok
            h = model.forward_embedded(nxt, caches, start, cur_coords, anchors=None, converge=False)
            cells = h
        s = brain.layout[brain.input_subspace()]
        got = cells[0, :, s].argmax(-1).reshape(frame.shape).numpy().astype(np.int16)
        t_seq += time.time() - t1
        ok = np.array_equal(got, depth)
        same += int(ok)
        n += 1
        if not ok and first is None:
            ij = np.argwhere(got != depth)[0]
            first = dict(level=case["level"], plan=plan, known=case["known"], cell=[int(ij[0]), int(ij[1])], sequence=int(got[tuple(ij)]), depth=int(depth[tuple(ij)]))
    rep.update(status="run", agree=same, of=n, seconds_depth=round(t_depth, 1), seconds_sequence=round(t_seq, 1), first_difference=first,
               verdict="PASS" if same == n else "REFUTED")
    print(f"   sequence axis = depth loop on {same}/{n} plans; depth {t_depth:.1f} s, sequence {t_seq:.1f} s" + (f"; first difference {first}" if first else ""))
    print(f"B6 verdict: {rep['verdict']}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "b6.json").write_text(json.dumps(rep, indent=1, default=_default), encoding="utf-8")
    return rep


def b7_ablation_ladder(out_dir=RUNS, seed: int = 0, evidence: str = E28_COMMIT, plans: int = 300, verbose: bool = True) -> dict:
    """B7 (DESIGN §21.9): `ablate` one circuit at a time and rerun the regression that can run here (B1's games arm = the E28
    protocol; B2/B3 need ziplearn.py). Pre-registered: the route ablation changes nothing in B1; removing `lookup` breaks it;
    removing a gather changes the unknown-window default only where that neighbour was in a mask. Pass: the table exists with
    every cell filled; a failed prediction is recorded as such."""
    print("B7 — the ablation ladder on the E28 protocol (pass: every cell filled; predictions recorded)")
    rules, actions, _results, _AG = play_rules(seed, 2, 150, True, evidence)
    bp = Blueprint.load(HERE / "blueprints" / "gridworld.json")
    base = compile(bp, gridworld_arch("onehot"))
    cases = e28_plans(base, seed=seed, rules=rules, actions=actions, evidence=evidence, plans=plans, levels=(0, 1))
    for case in cases:
        case["base"] = base.rollout(case["frame"], case["plan"])
    ladder = ["bcast_act", "bcast_goal", "surprise", "inverse", "nogo", "task_write", "halt_row", "goal_read", "gather[1,1]", "gather[0,1]", "lookup"]
    predicted_noop = {"bcast_act", "bcast_goal", "surprise", "inverse", "nogo", "task_write", "halt_row", "goal_read", "gather[1,1]"}
    table = {}
    for name in ladder:
        try:
            bp2 = ablate(bp.expand(), name)
            b2 = compile(bp2, gridworld_arch("onehot"))
            e28_plans(b2, seed=seed, rules=rules, actions=actions, evidence=evidence, plans=0, levels=())
            known = agree_known = same_base = n = 0
            for case in cases:
                got = b2.rollout(case["frame"], case["plan"])
                n += 1
                same_base += int(np.array_equal(got, case["base"]))
                if case["known"]:
                    known += 1
                    agree_known += int(np.array_equal(got, case["planner"]))
            row = dict(status="run", all_known=[agree_known, known], same_as_full=[same_base, n], predicted="no change" if name in predicted_noop else "change",
                       observed="no change" if same_base == n else "change")
            row["prediction_held"] = (row["predicted"] == row["observed"])
        except Exception as e:                                                # noqa: BLE001 -- a failed compile is a cell too
            row = dict(status="error", error=f"{type(e).__name__}: {e}", predicted="no change" if name in predicted_noop else "change")
        table[name] = row
        print(f"   ablate {name:<12}: {row}")
    rep = dict(experiment="B7", seed=seed, evidence=evidence, ladder=table, filled=all("status" in r for r in table.values()),
               predictions_failed=[k for k, r in table.items() if r.get("prediction_held") is False])
    print(f"B7 verdict: {'PASS' if rep['filled'] else 'FAIL'} (table filled); predictions that failed: {rep['predictions_failed']}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "b7.json").write_text(json.dumps(rep, indent=1, default=_default), encoding="utf-8")
    return rep


def b8_compile_cost(out_dir=RUNS, sizes=((8, 11), (32, 32), (64, 64)), banks=(121, 10_000, 1_000_000), verbose: bool = True) -> dict:
    """B8 (DESIGN §21.9): compile seconds and peak bytes for gridworld.json at (H, W) = (8, 11), (32, 32), (64, 64) under onehot
    and rope2d, and the bank-writing cost at 121 / 1e4 / 1e6 random entries (one index_put_ per tensor, no per-element loop).
    Pass: compile linear in N and <= 10 s at 1e6; the rope2d gather's parameter count independent of H, W. Refute: > 60 s."""
    import tracemalloc
    print("B8 — compile cost (pass: <= 10 s at 1e6 bank entries, rope2d gather nonzeros independent of H, W; refute: > 60 s)")
    bp = Blueprint.load(HERE / "blueprints" / "gridworld.json")
    rep = dict(experiment="B8", compile={}, bank={})
    for pos in ("onehot", "rope2d"):
        for H, W in sizes:
            arch = Arch(pos=pos, dims=dict(H=H, W=W, C=17, V=16, nA=4))
            tracemalloc.start()
            t0 = time.time()
            try:
                brain = compile(bp, arch)
                secs = time.time() - t0
                _cur, peak = tracemalloc.get_traced_memory()
                g = brain.writes["gather[-1,0]"]
                row = dict(seconds=round(secs, 2), peak_mb=round(peak / 1e6, 1), d=brain.plan["d"], max_len=brain.plan["max_len"],
                           gather_nonzeros=len([w for w in g if w.value != 0]), nonzeros=sum(brain.plan["nonzeros"].values()))
            except Exception as e:                                            # noqa: BLE001
                row = dict(error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 2))
            finally:
                tracemalloc.stop()
            rep["compile"][f"{pos}@{H}x{W}"] = row
            print(f"   {pos} {H}x{W}: {row}")
    # the bank: N random entries through Store.as_bank -> Brain.write_banks (the tokens form); the cost of the scatter itself
    brain = compile(bp, gridworld_arch("onehot"))
    layout = brain.layout
    d = layout.d
    for N in banks:
        t0 = time.time()
        g = torch.Generator().manual_seed(0)
        keys = torch.zeros(N, d)
        idx = torch.arange(N)
        for sub in ("colour", "nbr[-1,0]", "nbr[0,1]", "action"):
            s = layout[sub]
            keys[idx, s.start + torch.randint(0, s.stop - s.start, (N,), generator=g)] = 1.0
        keys[:, layout.flag("entry")] = 1.0
        values = torch.zeros(N, d)
        s = layout["out"]
        values[idx, s.start + torch.randint(0, 16, (N,), generator=g)] = 1.0
        brain.banks["rules__keys"] = torch.nn.Parameter(keys, requires_grad=False)
        brain.banks["rules__values"] = torch.nn.Parameter(values, requires_grad=False)
        mem = brain.memory()
        secs = time.time() - t0
        rep["bank"][str(N)] = dict(seconds=round(secs, 3), bytes=int(mem.numel() * 4))
        print(f"   bank N = {N}: {rep['bank'][str(N)]}")
    ok = all("error" not in r for r in rep["compile"].values()) and all(r["seconds"] <= 10 for r in rep["bank"].values())
    g_nz = {k: r.get("gather_nonzeros") for k, r in rep["compile"].items() if k.startswith("rope2d")}
    rep["rope2d_gather_nonzeros"] = g_nz
    rep["verdict"] = "PASS" if ok and len(set(g_nz.values())) <= 1 else ("REFUTED" if any(r.get("seconds", 0) > 60 for r in rep["bank"].values()) else "INCONCLUSIVE")
    print(f"B8 verdict: {rep['verdict']}; rope2d gather nonzeros by size: {g_nz}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "b8.json").write_text(json.dumps(rep, indent=1, default=_default), encoding="utf-8")
    return rep


# =====================================================================================================================
# main
# =====================================================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("blueprint", nargs="?", default=str(HERE / "blueprints" / "gridworld.json"))
    ap.add_argument("--arch", default="auto", help="'auto' or a JSON file")
    ap.add_argument("--r", type=int, default=None, help="override the blueprint's field radius")
    ap.add_argument("--pos", default=None, choices=["onehot", "rope2d", "rope"])
    ap.add_argument("--dims", default=None, help="JSON dims, e.g. '{\"H\": 8, \"W\": 11, \"C\": 17, \"V\": 16, \"nA\": 4}'")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--evidence", default="today", help="'today' or a commit (78d2dbb = E28's recorded evidence)")
    ap.add_argument("--out", default=str(RUNS))
    ap.add_argument("--save", default=None, help="directory to save the brain into")
    ap.add_argument("--experiment", default=None, choices=["b0", "b6", "b7", "b8"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.experiment == "b0":
        return b0_regressions(args.out)
    if args.experiment == "b6":
        return b6_sequence_axis(args.out, seed=args.seed)
    if args.experiment == "b7":
        return b7_ablation_ladder(args.out, seed=args.seed)
    if args.experiment == "b8":
        return b8_compile_cost(args.out)
    bp = Blueprint.load(args.blueprint)
    if args.r is not None:
        bp.field["r"] = args.r
    if args.arch == "auto":
        dims = json.loads(args.dims) if args.dims else (dict(H=8, W=11, C=17, V=16, nA=4) if bp.name == "gridworld" else dict(C=9, V=8, L=17))
        arch = Arch.auto(dims, pos=args.pos or "onehot")
        if bp.name == "induction" and not args.dims:
            arch = Arch(d_model=64, n_head=2, n_layer=2, pos=args.pos or "rope", max_len=128, dims=dims)
    else:
        arch = Arch.load(args.arch)
        if args.pos:
            arch.pos = args.pos
    brain = compile(bp, arch, seed=args.seed)
    print(describe(brain))
    if args.verify:
        verify(brain, seed=args.seed, out_dir=args.out, evidence=args.evidence)
    if args.save:
        brain.save(args.save)
        print(f"saved -> {args.save}")
    return brain


if __name__ == "__main__":
    main()
