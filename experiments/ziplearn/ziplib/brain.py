"""ziplib.brain — the compiled artefact and its runtime plumbing (DESIGN §21.2, §21.4, §21.8; the module spec's `brain.py`).

A `Brain` is an `h1_lid.LoopedModel` whose weights were WRITTEN by `brainbuilder.compile` (`norm="none"`: every code is
one-hot, nothing to normalise), plus what the model cannot carry on its own: the resolved `Layout` and `Codec` (data <->
tokens), the expanded `Blueprint` and the resolved `Arch`, the `BoundaryOp` (the written operator between passes), the
STATE the learner fills -- `banks` (one (keys, values) pair per store: the memory tokens, split for the k/v cache),
`W` (the `Pool` value matrices), `registers` (the register tokens' initial contents) -- the `stores` (the evidence the
banks are regenerated from), the sharpness `M` (nats: the softmax gap one match buys, `Arch.sharpness`), the `Write`
lists per circuit (what `anatomy` traces every nonzero to), and `verified=False` until `brainbuilder.verify` passes.

  run(tokens, coords, anchors, max_passes)   MOVES `WrittenSim.rollout`'s loop (e28.py lines 172-191): the sequence is
                                             nulls + registers + memory tokens + the given tokens; a plan of n anchors is
                                             n passes (anchor k before pass k, the BoundaryOp after it); without anchors
                                             the loop runs until the halt flag, register convergence or `max_passes`, the
                                             anchor read off the ACTION register by the operator itself. Plain torch.
  rollout(frame, plan)                       E28's interface on a frame: encode, run, decode the input subspace.
  forward(tok)                               a sequence brain as a language model: head(norm(forward_embedded(embed(tok)))).
  think(tokens, coords, n)                   the sequence axis (DESIGN §18, §21.4): append a thought position whose embedding
                                             is BoundaryOp(final residual of the last position) through `loop.sequence.feedback`,
                                             via `model.forward_embedded(h, caches, start)`; the same halt rule, or n thoughts.
  rows.write(key_row, threshold, value_row)  claims the next free MLP row (the headroom) and writes a `Row` into it.
  param_groups()                             {"structure": [...], "state": [...]} -- the two parameter classes of DESIGN §21.8.
  readout(name, x)                           "action": the ACTION register's `action` through the head; "pred": the cells' `pred`;
                                             the head sees only that subspace (both Readouts share `head.weight`).
  halt_prob(x)                               a sigmoid over the halt row's pre-activation (the PonderNet form of the same tensor).
  anatomy()                                  every nonzero traced to its circuit name (from the kept `Write` lists).
  save(dir) / load(dir)                      `state.safetensors` (the tensor names h1_lid uses; the format written here without
                                             the library, which this venv lacks), `blueprint.json`, `layout.json`, `arch.json`,
                                             `plan.json` (the resolved placement), `<store>.npz` per store (the evidence).

Nothing here decides, values or looks: the brain is driven by the harness (`brainbuilder`, later `ziplearn`) and computes.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .layout import Layout
from .codec import Codec
from .blueprint import Blueprint, Arch
from .instructions import Write, Slot, Row, Pool, scatter, boundary_op, InstructionError

STRUCTURE_PREFIXES = ("blocks.", "prelude.", "coda.", "emb.", "head.", "pos", "alpha", "norm.", "res_final.", "mix_")


def _h1():
    try:
        import h1_lid  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers"))
        import h1_lid  # noqa: F401
    return sys.modules["h1_lid"]


# ── safetensors, the format (the library is not in this venv; the file is byte-compatible with it) ──────────────────────
_DTYPES = {torch.float32: "F32", torch.float64: "F64", torch.float16: "F16", torch.bfloat16: "BF16",
           torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
_NP = {"F32": np.float32, "F64": np.float64, "F16": np.float16, "I64": np.int64, "I32": np.int32, "I16": np.int16,
       "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def save_safetensors(tensors: dict[str, torch.Tensor], path) -> None:
    """The safetensors layout: 8-byte little-endian header length, a JSON header {name: {dtype, shape, data_offsets}},
    then the tensors' raw little-endian bytes in header order."""
    header, blobs, at = {}, [], 0
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            raw = t.view(torch.int16).numpy().tobytes()
        else:
            raw = t.numpy().tobytes()
        header[name] = {"dtype": _DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    header["__metadata__"] = {"format": "pt", "writer": "ziplib.brain"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for raw in blobs:
            f.write(raw)


def load_safetensors(path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        arr = np.frombuffer(data[a:b], dtype=_NP[info["dtype"]]).reshape(info["shape"]).copy()
        out[name] = torch.from_numpy(arr)
    return out


# ── the MLP headroom ────────────────────────────────────────────────────────────────────────────────────────────────────
class RowPool:
    """The free MLP hidden units of the core blocks: `write(key_row, threshold, value_row, layer=None) -> r` claims the
    next free unit of `layer` (default: the deepest core layer with room) and writes a `Row` at the brain's MLP sharpness
    (DESIGN §21.7's rows form; `ziplearn.sleep` -> `Store.consolidate` -> here). `free(layer)` = units left."""

    def __init__(self, brain, used: dict):
        self.brain = brain
        self.used = {int(k): int(v) for k, v in used.items()}            # layer -> hidden units claimed (compile + writes)

    def capacity(self, layer: int) -> int:
        return self.brain.model.blocks[layer].mlp[0].weight.shape[0]

    def free(self, layer: int) -> int:
        return self.capacity(layer) - self.used.get(layer, 0)

    def write(self, key_row, threshold, value_row, layer: int | None = None, M: float | None = None, name: str | None = None) -> int:
        layers = list(range(len(self.brain.model.blocks)))
        if layer is None:
            with_room = [L for L in reversed(layers) if self.free(L) > 0]
            if not with_room:
                raise InstructionError("rows.write: every core MLP is full")
            layer = with_room[0]
        if self.free(layer) <= 0:
            raise InstructionError(f"rows.write: layer {layer}'s MLP has no free row ({self.capacity(layer)} used)")
        r = self.used.get(layer, 0)
        key = torch.as_tensor(key_row, dtype=torch.float32).reshape(-1).tolist()
        value = torch.as_tensor(value_row, dtype=torch.float32).reshape(-1).tolist()
        writes = Row.emit(dict(key=key, threshold=float(threshold), value=value), self.brain.layout, self.brain.arch_view(),
                          Slot(layer=layer, row=r, name=name or f"row@{layer}:{r}"), M if M is not None else self.brain.M)
        scatter(writes, self.brain.tensors())
        self.used[layer] = r + 1
        self.brain.writes[name or f"rows/{layer}:{r}"] = list(writes)
        return r


# ── the brain ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class Brain(nn.Module):
    """The compiled artefact (module docstring). Built by `brainbuilder.compile`; `Brain.load(dir)` rebuilds one from
    what `save(dir)` wrote."""

    def __init__(self, model, layout: Layout, codec: Codec, blueprint: Blueprint, arch: Arch, plan: dict, M: float,
                 writes: dict | None = None, verified: bool = False):
        super().__init__()
        self.model = model
        self.layout, self.codec, self.blueprint, self.arch = layout, codec, blueprint, arch
        self.plan = plan                                                 # the resolved placement (json-able)
        self.M = float(M)                                                # nats: the gap one match buys
        self.verified = verified
        self.writes: dict[str, list[Write]] = dict(writes or {})
        d = layout.d
        self.banks = nn.ParameterDict()
        for s in blueprint.stores:
            self.banks[f"{s.name}__keys"] = nn.Parameter(torch.zeros(0, d), requires_grad=False)
            self.banks[f"{s.name}__values"] = nn.Parameter(torch.zeros(0, d), requires_grad=False)
        self.W = nn.ParameterDict()
        for c in blueprint.circuits:                                     # the Pool value matrices: STATE, zero until counted
            if c.instr == "Pool" and isinstance(c.params.get("W"), str):
                w_dst = layout.width(c.params["dst"])
                w_src = layout.width(Pool._src(c.params)[0][0])
                self.W[c.params["W"]] = nn.Parameter(torch.zeros(w_dst, w_src), requires_grad=False)
        regs = [codec.register(r.name, k) for r in blueprint.registers for k in range(int(r.slots))]
        self.registers = nn.Parameter(torch.stack(regs) if regs else torch.zeros(0, d), requires_grad=False)
        self.register_names = [(r.name, k) for r in blueprint.registers for k in range(int(r.slots))]
        self.stores: dict[str, list] = {s.name: [] for s in blueprint.stores}   # name -> [(Store, consts)]
        self.rows = RowPool(self, plan.get("rows_used", {}))
        self.noise_gain = 0.0                                            # the noise register's gain (typed; NOT DESIGNED how to price it)
        self.generator = torch.Generator().manual_seed(int(plan.get("seed", 0)))
        self.last = None                                                 # the last run's full final residual (1, T, d)
        self.last_layout = None                                          # (n_nulls, n_registers, n_memory, n_input) of the last run

    # -- views ----------------------------------------------------------------------------------------------------------
    @property
    def boundary(self):
        return self.model.boundary

    @property
    def d(self) -> int:
        return self.layout.d

    @property
    def pos(self) -> str:
        return self.arch.pos

    def arch_view(self):
        """The duck-typed arch the instructions read (pos, dims, max_len, p_star, head_dim)."""
        from types import SimpleNamespace
        return SimpleNamespace(pos=self.arch.pos, dims=dict(self.arch.dims), max_len=self.plan["max_len"], p_star=self.arch.p_star,
                               head_dim=self.plan.get("head_dim"))

    def tensors(self) -> dict[str, torch.Tensor]:
        """The model's parameters by their state_dict names (what `Write` addresses)."""
        return dict(self.model.named_parameters())

    # -- the null classes ------------------------------------------------------------------------------------------------
    @property
    def null_names(self) -> list[str]:
        return list(self.plan.get("nulls", []))

    def nulls(self) -> torch.Tensor:
        names = self.null_names
        return self.codec.nulls(names) if names else torch.zeros(0, self.d)

    # -- STATE: the banks (memory tokens) and W --------------------------------------------------------------------------
    def attach_stores(self, name: str, stores) -> None:
        """Give the brain the evidence of store `name`: a `Store`, or a list of `(Store, consts)` pairs (E28: one store per
        action with `consts = {"action": ai}`, the tokens concatenated in that order). `write_banks()` regenerates the bank."""
        if name not in self.stores:
            raise KeyError(f"no store {name!r} in the blueprint; it has {list(self.stores)}")
        if not isinstance(stores, (list, tuple)):
            stores = [(stores, {})]
        self.stores[name] = [(s, dict(c)) for s, c in stores]

    @torch.no_grad()
    def write_banks(self) -> dict[str, int]:
        """Regenerate every bank from its stores (`Store.as_bank`; keys + values = E28's memory tokens). Returns the
        entry counts. Capacity is the blueprint's `capacity` (an over-full store is an error naming it)."""
        counts = {}
        for spec in self.blueprint.stores:
            parts_k, parts_v = [], []
            for store, consts in self.stores.get(spec.name, []):
                k, v = store.as_bank(self.codec, value=spec.value, **consts)
                parts_k.append(k)
                parts_v.append(v)
            keys = torch.cat(parts_k, 0) if parts_k else torch.zeros(0, self.d)
            values = torch.cat(parts_v, 0) if parts_v else torch.zeros(0, self.d)
            if spec.capacity and keys.shape[0] > spec.capacity:
                raise ValueError(f"store {spec.name!r}: {keys.shape[0]} entries exceed its capacity {spec.capacity} "
                                 f"(the sequence length M was derived for)")
            self.banks[f"{spec.name}__keys"] = nn.Parameter(keys, requires_grad=False)
            self.banks[f"{spec.name}__values"] = nn.Parameter(values, requires_grad=False)
            counts[spec.name] = int(keys.shape[0])
        return counts

    def memory(self) -> torch.Tensor:
        """The memory tokens of every store, in blueprint order: keys + values (the tokens form of the bank)."""
        parts = [self.banks[f"{s.name}__keys"] + self.banks[f"{s.name}__values"] for s in self.blueprint.stores]
        return torch.cat(parts, 0) if parts else torch.zeros(0, self.d)

    @torch.no_grad()
    def write_W(self, name: str) -> int:
        """Scatter `W[name]` into the value rows of every Pool head that names it (`Pool.value_writes`); returns the count."""
        n = 0
        for c in self.blueprint.circuits:
            if c.instr == "Pool" and c.params.get("W") == name:
                slot = self.slot_of(c.name)
                sign = float(c.params.get("sign", 1) or 1)
                ws = Pool.value_writes(c.params, self.codec, slot, self.W[name].detach(), sign)
                # clear the head's value rows for the src dims first (a re-count replaces, never accumulates)
                A_v = slot.d if slot.d is not None else self.d
                hd = slot.hd if slot.hd is not None else self.d
                qkv = self.tensors()[f"blocks.{slot.layer}.attn.qkv.weight"]
                for sub, _s in Pool._src(c.params):
                    S = self.layout[sub]
                    qkv[2 * A_v + slot.head * hd: 2 * A_v + slot.head * hd + self.layout.width(c.params["dst"]), S] = 0.0
                scatter(ws, self.tensors())
                self.writes[f"{c.name}/W"] = list(ws)
                n += len(ws)
        return n

    def slot_of(self, circuit: str) -> Slot:
        """The placement the compiler resolved for `circuit` (from plan.json)."""
        p = self.plan["slots"][circuit]
        return Slot(layer=p["layer"], head=p["head"], hd=p.get("hd"), d=p.get("d"), row=p.get("row"),
                    registers=self.plan.get("registers"), n_zero=p.get("n_zero", 0), name=circuit)

    # -- the sequence ----------------------------------------------------------------------------------------------------
    def sequence(self, tokens: torch.Tensor, coords: torch.Tensor | None = None):
        """nulls + registers + memory + tokens as one (1, T, d) tensor, with the (T, 2) coordinates (non-cells at (0, 0))
        and the prefix lengths (n_nulls, n_registers, n_memory, n_tokens)."""
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
        if tokens.dim() == 3:
            tokens = tokens[0]
        nulls, regs, mem = self.nulls(), self.registers.detach(), self.memory().detach()
        x = torch.cat([nulls, regs, mem, tokens], 0)[None]
        n_pre = nulls.shape[0] + regs.shape[0] + mem.shape[0]
        if coords is None:
            if self.pos == "onehot":
                coords_all = self.codec.coords_for(x[0])
            else:
                raise ValueError("Brain.sequence: under rope/rope2d the token coordinates must be given (coords)")
        else:
            coords = torch.as_tensor(coords, dtype=torch.long).reshape(-1, 2)
            coords_all = torch.cat([torch.zeros(n_pre, 2, dtype=torch.long), coords], 0)
        return x, coords_all, (nulls.shape[0], regs.shape[0], mem.shape[0], tokens.shape[0])

    def _model_coords(self, coords_all):
        """What `Attn.forward(coords=)` gets: None under onehot (position is in the residual) and under 1-D rope (the token
        index is the position, E18); the (T, 2) list under rope2d."""
        return coords_all if self.pos == "rope2d" else None

    def _fill_noise(self, x, n_nulls, n_regs):
        """Refill the registers' `noise` field from the seeded generator (DESIGN §21.1 item 4, §21.8a: noise is an input)."""
        if self.noise_gain <= 0 or "noise" not in self.layout or n_regs == 0:
            return x
        s = self.layout["noise"]
        x = x.clone()
        x[0, n_nulls:n_nulls + n_regs, s] = self.noise_gain * torch.randn(n_regs, s.stop - s.start, generator=self.generator)
        return x

    def anchor_vectors(self, anchors):
        """`anchors` as (width,) one-hot tensors: ints index the anchor subspace, tensors pass through."""
        if anchors is None:
            return None
        a_slice = self.boundary.anchor if self.boundary is not None else None
        if a_slice is None:
            raise ValueError("Brain.run: the blueprint's boundary has no anchor to inject")
        width = a_slice[1] - a_slice[0]
        out = []
        for a in anchors:
            if isinstance(a, (int, np.integer)):
                v = torch.zeros(width)
                v[int(a)] = 1.0
            else:
                v = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
                if v.numel() != width:
                    raise ValueError(f"Brain.run: an anchor of {v.numel()} entries for a {width}-wide anchor subspace")
            out.append(v)
        return out

    # -- run: the depth axis (MOVES WrittenSim.rollout's loop) ---------------------------------------------------------------
    @torch.no_grad()
    def run(self, tokens, coords=None, anchors=None, max_passes: int = 8):
        """One rollout. `tokens` (T, d) from the codec (a frame's cells), `coords` (T, 2) (None under onehot: read off the
        tokens). `anchors`: the plan -- one per pass (ints = symbol in the anchor subspace, or vectors); the passes are
        exactly len(anchors) (E28: n actions = n passes) and register convergence does not stop them. `anchors=None`: the
        executive's own loop -- up to `max_passes`, halting on the halt flag or register convergence, the anchor read off
        the ACTION register by the operator (`anchor_token`); the noise field refilled before each pass when `noise_gain`
        > 0. Returns (the given tokens' final residuals (T, d), passes run); `self.last` keeps the whole sequence."""
        x, coords_all, (n0, n_reg, n_mem, n_tok) = self.sequence(tokens, coords)
        model = self.model
        mc = self._model_coords(coords_all)
        vecs = self.anchor_vectors(anchors)
        if vecs is not None:
            model.active_k = len(vecs)
            h = model.forward_embedded(x, None, 0, mc, anchors=vecs)
            passes = model.last_passes
        elif self.noise_gain > 0 and n_reg:
            model.active_k = 1                                          # one pass per call: the noise is refilled between passes
            h, passes, prev = x, 0, None
            for k in range(max_passes):
                h = self._fill_noise(h, n0, n_reg)
                h = model.forward_embedded(h, None, 0, mc, anchors=None, converge=False)
                passes += 1
                halted = self.boundary is not None and self.boundary.halt_flag is not None and bool(
                    (h[0, :, self.boundary.halt_flag] > 0.5).any())
                regs = h[0, n0:n0 + n_reg].clone()
                if halted or (prev is not None and bool((regs == prev).all())):
                    break
                prev = regs
        else:
            model.active_k = max(1, int(max_passes))
            h = model.forward_embedded(x, None, 0, mc, anchors=None)
            passes = model.last_passes
        self.last, self.last_layout = h, (n0, n_reg, n_mem, n_tok)
        return h[0, n0 + n_reg + n_mem:], passes

    def input_subspace(self) -> str:
        c = self.codec.cell_class
        inputs = c.with_source("input")
        if not inputs:
            raise ValueError(f"token class {c.name!r} has no 'input' field")
        return inputs[0]

    @torch.no_grad()
    def rollout(self, frame, plan, **consts) -> np.ndarray:
        """E28's interface: the frame after the plan (anchor symbols), decoded as the argmax of the input subspace."""
        frame = np.asarray(frame)
        x, coords = self.codec.encode_frame(frame, **consts)
        out, _passes = self.run(x, coords, anchors=list(plan))
        s = self.layout[self.input_subspace()]
        return out[:, s].argmax(-1).reshape(frame.shape).cpu().numpy().astype(np.int16)

    # -- a sequence brain as a language model -------------------------------------------------------------------------------
    def forward(self, tok, passes: int | None = None):
        """logits (B, T, n_vocab) for token ids: `head(norm(forward_embedded(embed(tok))))` with `passes` core passes
        (default: the plan's `passes`, 1 for a loop with no anchor, no halt and no registers -- E18)."""
        model = self.model
        model.active_k = int(passes if passes is not None else self.plan.get("passes", 1))
        h = model.embed(tok)
        T = tok.shape[1]
        coords = torch.stack([torch.zeros(T, dtype=torch.long), torch.arange(T)], 1) if self.pos == "rope2d" else None
        return model.head(model.norm(model.forward_embedded(h, None, 0, coords)))

    # -- think: the sequence axis -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def think(self, tokens, coords=None, n: int | None = None, max_thoughts: int = 8):
        """Append thought positions: the embedding of thought k is `BoundaryOp(final residual of the last position)` mapped
        through `loop.sequence.feedback` ([[from, to]] subspace copies; identity when the blueprint gives none), fed as a
        new position through the k/v caches (`forward_embedded(h, caches, start)`); stops after `n` thoughts, on the halt
        flag, or after `max_thoughts`. Returns (the whole sequence's final residuals (T + k, d), thoughts appended)."""
        x, coords_all, (n0, n_reg, n_mem, n_tok) = self.sequence(tokens, coords)
        model = self.model
        model.active_k = int(self.plan.get("passes", 1))
        caches = model.new_caches()
        h = model.forward_embedded(x, caches, 0, self._model_coords(coords_all))
        outs = [h[0]]
        feedback = ((self.blueprint.loop.sequence or {}).get("feedback") or []) if self.blueprint.loop.sequence else []
        T = h.shape[1]
        limit = int(n) if n is not None else int(max_thoughts)
        k = 0
        for k in range(1, limit + 1):
            last = h[:, -1:, :]
            if self.boundary is not None:
                last, halted = self.boundary(last, None, None)
            else:
                halted = torch.zeros(1, dtype=torch.bool)
            emb = torch.zeros_like(last)
            if feedback:
                for src, dst in feedback:
                    f, t = self.layout[src], self.layout[dst]
                    w = min(f.stop - f.start, t.stop - t.start)
                    emb[..., t.start:t.start + w] = last[..., f.start:f.start + w]
            else:
                emb = last
            c = torch.tensor([[0, T + k - 1]], dtype=torch.long)
            h = model.forward_embedded(emb, caches, T + k - 1, c if self.pos == "rope2d" else None)
            outs.append(h[0])
            if n is None and bool(halted.all()):
                break
        return torch.cat(outs, 0), k

    # -- read-outs -----------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def readout(self, name: str, x: torch.Tensor | None = None) -> torch.Tensor:
        """Logits through `head` from ONE subspace: "action" reads the ACTION register's `action` of the last run (or of
        `x`), "pred" reads `pred` of the given tokens (default: the last run's input tokens). Any other name is a subspace
        name read from `x` (or the last run's input tokens)."""
        if x is None:
            if self.last is None:
                raise ValueError("readout: run the brain first, or pass x")
            n0, n_reg, n_mem, _ = self.last_layout
            if name == "action":
                idx = [i for i, (r, _k) in enumerate(self.register_names) if r == "ACTION"]
                x = self.last[0, n0 + idx[0]:n0 + idx[0] + 1] if idx else self.last[0, n0 + n_reg + n_mem:]
            else:
                x = self.last[0, n0 + n_reg + n_mem:]
        sub = {"action": "action", "pred": "pred"}.get(name, name)
        m = torch.zeros(self.d)
        m[self.layout[sub]] = 1.0
        return self.model.head(torch.as_tensor(x) * m)

    @torch.no_grad()
    def halt_prob(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """sigmoid(the halt row's pre-activation) per token: the PonderNet-differentiable read of the same tensor the hard
        halt reads (DESIGN §21.8 obligation 3b). None when the blueprint has no halt row."""
        halt = self.plan.get("halt_row")
        if halt is None:
            return None
        if x is None:
            x = self.last[0] if self.last is not None else None
        if x is None:
            raise ValueError("halt_prob: run the brain first, or pass x")
        blk = self.model.blocks[halt["layer"]]
        w, b = blk.mlp[0].weight[halt["row"]], blk.mlp[0].bias[halt["row"]]
        return torch.sigmoid(torch.as_tensor(x) @ w + b)

    # -- DESIGN §21.8: the two parameter classes ---------------------------------------------------------------------------------
    def param_groups(self) -> dict[str, list]:
        """STRUCTURE = the model's written tensors (attention, MLP rows, routes, read-out, embedding, the BoundaryOp's
        buffers are not parameters); STATE = banks, W, registers."""
        structure = [p for _n, p in self.model.named_parameters()]
        state = list(self.banks.values()) + list(self.W.values()) + [self.registers]
        return {"structure": structure, "state": state}

    # -- anatomy -----------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def anatomy(self) -> list[dict]:
        """One row per circuit: instr, layer, head/row, the tensors it wrote and how many nonzeros; a final row per tensor
        with the nonzeros no circuit accounts for (STATE the learner wrote, or a bug). From the kept `Write` lists."""
        rows = []
        accounted = {}
        for name, ws in self.writes.items():
            by_t = {}
            for w in ws:
                if w.value != 0.0:
                    by_t.setdefault(w.tensor, set()).add(w.index)
            for t, idx in by_t.items():
                accounted.setdefault(t, set()).update(idx)
            slot = self.plan["slots"].get(name.split("/")[0], {})
            rows.append(dict(circuit=name, instr=self.plan.get("instr", {}).get(name.split("/")[0], "-"), layer=slot.get("layer"),
                             head=slot.get("head"), row=slot.get("row"), nonzeros={t: len(i) for t, i in by_t.items()},
                             total=sum(len(i) for i in by_t.values())))
        for t, tensor in self.tensors().items():
            nz = set(map(tuple, (tensor != 0).nonzero().tolist()))
            extra = nz - accounted.get(t, set())
            if extra:
                rows.append(dict(circuit="(untraced)", instr="-", layer=None, head=None, row=None, nonzeros={t: len(extra)}, total=len(extra)))
        return rows

    # -- save / load ----------------------------------------------------------------------------------------------------------------
    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach() for k, v in self.model.state_dict().items()}
        tensors.update({f"banks.{k}": v.detach() for k, v in self.banks.items()})
        tensors.update({f"W.{k}": v.detach() for k, v in self.W.items()})
        tensors["registers"] = self.registers.detach()
        save_safetensors(tensors, d / "state.safetensors")
        self.blueprint.save(d / "blueprint.json")
        (d / "layout.json").write_text(self.layout.to_json(), encoding="utf-8")
        self.arch.save(d / "arch.json")
        plan = dict(self.plan)
        plan["rows_used"] = dict(self.rows.used)
        plan["verified"] = bool(self.verified)
        plan["M"] = self.M
        plan["stores"] = {name: [dict(file=f"{name}.{i}.npz", consts=consts) for i, (_s, consts) in enumerate(pairs)]
                          for name, pairs in self.stores.items()}
        (d / "plan.json").write_text(json.dumps(plan, indent=1, default=_json_default), encoding="utf-8")
        for name, pairs in self.stores.items():
            for i, (store, _consts) in enumerate(pairs):
                store.save_npz(d / f"{name}.{i}.npz")
        return d

    @classmethod
    def load(cls, directory) -> "Brain":
        from .store import Store
        d = Path(directory)
        plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
        blueprint = Blueprint.load(d / "blueprint.json")
        layout = Layout.from_json((d / "layout.json").read_text(encoding="utf-8"))
        arch = Arch.load(d / "arch.json")
        codec = Codec(layout, blueprint.tokens, pos="rope2d" if arch.pos == "rope" else arch.pos)
        model = materialise(plan, layout, blueprint, codec)
        brain = cls(model, layout, codec, blueprint, arch, plan, plan["M"], verified=bool(plan.get("verified", False)))
        tensors = load_safetensors(d / "state.safetensors")
        model.load_state_dict({k: v for k, v in tensors.items() if not (k.startswith("banks.") or k.startswith("W.") or k == "registers")})
        for k, v in tensors.items():
            if k.startswith("banks."):
                brain.banks[k[len("banks."):]] = nn.Parameter(v, requires_grad=False)
            elif k.startswith("W."):
                brain.W[k[len("W."):]] = nn.Parameter(v, requires_grad=False)
        brain.registers = nn.Parameter(tensors["registers"], requires_grad=False)
        for name, entries in plan.get("stores", {}).items():
            brain.stores[name] = [(Store.load_npz(d / e["file"]), dict(e["consts"])) for e in entries]
        brain.writes = {k: [Write(w[0], tuple(w[1]), w[2]) for w in ws] for k, ws in plan.get("writes", {}).items()}
        return brain


def _json_default(o):
    if isinstance(o, Write):
        return [o.tensor, list(o.index), o.value]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not json-able: {type(o).__name__}")


def materialise(plan: dict, layout: Layout, blueprint: Blueprint, codec: Codec):
    """The model skeleton from a resolved plan (`brainbuilder.compile` step 7; `Brain.load`): `LoopedModel(d, n_head per
    core layer, max_len, pos, n_vocab, core_layers, boundary, norm="none", n_zero per layer)`, every parameter zeroed,
    `Attn.causal` as the plan says."""
    H1 = _h1()
    b = blueprint.loop.boundary
    bsp = plan["boundary"]
    op = boundary_op(layout, keep=b.keep or None, clear=b.clear or None, quantise=b.quantise, commit=[tuple(c) for c in b.commit],
                     anchor=b.anchor, halt=b.halt, register_dims=bsp.get("register_dims", []), cls_flags=bsp.get("cls_flags"),
                     cell_flag=bsp.get("cell_flag"), register_flag=bsp.get("register_flag"), anchor_token=bsp.get("anchor_token"),
                     tie_tol=bsp.get("tie_tol", 1e-3)) if b is not None else None
    model = H1.LoopedModel(d_model=plan["d"], n_head=list(plan["n_head"]), max_len=plan["max_len"], pos=plan["model_pos"],
                           n_vocab=plan["n_vocab"], loops=max(1, int(plan.get("max_passes", 64))), tied=True, n_prelude=0, n_coda=0,
                           core_layers=len(plan["n_head"]), boundary=op, norm=plan.get("norm", "none"),
                           n_zero=list(plan.get("n_zero", [0] * len(plan["n_head"]))))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if isinstance(model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model, nn.LayerNorm):
                p.fill_(1.0 if name.endswith("weight") else 0.0)                 # a LayerNorm at its default (e18.write line 49-52)
            else:
                p.zero_()
    for blk in model.blocks:
        blk.attn.causal = bool(plan["causal"])
    model.eval()
    return model


__all__ = ["Brain", "RowPool", "materialise", "save_safetensors", "load_safetensors", "STRUCTURE_PREFIXES"]
