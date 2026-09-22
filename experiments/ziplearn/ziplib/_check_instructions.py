"""Check that ziplib.instructions reproduces the written tensors of e28.WrittenSim and e18.write EXACTLY (to 1e-12).

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/ziplib/_check_instructions.py       (CPU, seconds)

E28 (the layout of `_check_codec.e28_layout_and_codec`, a 7 x 9 frame, synthetic E24-shaped rules with four offsets): the four
`Gather` heads emitted into zero (3d x d) / (d x d) tensors must equal `WrittenSim.gather.qkv.weight` / `proj.weight`, and the
`Match` head (`lookup`: q = k = [colour, nbr[*], action], v = out -> colour, key_class entry, null border, weight(action) auto,
replace) must equal `WrittenSim.lookup.qkv.weight` / `proj.weight`. E18 (`Model(d 64, 2 layers, 2 heads, rope, max_len 128)`
after `e18.write`, the layout tok 0..8 / prev 10..18 / out 20..27 / bias 63 with padding subspaces): `Gather(offset (0, -1))`
at M = 3 must equal `blocks.0.attn`, `Match(q tok, k prev, v tok -> out, add)` at M = 2 must equal `blocks.1.attn`, `Readout(out,
V, 10)` must equal `head.weight`; `Gather.needs` must ask for E18's four position pairs. Then the instructions' own unit tests
run on the written modules (a duck-typed brain), `Row` / `Compare` / `Pool` / `Broadcast` are exercised on a fresh Block, and
`boundary_op` is built from E28's boundary and applied.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                          # experiments/ziplearn: e28, e18, arcgames
sys.path.insert(0, str(HERE.parent.parent / "transformers"))                   # h1_lid
import h1_lid as H1                                                            # noqa: E402
import e28 as E28                                                              # noqa: E402
import e18 as E18                                                              # noqa: E402
from _check_codec import synthetic_rules, e28_layout_and_codec                 # noqa: E402
from ziplib.layout import Layout, Subspace                                     # noqa: E402
from ziplib.codec import Codec, TokenClass                                     # noqa: E402
from ziplib.instructions import (Gather, Match, Pool, Broadcast, Row, Compare, Readout, Slot, scatter,   # noqa: E402
                                 boundary_op, INSTRUCTIONS, InstructionError, derived_M, GATHER_GAP_MIN)

TOL = 1e-12
ok = True


def check(name, err, tol=TOL):
    global ok
    good = err <= tol
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {name}: max |new - old| = {err:.3e}")


def max_err(a, b):
    a, b = torch.as_tensor(a).detach().to(torch.float64), torch.as_tensor(b).detach().to(torch.float64)
    return float((a - b).abs().max())


def zeros_like_attn(d):
    return {"qkv": torch.zeros(3 * d, d), "proj": torch.zeros(d, d)}           # float32, the model's dtype (E18 rounds M cos theta to it)


def rename(writes, layer):
    """The Write tensor names of layer `layer` -> the short keys of `zeros_like_attn`."""
    return {f"blocks.{layer}.attn.qkv.weight": "qkv", f"blocks.{layer}.attn.proj.weight": "proj"}


def scatter_attn(writes, layer, d):
    t = zeros_like_attn(d)
    names = rename(writes, layer)
    from ziplib.instructions import Write
    counts = scatter([Write(names[w.tensor], w.index, w.value) for w in writes], t)
    return t, counts


# ---------------------------------------------------------------------------------------------------------------------
# 1. E28: the four Gather heads and the Match head
# ---------------------------------------------------------------------------------------------------------------------
def check_e28(seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rules, actions = synthetic_rules(rng)
    H, W = 7, 9
    sim = E28.WrittenSim(rules, actions, H, W)
    layout, codec, nbr = e28_layout_and_codec(sim)
    d, hd, M = sim.d, sim.hd, sim.M
    sim.offs = [(int(a), int(b)) for a, b in sim.offs]
    print(f"E28: WrittenSim d = {d}, hd = {hd}, {sim.n_head} gather heads over {sim.offs}, M = {M}; layout {layout}")
    arch = dict(pos="onehot")
    # the gathers
    all_w = []
    for j, (di, dj) in enumerate(sim.offs):
        params = dict(offset=[di, dj], src="colour", dst=nbr[j], null="border", cls="cell")
        need = Gather.needs(params, codec, arch)
        assert need.heads == 1 and need.hd_min == max(H + W + 2, sim.C), need
        all_w += Gather.emit(params, codec, arch, Slot(layer=0, head=j, hd=hd, d=d, name=f"gather{(di, dj)}"), M)
    t, counts = scatter_attn(all_w, 0, d)
    check(f"Gather x {len(sim.offs)}: qkv.weight = WrittenSim.gather.qkv.weight ({counts['qkv']} nonzeros)", max_err(t["qkv"], sim.gather.qkv.weight))
    check(f"Gather x {len(sim.offs)}: proj.weight = WrittenSim.gather.proj.weight ({counts['proj']} nonzeros)", max_err(t["proj"], sim.gather.proj.weight))
    check("Gather: nonzero count equals WrittenSim's", 0.0 if counts["qkv"] + counts["proj"] == int((sim.gather.qkv.weight != 0).sum() + (sim.gather.proj.weight != 0).sum()) else 1.0)
    # the lookup
    lookup = dict(q=["colour"] + nbr + ["action"], k=["colour"] + nbr + ["action"], v="out", dst="colour", key_class="entry",
                  null="border", weights={"action": "auto"}, mode="replace")
    need = Match.needs(lookup, codec, arch)
    assert need.whole_window and need.hd_min == d, need
    ws = Match.weights(lookup, layout)
    check(f"Match weights: auto(action) = {ws[-1]:g} = E28's g = nO + 2 = {len(sim.offs) + 2}", abs(ws[-1] - (len(sim.offs) + 2)))
    lw = Match.emit(lookup, codec, arch, Slot(layer=1, head=0, hd=d, d=d, name="lookup"), M)
    t, counts = scatter_attn(lw, 1, d)
    check(f"Match: qkv.weight = WrittenSim.lookup.qkv.weight ({counts['qkv']} nonzeros)", max_err(t["qkv"], sim.lookup.qkv.weight))
    check(f"Match: proj.weight = WrittenSim.lookup.proj.weight ({counts['proj']} nonzeros)", max_err(t["proj"], sim.lookup.proj.weight))
    check("Match: nonzero count equals WrittenSim's", 0.0 if counts["qkv"] + counts["proj"] == int((sim.lookup.qkv.weight != 0).sum() + (sim.lookup.proj.weight != 0).sum()) else 1.0)
    # the same Match with null = zero (gridworld.json's choice): same function, one sink address moved
    lw2 = Match.emit({**lookup, "null": "zero"}, codec, arch, Slot(layer=1, head=0, hd=d, d=d), M)
    t2, _ = scatter_attn(lw2, 1, d)
    diff = int((t2["qkv"] != t["qkv"]).sum())
    print(f"       (null = zero instead of border moves {diff} qkv entries: the sink channel/key; rollouts are unaffected, both sinks have value 0)")
    # the unit tests, on the written modules
    brain = SimpleNamespace(model=SimpleNamespace(blocks=[SimpleNamespace(attn=sim.gather), SimpleNamespace(attn=sim.lookup)]),
                            codec=codec, M=M, arch=arch)
    for j, (di, dj) in enumerate(sim.offs):
        params = dict(offset=[di, dj], src="colour", dst=nbr[j], null="border", cls="cell")
        r = Gather.test(params, codec, brain, n=4 * H * W, slot=Slot(layer=0, head=j, hd=hd, d=d))
        check(f"Gather.test {(di, dj)}: argmax {r.argmax_acc:.3f}, min gap {r.min_gap:.1f} >= M {M:g}, dst err {r.subspace_err:.1e}", 0.0 if r.passed else 1.0)
    r = Match.test(lookup, codec, brain, n=128, slot=Slot(layer=1, head=0, hd=d, d=d))
    check(f"Match.test lookup: argmax {r.argmax_acc:.3f}, min gap {r.min_gap:.1f} >= M {M:g}, dst err {r.subspace_err:.1e}, keys on null {r.extra.get('keys_on_null_acc')}", 0.0 if r.passed else 1.0)
    # the boundary: E28's clear + re-quantise as a BoundaryOp on one rollout step
    op = boundary_op(layout, clear=nbr, quantise=[dict(read=["colour"], write="colour")], anchor=dict(sub="action", **{"from": "runner"}),
                     cls_flags=[layout.flag("cell")])
    frame = rng.integers(0, E28.V, size=(H, W)).astype(np.int16)
    x = torch.cat([sim.memory, sim.encode(frame)], 0)[None]
    n = sim.n_mem
    a = actions[0]
    x_ref = x.clone()
    x_ref[0, n:, sim.A] = 0.0
    x_ref[0, n:, sim.A.start + actions.index(a)] = 1.0
    x_ref = x_ref + sim.gather(x_ref)
    x_ref = x_ref + sim.lookup(x_ref)
    x_new = x_ref.clone()
    tcol = x_ref[0, n:, sim.T.start:sim.T.start + E28.V].clone()
    ties_ref = int(((tcol.max(1).values - 1).abs() > 1e-3).sum())
    x_ref[0, n:, sim.T] = 0.0
    x_ref[0, n:, sim.T.start:sim.T.start + E28.V] = torch.nn.functional.one_hot(tcol.argmax(1), E28.V).float()
    for sl in sim.N:
        x_ref[0, n:, sl] = 0.0
    anchor = torch.zeros(len(actions))
    anchor[actions.index(a)] = 1.0
    y, halted = op(x_new, anchor_vec=anchor)
    x_ref[0, n:, sim.A] = anchor
    check(f"boundary_op(clear nbr[*], quantise colour, anchor action) = E28 rollout lines 178-189 (ties {op.ties} vs {ties_ref})",
          max_err(y, x_ref) + abs(op.ties - ties_ref))
    return layout, codec, sim


# ---------------------------------------------------------------------------------------------------------------------
# 2. E18: the previous-token head, the induction head, the read-out
# ---------------------------------------------------------------------------------------------------------------------
def e18_layout_and_codec():
    """E18's residual layout (e18.py line 38): tok 0..8, prev 10..18, out 20..27, bias 63 -- padding subspaces fill the gaps."""
    subs = [Subspace("tok", "C"), Subspace("_pad1", 1, "scalar"), Subspace("prev", "C"), Subspace("_pad2", 1, "scalar"),
            Subspace("out", "V", "dist"), Subspace("_pad3", E18.BIAS - 20 - E18.V, "scalar"), Subspace("bias", 1, "flag")]
    layout = Layout.allocate(subs, d=E18.D, dims=dict(C=E18.NT, V=E18.V))
    assert layout["tok"].start == E18.T_SUB and layout["prev"].start == E18.P_SUB and layout["out"].start == E18.O_SUB and layout.flag("bias") == E18.BIAS
    codec = Codec(layout, [TokenClass("tok", "bias", {"tok": "input", "row": "coord.row", "col": "coord.col", "bias": "const:1"})], pos="rope2d")
    return layout, codec


def check_e18():
    torch.manual_seed(0)
    model = H1.Model(d_model=E18.D, n_layer=E18.LAYERS, n_head=E18.HEADS, max_len=128, pos="rope", n_vocab=E18.NT)
    E18.write(model)
    layout, codec = e18_layout_and_codec()
    arch = dict(pos="rope", max_len=128, dims=dict(C=E18.NT, V=E18.V))          # a sequence: the profile spans max_len - 1 = 127
    print(f"E18: Model d = {E18.D}, hd = {E18.HD}, rope, max_len 128; layout {layout}")
    d, hd = E18.D, E18.HD
    gather = dict(offset=[0, -1], src="tok", dst="prev", null=None, cls="tok", M=3.0)
    need = Gather.needs(gather, codec, arch)
    check(f"Gather.needs (rope, M = 3, extent 127): position_pairs = {need.position_pairs} = E18's four highest-frequency pairs (profile M*gap: "
          f"{[round(3 * Gather.profile(gather, arch, (0, n), hd, 'rope'), 3) for n in range(1, 6)]} for 1..5 pairs, threshold {GATHER_GAP_MIN})",
          0.0 if need.position_pairs == 4 else 1.0)
    gw = Gather.emit(gather, codec, arch, Slot(layer=0, head=0, hd=hd, d=d, name="prev_token"), M=None)
    t, counts = scatter_attn(gw, 0, d)
    check(f"E18 Gather: blocks.0.attn.qkv.weight ({counts['qkv']} nonzeros)", max_err(t["qkv"], model.blocks[0].attn.qkv.weight))
    check(f"E18 Gather: blocks.0.attn.proj.weight ({counts['proj']} nonzeros)", max_err(t["proj"], model.blocks[0].attn.proj.weight))
    induction = dict(q=["tok"], k=["prev"], v="tok", dst="out", key_class=None, null=None, weights={}, mode="add", M=2.0)
    need = Match.needs(induction, codec, arch)
    assert not need.whole_window and need.content_dims == E18.NT and need.hd_min == 2 * E18.NT, need
    drift = Match.drift(induction, codec, arch, Slot(layer=1, head=0, hd=hd, d=d))
    mw = Match.emit(induction, codec, arch, Slot(layer=1, head=0, hd=hd, d=d, name="induction"), M=None)
    t, counts = scatter_attn(mw, 1, d)
    check(f"E18 Match: blocks.1.attn.qkv.weight ({counts['qkv']} nonzeros; content drift max_len*theta = {drift:.2f}, E18 scores 1.000 at it)",
          max_err(t["qkv"], model.blocks[1].attn.qkv.weight))
    check(f"E18 Match: blocks.1.attn.proj.weight ({counts['proj']} nonzeros)", max_err(t["proj"], model.blocks[1].attn.proj.weight))
    rw = Readout.emit(dict(sub="out", vocab="V", M_out=10.0), codec, arch, Slot(), M=None)
    head = torch.zeros_like(model.head.weight)
    from ziplib.instructions import Write
    scatter([Write("head", w.index, w.value) for w in rw], {"head": head})
    check(f"E18 Readout: head.weight ({len(rw)} nonzeros)", max_err(head, model.head.weight))
    # the unit tests on the written model (1-D rope: one row of 17 positions = E18's sequence length)
    brain = SimpleNamespace(model=model, codec=codec, M=3.0, arch=arch)
    r = Gather.test(gather, codec, brain, n=17 * 8, slot=Slot(layer=0, head=0, hd=hd, d=d), H=1, W=17)
    check(f"E18 Gather.test: argmax {r.argmax_acc:.3f}, min gap {r.min_gap:.2f} (M 3), dst err {r.subspace_err:.1e}", 0.0 if r.argmax_acc == 1.0 else 1.0)
    print(f"       (E18's gap {r.min_gap:.2f} < M = 3 and dst err {r.subspace_err:.2f} > 1e-3: E18 is written at softmax mass ~0.7 on the target, "
          f"not at p* = 0.99 -- the argmax is exact, the copy is not; e18 scores 1.000 because the read-out's argmax survives)")
    r = Match.test(induction, codec, brain, n=8, slot=Slot(layer=1, head=0, hd=hd, d=d))
    print(f"       E18 Match.test: argmax {r.argmax_acc:.3f}, min gap {r.min_gap:.2f}, dst err {r.subspace_err:.2f} (M_ind = 2: the same softness)")
    check("E18 Match.test: argmax on the induction target", 0.0 if r.argmax_acc == 1.0 else 1.0)
    r = Readout.test(dict(sub="out", vocab="V", M_out=10.0), codec, brain, n=64)
    check(f"E18 Readout.test: argmax {r.argmax_acc:.3f}, gap {r.min_gap:.1f} >= M_out 10", 0.0 if r.passed else 1.0)
    acc, _ = E18.accuracy(model)
    check(f"E18 accuracy after write: {acc:.3f} (must stay 1.000)", abs(acc - 1.0))


# ---------------------------------------------------------------------------------------------------------------------
# 3. the NEW instructions on a gridworld-shaped layout: Row, Compare, Pool, Broadcast
# ---------------------------------------------------------------------------------------------------------------------
def gridworld_layout(r=1, with_slot=True):
    bp = json.load(open(HERE.parent / "blueprints" / "gridworld.json"))
    offs = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1) if (di, dj) != (0, 0)]
    subs = []
    for s in bp["subspaces"]:
        if s["name"] == "nbr[o]":
            subs += [Subspace(f"nbr[{di},{dj}]", s["width"], s["kind"]) for di, dj in offs]
        else:
            subs.append(Subspace(s["name"], s["width"], s["kind"]))
    if with_slot:
        subs.append(Subspace("slot", 4))
    dims = dict(H=8, W=11, C=17, V=16, nA=4)
    layout = Layout.allocate(subs, dims=dims)
    nbr = [f"nbr[{di},{dj}]" for di, dj in offs]
    classes = []
    for t in bp["tokens"]:
        fields = {}
        for k, v in t["fields"].items():
            if k == "nbr[*]":
                fields.update({n: v for n in nbr})
            else:
                fields[k] = v
        classes.append(TokenClass(t["class"], t["flag"], fields))
    return layout, Codec(layout, classes, pos="onehot"), nbr, dims


def check_new():
    layout, codec, nbr, dims = gridworld_layout()
    d = layout.d
    arch = dict(pos="onehot", max_len=350, p_star=0.99, dims=dims)
    M = derived_M(arch)
    print(f"gridworld (r = 1, + slot(4)): d_layout = {layout.d_layout}, M = {M:.3f}")
    check("derived M = ln(349 * 0.99 / 0.01) = 10.45 (DESIGN §21.2.2)", abs(M - math.log(349 * 99)) + abs(round(M, 2) - 10.45))
    blk = H1.Block(d, 1, "learned", norm="layer" if False else "none")
    blk.attn.causal = False
    with torch.no_grad():
        for p in blk.parameters():
            p.zero_()
    # -- Row (halt_row) and Compare (surprise) into the MLP
    rw = Row.emit(dict(key={"goal_met": 1}, threshold=0.5, value={"halt": 1}), codec, arch, Slot(layer=0, row=0), M)
    cw = Compare.emit(dict(a="pred", b="colour", flag="surprise"), codec, arch, Slot(layer=0, row=1), M)
    sd = {f"blocks.0.{k}": v for k, v in blk.state_dict().items()}
    scatter(rw + cw, sd)
    brain = SimpleNamespace(model=SimpleNamespace(blocks=[blk]), codec=codec, M=M, arch=arch)
    r = Row.test(dict(key={"goal_met": 1}, threshold=0.5, value={"halt": 1}), codec, brain, slot=Slot(layer=0, row=0))
    check(f"Row.test halt_row: halt = goal_met to {r.subspace_err:.1e} (GELU at M/2 = {M / 2:.2f})", 0.0 if r.passed else 1.0)
    r = Compare.test(dict(a="pred", b="colour", flag="surprise"), codec, brain, slot=Slot(layer=0, row=1))
    check(f"Compare.test surprise: flag = [pred != colour] to {r.subspace_err:.1e} ({Compare.needs(dict(a='pred', b='colour', flag='surprise'), codec, arch).rows} rows)", 0.0 if r.passed else 1.0)
    # -- Broadcast(ACTION.action -> action) and Pool(ACTION <- W (goal - colour)) as heads of a one-head layer
    regs = {"ACTION": 0, "GOAL": 1, "TASK": 2, "STEP": 3}
    att = H1.Attn(d, 1, "learned")
    att.causal = False
    with torch.no_grad():
        for p in att.parameters():
            p.zero_()
    bw = Broadcast.emit(dict(reg="ACTION", sub="action", dst="action", cls="cell"), codec, arch, Slot(layer=0, head=0, hd=d, d=d, registers=regs), M)
    sd = {"blocks.0.attn.qkv.weight": att.qkv.weight, "blocks.0.attn.proj.weight": att.proj.weight}
    scatter(bw, sd)
    frame = np.random.default_rng(0).integers(0, 16, size=(dims["H"], dims["W"]))
    cells, coords = codec.encode_frame(frame)
    registers = torch.stack([codec.register(n, k) for n, k in regs.items()])
    registers[0, layout["action"].start + 2] = 1.0                           # ACTION holds action 2
    registers[1, layout["goal"].start + 5] = 1.0                             # GOAL holds colour 5
    x = torch.cat([codec.nulls(), registers, cells], 0)[None]
    with torch.no_grad():
        y = x + att(x)
    nc = 2 + 4
    want = torch.zeros(dims["H"] * dims["W"], 4)
    want[:, 2] = 1.0
    err = max_err(y[0, nc:, layout["action"]], want)
    others = float(y[0, :nc, layout["action"]].sub(x[0, :nc, layout["action"]]).abs().max())
    check(f"Broadcast ACTION.action -> every cell's action (err {err:.1e}; leak into nulls/registers {others:.1e}; p* = 0.99 leak bound {(1 - 0.99):.2f})", err, tol=0.02)
    # Pool: W (4 x 17) random, src = goal - colour, into ACTION.util
    att2 = H1.Attn(d, 1, "learned")
    att2.causal = False
    with torch.no_grad():
        for p in att2.parameters():
            p.zero_()
    pool = dict(reg="ACTION", key_class="cell", src="goal - colour", W="inverse", dst="util", gate=None, sign=1)
    slot = Slot(layer=0, head=0, hd=d, d=d, registers=regs)
    Wm = torch.randn(4, 17, generator=torch.Generator().manual_seed(1))
    pw = Pool.emit(pool, codec, arch, slot, M) + Pool.value_writes(pool, codec, slot, Wm)
    scatter(pw, {"blocks.0.attn.qkv.weight": att2.qkv.weight, "blocks.0.attn.proj.weight": att2.proj.weight})
    x2 = y.clone()
    x2[0, nc:, layout["goal"].start + 5] = 1.0                                # as if bcast_goal ran: every cell holds goal 5
    with torch.no_grad():
        y2 = x2 + att2(x2)
    src = x2[0, nc:, layout["goal"]] - x2[0, nc:, layout["colour"]]
    want = (src @ Wm.T).mean(0)
    err = max_err(y2[0, 2, layout["util"]], want)
    leak = float((y2[0, [0, 1, 3, 4, 5], :] - x2[0, [0, 1, 3, 4, 5], :]).abs().max())
    check(f"Pool ACTION.util = mean_cells W (goal - colour) (err {err:.1e}); other registers/nulls untouched (leak {leak:.1e})", err + leak, tol=0.05)
    print(f"       (the 0.99 softmax mass on the cells leaks {1 - 0.99:.0%} of the value onto the sink: the boundary's quantise absorbs it)")
    # gated Pool: TASK reads cells whose surprise flag is set; none set -> lands on zero (a no-op)
    att3 = H1.Attn(d, 1, "learned")
    att3.causal = False
    with torch.no_grad():
        for p in att3.parameters():
            p.zero_()
    gated = dict(reg="TASK", key_class="cell", src="task", W=None, dst="task", gate="surprise", sign=1)
    scatter(Pool.emit(gated, codec, arch, slot, M), {"blocks.0.attn.qkv.weight": att3.qkv.weight, "blocks.0.attn.proj.weight": att3.proj.weight})
    with torch.no_grad():
        y3 = x2 + att3(x2)
    noop = float((y3 - x2).abs().max())
    x4 = x2.clone()
    x4[0, nc + 3, layout.flag("surprise")] = 1.0
    x4[0, nc + 3, layout["task"].start + 6] = 1.0
    with torch.no_grad():
        y4 = x4 + att3(x4)
    fired = float(y4[0, 4, layout["task"].start + 6])
    check(f"gated Pool: no surprise -> no-op (|dy| {noop:.1e}); one surprised cell -> TASK.task reads it ({fired:.3f})", noop + abs(fired - 1.0), tol=0.02)
    # the gridworld boundary: keep/clear partition the layout (nbr[*] expands), two quantisations, the commit, the halt
    bp = json.load(open(HERE.parent / "blueprints" / "gridworld.json"))["loop"]["boundary"]
    op = boundary_op(layout, keep=bp["keep"] + ["slot"], clear=bp["clear"], quantise=bp["quantise"], commit=bp["commit"], anchor=bp["anchor"],
                     halt=bp["halt"], register_dims=list(range(*layout["action"].indices(d))), cls_flags=[layout.flag("cell"), layout.flag("register")],
                     cell_flag=layout.flag("cell"))
    xb = x2.clone()
    xb[0, nc:, layout["util"]] = torch.randn(1, generator=torch.Generator().manual_seed(2)).abs()
    xb[0, 2, layout["util"].start + 1] = 5.0                                  # ACTION's util favours action 1
    yb, halted = op(xb, anchor_vec=torch.tensor([0.0, 1.0, 0.0, 0.0]))
    good = float(yb[0, nc:, layout["util"]].abs().max()) == 0.0 and float(yb[0, nc:, layout[nbr[0]]].abs().max()) == 0.0    # cleared
    good &= bool((yb[0, 2, layout["action"]] == torch.tensor([0.0, 1.0, 0.0, 0.0])).all())                          # ACTION quantised util+noise -> action
    good &= bool((yb[0, nc:, layout["action"]] == torch.tensor([0.0, 1.0, 0.0, 0.0])).all())                        # the anchor into every cell
    good &= bool((yb[0, nc:, layout["colour"]][:, :16] == yb[0, nc:, layout["pred"]]).all())                        # commit pred -> colour
    good &= not bool(halted.any())
    check(f"gridworld boundary_op: keep {len(bp['keep']) + 1} / clear {bp['clear']} partition, quantise pred and util+noise -> action, commit, anchor, halt {bp['halt']}", 0.0 if good else 1.0)
    try:
        boundary_op(layout, keep=bp["keep"], clear=bp["clear"])
        check("boundary_op: a subspace in neither keep nor clear is refused", 1.0)
    except InstructionError as e:
        check(f"boundary_op: a subspace in neither keep nor clear is refused ({str(e)[:60]}...)", 0.0)
    from ziplib.instructions import Branch
    bw = Branch.emit(dict(mixer="attn", table={"surprise": 1, "cell": 0}), codec, arch, Slot(layer=2), M)
    check(f"Branch: {[(w.tensor, w.index, round(w.value, 2)) for w in bw]}", 0.0 if all(w.tensor == "blocks.2.res_attn.w" and w.value == M for w in bw) and len(bw) == 2 else 1.0)
    # the whitelist and the params guard
    check(f"INSTRUCTIONS = {sorted(INSTRUCTIONS)}", 0.0 if sorted(INSTRUCTIONS) == ["Branch", "Broadcast", "Compare", "Gather", "Match", "Pool", "Readout", "Row"] else 1.0)
    try:
        Gather.emit(dict(offset=[0, 1], src="colour", dst=nbr[0], null="border", cls="cell", colour=3), codec, arch, Slot(hd=d, d=d), M)
        check("unknown/content param refused", 1.0)
    except InstructionError as e:
        check(f"unknown/content param refused ({e})", 0.0)
    try:
        Gather.pairs_needed(dict(offset=[0, 1], src="colour", dst=nbr[0]), dict(pos="rope2d", max_len=350, dims=dims), 32, M, "rope2d")
        print("       rope2d column offset: pairs found")
    except InstructionError as e:
        print(f"       NOTE rope2d column offset under the wave-1 theta split: {e}")


def main():
    check_e28()
    check_e18()
    check_new()
    print("PASS: ziplib.instructions reproduces WrittenSim.gather/lookup and e18.write's tensors to 1e-12" if ok else "FAIL: see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
