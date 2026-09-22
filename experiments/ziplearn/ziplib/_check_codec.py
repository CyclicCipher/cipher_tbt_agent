"""Check that ziplib.Layout + ziplib.Codec reproduce e28.WrittenSim's layout and tokens EXACTLY (to 1e-12).

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/ziplib/_check_codec.py            (CPU, seconds)
    ... _check_codec.py --play      the rules from a played LockPath (levels 0-1, sleep), as e28.main does (~1 min)

Default: the rules are reconstructed as `LocalRule`s with E24-shaped masks (centre + 2-3 neighbours, two actions with
different masks, random tables with BORDER = -1 among the colours), and `WrittenSim.__init__` builds its layout from
them. Compared, on a 7 x 9 frame: every slice of the layout (`T`, `N[j]`, `A`, `E/F/S/Z`, `ROW`, `COL`, `OUT`, `d`);
`encode_frame` vs `WrittenSim.encode`; `nulls()` vs `memory[:2]`; `entry()` per table entry vs `memory[2:]`;
`coords_for`; `decode` round-trips; the `LayoutError` on d too small; `channels` against e18.py's pair choice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # experiments/ziplearn: e28, arcgames
import e28 as E28                                                     # noqa: E402
from arcgames import LocalRule, BORDER, V                             # noqa: E402
from ziplib.layout import Layout, Subspace, LayoutError, channels     # noqa: E402
from ziplib.codec import Codec, TokenClass, coords_for                # noqa: E402

TOL = 1e-12


def synthetic_rules(rng):
    """Two actions, radius-1 rules with different E24-shaped masks and random tables (colours 0..V-1 and BORDER)."""
    def rule(cells, n_entries):
        r = LocalRule(1)
        side = 3
        r.mask[:] = False
        for (di, dj) in cells:
            r.mask[(di + 1) * side + (dj + 1)] = True
        keys = set()
        for _ in range(n_entries):
            cols = rng.integers(-1, V, size=int(r.mask.sum())).astype(np.int16)      # -1 = BORDER
            k = cols.tobytes()
            if k in keys:
                continue
            keys.add(k)
            ys = rng.integers(0, V, size=int(rng.integers(1, 3)))
            r.table[k] = {int(y): int(rng.integers(1, 9)) for y in ys}
        return r
    actions = ["up", "left"]
    rules = {"up": rule([(0, 0), (-1, 0), (0, 1)], 40), "left": rule([(0, 0), (1, 0), (0, -1), (0, 1)], 60)}
    return rules, actions


def played_rules(seed):
    """As e28.main: play LockPath levels 0-1 with the sleep pass; the best rule per action."""
    from arcgames import play
    from tasks.games import LockPath
    from tasks.harness import Environment
    env = Environment(LockPath())
    _results, _stats, player = play(env, budget_per_level=150, max_levels=2, seed=seed, sleep=True, return_player=True)
    actions = [a for a in player.actions if player.models[a].n_obs]
    return {a: player.models[a].best() for a in actions}, actions


def e28_layout_and_codec(sim):
    """The E28 residual layout in ziplib terms, from the sim's offsets and sizes (the same declaration order as
    WrittenSim.__init__ lines 75-84), and the four E28 token classes."""
    H, W = sim.H, sim.W
    dims = dict(C=sim.C, nA=len(sim.actions), H=H, W=W, V=V)
    subs = [Subspace("colour", "C")]
    nbr = [f"nbr[{di},{dj}]" for di, dj in sim.offs]
    subs += [Subspace(n, "C") for n in nbr]
    subs += [Subspace("action", "nA"),
             Subspace("cell", 1, "flag"), Subspace("entry", 1, "flag"), Subspace("border", 1, "flag"), Subspace("zero", 1, "flag"),
             Subspace("row", "H"), Subspace("col", "W"), Subspace("out", "V")]
    layout = Layout.allocate(subs, d=sim.d, dims=dims)
    classes = [
        TokenClass("cell", "cell", {"colour": "input", "row": "coord.row", "col": "coord.col"}),
        TokenClass("entry", "entry", {"colour": "store", **{n: "store" for n in nbr}, "action": "store", "out": "store"}),
        TokenClass("border", "border", {"colour": "const:-1"}),          # colour = BORDER = the last colour dim (V)
        TokenClass("zero", "zero", {}),
    ]
    return layout, Codec(layout, classes, pos="onehot"), nbr


def max_err(a, b):
    return float((torch.as_tensor(a, dtype=torch.float64) - torch.as_tensor(b, dtype=torch.float64)).abs().max()) if torch.as_tensor(a).numel() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true", help="rules from a played LockPath (e28.main's step 1) instead of synthetic ones")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    rules, actions = played_rules(args.seed) if args.play else synthetic_rules(rng)
    H, W = 7, 9
    sim = E28.WrittenSim(rules, actions, H, W)
    print(f"WrittenSim: d = {sim.d}, hd = {sim.hd}, {sim.n_head} heads, offsets {sim.offs}, {sim.n_mem - 2} entries + 2 nulls")
    layout, codec, nbr = e28_layout_and_codec(sim)
    print(f"ziplib:     {layout}")
    ok = True

    def check(name, err, tol=TOL):
        nonlocal ok
        good = err <= tol
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: max |new - old| = {err:.3e}")

    # 1. the layout: every slice where WrittenSim put it
    same = [layout["colour"] == sim.T, layout["action"] == sim.A, layout["row"] == sim.ROW, layout["col"] == sim.COL,
            layout["out"] == sim.OUT, layout.flag("cell") == sim.E, layout.flag("entry") == sim.F,
            layout.flag("border") == sim.S, layout.flag("zero") == sim.Z, layout.d == sim.d,
            layout.d_layout == sim.OUT.stop] + [layout[n] == sl for n, sl in zip(nbr, sim.N)]
    check(f"layout slices ({len(same)} compared; d_layout {layout.d_layout}, d {layout.d}, {layout.n_free} spare)", 0.0 if all(same) else 1.0)
    # 2. encode_frame vs WrittenSim.encode on a random frame (colours 0..V-1)
    frame = rng.integers(0, V, size=(H, W)).astype(np.int16)
    x_new, coords = codec.encode_frame(frame)
    x_old = sim.encode(frame)
    check(f"encode_frame ({H}x{W} = {H * W} cell tokens)", max_err(x_new, x_old))
    ij = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), -1).reshape(-1, 2)
    check("encode_frame coords = (i, j) in raster order", max_err(coords, ij))
    # 3. the nulls vs memory[:2]
    check("nulls() = [border, zero] (e28.py lines 151-155)", max_err(codec.nulls(), sim.memory[:2]))
    # 4. every table entry vs memory[2:], in WrittenSim's order (actions, then table order)
    mem = []
    for ai, a in enumerate(actions):
        r = rules[a]
        side = 2 * r.r + 1
        cells = [(idx // side - r.r, idx % side - r.r) for idx in np.flatnonzero(r.mask)]
        for key, counts in r.table.items():
            colours = np.frombuffer(key, dtype=np.int16)
            k = {("colour" if o == (0, 0) else f"nbr[{o[0]},{o[1]}]"): int(c) for o, c in zip(cells, colours)}   # -1 stays -1
            k["action"] = ai
            mem.append(codec.entry(k, {"out": int(max(counts, key=counts.get))}))
    mem = torch.stack(mem)
    check(f"entry() x {len(mem)} = memory[2:] (e28.py lines 134-150; BORDER -1 -> dim {sim.C - 1})", max_err(mem, sim.memory[2:]))
    check("entry(): a dropped cell is a wildcard (all-zero subspace)",
          0.0 if all(float(mem[:, layout[n]].sum(1).max()) <= 1.0 for n in nbr) and any(float(mem[:, layout[n]].sum(1).min()) == 0.0 for n in nbr) else 1.0)
    # 5. coords_for over the full E28 sequence: nulls + entries at (0, 0), cells at (i, j)
    seq = torch.cat([sim.memory, x_old], 0)
    cf = coords_for(seq, codec)
    want = torch.cat([torch.zeros(sim.n_mem, 2, dtype=torch.long), ij], 0)
    check(f"coords_for over {seq.shape[0]} tokens (nulls + entries -> (0, 0))", max_err(cf, want))
    # 6. decode round-trips a cell and an entry
    dec = codec.decode(x_new[13])
    i, j = divmod(13, W)
    good = dec["colour"] == int(frame[i, j]) and dec["row"] == i and dec["col"] == j and dec["cell"] == 1 and dec["entry"] == 0 and dec["action"] is None
    check("decode(cell token) -> colour, row, col, flags; empty action -> None", 0.0 if good else 1.0)
    dec = codec.decode(mem[0])
    good = dec["entry"] == 1 and dec["cell"] == 0 and dec["out"] is not None and dec["action"] == 0 and codec.token_class_of(mem[0]) == "entry"
    check("decode(entry token) -> class entry, out, action", 0.0 if good else 1.0)
    # 7. the layout refuses superposition
    try:
        Layout.allocate([Subspace("a", 10), Subspace("b", 5)], d=12)
        check("LayoutError when d_layout > d", 1.0)
    except LayoutError as e:
        check(f"LayoutError when d_layout > d ({e})", 0.0)
    # 8. free-list and JSON round trip
    lay2 = Layout.from_json(layout.to_json())
    good = all(lay2[n] == layout[n] for n in layout.names) and lay2.d == layout.d and lay2.kinds == layout.kinds
    sp = layout.free(1, "spare") if layout.n_free else None
    good &= sp is None or (sp.start == sim.OUT.stop and layout["spare"] == sp)
    check("to_json/from_json round trip; free(1) claims the first spare dim", 0.0 if good else 1.0)
    # 9. channels: E18's pair choice (HD = 32: content NT = 9 dims -> pairs 7..15; position 4 pairs -> 0..3)
    plan = channels(32, dict(content_dims=9, position_pairs=4), name="l1h0")
    good = plan.content_pairs == list(range(16 - 9, 16)) and plan.position_pairs == [0, 1, 2, 3] and plan.content_dims == [2 * c for c in range(7, 16)]
    try:
        channels(32, dict(content_dims=13, position_pairs=4), name="l1h0")
        good = False
    except LayoutError:
        pass
    check("channels(32, content 9, position 4) = e18.py lines 56/67; conflict raises naming the head", 0.0 if good else 1.0)
    print("PASS: ziplib.Layout + Codec reproduce WrittenSim's layout and tokens to 1e-12" if ok else "FAIL: see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
