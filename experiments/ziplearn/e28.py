"""E28 — the written looped block: the world model ZipLearner learned on a game, written into one attention block that
is looped once per action of a plan, with no training, and checked as the simulator the planner would run.

What is written. After playing LockPath (levels 0–1, with the sleep pass) each action's rule is a TABLE from the
colours of a few cells around a cell (the mask) to the cell's new colour. The table becomes MEMORY TOKENS at the
front of the sequence, one per entry: its window colours in per-offset subspaces, its action, its outcome. The frame's
cells are the other tokens, each carrying its colour and its row and column as one-hot codes. One pass of the block is
two attention layers over that sequence (residual adds, no normalisation: every code is exactly one-hot, so none is
needed; no MLP: the block has nothing to compute that attention does not):
  gather   one head per neighbour offset. The query is the cell's coordinates SHIFTED by the offset (a written
           permutation of the coordinate code); the key is a token's own coordinates. The head lands on the one
           neighbour at that offset and copies its colour into the offset's subspace. A null token whose key scores
           between a full and a partial match catches the cells whose neighbour is off the grid and gives them the
           BORDER colour, as the rule's window does.
  lookup   one head. The query is the cell's own colour, the gathered neighbour colours and the current action; the
           keys are the memory tokens' windows; the value is the entry's change of colour (new minus old), added to
           the cell's colour subspace. A window never stored lands on its NEAREST stored window (most matching
           cells), so the block's default for the unknown is "the closest rule" where the planner's is "unchanged".
The loop. Pass k applies action a_k: the action's one-hot is the anchor re-injected into every cell token before the
pass (the paper's α·anchor); between passes the cell tokens' gathered subspaces are cleared and the colour subspace is
re-quantised to a one-hot (the boundary operator; the paper's is RMSNorm). Memory tokens attend only to the null token,
so they are never overwritten and serve every pass. The lengths: d = 200-ish, memory tokens = the table's entries.

Test. Frames from random walks in the true game (snapshot/restore, levels 0 and 1); random plans of 1–4 actions.
Agreement of the block's rollouts with (a) the planner's own rollouts (its LocalRule predictions) and (b) the true
game, split by whether every window along the plan was known to the table. Pre-registered: (a) >= 0.98 exact on
plans whose windows were all known (refute: < 0.9); (b) reported, with the nearest-rule default's accuracy against
the planner's "unchanged" on plans that met unknown windows.

    python experiments/ziplearn/e28.py           (CPU, ~1 min) -> runs/e28/e28.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "transformers"))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import h1_lid as H1                                           # noqa: E402
from arcgames import play, crop_box, BORDER, V               # noqa: E402
from tasks.games import LockPath                             # noqa: E402
from tasks.harness import Environment                        # noqa: E402


class WrittenSim(nn.Module):
    """The learned rules of a game written into one looped attention block (module docstring)."""

    def __init__(self, rules, actions, H, W, M=30.0):
        super().__init__()
        self.actions = list(actions)
        self.H, self.W = H, W
        C = V + 1                                                    # colours 0..V-1, index V = BORDER
        self.C, self.BORDER = C, V
        offs = []                                                    # the union of the masks' offsets, centre excluded
        for a in self.actions:
            r = rules[a]
            side = 2 * r.r + 1
            for idx in np.flatnonzero(r.mask):
                o = (idx // side - r.r, idx % side - r.r)
                if o != (0, 0) and o not in offs:
                    offs.append(o)
        self.offs = offs
        nA, nO = len(self.actions), len(offs)
        g = nO + 2                                                   # the action's weight: one action match outweighs every cell match
        # -- residual layout ------------------------------------------------------------------------------------
        base = 0
        self.T = slice(base, base + C); base += C                    # own colour (+BORDER)
        self.N = []
        for _ in offs:
            self.N.append(slice(base, base + C)); base += C          # gathered neighbour colours, one subspace per offset
        self.A = slice(base, base + nA); base += nA                  # the action (the per-pass anchor)
        self.E, self.F, self.S, self.Z = base, base + 1, base + 2, base + 3; base += 4   # token class: cell, entry, border, zero
        self.ROW = slice(base, base + H); base += H
        self.COL = slice(base, base + W); base += W
        self.OUT = slice(base, base + V); base += V                  # a memory entry's new colour
        n_head = max(1, nO)
        hd = max(int(math.ceil(base / n_head)), H + W + 2, C)
        self.d, self.hd, self.n_head, self.M = hd * n_head, hd, n_head, M
        d = self.d
        # -- the two attention layers, written -------------------------------------------------------------------
        self.gather = H1.Attn(d, n_head, "learned"); self.gather.causal = False
        self.lookup = H1.Attn(d, 1, "learned"); self.lookup.causal = False
        with torch.no_grad():
            for lin in (self.gather.qkv, self.gather.proj, self.lookup.qkv, self.lookup.proj):
                lin.weight.zero_(); lin.bias.zero_()
            Q, K, Vv = 0, d, 2 * d                                   # row offsets of q | k | v in qkv.weight
            Wg, Pg = self.gather.qkv.weight, self.gather.proj.weight
            for j, (di, dj) in enumerate(offs):
                h0 = j * hd
                for r_ in range(H):                                  # query: my row shifted by di; key: your row
                    if 0 <= r_ + di < H:
                        Wg[Q + h0 + r_ + di, self.ROW.start + r_] = M
                    Wg[K + h0 + r_, self.ROW.start + r_] = M
                for c_ in range(W):
                    if 0 <= c_ + dj < W:
                        Wg[Q + h0 + H + c_ + dj, self.COL.start + c_] = M
                    Wg[K + h0 + H + c_, self.COL.start + c_] = M
                b = h0 + H + W                                       # cells: the border token catches an off-grid neighbour (1.5 matches)
                Wg[Q + b, self.E] = M
                Wg[K + b, self.S] = 1.5 * M
                for flag in (self.F, self.S, self.Z):                # entries and the two nulls: the zero token (value 0)
                    Wg[Q + b + 1, flag] = M
                Wg[K + b + 1, self.Z] = 1.5 * M
                for c in range(C):                                   # value: the attended token's colour -> subspace j
                    Wg[Vv + h0 + c, self.T.start + c] = 1.0
                    Pg[self.N[j].start + c, h0 + c] = 1.0
            Wl, Pl = self.lookup.qkv.weight, self.lookup.proj.weight
            for sl in [self.T] + self.N:                             # window colours: query and key alike
                for c in range(C):
                    Wl[Q + sl.start + c, sl.start + c] = M
                    Wl[K + sl.start + c, sl.start + c] = M
            for a in range(nA):
                Wl[Q + self.A.start + a, self.A.start + a] = g * M
                Wl[K + self.A.start + a, self.A.start + a] = M
            Wl[Q + self.F, self.E] = M                               # a cell's query: memory entries only
            Wl[K + self.F, self.F] = (2 + nO + g) * M
            for flag in (self.F, self.S, self.Z):                    # entries and the nulls: the border token (value 0)
                Wl[Q + self.S, flag] = M
            Wl[K + self.S, self.S] = (3 + nO + g) * M
            for c in range(V):                                       # value: new colour minus old colour -> T
                Wl[Vv + c, self.OUT.start + c] = 1.0
                Wl[Vv + c, self.T.start + c] = -1.0
                Pl[self.T.start + c, c] = 1.0
        # -- the memory tokens: one per table entry -----------------------------------------------------------------
        mem = []
        for ai, a in enumerate(self.actions):
            r = rules[a]
            side = 2 * r.r + 1
            cells = [(idx // side - r.r, idx % side - r.r) for idx in np.flatnonzero(r.mask)]
            for key, counts in r.table.items():
                colours = np.frombuffer(key, dtype=np.int16)
                new = max(counts, key=counts.get)
                v = torch.zeros(d)
                for o, col in zip(cells, colours):
                    col = self.BORDER if col == BORDER else int(col)
                    sl = self.T if o == (0, 0) else self.N[offs.index(o)]
                    v[sl.start + col] = 1.0
                v[self.A.start + ai] = 1.0
                v[self.F] = 1.0
                v[self.OUT.start + int(new)] = 1.0
                mem.append(v)
        border = torch.zeros(d)
        border[self.S] = 1.0
        border[self.T.start + self.BORDER] = 1.0                     # the border token: its colour is BORDER
        zero = torch.zeros(d)
        zero[self.Z] = 1.0                                           # the zero token: what the entries attend to
        self.register_buffer("memory", torch.stack([border, zero] + mem))
        self.n_mem = self.memory.shape[0]                            # 2 nulls + the entries
        self.ties = 0                                                # cells whose colour was not one-hot before the snap

    def encode(self, frame):
        Hh, Ww = frame.shape
        x = torch.zeros(Hh * Ww, self.d)
        for i in range(Hh):
            for j in range(Ww):
                t = i * Ww + j
                x[t, self.T.start + int(frame[i, j])] = 1.0
                x[t, self.ROW.start + i] = 1.0
                x[t, self.COL.start + j] = 1.0
                x[t, self.E] = 1.0
        return x

    @torch.no_grad()
    def rollout(self, frame, plan):
        Hh, Ww = frame.shape
        x = torch.cat([self.memory, self.encode(frame)], 0)[None]    # (1, n_mem + H*W, d)
        n = self.n_mem
        for a in plan:
            # the boundary operator: the anchor (this pass's action) into every cell token
            x[0, n:, self.A] = 0.0
            x[0, n:, self.A.start + self.actions.index(a)] = 1.0
            x = x + self.gather(x)                                   # gather neighbours
            x = x + self.lookup(x)                                   # look the window up: colour <- new colour
            # the boundary operator: re-quantise the colour, clear the scratch subspaces
            t = x[0, n:, self.T.start:self.T.start + V].clone()
            self.ties += int(((t.max(1).values - 1).abs() > 1e-3).sum())
            x[0, n:, self.T] = 0.0
            x[0, n:, self.T.start:self.T.start + V] = nn.functional.one_hot(t.argmax(1), V).float()
            for sl in self.N:
                x[0, n:, sl] = 0.0
        cols = x[0, n:, self.T.start:self.T.start + V].argmax(1)
        return cols.reshape(Hh, Ww).numpy().astype(np.int16)


def true_rollout(game, plan):
    """The game's own answer, or None if the plan ends the level or dies (no frame to compare)."""
    snap = game.snapshot()
    ok = True
    for a in plan:
        game.apply(a, None)
        if game.level_complete() or game.is_dead():
            ok = False
            break
    frame = np.array(game.render()[-1], dtype=np.int16) if ok else None
    game.restore(snap)
    return frame


def walk_states(game, actions, rng, n):
    """Snapshots of n states reached by a random walk that never completes the level or dies."""
    states = [game.snapshot()]
    for _ in range(n - 1):
        for _try in range(8):
            a = actions[int(rng.integers(len(actions)))]
            snap = game.snapshot()
            game.apply(a, None)
            if game.level_complete() or game.is_dead():
                game.restore(snap)
                continue
            break
        states.append(game.snapshot())
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=int, default=300, help="random plans per level")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "runs" / "e28"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    # 1. play LockPath levels 0-1 with the sleep pass: the rules
    env = Environment(LockPath())
    results, stats, player = play(env, budget_per_level=150, max_levels=2, seed=args.seed, sleep=True, return_player=True)
    actions = [a for a in player.actions if player.models[a].n_obs]
    rules = {a: player.models[a].best() for a in actions}
    print("E28: rules learned on LockPath levels 0-1 (with sleep): " + "; ".join(
        f"{a.name}: radius {rules[a].r}, {int(rules[a].mask.sum())} cells, {len(rules[a].table)} entries" for a in actions))
    print("   play: " + ", ".join(f"level {r['level']} {'solved' if r['solved'] else 'unsolved'} in {r['actions']}" for r in results))
    # 2. the written block, sized for the largest playfield among the levels tested
    game = LockPath()
    boxes = {}
    for lvl in args.levels:
        game.load_level(lvl)
        boxes[lvl] = crop_box(np.array(game.render()[-1]))
    Hmax = max(b[2] - b[0] for b in boxes.values())
    Wmax = max(b[3] - b[1] for b in boxes.values())
    sim = WrittenSim(rules, actions, Hmax, Wmax)
    print(f"   written block: d = {sim.d}, {sim.n_head} gather heads over offsets {sim.offs}, one lookup head, "
          f"{sim.n_mem - 2} memory tokens + 2 nulls, up to {Hmax * Wmax} cell tokens; weights written, none trained")
    # 3. rollouts: written vs planner vs truth, per level
    report = {}
    for lvl in args.levels:
        game.load_level(lvl)
        box = boxes[lvl]
        crop = lambda f: f[box[0]:box[2], box[1]:box[3]]
        states = walk_states(game, actions, rng, 40)
        tally = dict(n=0, known=0, wp_known=0, wp_all=0, wt_known=0, pt_known=0, wt_unk=0, pt_unk=0, wp_unk=0, skipped=0)
        by_len = {k: [0, 0] for k in range(1, 5)}
        for _ in range(args.plans):
            game.restore(states[int(rng.integers(len(states)))])
            plan = [actions[int(rng.integers(len(actions)))] for _ in range(int(rng.integers(1, 5)))]
            frame = crop(np.array(game.render()[-1], dtype=np.int16))
            truth = true_rollout(game, plan)
            if truth is None:
                tally["skipped"] += 1
                continue
            truth = crop(truth)
            f, known = frame.copy(), True
            for a in plan:
                f, unknown = rules[a].predict(f)
                known = known and unknown == 0
            planner = f
            written = sim.rollout(frame, plan)
            wp, wt, pt = (np.array_equal(written, planner), np.array_equal(written, truth), np.array_equal(planner, truth))
            tally["n"] += 1
            tally["wp_all"] += wp
            if known:
                tally["known"] += 1
                tally["wp_known"] += wp; tally["wt_known"] += wt; tally["pt_known"] += pt
                by_len[len(plan)][0] += wp; by_len[len(plan)][1] += 1
            else:
                tally["wt_unk"] += wt; tally["pt_unk"] += pt; tally["wp_unk"] += wp
        n, k, u = tally["n"], tally["known"], tally["n"] - tally["known"]
        report[lvl] = dict(tally, by_len={str(a): b for a, b in by_len.items()}, playfield=[box[2] - box[0], box[3] - box[1]])
        print(f"   level {lvl} ({box[2] - box[0]}x{box[3] - box[1]}), {n} plans ({tally['skipped']} skipped: ended the level or died):")
        print(f"      all windows known ({k}): written = planner {tally['wp_known']}/{k}" +
              (f" ({tally['wp_known'] / k:.3f})" if k else "") + f"; written = truth {tally['wt_known']}/{k}; planner = truth {tally['pt_known']}/{k}"
              + "; by plan length " + ", ".join(f"{a}: {b[0]}/{b[1]}" for a, b in by_len.items()))
        print(f"      unknown windows met ({u}): written = truth {tally['wt_unk']}/{u} (nearest rule); planner = truth {tally['pt_unk']}/{u} "
              f"('unchanged'); written = planner {tally['wp_unk']}/{u}")
    print(f"   colour subspaces not one-hot before re-quantisation (ties between nearest rules): {sim.ties}")
    # verdict on the training levels' all-known plans
    kn = sum(report[l]["known"] for l in args.levels if l < 2)
    ok = sum(report[l]["wp_known"] for l in args.levels if l < 2)
    ratio = ok / max(1, kn)
    verdict = "PASS" if ratio >= 0.98 else "REFUTED" if ratio < 0.9 else "INCONCLUSIVE"
    print(f"\nE28 verdict: {verdict} — written block = planner on {ok}/{kn} all-known plans over levels 0-1 ({ratio:.3f})")
    json.dump(dict(rules={a.name: dict(radius=rules[a].r, cells=int(rules[a].mask.sum()), entries=len(rules[a].table)) for a in actions},
                   play=results, d=sim.d, heads=sim.n_head, offsets=sim.offs, memory_tokens=sim.n_mem - 2, ties=sim.ties,
                   levels=report, verdict=verdict, ratio=ratio), open(out / "e28.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
