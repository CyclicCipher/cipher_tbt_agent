"""A satisficing active-inference agent (docs/AGENT.md, part 14; arc_agent/claude/HOMEOSTASIS.md).

Values are homeostatic variables with acceptable ranges, not a scalar utility. Progress is acceptable when a
reachable state has predicted P(WIN) >= ALPHA; understanding when no reachable contact context is untried. The agent
takes the fewest-action acceptable plan; failing that, the most epistemic plan (goal information, plus contact
novelty while understanding is out of range); failing that, a random action. The goal is the hidden cause of the WIN
signal, inferred from evidence over cell(f, c) and pos(f, x, y) hypotheses. The world model is part 13's frame agent.
    python -m arc_agent.claude.homeostatic_agent      (from experiments/symbolics)
"""
import json
import math
import time
from collections import Counter, deque

import numpy as np

from arc_agent.claude.push_block_rebuild import EP_MAX, H, W, FrameAgent, PushWorld, mode, rot

ALPHA, EPS_G, DEPTH, COST = 0.5, 0.01, 12, 0.01
LEVELS, SEEDS, CAP = 8, 3, 3000
WORLDS = (("W1 push_block", {}), ("W2 random layouts", dict(random_layout=True)),
          ("W3 random layouts + patroller", dict(random_layout=True, patroller=True)))


def key(f):
    return (tuple(map(tuple, f.body.tolist())), tuple(f.content.tolist()))


def offsets(f):
    return [tuple(c) for c in rot(f.body, f.pose[2]).tolist()]


def inb(x, y):
    return 0 <= x < W and 0 <= y < H


# ---- goal discovery: hypotheses about what causes WIN ---------------------------------------------------------------
class GoalModel:
    def __init__(self):
        self.err = Counter()

    def hypotheses(self, agent):
        colours = sorted(set(int(c) for c in np.unique(agent.static) if c >= 0))
        hs = []
        for f in agent.frames:
            k = key(f)
            hs += [("cell", k, c) for c in colours] + [("pos", k, x, y) for x in range(W) for y in range(H)]
        return hs

    @staticmethod
    def holds(h, xy, offs, static):
        if h[0] == "pos":
            return xy == (h[2], h[3])
        pts = [(xy[0] + cx, xy[1] + cy) for cx, cy in offs]
        return all(inb(x, y) and static[y, x] == h[2] for x, y in pts)

    def posterior(self, agent):
        hs = self.hypotheses(agent)
        n_cell = sum(h[0] == "cell" for h in hs) or 1
        n_pos = sum(h[0] == "pos" for h in hs) or 1
        r = math.log(EPS_G / (1 - EPS_G))
        lw = {h: math.log(0.5 / (n_cell if h[0] == "cell" else n_pos)) + r * self.err[h] for h in hs}
        if not lw:
            return {}
        m = max(lw.values())
        z = sum(math.exp(v - m) for v in lw.values())
        return {h: math.exp(v - m) / z for h, v in lw.items()}

    def observe(self, agent, win):
        frames = {key(f): f for f in agent.frames}
        for h in self.hypotheses(agent):
            f = frames[h[1]]
            if self.holds(h, (f.pose[0], f.pose[1]), offsets(f), agent.static) != win:
                self.err[h] += 1

    def masses(self, agent, post):
        """M[i][(x, y)]: posterior mass of the hypotheses about frame i that hold with frame i at (x, y)."""
        out = []
        for f in agent.frames:
            k, offs, M = key(f), offsets(f), Counter()
            for h, p in post.items():
                if h[1] != k:
                    continue
                if h[0] == "pos":
                    M[(h[2], h[3])] += p
                else:
                    for x in range(W):
                        for y in range(H):
                            if self.holds(h, (x, y), offs, agent.static):
                                M[(x, y)] += p
            out.append(M)
        return out

    def best(self, agent):
        post = self.posterior(agent)
        h = max(post, key=post.get)
        return (h[0], list(h[1][1]), *h[2:]), post[h]


# ---- the learned dynamics as plain tuples, for search (mirrors FrameAgent.resolve for translations) -------------
def compact(agent):
    n = len(agent.frames)
    drv = {i: agent.choice(i) for i in range(n)}
    inc = {i for i in range(n) if drv[i] == "action"}
    grew = True
    while grew:
        grew = False
        for i in range(n):
            if i not in inc and drv[i].startswith("push") and int(drv[i].split(":")[1]) in inc:
                inc.add(i)
                grew = True
    idx = sorted(inc)
    frames = []
    for i in idx:
        f = agent.frames[i]
        if drv[i] == "action":
            table = {}
            for a in range(4):
                e = mode(f.act.get(a, Counter()))
                table[a] = (e[0], e[1]) if e is not None and e[2] == 0 else None
            frames.append(("action", offsets(f), table, None))
        else:
            frames.append(("push", offsets(f), None, idx.index(int(drv[i].split(":")[1]))))
    start = tuple((agent.frames[i].pose[0], agent.frames[i].pose[1]) for i in idx)
    return idx, frames, start


def sim(frames, state, a):
    new, moved = list(state), [None] * len(frames)
    for k, (kind, offs, table, _) in enumerate(frames):
        if kind == "action" and table.get(a):
            d = table[a]
            x, y = state[k][0] + d[0], state[k][1] + d[1]
            if all(inb(x + cx, y + cy) for cx, cy in offs):
                new[k], moved[k] = (x, y), d
    for k, (kind, offs, _, j) in enumerate(frames):
        if kind == "push" and moved[j]:
            d = moved[j]
            pc = {(new[j][0] + cx, new[j][1] + cy) for cx, cy in frames[j][1]}
            if pc & {(state[k][0] + cx, state[k][1] + cy) for cx, cy in offs}:
                x, y = state[k][0] + d[0], state[k][1] + d[1]
                if all(inb(x + cx, y + cy) for cx, cy in offs):
                    new[k] = (x, y)
                else:
                    new[j] = state[j]                      # solidity
    return tuple(new)


def contact(agent, frames, idx, state, k, a, fixed):
    """What the body frame k would move into: another frame, a board colour, or nothing."""
    d = frames[k][2].get(a)
    if not d:
        return None
    dest = {(state[k][0] + d[0] + cx, state[k][1] + d[1] + cy) for cx, cy in frames[k][1]}
    for j, (_, offs, _, _) in enumerate(frames):
        if j != k and dest & {(state[j][0] + cx, state[j][1] + cy) for cx, cy in offs}:
            return ("frame", key(agent.frames[idx[j]]))
    for fk, cells in fixed:
        if dest & cells:
            return ("frame", fk)
    for x, y in dest:
        if inb(x, y) and agent.static[y, x] > 0:
            return ("colour", int(agent.static[y, x]))
    return None


def hb(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def plan(agent, goal, counts, epistemic=True):
    """The fewest-action acceptable plan; else the most epistemic; else None. Returns (path, kind)."""
    idx, frames, start = compact(agent)
    if not idx:
        return None, "none"
    post = goal.posterior(agent)
    M = goal.masses(agent, post)
    others = [i for i in range(len(agent.frames)) if i not in idx]
    const = sum(M[i][(agent.frames[i].pose[0], agent.frames[i].pose[1])] for i in others)
    fixed = [(key(agent.frames[i]), {(agent.frames[i].pose[0] + cx, agent.frames[i].pose[1] + cy) for cx, cy in offsets(agent.frames[i])})
             for i in others]
    bodies = [k for k, fr in enumerate(frames) if fr[0] == "action"]
    seen = {start: ([], frozenset())}
    q = deque([start])
    accept, untried, scored = None, False, []
    while q:
        s = q.popleft()
        path, ctx = seen[s]
        pwin = const + sum(M[idx[k]][s[k]] for k in range(len(idx)))
        if pwin >= ALPHA and path and (accept is None or (len(path), -pwin) < (len(accept[0]), -accept[1])):
            accept = (path, pwin)
        nov = sum(1 / (1 + counts[c]) for c in ctx)
        scored.append((hb(pwin), nov, len(path), path))
        if len(path) >= DEPTH:
            continue
        for a in range(4):
            c2 = set(ctx)
            for k in bodies:
                c = contact(agent, frames, idx, s, k, a, fixed)
                if c is not None:
                    cc = (key(agent.frames[idx[k]]), a, c)
                    c2.add(cc)
                    untried |= counts[cc] == 0
            ns = sim(frames, s, a)
            if ns not in seen:
                seen[ns] = (path + [a], frozenset(c2))
                q.append(ns)
    if accept is not None:
        return list(accept[0]), "satisfice"
    if not epistemic:
        return None, "random"
    best = max(scored, key=lambda r: (r[0] + (r[1] if untried else 0) - COST * r[2], -r[2]))
    if best[0] + (best[1] if untried else 0) - COST * best[2] <= 1e-3 or not best[3]:
        return None, "random"
    return list(best[3]), "explore"


def record_contacts(agent, counts, a):
    idx, frames, start = compact(agent)
    others = [i for i in range(len(agent.frames)) if i not in idx]
    fixed = [(key(agent.frames[i]), {(agent.frames[i].pose[0] + cx, agent.frames[i].pose[1] + cy) for cx, cy in offsets(agent.frames[i])})
             for i in others]
    for k, fr in enumerate(frames):
        if fr[0] == "action":
            c = contact(agent, frames, idx, start, k, a, fixed)
            if c is not None:
                counts[(key(agent.frames[idx[k]]), a, c)] += 1


# ---- the true world, for levels and optima --------------------------------------------------------------------------
def restart(world, layout):
    world.player, world.box, world.target = layout
    world.t, world.pat_x = 0, 1
    return world.frame()


def optimum(world):
    start, target = (world.player, world.box), world.target
    seen, q = {start: 0}, deque([start])
    while q:
        p, b = q.popleft()
        if b == target:
            return seen[(p, b)]
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            np_, nb = (p[0] + dx, p[1] + dy), b
            if not inb(*np_):
                continue
            if np_ == b:
                nb = (b[0] + dx, b[1] + dy)
                if not inb(*nb):
                    continue
            if (np_, nb) not in seen:
                seen[(np_, nb)] = seen[(p, b)] + 1
                q.append((np_, nb))
    return None


def run(kw, seed, agent_kind):
    rng = np.random.default_rng(seed)
    world = PushWorld(np.random.default_rng(100 + seed), **kw)
    agent, goal, counts = FrameAgent(), GoalModel(), Counter()
    F = world.reset()
    layout = (world.player, world.box, world.target)
    agent.start(F)
    level, total, lsteps, attempts, path = 0, 0, 0, 0, []
    first_win, levels, opt, pos_goal = None, [], optimum(world), None
    kinds = Counter()
    while level < LEVELS and total < CAP:
        if not path:
            if agent_kind == "positional":
                path = (agent.plan(pos_goal) or []) if pos_goal is not None else []
                kind = "plan" if path else "random"
            else:
                path, kind = plan(agent, goal, counts, epistemic=(agent_kind == "homeostatic"))
                path = path or []
            if not path:
                path, kind = [int(rng.integers(4))], "random"
        kinds[kind] += 1
        a = path.pop(0)
        pred = agent.predict(a)
        record_contacts(agent, counts, a)
        F2, win = world.step(a)
        total, lsteps = total + 1, lsteps + 1
        agent.observe(F, a, F2, world.t)
        goal.observe(agent, win)
        if pred is None or not np.array_equal(pred, F2):
            path = []
        F = F2
        if win:
            if first_win is None:
                first_win = total
                pos_goal = {i: f.pose for i, f in enumerate(agent.frames)}
            best, p = goal.best(agent)
            levels.append(dict(level=level, steps=lsteps, optimum=opt, first_attempt=attempts == 0,
                               map_goal=best, map_p=p))
            level, lsteps, attempts, path = level + 1, 0, 0, []
            F = world.reset()
            layout, opt = (world.player, world.box, world.target), optimum(world)
            agent.start(F)
        elif world.t >= EP_MAX:
            attempts, path = attempts + 1, []
            F = restart(world, layout)
            agent.start(F)
    return dict(first_win=first_win, levels=levels, total_steps=total, decisions=dict(kinds))


def main():
    t0 = time.time()
    out = {}
    for name, kw in WORLDS:
        for agent_kind in ("homeostatic", "no epistemic", "positional"):
            runs = [run(kw, seed, agent_kind) for seed in range(SEEDS)]
            out[f"{name} / {agent_kind}"] = runs
            fw = [r["first_win"] for r in runs]
            later = [l for r in runs for l in r["levels"] if l["level"] >= 1]
            l3 = [l for r in runs for l in r["levels"] if l["level"] >= 2]
            print(f"{name:30s} {agent_kind:12s} first WIN at {fw} | levels done {[len(r['levels']) for r in runs]} | "
                  f"levels 2-8 won in first attempt {np.mean([l['first_attempt'] for l in later]) if later else float('nan'):.2f} | "
                  f"level 3+ excess over optimum {np.mean([l['steps'] - l['optimum'] for l in l3]) if l3 else float('nan'):.2f} | "
                  f"decisions {dict(sum((Counter(r['decisions']) for r in runs), Counter()))} | {time.time() - t0:.0f} s", flush=True)
            if agent_kind == "homeostatic":
                for r in runs:
                    print("      MAP goal after each WIN: " + "; ".join(f"{l['map_goal']} ({l['map_p']:.2f})" for l in r["levels"][:3]), flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/homeostatic_agent.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
