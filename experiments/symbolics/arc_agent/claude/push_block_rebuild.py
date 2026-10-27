"""The push_block rebuild of the symbolic agent (docs/COORDINATES.md, part 13; arc_agent/claude/DESIGN.md).

Columns are frames discovered when cells move together, not colours. Each frame's motion is explained by competing
drivers (static, action, time with a period, contact with another frame), chosen by accumulated evidence: attention
over drivers by track record. The goal is the frames' poses at the first WIN; play searches the learned model.
    python -m arc_agent.claude.push_block_rebuild      (from experiments/symbolics)
"""
import json
import time
from collections import Counter, deque

import numpy as np

H = W = 9
EPS, KEL, NMAX = 0.01, 25, 8
SEEDS, STEPS, EP_MAX, ATTEMPTS = 3, 200, 50, 5
DIRS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}      # ACTION1..ACTION4: up, down, left, right, as (dx, dy)
IDENT = (0, 0, 0)
UNK = -2                                                     # a cell hidden by a known frame
CAND = [(dx, dy, 0) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)] + \
       [("rot", cx, cy, r) for r in (1, 2, 3) for cx in range(W) for cy in range(H)]


# ---- the group: translations and 90-degree rotations of the plane; poses are (tx, ty, r) --------------------------
def rot(p, r):
    p = np.array(p).reshape(-1, 2)
    for _ in range(r % 4):
        p = np.stack([-p[:, 1], p[:, 0]], -1)
    return p


def apply(g, pts):
    return rot(pts, g[2]) + np.array(g[:2])


def compose(g1, g2):
    t = np.array(g1[:2]) + rot(g2[:2], g1[2])[0]
    return (int(t[0]), int(t[1]), (g1[2] + g2[2]) % 4)


def inverse(g):
    t = -rot(g[:2], -g[2])[0]
    return (int(t[0]), int(t[1]), (-g[2]) % 4)


def as_element(c):
    if c[0] == "rot":                                       # rotation by r about cell (cx, cy)
        _, cx, cy, r = c
        return compose((cx, cy, 0), compose((0, 0, r), (-cx, -cy, 0)))
    return c


def inside(pts):
    return bool(np.all((pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)))


def get(F, pts):
    return F[pts[:, 1], pts[:, 0]]


# ---- the worlds -----------------------------------------------------------------------------------------------------
class PushWorld:
    """Gemini's push_block (arc_agent/gemini/arc3_worlds.py); optionally a random layout and a time-driven patroller."""
    def __init__(self, rng, random_layout=False, patroller=False):
        self.rng, self.random_layout, self.patroller = rng, random_layout, patroller
        self.reset()

    def reset(self):
        if self.random_layout:
            while True:
                p = [(int(self.rng.integers(W)), int(self.rng.integers(1, H))) for _ in range(3)]
                if len(set(p)) == 3 and 1 <= p[1][0] <= W - 2 and 1 <= p[1][1] <= H - 2:
                    break
            self.player, self.box, self.target = p
        else:
            self.player, self.box, self.target = (2, 4), (4, 4), (6, 4)
        self.t, self.pat_x = 0, 1
        return self.frame()

    def frame(self):
        g = np.zeros((H, W), int)
        g[self.target[1], self.target[0]] = 2
        g[self.box[1], self.box[0]] = 4
        g[self.player[1], self.player[0]] = 1
        if self.patroller:
            g[0, self.pat_x] = 6
        return g

    def step(self, a):
        self.t += 1
        dx, dy = DIRS[a]
        nx, ny = self.player[0] + dx, self.player[1] + dy
        if 0 <= nx < W and 0 <= ny < H:
            if (nx, ny) == self.box:
                bx, by = self.box[0] + dx, self.box[1] + dy
                if 0 <= bx < W and 0 <= by < H:
                    self.box, self.player = (bx, by), (nx, ny)
            else:
                self.player = (nx, ny)
        if self.patroller:
            self.pat_x += 1 if (self.t - 1) % 6 < 3 else -1
        return self.frame(), self.box == self.target


# ---- the rebuilt agent ----------------------------------------------------------------------------------------------
class Frame:
    def __init__(self, body, content, pose):
        self.body, self.content, self.pose = body, content, pose
        self.evidence = {"static": 0.0, "action": 0.0, **{f"time{n}": 0.0 for n in range(1, NMAX + 1)}}
        self.act, self.tim = {}, {n: {} for n in range(1, NMAX + 1)}

    def cells(self, pose=None):
        return apply(self.pose if pose is None else pose, self.body)


def mode(counter):
    return counter.most_common(1)[0][0] if counter else None


class FrameAgent:
    ORDER = ["static", "action"] + [f"time{n}" for n in range(1, NMAX + 1)]

    def __init__(self):
        self.static, self.frames, self.top, self.t, self.revisions = None, [], Counter(), 0, 0

    # -- the vote: attention over drivers by accumulated evidence (hard maximum; simpler driver on ties) --
    def choice(self, i):
        ev = self.frames[i].evidence
        names = self.ORDER + sorted(k for k in ev if k.startswith("push"))
        return max(names, key=lambda k: (round(ev[k], 9), -names.index(k)))

    def raw(self, i, m, a, s):
        f = self.frames[i]
        if m == "static":
            return IDENT
        if m == "action":
            return mode(f.act.get(a, Counter()))
        if m.startswith("time"):
            n = int(m[4:])
            return mode(f.tim[n].get(s % n, Counter()))
        return None

    def resolve(self, drivers, a, s):
        """Predicted element of every frame, given each frame's driver."""
        out = {}
        for i, m in drivers.items():
            if not m.startswith("push"):
                e = self.raw(i, m, a, s)
                if e is not None and not inside(self.frames[i].cells(compose(e, self.frames[i].pose))):
                    e = IDENT                                   # the grid is bounded: such a move doesn't happen
                out[i] = e
        for i, m in drivers.items():
            if m.startswith("push"):
                j = int(m.split(":")[1])
                ej = out.get(j)
                e = IDENT
                if ej is not None and ej != IDENT:
                    fj, fi = self.frames[j], self.frames[i]
                    if set(map(tuple, fj.cells(compose(ej, fj.pose)).tolist())) & set(map(tuple, fi.cells().tolist())):
                        e = (ej[0], ej[1], 0)
                        if not inside(fi.cells(compose(e, fi.pose))):
                            e, out[j] = IDENT, IDENT            # solidity: the pushed can't move, so neither does the pusher
                out[i] = e
        return out

    def predict_poses(self, a, s):
        drivers = {i: self.choice(i) for i in range(len(self.frames))}
        el = self.resolve(drivers, a, s)
        return {i: (compose(e, self.frames[i].pose) if e is not None else None) for i, e in el.items()}

    def render(self, poses):
        F = np.where(self.static >= 0, self.static, 0)
        for i in sorted(range(len(self.frames)), key=lambda i: self.top[i]):
            if poses.get(i) is None:
                continue
            pts = self.frames[i].cells(poses[i])
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            F[pts[ok, 1], pts[ok, 0]] = self.frames[i].content[ok]
        return F

    def predict(self, a):
        if self.static is None:
            return None
        poses = self.predict_poses(a, self.t + 1)
        if any(p is None for p in poses.values()):
            return None
        return self.render(poses)

    def start(self, F):
        """A new episode: find the known frames anywhere, and rebuild the board memory."""
        self.t = 0
        self.static = F.copy()
        for f in self.frames:
            hits = [(tx, ty, r) for r in range(4) for tx in range(-8, 17) for ty in range(-8, 17)
                    if inside(apply((tx, ty, r), f.body)) and np.array_equal(get(F, apply((tx, ty, r), f.body)), f.content)]
            if hits:
                f.pose = min(hits, key=lambda g: (g[2] != f.pose[2], abs(g[0] - f.pose[0]) + abs(g[1] - f.pose[1])))
            pts = f.cells()
            if inside(pts):
                self.static[pts[:, 1], pts[:, 0]] = -1

    def recognize(self, f, F2, pred):
        best = None
        for r in range(4):
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    g = (f.pose[0] + dx, f.pose[1] + dy, r)
                    pts = apply(g, f.body)
                    if inside(pts) and np.array_equal(get(F2, pts), f.content):
                        d = (0 if pred is None else (abs(g[0] - pred[0]) + abs(g[1] - pred[1]) + (g[2] != pred[2])))
                        if best is None or d < best[0]:
                            best = (d, g)
        return best[1] if best else pred

    def split_refuted(self, F2, pred_poses):
        """Revision (part 12): a multi-cell frame matched nowhere in F2 is refuted as one rigid set. Each cell is
        matched on its own (nearest same-colour cell to its predicted place, within 2); cells that moved alike form a
        new frame. The parts inherit the parent's evidence and tables, and the frames' indices are renumbered."""
        i = 0
        while i < len(self.frames):
            f = self.frames[i]
            if len(f.body) < 2 or self.recognize(f, F2, None) is not None:
                i += 1
                continue
            groups = {}
            pred = pred_poses.get(i) or f.pose
            for k, b in enumerate(f.body):
                p0 = apply(f.pose, b[None])[0]
                pp = apply(pred, b[None])[0]
                opts = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
                        if 0 <= p0[0] + dx < W and 0 <= p0[1] + dy < H and F2[p0[1] + dy, p0[0] + dx] == f.content[k]]
                if not opts:
                    continue
                d = min(opts, key=lambda o: abs(p0[0] + o[0] - pp[0]) + abs(p0[1] + o[1] - pp[1]))
                groups.setdefault(d, []).append(k)
            if len(groups) < 2:
                i += 1
                continue
            parts = []
            for d, ks in groups.items():
                child = Frame(f.body[ks].copy(), f.content[ks].copy(), f.pose)
                child.evidence = dict(f.evidence)
                child.act = {a: Counter(c) for a, c in f.act.items()}
                child.tim = {n: {ph: Counter(c) for ph, c in t.items()} for n, t in f.tim.items()}
                parts.append(child)
            self.replace(i, parts)
            self.revisions += 1
            i += len(parts)

    def replace(self, i, parts):
        """Swap frame i for its parts; renumber the contact drivers ('push:j') of every frame."""
        old = self.frames
        new = old[:i] + parts + old[i + 1:]
        index = {}
        for j in range(len(old)):
            index[j] = [i + k for k in range(len(parts))] if j == i else [j if j < i else j + len(parts) - 1]
        for f in new:
            ev = {k: v for k, v in f.evidence.items() if not k.startswith("push")}
            for k, v in f.evidence.items():
                if k.startswith("push"):
                    for j2 in index[int(k.split(":")[1])]:
                        ev[f"push:{j2}"] = v
            f.evidence = ev
        for k, child in enumerate(parts):
            for k2 in range(len(parts)):
                if k2 != k:
                    child.evidence[f"push:{i + k2}"] = child.evidence["static"]
            child.evidence.pop(f"push:{i + k}", None)
        top = Counter()
        for j, c in self.top.items():
            for j2 in index.get(j, []):
                top[j2] = c
        self.frames, self.top = new, top

    def erased(self, F, poses):
        """F with the known frames removed; the cells they cover are unknown (UNK)."""
        G = F.copy()
        for i, f in enumerate(self.frames):
            if poses.get(i) is None:
                continue
            pts = f.cells(poses[i])
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            G[pts[ok, 1], pts[ok, 0]] = UNK
        return G

    def discover(self, Fb, F2b):
        """Explain the residual change as one set moving by one group element (part 12)."""
        known = (Fb != UNK) & (F2b != UNK)
        changed = known & (Fb != F2b)
        hidden2 = F2b == UNK
        if not changed.any():
            return None
        pts = np.argwhere(np.ones((H, W), bool))[:, ::-1]      # all cells as (x, y)
        best = None
        for c in CAND:
            g = as_element(c)
            q = apply(g, pts)
            ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
            src, dst = pts[ok], q[ok]
            A = (get(Fb, src) != 0) & (get(Fb, src) != UNK) & (get(F2b, dst) == get(Fb, src))
            S = A & (get(changed, src) | get(changed, dst) | get(hidden2, src))
            img = np.zeros((H, W), bool)
            img[dst[S, 1], dst[S, 0]] = True
            S = S & (get(changed, src) | get(hidden2, src) | get(img, src))
            if not S.any():
                continue
            pred = Fb.copy()
            under = np.where(self.static >= 0, self.static, 0)
            pred[src[S, 1], src[S, 0]] = get(under, src[S])
            pred[dst[S, 1], dst[S, 0]] = get(Fb, src[S])
            err = int(np.sum((pred != F2b) & (F2b != UNK) & (pred != UNK)))
            key = (err, g[2] != 0, abs(g[0]) + abs(g[1]))
            if best is None or key < best[0]:
                best = (key, g, src[S])
        if best is None or best[0][0] >= int(changed.sum()):
            return None
        return best[1], best[2]

    def observe(self, F, a, F2, s):
        self.split_refuted(F2, self.predict_poses(a, s))
        n = len(self.frames)
        drivers = {i: self.choice(i) for i in range(n)}
        scored = {}                                         # each driver's prediction, the others at their choice
        for i in range(n):
            for m in self.frames[i].evidence:
                d = dict(drivers)
                d[i] = m
                scored[(i, m)] = self.resolve(d, a, s).get(i)
        pred_poses = self.predict_poses(a, s)
        old = {i: f.pose for i, f in enumerate(self.frames)}
        for i, f in enumerate(self.frames):
            f.pose = self.recognize(f, F2, pred_poses.get(i)) or f.pose
        for i, f in enumerate(self.frames):
            e = compose(f.pose, inverse(old[i]))
            for m in f.evidence:
                p = scored[(i, m)]
                f.evidence[m] += np.log(1 / KEL) if p is None else (np.log(1 - EPS) if p == e else np.log(EPS / KEL))
            f.act.setdefault(a, Counter())[e] += 1
            for k in range(1, NMAX + 1):
                f.tim[k].setdefault(s % k, Counter())[e] += 1
        # frames the model doesn't know yet: explain what is left of the change
        Fb, F2b = self.erased(F, old), self.erased(F2, {i: f.pose for i, f in enumerate(self.frames)})
        for _ in range(3):
            found = self.discover(Fb, F2b)
            if found is None:
                break
            g, src = found
            new = Frame(src.copy(), get(Fb, src).copy(), g)
            new.act.setdefault(a, Counter())[g] += 1
            for k in range(1, NMAX + 1):
                new.tim[k].setdefault(s % k, Counter())[g] += 1
            j = len(self.frames)
            for i, f in enumerate(self.frames):
                f.evidence[f"push:{j}"] = f.evidence["static"]
                new.evidence[f"push:{i}"] = 0.0
            self.frames.append(new)
            self.static[src[:, 1], src[:, 0]] = -1
            under = np.where(self.static >= 0, self.static, 0)
            Fb[src[:, 1], src[:, 0]] = get(under, src)
            dst = new.cells()
            F2b[dst[:, 1], dst[:, 0]] = get(under, dst)
        covered = np.zeros((H, W), bool)
        for i, f in enumerate(self.frames):
            pts = f.cells()
            if inside(pts):
                covered[pts[:, 1], pts[:, 0]] = True
        self.static[~covered] = F2[~covered]
        for i in range(len(self.frames)):                   # who is drawn on top, from overlaps actually seen
            for j in range(i + 1, len(self.frames)):
                ci = set(map(tuple, self.frames[i].cells().tolist()))
                for (x, y) in ci & set(map(tuple, self.frames[j].cells().tolist())):
                    v = F2[y, x]
                    if v in self.frames[i].content and v not in self.frames[j].content:
                        self.top[i] += 1
                    elif v in self.frames[j].content and v not in self.frames[i].content:
                        self.top[j] += 1
        self.t = s

    def plan(self, goal, depth=12):
        """Breadth-first search over the learned model, from the current poses to the goal poses."""
        start = tuple(f.pose for f in self.frames)
        saved = [f.pose for f in self.frames]
        q, seen = deque([(start, [])]), {start}
        while q:
            state, path = q.popleft()
            if all(state[i] == goal[i] for i in goal if i < len(state)):
                for f, p in zip(self.frames, saved):
                    f.pose = p
                return path
            if len(path) >= depth:
                continue
            for a in range(4):
                for f, p in zip(self.frames, state):
                    f.pose = p
                nxt = self.predict_poses(a, self.t + len(path) + 1)
                ns = tuple(nxt[i] if nxt[i] is not None else state[i] for i in range(len(state)))
                if ns not in seen:
                    seen.add(ns)
                    q.append((ns, path + [a]))
        for f, p in zip(self.frames, saved):
            f.pose = p
        return None


class ColourObjects:
    """The symbolic agent's assumption: each colour is an object, moved by a translation per (action, colour)."""
    def __init__(self):
        self.rule = {}

    def predict(self, F, a):
        out = np.zeros_like(F)
        for c in range(1, 16):
            pts = np.argwhere(F == c)[:, ::-1]
            if len(pts):
                q = pts + np.array(self.rule.get((a, c), (0, 0)))
                ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
                out[q[ok, 1], q[ok, 0]] = c
        return out

    def observe(self, F, a, F2):
        for c in range(1, 16):
            p1 = np.argwhere(F == c)[:, ::-1]
            if not len(p1):
                continue
            p2 = set(map(tuple, np.argwhere(F2 == c)[:, ::-1].tolist()))
            best = max(((sum(tuple(v) in p2 for v in (p1 + np.array([dx, dy])).tolist()), -abs(dx) - abs(dy), (dx, dy))
                        for dx in range(-2, 3) for dy in range(-2, 3)))
            self.rule[(a, c)] = best[2]


# ---- runs -----------------------------------------------------------------------------------------------------------
def prediction_run(world, rng):
    agent, colour = FrameAgent(), ColourObjects()
    F = world.reset()
    agent.start(F)
    res = Counter()
    for _ in range(STEPS):
        a = int(rng.integers(4))
        p, cp = agent.predict(a), colour.predict(F, a)
        F2, win = world.step(a)
        res["n"] += 1
        res["rebuild"] += p is not None and np.array_equal(p, F2)
        res["colour"] += np.array_equal(cp, F2)
        res["no change"] += np.array_equal(F, F2)
        agent.observe(F, a, F2, world.t)
        colour.observe(F, a, F2)
        F = F2
        if win or world.t >= EP_MAX:
            F = world.reset()
            agent.start(F)
    frames = []
    for i, f in enumerate(agent.frames):
        ranked = sorted(f.evidence.items(), key=lambda kv: -kv[1])
        frames.append(dict(colours=sorted(set(f.content.tolist())), size=len(f.body), driver=agent.choice(i),
                           margin=float(ranked[0][1] - ranked[1][1]), runner_up=ranked[1][0]))
    return {k: res[k] / res["n"] for k in ("rebuild", "colour", "no change")}, frames, agent.revisions


def play_run(rng):
    world, agent = PushWorld(rng), FrameAgent()
    F = world.reset()
    agent.start(F)
    explore, goal = 0, None
    while goal is None:
        a = int(rng.integers(4))
        F2, win = world.step(a)
        agent.observe(F, a, F2, world.t)
        explore += 1
        F = F2
        if win:
            goal = {i: f.pose for i, f in enumerate(agent.frames)}
        if win or world.t >= EP_MAX:
            F = world.reset()
            agent.start(F)
    attempts = []
    for _ in range(ATTEMPTS):
        F = world.reset()
        agent.start(F)
        path, steps, win = agent.plan(goal) or [], 0, False
        while not win and world.t < EP_MAX:
            a = path[steps] if steps < len(path) else int(rng.integers(4))
            F2, win = world.step(a)
            agent.observe(F, a, F2, world.t)
            F, steps = F2, steps + 1
        attempts.append(steps if win else None)
    return explore, attempts


def main():
    t0 = time.time()
    out = {}
    for name, kw in (("W1 push_block", {}), ("W2 random layouts", dict(random_layout=True)),
                     ("W3 random layouts + patroller", dict(random_layout=True, patroller=True))):
        out[name] = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            acc, frames, revisions = prediction_run(PushWorld(np.random.default_rng(100 + seed), **kw), rng)
            out[name].append(dict(accuracy=acc, frames=frames, revisions=revisions))
            print(f"{name:32s} seed {seed}: exact frames rebuild {acc['rebuild']:.3f}, colour {acc['colour']:.3f}, "
                  f"no change {acc['no change']:.3f} | revisions {revisions}", flush=True)
            for f in frames:
                print(f"      frame colours {f['colours']} ({f['size']} cell): driver {f['driver']:8s} lead {f['margin']:6.1f} nats over {f['runner_up']}",
                      flush=True)
    out["play W1"] = []
    for seed in range(SEEDS):
        explore, attempts = play_run(np.random.default_rng(200 + seed))
        out["play W1"].append(dict(exploration_steps=explore, attempt_steps=attempts))
        print(f"play W1 seed {seed}: first WIN after {explore} exploration steps; later attempts won in {attempts} steps", flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/push_block_rebuild.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
