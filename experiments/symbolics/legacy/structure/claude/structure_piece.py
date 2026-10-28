"""The moving piece (docs/COORDINATES.md, part 12; arc_agent/claude/DESIGN.md, F3): objects as frames that move
together. The learner's only geometry is the board's: translations and 90-degree rotations on a 12 x 12 torus. It
explains the first frame change as one set of cells moving by one group element; that set is the object. Afterwards
it reads the pose off each frame, and learns whether each action is a world-frame element (pose' = g.pose) or a
body-frame one (pose' = pose.h).
    python -m structure.claude.structure_piece      (from experiments/symbolics)
"""
import json
import time

import numpy as np

N, SEEDS, TRIALS, STEPS, TRANSFER = 12, 3, 20, 30, 10
MOVES = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}    # world translations; action 4 rotates about the pivot
G = [((tx, ty), r) for r in range(4) for tx in range(N) for ty in range(N)]
CELLS = np.array([(x, y) for x in range(N) for y in range(N)])


def rot(p, r):
    p = np.array(p) % N
    for _ in range(r % 4):
        p = np.stack([-p[..., 1], p[..., 0]], -1) % N
    return p


def apply(g, pts):
    return (rot(pts, g[1]) + np.array(g[0])) % N


def compose(g1, g2):                                       # g1 after g2
    return (tuple(int(v) for v in (np.array(g1[0]) + rot(g2[0], g1[1])) % N), (g1[1] + g2[1]) % 4)


def inverse(g):
    return (tuple(int(v) for v in (-rot(g[0], -g[1])) % N), (-g[1]) % 4)


def idx(pts):
    pts = np.atleast_2d(pts)
    return pts[:, 0] * N + pts[:, 1]


# ---- the world --------------------------------------------------------------------------------------------------------
def make_piece(rng):
    while True:
        cells = [(0, 0)]
        while len(cells) < 5:
            x, y = cells[int(rng.integers(len(cells)))]
            dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(4))]
            if (x + dx, y + dy) not in cells:
                cells.append((x + dx, y + dy))
        shape, colours = np.array(cells), rng.integers(4, 7, 5)
        canon = lambda pts: sorted(zip(map(tuple, (pts - pts.min(0)).tolist()), colours.tolist()))
        base = canon(shape)
        rots = [canon(np.stack([-shape[:, 1], shape[:, 0]], -1) if r == 1 else
                      (-shape if r == 2 else np.stack([shape[:, 1], -shape[:, 0]], -1))) for r in (1, 2, 3)]
        if all(rr != base for rr in rots):
            return shape, colours


class World:
    def __init__(self, rng):
        self.shape, self.colours = make_piece(rng)
        self.pose = ((int(rng.integers(N)), int(rng.integers(N))), int(rng.integers(4)))
        self.board = np.zeros((N, N), int)
        occupied = set(idx(apply(self.pose, self.shape)).tolist())
        free = [c for c in range(N * N) if c not in occupied]
        spots = rng.choice(free, 9, replace=False)
        self.board.flat[spots[:8]] = rng.integers(1, 4, 8)
        self.board.flat[spots[8]] = self.colours[int(rng.integers(5))]       # the distractor shares a piece colour

    def piece_cells(self):
        return set(idx(apply(self.pose, self.shape)).tolist())

    def frame(self):
        f = self.board.copy()
        f.flat[idx(apply(self.pose, self.shape))] = self.colours
        return f

    def step(self, a):
        self.pose = compose((MOVES[a], 0), self.pose) if a in MOVES else compose(self.pose, ((0, 0), 1))


# ---- the learner: no object primitive ------------------------------------------------------------------------------
class Learner:
    def __init__(self, carried=None, revise=False):
        self.discovered, self.static, self.revise, self.revisions = False, None, revise, 0
        self.obs = {a: {"world": [], "body": []} for a in range(5)}
        self.kind = {}                                     # carried frame types
        if carried:
            for a, (kind, g) in carried.items():
                self.kind[a] = kind
                if kind == "world":
                    self.obs[a]["world"].append(g)

    def rule(self, a):
        w, b = self.obs[a]["world"], self.obs[a]["body"]
        wc, bc = len(w) > 0 and len(set(w)) == 1, len(b) > 0 and len(set(b)) == 1
        if self.kind.get(a) == "body":
            return ("body", b[0]) if bc else None
        if wc:
            return ("world", w[0])
        if bc:
            return ("body", b[0])
        return None

    def render(self, pose):
        f = np.where(self.static >= 0, self.static, 0)
        f.flat[idx(apply(pose, self.body))] = self.content
        return f

    def predict(self, a):
        if not self.discovered:
            return None
        r = self.rule(a)
        if r is None:
            return None
        pose = compose(r[1], self.pose) if r[0] == "world" else compose(self.pose, r[1])
        return self.render(pose)

    def discover(self, F, F2):
        if self.static is None:
            self.static = F.copy()
        changed = (F != F2).ravel()
        Ff, F2f = F.ravel(), F2.ravel()
        best = None
        for g in G[1:]:
            gi = idx(apply(g, CELLS))
            A = (Ff != 0) & (F2f[gi] == Ff)
            S = A & (changed | changed[gi])
            img = np.zeros(N * N, bool)
            img[gi[S]] = True
            S = S & (changed | img)                         # an unchanged cell that isn't re-covered didn't move
            pred = Ff.copy()
            under = self.static.ravel()[S]
            pred[S] = np.where(under >= 0, under, 0)
            pred[gi[S]] = Ff[S]
            err = int(np.sum(pred != F2f))
            key = (err, g[1] != 0, min(g[0][0], N - g[0][0]) + min(g[0][1], N - g[0][1]))
            if best is None or key < best[0]:
                best = (key, g, S)
        _, g, S = best
        self.body, self.content = CELLS[S], Ff[S]           # body frame = world frame at discovery
        self.static.flat[S] = -1
        self.discovered, self.pose = True, ((0, 0), 0)
        return g, set(np.flatnonzero(S).tolist())

    def observe(self, F, a, F2):
        if not self.discovered:
            new, _ = self.discover(F, F2)
        else:
            best, new = -1, None
            for g in G:
                m = int(np.sum(F2.ravel()[idx(apply(g, self.body))] == self.content))
                if m > best:
                    best, new = m, g
            if self.revise and not np.array_equal(self.render(new), F2):
                # exploratory (after P1 failed): the object model is refuted, so re-explain this transition and carry
                # the body-frame observations into the new body frame by conjugation, h -> P h P^-1
                P = self.pose
                for o in self.obs.values():
                    o["body"] = [compose(compose(P, h), inverse(P)) for h in o["body"]]
                new, _ = self.discover(F, F2)
                self.revisions += 1
        self.obs[a]["world"].append(compose(new, inverse(self.pose)))
        self.obs[a]["body"].append(compose(inverse(self.pose), new))
        self.pose = new
        covered = np.zeros(N * N, bool)
        covered[idx(apply(new, self.body))] = True
        self.static.flat[~covered] = F2.ravel()[~covered]

    def classes(self):
        out = {}
        for a in range(5):
            w, b = self.obs[a]["world"], self.obs[a]["body"]
            wc, bc = len(w) > 0 and len(set(w)) == 1, len(b) > 0 and len(set(b)) == 1
            out[a] = "both" if wc and bc else "world" if wc else "body" if bc else "neither"
        return out


class ColourObjects:
    """The symbolic agent's assumption: each colour is an object, moved by a translation learned per (action, colour)."""
    def __init__(self):
        self.rule = {}

    def predict(self, F, a):
        out = np.zeros_like(F)
        for c in range(1, 7):
            pts = np.argwhere(F == c)
            if len(pts):
                d = np.array(self.rule.get((a, c), (0, 0)))
                q = (pts + d) % N                           # frames are indexed [x, y], so argwhere gives (x, y)
                out[q[:, 0], q[:, 1]] = c
        return out

    def observe(self, F, a, F2):
        for c in range(1, 7):
            p1, p2 = np.argwhere(F == c), set(map(tuple, np.argwhere(F2 == c).tolist()))
            if not len(p1):
                continue
            best = None
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    q = (p1 + np.array([dx, dy])) % N
                    hits = sum(tuple(v) in p2 for v in q.tolist())
                    key = (-hits, abs(dx) + abs(dy))
                    if best is None or key < best[0]:
                        best = (key, (dx, dy))
            self.rule[(a, c)] = best[1]


def run_trial(world, learner, rng, steps):
    rec = dict(frames=0, learner_ok=0, colour_ok=0, nochange_ok=0, first_sightings=0, tie_errors=0, other_errors=0,
               pose_ok=0, pose_checked=0, errors_all=0)
    colour = ColourObjects()
    seen_actions = set()
    F = world.frame()
    for t in range(steps):
        a = int(rng.integers(5))
        pred = learner.predict(a)
        cpred = colour.predict(F, a)
        world.step(a)
        F2 = world.frame()
        ok = pred is not None and np.array_equal(pred, F2)
        rec["errors_all"] += not ok
        if t >= 1:
            rec["frames"] += 1
            rec["learner_ok"] += ok
            rec["colour_ok"] += np.array_equal(cpred, F2)
            rec["nochange_ok"] += np.array_equal(F, F2)
            if not ok:
                if a not in seen_actions:
                    rec["first_sightings"] += 1
                elif pred is not None:
                    rec["tie_errors"] += 1
                else:
                    rec["other_errors"] += 1
        learner.observe(F, a, F2)
        rec["pose_checked"] += 1
        rec["pose_ok"] += set(idx(apply(learner.pose, learner.body)).tolist()) == world.piece_cells()
        colour.observe(F, a, F2)
        seen_actions.add(a)
        F = F2
    return rec


def main(revise=False):
    t0 = time.time()
    trials = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        for _ in range(TRIALS):
            world = World(rng)
            learner = Learner(revise=revise)
            piece_before = world.piece_cells()
            rec = run_trial(world, learner, rng, STEPS)
            rec["final_model_exact"] = set(idx(apply(learner.pose, learner.body)).tolist()) == world.piece_cells()
            trials.append((rec, learner, piece_before))
    out = []                                               # the learner's body set is the set it discovered
    for rec, learner, piece_before in trials:
        rec["segmentation_exact"] = set(idx(learner.body).tolist()) == piece_before if not revise else None   # P1: the set discovered at the first transition
        rec["revisions"] = learner.revisions
        cls = learner.classes()
        rec["classes"] = cls
        rec["classes_right"] = all(cls[a] == "world" for a in range(4)) and cls[4] == "body"
        out.append(rec)
    # transfer: a new board and piece; carried rules vs a fresh learner
    transfer = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(100 + seed)
        for (rec, learner, _) in trials[seed * TRIALS:(seed + 1) * TRIALS]:
            carried = {}
            for a, c in learner.classes().items():
                if c in ("world", "both"):
                    carried[a] = ("world", learner.obs[a]["world"][0])
                elif c == "body":
                    carried[a] = ("body", None)
            state = rng.integers(1 << 30)
            res = {}
            for name, lrn in (("carried", Learner(carried, revise)), ("fresh", Learner(revise=revise))):
                r2 = np.random.default_rng(state)
                res[name] = run_trial(World(r2), lrn, r2, TRANSFER)["errors_all"]
            transfer.append(res)
    m = lambda k: float(np.mean([r[k] for r in out]))
    frames = sum(r["frames"] for r in out)
    summary = dict(
        segmentation_exact=None if revise else m("segmentation_exact"), final_model_exact=m("final_model_exact"),
        classes_right=m("classes_right"),
        learner_exact_frames=sum(r["learner_ok"] for r in out) / frames,
        colour_exact_frames=sum(r["colour_ok"] for r in out) / frames,
        nochange_exact_frames=sum(r["nochange_ok"] for r in out) / frames,
        first_sighting_errors_per_trial=m("first_sightings"), tie_errors_per_trial=m("tie_errors"),
        other_errors_per_trial=m("other_errors"), max_tie_errors=int(max(r["tie_errors"] for r in out)),
        pose_correct=sum(r["pose_ok"] for r in out) / sum(r["pose_checked"] for r in out),
        transfer_carried_errors=float(np.mean([t["carried"] for t in transfer])),
        transfer_fresh_errors=float(np.mean([t["fresh"] for t in transfer])),
        revisions_per_trial=m("revisions"))
    for k, v in summary.items():
        print(f"{k:34s} {v:.3f}" if isinstance(v, float) else f"{k:34s} {v}")   # None = not measured in this variant
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(dict(summary=summary, trials=[{k: v for k, v in r.items()} for r in out], transfer=transfer),
              open("runs/structure_piece_revise.json" if revise else "runs/structure_piece.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    import sys
    main(revise="--revise" in sys.argv)
