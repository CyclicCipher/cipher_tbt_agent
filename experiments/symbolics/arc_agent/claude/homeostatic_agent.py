"""A satisficing active-inference agent, version 2 (docs/AGENT.md, part 15; arc_agent/claude/HOMEOSTASIS.md).

Values are homeostatic variables with acceptable ranges, not a scalar utility. Progress is acceptable when some plan
reaches WIN with posterior probability >= ALPHA; understanding when no plan within the horizon has expected
information gain >= THETA bits. The agent holds explicit competing hypotheses: each frame's driver, each unmoved board
object's response to contact, and the cause of WIN. It plans over the posterior, and values a plan's information as
its exact expected information gain: every joint hypothesis predicts deterministically, so EIG = H(observations).
Version 1 (count-based novelty, part 14) is at commit f173b08.
    python -m arc_agent.claude.homeostatic_agent      (from experiments/symbolics)
"""
import json
import math
import time
from collections import Counter, deque
from itertools import product

import numpy as np

from arc_agent.claude.efference import Efference
from arc_agent.claude.inverse_model import nav, push_plan
from arc_agent.claude.push_block_rebuild import DIRS, EP_MAX, H, KEL, W, FrameAgent, PushWorld, compose, inverse, mode, rot
from arc_agent.claude.wall_world import WallWorld, solve

ALPHA, THETA, STEP_BITS, EPS = 0.5, 0.05, 0.02, 0.01
DEPTH, DEPTH_SAT, PMIN, MAXCOMBOS, DEEP_COMBOS, TOPK = 10, 30, 0.02, 12, 3, 5
HUNGER, GOAL_K = 25, 8                                                 # part 16: steps without a WIN at which hunger reaches 1
LEVELS, SEEDS, CAP = 8, 3, 1500
AFF = ("passable", "solid", "pushable")
LOG_R = math.log(EPS / (1 - EPS))
UNKNOWN_BITS = math.log2(KEL)                               # an action never seen: one of KEL displacements
WORLDS = (("W1 push_block", {}), ("W2 random layouts", dict(random_layout=True)),
          ("W3 random layouts + patroller", dict(random_layout=True, patroller=True)))


def inb(x, y):
    return 0 <= x < W and 0 <= y < H


def entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


def h_none(m):
    return -(1 - m) * math.log2(1 - m) if 0 < m < 1 else 0.0


def normalized(lw, prune=PMIN):
    mx = max(lw.values())
    p = {k: math.exp(v - mx) for k, v in lw.items()}
    z = sum(p.values())
    p = {k: v / z for k, v in p.items() if v / z >= prune}
    z = sum(p.values())
    return {k: v / z for k, v in p.items()}


# ---- things: discovered frames, and board objects that haven't moved yet ------------------------------------------
class Thing:
    def __init__(self, key, cells, frame, t):
        self.key, self.cells, self.frame, self.t = key, cells, frame, t
        x0, y0 = cells[0]
        self.shape = (tuple((x - x0, y - y0) for x, y in cells), key[1])   # what carries across levels

    def at(self, t):
        return [(x + t[0], y + t[1]) for x, y in self.cells]


def things(agent):
    """Frames first (thing index = frame index), then connected same-colour board objects not under a frame. A board
    object's key equals the key of the frame it becomes when it is first pushed from where it lies."""
    out, covered = [], set()
    for i, f in enumerate(agent.frames):
        th = Thing((tuple(map(tuple, f.body.tolist())), tuple(f.content.tolist())),
                   [tuple(c) for c in rot(f.body, f.pose[2]).tolist()], i, (f.pose[0], f.pose[1]))
        out.append(th)
        covered |= set(th.at(th.t))
    S, seen = agent.static, set()
    for y in range(H):
        for x in range(W):
            if (x, y) in seen or (x, y) in covered or S[y, x] <= 0:
                continue
            c, comp, q = S[y, x], [], [(x, y)]
            seen.add((x, y))
            while q:
                px, py = q.pop()
                comp.append((px, py))
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if inb(nx, ny) and (nx, ny) not in seen and (nx, ny) not in covered and S[ny, nx] == c:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            comp.sort(key=lambda p: (p[1], p[0]))
            out.append(Thing((tuple(comp), tuple(int(S[py, px]) for px, py in comp)), comp, None, (0, 0)))
    return out


# ---- beliefs: drivers, affordances, the cause of WIN ----------------------------------------------------------------
def family(m):
    return "time" if m.startswith("time") else "push" if m.startswith("push") else m


def driver_posterior(f):
    """Prior: a quarter per family (static, action, time, contact), split evenly inside it; times the evidence."""
    sizes = Counter(family(m) for m in f.evidence)
    return normalized({m: -math.log(len(sizes) * sizes[family(m)]) + float(v) for m, v in f.evidence.items()})


class Affordances:
    """How a board object that hasn't moved yet responds when the body moves into it; keyed by shape and colour."""
    def __init__(self):
        self.lw = {}

    def posterior(self, shape):
        return normalized(self.lw.get(shape, {a: 0.0 for a in AFF}))

    def observe(self, shape, outcome):
        w = self.lw.setdefault(shape, {a: 0.0 for a in AFF})
        for a in AFF:
            w[a] += math.log(1 - EPS) if a == outcome else math.log(EPS / 2)


class GoalModel:
    """What causes WIN, about kinds of thing (shape and colours), so that what is refuted in one level stays refuted in
    the next: cell(kind, c), some thing of the kind lies wholly on board colour c; at(kind, x, y), some thing of the
    kind has its first cell at (x, y). Prior: half per family, split evenly (description length). Evidence: the WIN
    signal at every step, against the union of what holds for all things. Hypotheses cover every kind and colour seen
    so far, present or not: a kind that is absent (say, hidden under another thing) makes none of its hypotheses hold,
    so they predict no WIN, and a WIN refutes them. By the same rule, a hypothesis first registered now (a new kind or
    colour) was false at every WIN so far, since its kind or colour wasn't there: it starts charged with those WINs
    (part 17, from a build check)."""
    def __init__(self, charge_new=False, prior_fn=None):
        self.err, self.kinds, self.cols = Counter(), {}, set()
        self.charge_new, self.wins, self.known = charge_new, 0, set()
        self.prior_fn = prior_fn                            # part 18: prior_fn(hypotheses, things, static) -> {h: p}

    def hypotheses(self, th, static):
        self.kinds.update(dict.fromkeys(t.shape for t in th))
        self.cols |= set(int(c) for c in np.unique(static) if c >= 0)
        cols = sorted(self.cols)
        hs = [("cell", k, c) for k in self.kinds for c in cols] + \
             [("at", k, x, y) for k in self.kinds for x in range(W) for y in range(H)]
        if self.charge_new and self.wins:
            for h in hs:
                if h not in self.known:
                    self.err[h] += self.wins
        self.known.update(hs)
        return hs

    @staticmethod
    def holds(t, trans, static):
        cells = t.at(trans)
        if not all(inb(x, y) for x, y in cells):
            return frozenset()
        out = {("at", t.shape, cells[0][0], cells[0][1])}
        cols = {int(static[y, x]) for x, y in cells}
        if len(cols) == 1 and min(cols) >= 0:
            out.add(("cell", t.shape, cols.pop()))
        return frozenset(out)

    def posterior(self, th, static):
        hs = self.hypotheses(th, static)
        if self.prior_fn is not None:
            pri = self.prior_fn(hs, th, static)
            return normalized({h: math.log(pri[h]) + LOG_R * self.err[h] for h in hs}, prune=0.0)
        n = Counter(h[0] for h in hs)
        return normalized({h: math.log(0.5 / n[h[0]]) + LOG_R * self.err[h] for h in hs}, prune=0.0)

    def observe(self, th, static, win):
        hold = set()
        for t in th:
            hold |= self.holds(t, t.t, static)
        for h in self.hypotheses(th, static):
            if (h in hold) != win:
                self.err[h] += 1
        self.wins += win


def belief_entropy(agent, goal, aff):
    th = things(agent)
    h = entropy(goal.posterior(th, agent.static).values())
    for t in th:
        h += entropy((driver_posterior(agent.frames[t.frame]) if t.frame is not None else aff.posterior(t.shape)).values())
    return h


# ---- the model combinations, and how one of them moves the world --------------------------------------------------
def fkey(f):
    return (tuple(map(tuple, f.body.tolist())), tuple(f.content.tolist()))


def action_table(f, eff=None):
    """What each action is meant to do to this body, and the value in bits of trying it when that is unknown. With an
    efference copy (part 16): its most probable displacement if that has probability >= 1 - PMIN, else unknown, worth
    its posterior's entropy. Without one (part 15b): the most frequent observed effect, blocked moves included."""
    table, bits = {}, {}
    for a in range(4):
        if eff is None:
            e = mode(f.act.get(a, Counter()))
            table[a], bits[a] = (None if e is None or e[2] != 0 else (e[0], e[1])), UNKNOWN_BITS
        else:
            d, p, h = eff.intended(fkey(f), a)
            table[a], bits[a] = (d if p >= 1 - PMIN else None), eff.posterior(fkey(f), a)
    return table, bits


def outcome_bits(post, thing, t, solid):
    """Part 18 fix: the information in trying an uncertain action is the entropy of its predicted *outcome* here, not
    of its displacement. At the edge, every outward displacement predicts the same outcome: blocked."""
    if not isinstance(post, dict):
        return post
    out = Counter()
    for d, p in post.items():
        cells = thing.at((t[0] + d[0], t[1] + d[1]))
        moved = d != (0, 0) and all(inb(x, y) for x, y in cells) and not solid & set(cells)
        out[d if moved else (0, 0)] += p
    return entropy(out.values())


def time_table(f, n):
    out = {}
    for ph, c in f.tim[n].items():
        e = mode(c)
        out[ph] = (e[0], e[1]) if e is not None and e[2] == 0 else (0, 0)
    return out


def role(f, m, eff=None):
    if m == "static":
        return ("fixed",)
    if m == "action":
        return ("body",) + action_table(f, eff)
    if m.startswith("time"):
        return ("time", int(m[4:]), time_table(f, int(m[4:])))
    return ("push", int(m.split(":")[1]))


def model(agent, aff, eff=None):
    """The things; the endogenous ones (in the search) with their combinations; the exogenous ones' future places."""
    th = things(agent)
    endo, opts, exo = [], [], {}
    for k, t in enumerate(th):
        if t.frame is not None:
            f = agent.frames[t.frame]
            dp = driver_posterior(f)
            if all(family(m) in ("static", "time") for m in dp):
                m, traj, pos = max(dp, key=dp.get), [t.t], t.t
                tab = time_table(f, int(m[4:])) if m.startswith("time") else {}
                for d in range(1, DEPTH_SAT + 2):
                    dx, dy = tab.get((agent.t + d) % int(m[4:]), (0, 0)) if tab else (0, 0)
                    pos = (pos[0] + dx, pos[1] + dy)
                    traj.append(pos)
                exo[k] = traj
                continue
            endo.append(k)
            opts.append([(role(f, m, eff), p) for m, p in dp.items()])
        else:
            endo.append(k)
            opts.append([((a,), p) for a, p in aff.posterior(t.shape).items()])
    if math.prod(len(o) for o in opts) <= MAXCOMBOS:       # the full product
        raw = [(math.prod(c[1] for c in ch), tuple(c[0] for c in ch)) for ch in product(*opts)]
    else:                                                   # part 16: one at a time around the most probable combination
        base = [max(o, key=lambda c: c[1]) for o in opts]
        raw = [(math.prod(c[1] for c in base), tuple(c[0] for c in base))]
        for i, o in enumerate(opts):
            for c in o:
                if c is not base[i]:
                    ch = base[:i] + [c] + base[i + 1:]
                    raw.append((math.prod(x[1] for x in ch), tuple(x[0] for x in ch)))
    combos = sorted(raw, key=lambda c: -c[0])[:MAXCOMBOS]
    z = sum(c[0] for c in combos)
    where = {th[k].frame: i for i, k in enumerate(endo) if th[k].frame is not None}
    fixed = []
    for p, roles in combos:
        rr = tuple(("push", where.get(r[1])) if r[0] == "push" else r for r in roles)
        fixed.append((p / z, rr))
    return th, endo, fixed, exo


def sim(ths, roles, state, a, s):
    """One step under one model combination; None if a body's action has never been seen."""
    new, moved, solid = list(state), {}, set()
    for i, r in enumerate(roles):
        if r[0] == "solid":
            solid |= set(ths[i].at(state[i]))
    bodies = [i for i, r in enumerate(roles) if r[0] == "body"]
    for i in bodies:
        d = roles[i][1].get(a)
        if d is None:
            return None
        if d == (0, 0):
            continue
        nt = (state[i][0] + d[0], state[i][1] + d[1])
        cells = ths[i].at(nt)
        if all(inb(x, y) for x, y in cells) and not solid & set(cells):
            new[i], moved[i] = nt, d
    for i, r in enumerate(roles):
        if r[0] == "time":
            dx, dy = r[2].get(s % r[1], (0, 0))
            nt = (state[i][0] + dx, state[i][1] + dy)
            if all(inb(x, y) for x, y in ths[i].at(nt)):
                new[i] = nt
        elif r[0] in ("push", "pushable"):
            pushers = ([r[1]] if r[1] is not None else []) if r[0] == "push" else bodies
            mine = set(ths[i].at(state[i]))
            for j in pushers:
                if j in moved and set(ths[j].at(new[j])) & mine:
                    d = moved[j]
                    nt = (state[i][0] + d[0], state[i][1] + d[1])
                    cells = ths[i].at(nt)
                    if all(inb(x, y) for x, y in cells) and not solid & set(cells):
                        new[i] = nt
                    else:
                        new[j] = state[j]                   # solidity: the pusher stays too
                    break
    return tuple(new)


# ---- planning over the posterior --------------------------------------------------------------------------------------
def search(agent, th, endo, combos, exo, post, depth):
    """Breadth-first over action sequences. Each node: one state per combination, the partition of combinations into
    trajectory classes, and per class the goal hypotheses that have held and the running entropy of the first-WIN
    step. Returns (depth, path, P(WIN along the path), expected information gain in bits) for every node."""
    ths = [th[k] for k in endo]
    cache = {}

    def holds(k, trans):
        if (k, trans) not in cache:
            cache[(k, trans)] = frozenset(h for h in GoalModel.holds(th[k], trans, agent.static) if post.get(h, 0) > 1e-12)
        return cache[(k, trans)]

    def holds_state(state, d):
        out = set()
        for i, k in enumerate(endo):
            out |= holds(k, state[i])
        for k, traj in exo.items():
            out |= holds(k, traj[min(d, len(traj) - 1)])
        return frozenset(out)

    def mass(hs):
        return math.fsum(post[h] for h in hs)               # exactly rounded: independent of the set's hash order

    start = tuple(th[k].t for k in endo)
    states0, part0 = tuple(start for _ in combos), tuple(0 for _ in combos)
    q = deque([(states0, part0, {0: holds_state(start, 0)}, {0: 0.0}, [], 0.0)])
    seen, nodes = {(states0, part0)}, []
    pc = [p for p, _ in combos]
    while q:
        states, part, held, hk, path, unk = q.popleft()
        d = len(path)
        cls = Counter()
        for i, c in enumerate(part):
            cls[c] += pc[i]
        pwin = sum(p * mass(held[c]) for c, p in cls.items())
        eig = entropy(cls.values()) + sum(p * (hk[c] + h_none(mass(held[c]))) for c, p in cls.items()) + unk
        nodes.append((d, path, pwin, eig))
        if unk or d >= depth:
            continue
        for a in range(4):
            nxt, unknown = [], False
            ubits = 0.0
            for i, (_, roles) in enumerate(combos):
                s2 = sim(ths, roles, states[i], a, agent.t + d + 1)
                if s2 is None:                              # an action whose effect is unknown: worth its outcome's entropy
                    unknown, s2 = True, states[i]
                    solid = set()
                    for j, r in enumerate(roles):
                        if r[0] == "solid":
                            solid |= set(ths[j].at(states[i][j]))
                    ubits = max([ubits] + [outcome_bits(r[2][a], ths[j], states[i][j], solid)
                                           for j, r in enumerate(roles) if r[0] == "body" and r[1].get(a) is None])
                nxt.append(s2)
            canon, npart = {}, []
            for i in range(len(combos)):
                npart.append(canon.setdefault((part[i], nxt[i]), i))
            nxt, npart = tuple(nxt), tuple(npart)
            if not unknown:                                 # an untried action is a leaf of its own, never a duplicate
                if (nxt, npart) in seen:
                    continue
                seen.add((nxt, npart))
            nheld, nhk = {}, {}
            for (parent, s2), c in canon.items():
                hs = holds_state(s2, d + 1)
                m = mass(hs - held[parent])
                nheld[c] = held[parent] | hs
                nhk[c] = hk[parent] + (-m * math.log2(m) if m > 0 else 0.0)
            q.append((nxt, npart, nheld, nhk, path + [a], ubits if unknown else 0.0))
    return nodes


def win_probability(agent, th, endo, combos, exo, post, path):
    """P(WIN somewhere along a fixed plan), averaged over the model combinations."""
    ths, total = [th[k] for k in endo], 0.0
    for p, roles in combos:
        state, held = tuple(th[k].t for k in endo), set()
        for d, a in enumerate(path):
            nxt = sim(ths, roles, state, a, agent.t + d + 1)
            state = state if nxt is None else nxt
            for i, k in enumerate(endo):
                held |= GoalModel.holds(th[k], state[i], agent.static)
            for k, traj in exo.items():
                held |= GoalModel.holds(th[k], traj[min(d + 1, len(traj) - 1)], agent.static)
        total += p * math.fsum(post.get(h, 0.0) for h in held)
    return total


def goal_plans(agent, th, endo, combos, post):
    """Part 17: plans read off the inverse model for the GOAL_K most probable goal hypotheses, under the most probable
    model. None if the model has more than one body, or a body or pushed object of more than one cell."""
    roles, ths = combos[0][1], [th[k] for k in endo]
    bodies = [i for i, r in enumerate(roles) if r[0] == "body"]
    pushed = [i for i, r in enumerate(roles) if r[0] in ("push", "pushable")]
    if len(bodies) != 1 or any(len(ths[i].cells) > 1 for i in bodies + pushed):
        return None
    b = bodies[0]
    table, solid, objects = roles[b][1], set(), {}
    for i, r in enumerate(roles):
        if r[0] == "solid":
            solid |= set(ths[i].at(ths[i].t))
    for i in pushed:
        objects[i] = set(ths[i].at(ths[i].t))
    plans = []
    for h, p in sorted(post.items(), key=lambda kv: -kv[1])[:GOAL_K]:
        if p < 1e-9:
            break
        for i, t in enumerate(ths):
            if t.shape != h[1] or (i != b and i not in pushed):
                continue
            ok = lambda tt, t=t, h=h: h in GoalModel.holds(t, tt, agent.static)
            if i == b:
                block = solid.union(*objects.values())
                route = nav(t.t, t.cells, table, lambda x, y: inb(x, y) and (x, y) not in block, ok)
            else:
                rest = solid.union(*(c for j, c in objects.items() if j != i))
                free = lambda x, y, rest=rest: inb(x, y) and (x, y) not in rest
                route = push_plan(ths[b].cells, ths[b].t, t.cells, t.t, table, free, free, ok)
            if route:
                plans.append(route[:DEPTH_SAT])
    return plans


def beyond_horizon(agent, th, endo, combos, exo, post, nodes, macro=False):
    """Candidates from the 10-step search and, under each of the top combinations, from a 30-step search (the shortest
    acceptable plan and the plans most likely to win), scored under the whole posterior: [(-P(WIN), length, path)]."""
    best_first = lambda ns: sorted((n for n in ns if n[0] > 0 and n[2] > 1e-6), key=lambda n: (-n[2], n[0]))[:TOPK]
    cands = {tuple(n[1]) for n in best_first(nodes)}
    plans = goal_plans(agent, th, endo, combos, post) if macro else None
    if plans is not None:                                   # part 17: the inverse model instead of joint searches
        cands |= {tuple(p) for p in plans}
    for c in (combos[:DEEP_COMBOS] if plans is None else []):
        deep = search(agent, th, endo, [(1.0, c[1])], exo, post, DEPTH_SAT)
        acc = [n for n in deep if n[0] > 0 and n[2] >= ALPHA]
        if acc:
            cands.add(tuple(min(acc, key=lambda n: (n[0], -n[2]))[1]))
        cands |= {tuple(n[1]) for n in best_first(deep)}
    return sorted((-win_probability(agent, th, endo, combos, exo, post, list(p)), len(p), p) for p in cands)


def drive(c, since, u0, common=False):
    """Part 16: the deviation-weighted drive after a plan (Keramati & Gutkin's form, n = 3, m = 2). Progress: zero in
    its dead zone (P(WIN) >= ALPHA), else the unmet chance of winning times hunger, which grows with the steps since
    the last WIN. Understanding: zero once the reachable information is below THETA, else what the plan leaves of it."""
    d, _, pwin, eig = c
    h = (since + (DEPTH_SAT if common else d)) / HUNGER        # part 17: every plan judged at the same moment
    dp = 0.0 if pwin >= ALPHA else (1 - pwin) * h
    du = 0.0 if u0 < THETA else max(0.0, u0 - eig - THETA) / u0      # amended before the run: relative to what is reachable
    de = d / EP_MAX if common else 0.0                         # part 17: effort, the share of the budget used
    return (dp ** 3 + du ** 3 + de ** 3) ** 0.5


def plan(agent, goal, aff, epistemic=True, eff=None, since=None, macro=False, common=False, out=None):
    """Returns (path, kind, predicted information gain). since=None: the lexicographic order (part 15b); otherwise the
    deviation-weighted drive, with since = steps since the last WIN (part 16)."""
    th, endo, combos, exo = model(agent, aff, eff)
    if not any(r[0] == "body" for _, roles in combos for r in roles):
        return None, "random", 0.0
    post = goal.posterior(th, agent.static)
    nodes = search(agent, th, endo, combos, exo, post, DEPTH)
    if out is not None:                                     # part 17: what commitment needs to watch
        out.update(th=th, endo=endo, combos=combos, exo=exo, post=post)
    if since is not None:                                   # deviation-weighted: the plan that leaves the least drive
        u0 = max(n[3] for n in nodes)
        cands = [n for n in nodes if n[0] > 0] +                 [(len(p), list(p), -npw, 0.0) for npw, _, p in beyond_horizon(agent, th, endo, combos, exo, post, nodes, macro)]
        if not cands:
            return None, "random", 0.0
        n = min(cands, key=lambda c: (round(drive(c, since, u0, common), 12), c[0]))
        kind = "satisfice" if n[2] >= ALPHA else "explore" if n[3] >= THETA else "approach" if n[2] > 1e-6 else None
        return (list(n[1]), kind, n[3]) if kind else (None, "random", 0.0)
    acc = [n for n in nodes if n[0] > 0 and n[2] >= ALPHA]
    if acc:                                                 # progress: the fewest-action acceptable plan
        n = min(acc, key=lambda n: (n[0], -n[2]))
        return list(n[1]), "satisfice", n[3]
    if epistemic:                                           # understanding: the most informative plan, if worth it
        n = max(nodes, key=lambda n: (n[3] - STEP_BITS * n[0], -n[0]))
        if n[0] > 0 and n[3] - STEP_BITS * n[0] >= THETA:
            return list(n[1]), "explore", n[3]
    scored = beyond_horizon(agent, th, endo, combos, exo, post, nodes, macro)
    acc = [x for x in scored if -x[0] >= ALPHA]
    if acc:                                                 # progress beyond the horizon: the shortest acceptable plan
        _, _, path = min(acc, key=lambda x: (x[1], x[0]))
        return list(path), "satisfice (deep)", 0.0
    if scored and -scored[0][0] > 1e-6:                     # otherwise the plan most likely to win, the shorter on ties
        return list(scored[0][2]), "approach", 0.0
    return None, "random", 0.0


# ---- contact outcomes: what the body learned about a board object it moved into --------------------------------------
def bodies_before(agent, a, eff=None):
    out = []
    for i, f in enumerate(agent.frames):
        if agent.choice(i) == "action":
            d = action_table(f, eff)[0].get(a)
            if d and d != (0, 0):
                out.append((i, (f.pose[0], f.pose[1]), d))
    return out


def contact_update(agent, aff, th_before, nframes, bodies):
    blobs = [t for t in th_before if t.frame is None]
    for t in th_before:                                     # any frame that moved onto a board object passed over it
        if t.frame is None or t.frame >= len(agent.frames):
            continue
        f = agent.frames[t.frame]
        now = (f.pose[0], f.pose[1])
        if now != t.t:
            cells = {(x + now[0], y + now[1]) for x, y in rot(f.body, f.pose[2]).tolist()}
            for b in blobs:
                if cells & set(b.cells):
                    aff.observe(b.shape, "passable")
    for i, t0, d in bodies:
        if i >= len(agent.frames):
            continue
        f = agent.frames[i]
        dest = {(x + t0[0] + d[0], y + t0[1] + d[1]) for x, y in rot(f.body, f.pose[2]).tolist()}
        now = (f.pose[0], f.pose[1])
        for b in blobs:
            if not dest & set(b.cells):
                continue
            if any(set(map(tuple, g.body.tolist())) & set(b.cells) for g in agent.frames[nframes:]):
                aff.observe(b.shape, "pushable")
            elif now == t0 and all(inb(x, y) for x, y in dest):
                aff.observe(b.shape, "solid")


# ---- levels in the true world -----------------------------------------------------------------------------------------
def restart(world, layout):
    world.player, world.box, world.target = layout
    world.t, world.pat_x = 0, 1
    return world.frame()


def optimum(world):
    return solve(world.player, world.box, world.target, getattr(world, "walls", set()))


def trajectory(info, path, t0):
    """Part 17: the plan's predicted positions of what it moves (body and pushed objects), under the most probable model."""
    th, endo, combos = info["th"], info["endo"], info["combos"]
    roles, ths = combos[0][1], [th[k] for k in endo]
    movers = [i for i, r in enumerate(roles) if r[0] in ("body", "push", "pushable")]
    state, out = tuple(th[k].t for k in endo), []
    for d, a in enumerate(path):
        nxt = sim(ths, roles, state, a, t0 + d + 1)
        state = state if nxt is None else nxt
        out.append({ths[i].key: state[i] for i in movers})
    return out


def run(kw, seed, epistemic=True, eff_on=False, drives_on=False, walls=False, commit=False, macro=False, common=False,
        world=None, goal_factory=None, levels=None, cap=None, probe=None):
    rng = np.random.default_rng(seed)
    if world is None:
        world = WallWorld(np.random.default_rng(100 + seed)) if walls else PushWorld(np.random.default_rng(100 + seed), **kw)
    agent, aff = FrameAgent(score_discovery=True), Affordances()
    goal = goal_factory(agent, aff) if goal_factory else GoalModel(charge_new=commit)
    n_levels, n_cap = levels or LEVELS, cap or CAP
    snapshot = getattr(world, "snapshot", lambda: (world.player, world.box, world.target))
    restore = getattr(world, "restore", lambda lay: restart(world, lay))
    best = getattr(world, "optimum", lambda: optimum(world))
    eff = Efference() if eff_on else None
    F = world.reset()
    layout = snapshot()
    agent.start(F)
    level, total, lsteps, attempts, path, kind, since = 0, 0, 0, 0, [], None, 0
    first_win, levels, opt, decisions, gains, leads, pending = None, [], best(), [], [], [], None
    kinds_until_win = []
    deco_contacts, wall_bumps, wall_solid = 0, 0, None
    traj, pwin0, lengths, triggers, t_start = [], 0.0, [], Counter(), time.time()
    while level < n_levels and total < n_cap:
        if not path:
            if pending is not None:
                gains.append((pending[0], pending[1] - belief_entropy(agent, goal, aff)))
                pending = None
            info_out = {} if commit else None
            path, kind, info = plan(agent, goal, aff, epistemic, eff, since if drives_on else None, macro, common, info_out)
            path = path or []
            if kind == "explore":
                pending = (info, belief_entropy(agent, goal, aff))
            if not path:
                path, kind = [int(rng.integers(4))], "random"
            decisions.append((level, kind, since))
            lengths.append(len(path))
            if commit and info_out:
                traj = trajectory(info_out, path, agent.t)
                pwin0 = win_probability(agent, info_out["th"], info_out["endo"], info_out["combos"], info_out["exo"],
                                        info_out["post"], path)
            else:
                traj = []
        a = path.pop(0)
        if first_win is None:
            kinds_until_win.append(kind)
        th_before, nframes, bodies = things(agent), len(agent.frames), bodies_before(agent, a, eff)
        if eff is not None:                                 # the efference copy: each body's state before the step
            occupied, static_before, before = set(), agent.static.copy(), []
            for f in agent.frames:
                occupied |= {(x + f.pose[0], y + f.pose[1]) for x, y in rot(f.body, f.pose[2]).tolist()}
            for i, f in enumerate(agent.frames):
                if agent.choice(i) == "action":
                    cells = [tuple(c) for c in rot(f.body, f.pose[2]).tolist()]
                    own = {(x + f.pose[0], y + f.pose[1]) for x, y in cells}
                    before.append((i, fkey(f), (f.pose[0], f.pose[1]), f.pose, cells, own))
        pred = agent.predict(a)
        p0 = world.player
        F2, win = world.step(a)
        total, lsteps, since = total + 1, lsteps + 1, since + 1
        if walls:                                           # measures only (the agent never sees these)
            deco_contacts += world.player != p0 and world.player in world.deco
            dest = (p0[0] + DIRS[a][0], p0[1] + DIRS[a][1])
            wall_bumps += level == 0 and dest in world.walls
        agent.observe(F, a, F2, world.t)
        if eff is not None:
            for i, k, t0, pose0, cells, own in before:
                if i < len(agent.frames):
                    e = compose(agent.frames[i].pose, inverse(pose0))
                    free = lambda x, y, own=own: inb(x, y) and static_before[y, x] == 0 and ((x, y) not in occupied or (x, y) in own)
                    eff.observe(k, a, t0, e, cells, free)
        contact_update(agent, aff, th_before, nframes, bodies)
        goal.observe(things(agent), agent.static, win)
        if probe is not None:
            probe(agent, goal, aff, win)
        for f in agent.frames[nframes:]:                     # E3: the box's driver straight after its discovery
            pushes = [v for m, v in f.evidence.items() if m.startswith("push")]
            if 4 in f.content.tolist() and pushes:
                leads.append(float(max(pushes) - f.evidence["static"]))
        if not commit:
            if pred is None or not np.array_equal(pred, F2):
                path = []
        elif path:                                          # part 17: keep the plan unless a decision point arrives
            expected = traj.pop(0) if traj else {}
            where = {t.key: t.t for t in things(agent)}
            if any(where.get(k) != v for k, v in expected.items()):
                path, _ = [], triggers.update(["surprise"])
            elif kind in ("satisfice", "satisfice (deep)", "approach"):
                th_, endo_, combos_, exo_ = model(agent, aff, eff)
                pw = win_probability(agent, th_, endo_, combos_, exo_, goal.posterior(th_, agent.static), path)
                if pw < pwin0 - 0.1 or (pwin0 >= ALPHA and pw < ALPHA):
                    path, _ = [], triggers.update(["value"])
        F = F2
        if win:
            if first_win is None:
                first_win = total
            if walls and level == 0:                        # Q2: the wall's solidity at the end of the first level
                wc = sorted(world.walls, key=lambda c: (c[1], c[0]))
                shape = (tuple((x - wc[0][0], y - wc[0][1]) for x, y in wc), tuple(5 for _ in wc))
                wall_solid = aff.posterior(shape).get("solid", 0.0)
            levels.append(dict(level=level, steps=lsteps, optimum=opt, first_attempt=attempts == 0, deco=deco_contacts))
            level, lsteps, attempts, path, since, deco_contacts = level + 1, 0, 0, [], 0, 0
            F = world.reset()
            layout, opt = snapshot(), best()
            agent.start(F)
        elif world.t >= EP_MAX:
            attempts, path = attempts + 1, []
            F = restore(layout)
            agent.start(F)
    wrong = []                                              # Q1: the body's belief about each action, at the end
    for i, f in enumerate(agent.frames):
        if agent.choice(i) == "action":
            table, _ = action_table(f, eff)
            if eff is not None:
                wrong += [a for a in range(4) if eff.intended(fkey(f), a)[0] != DIRS[a] or eff.intended(fkey(f), a)[1] < 1 - PMIN]
            else:
                wrong += [a for a in range(4) if table[a] != DIRS[a]]
    return dict(first_win=first_win, levels=levels, total_steps=total, decisions=decisions, gains=gains,
                box_discovery_leads=leads, wrong_actions=wrong, wall_bumps_level1=wall_bumps, wall_solid=wall_solid,
                plan_lengths=lengths, triggers=dict(triggers), seconds=time.time() - t_start, kinds_until_win=kinds_until_win)


def main():
    t0 = time.time()
    out = {}
    for name, kw in WORLDS:
        for agent_kind, epistemic in (("version 2", True), ("version 2 without EIG", False)):
            runs = [run(kw, seed, epistemic) for seed in range(SEEDS)]
            out[f"{name} / {agent_kind}"] = runs
            later = [l for r in runs for l in r["levels"] if l["level"] >= 1]
            l3 = [l for r in runs for l in r["levels"] if l["level"] >= 2]
            dec3 = [k for r in runs for lv, k, _ in r["decisions"] if lv >= 2]
            gains = [g for r in runs for g in r["gains"]]
            print(f"{name:30s} {agent_kind:22s} first WIN at {[r['first_win'] for r in runs]} | levels done "
                  f"{[len(r['levels']) for r in runs]} | levels 2-8 first attempt "
                  f"{np.mean([l['first_attempt'] for l in later]) if later else float('nan'):.2f} | level 3+ excess "
                  f"{np.mean([l['steps'] - l['optimum'] for l in l3]) if l3 else float('nan'):.2f} | explore share in levels 3+ "
                  f"{np.mean([k == 'explore' for k in dec3]) if dec3 else float('nan'):.2f} | decisions "
                  f"{dict(Counter(k for r in runs for _, k, _ in r['decisions']))} | box lead after discovery "
                  f"{[round(x, 1) for r in runs for x in r['box_discovery_leads']]} | realized gain > 0 "
                  f"{np.mean([g[1] > 0 for g in gains]) if gains else float('nan'):.2f} of {len(gains)} | {time.time() - t0:.0f} s",
                  flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
    json.dump(out, open("runs/homeostatic_agent_v2.json", "w"), indent=1, default=str)


def part16():
    t0 = time.time()
    out = {}
    global LEVELS, CAP
    arms = (("L", False, False), ("LE", True, False), ("DW", False, True), ("DE", True, True))
    w4_arms = (("L", False, False), ("LE", True, False), ("DE", True, True), ("DW", False, True))
    worlds = WORLDS + (("W4 walls + distractors", None),)
    for name, kw in worlds:
        if kw is None:                                      # amendment 3: W4 has 5 levels and a 300-step cap, all arms
            LEVELS, CAP = 5, 300
        for arm, eff_on, drives_on in (w4_arms if kw is None else arms):
            runs = [run(kw or {}, seed, True, eff_on, drives_on, walls=kw is None) for seed in range(SEEDS)]
            out[f"{name} / {arm}"] = runs
            later = [l for r in runs for l in r["levels"] if l["level"] >= 1]
            l3 = [l for r in runs for l in r["levels"] if l["level"] >= 2]
            dec = [(k, since) for r in runs for _, k, since in r["decisions"]]
            early = [k == "explore" for k, since in dec if since < 20]
            late = [k == "explore" for k, since in dec if since >= 20]
            print(f"{name:30s} {arm:3s} first WIN {[r['first_win'] for r in runs]} | levels {[len(r['levels']) for r in runs]} | "
                  f"2-8 first attempt {np.mean([l['first_attempt'] for l in later]) if later else float('nan'):.2f} | "
                  f"3+ excess {np.mean([l['steps'] - l['optimum'] for l in l3]) if l3 else float('nan'):.2f} | "
                  f"2-8 mean steps {np.mean([l['steps'] for l in later]) if later else float('nan'):.1f}, "
                  f"distractor contacts/level {np.mean([l['deco'] for l in later]) if later else float('nan'):.2f} | "
                  f"explore share early {np.mean(early) if early else float('nan'):.2f} late {np.mean(late) if late else float('nan'):.2f} | "
                  f"wrong actions {[r['wrong_actions'] for r in runs]} | wall bumps L1 {[r['wall_bumps_level1'] for r in runs]}, "
                  f"solid {[None if r['wall_solid'] is None else round(r['wall_solid'], 3) for r in runs]} | {time.time() - t0:.0f} s",
                  flush=True)
    print(f"total {time.time() - t0:.1f} s")
    json.dump(out, open("runs/homeostatic_agent_part16.json", "w"), indent=1, default=str)


def part17():
    global LEVELS, CAP
    t0 = time.time()
    out = {}
    arms = (("LE17", False), ("DE17", True))
    for name, kw in WORLDS + (("W4 walls + distractors", None),):
        LEVELS, CAP = (5, 300) if kw is None else (8, 1500)
        for arm, drives_on in arms:
            runs = [run(kw or {}, seed, True, True, drives_on, walls=kw is None, commit=True, macro=True, common=drives_on)
                    for seed in range(SEEDS)]
            out[f"{name} / {arm}"] = runs
            later = [l for r in runs for l in r["levels"] if l["level"] >= 1]
            l3 = [l for r in runs for l in r["levels"] if l["level"] >= 2]
            steps = sum(r["total_steps"] for r in runs)
            print(f"{name:30s} {arm:4s} first WIN {[r['first_win'] for r in runs]} | levels {[len(r['levels']) for r in runs]} | "
                  f"2+ first attempt {np.mean([l['first_attempt'] for l in later]) if later else float('nan'):.2f} | "
                  f"3+ excess {np.mean([l['steps'] - l['optimum'] for l in l3]) if l3 else float('nan'):.2f} | "
                  f"decisions/step {sum(len(r['decisions']) for r in runs) / steps:.2f} | "
                  f"ms/step {1000 * sum(r['seconds'] for r in runs) / steps:.0f} | "
                  f"median plan length {np.median([x for r in runs for x in r['plan_lengths']]):.0f} | "
                  f"triggers {dict(sum((Counter(r['triggers']) for r in runs), Counter()))} | "
                  f"decisions {dict(Counter(k for r in runs for _, k, _ in r['decisions']))} | {time.time() - t0:.0f} s", flush=True)
    print(f"total {time.time() - t0:.1f} s")
    json.dump(out, open("runs/homeostatic_agent_part17.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    import sys
    part17() if "--part17" in sys.argv else part16() if "--part16" in sys.argv else main()
