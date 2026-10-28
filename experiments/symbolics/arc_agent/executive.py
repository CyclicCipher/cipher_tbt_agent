"""The executive (ARCHITECTURE.md section 6), a stand-in for the PFC: goal hypotheses, search over the brain's model,
and the choice of plan. Part 15b's lexicographic satisficing agent with part 17's commitment.

Goal hypotheses are about kinds (a column's content signature), so evidence carries across levels:
- ("on", kind, c): some column of this kind lies wholly on scene colour c;
- ("at", kind, state): some column of this kind is in this state;
- ("joint", states): the columns are in this joint state, a conjunction over columns (part 21: a level's WIN usually
  needs several columns in particular states at once, so visiting one column's state with the others wrong must not
  refute it).
Prior (part 22): each hypothesis has a type that carries across games ("on-rare": an "on" whose colour covers at most
RARE x the column's size on the screen, such as a marker, exit or slot; "on-common"; "at"; "joint"). A hypothesis's
prior is P(type) / (the number of hypotheses of that type); "at" is spread over AT_STATES states per kind and "joint"
over JOINT_STATES. P(type) is an overhypothesis (overhypothesis.py): at every WIN, the type of the most probable
unrefuted hypothesis that held, scored with even type priors, is one draw. Warm-up play on other games supplies the
first draws.

The choice of plan (part 22, part 17's DE17): the plan that leaves the least drive D = (dp^3 + du^3 + de^3)^(1/2):
dp = (1 - P(WIN along it)) x hunger (0 once P(WIN) >= ALPHA), hunger = (actions since the level started or the last
WIN + 30) / 25; du = the share of the reachable information it leaves unlearned; de = its length / 50. A hypothesis that holds without a WIN gains an error; at a WIN, every hypothesis that
didn't hold gains one. Hypotheses first registered after WINs start charged with them (part 17).
"""
import math
import time
from collections import deque

import numpy as np

from arc_agent.frames import H, W
from arc_agent.overhypothesis import Overhypothesis

ALPHA, THETA, STEP_BITS = 0.5, 0.05, 0.02
DEPTH, MAX_NODES, AT_STATES, JOINT_STATES = 60, 3000, 4096, 10 ** 6   # the node cap bounds the search
BUDGET = 0.3                                            # seconds per search: an anytime search, the best so far
LOG_EPS = math.log(0.01)
TYPES, RARE, EFFORT = ("on-rare", "on-common", "at", "joint"), 4, 50
FIXED_HUNGER = None                                     # ablation: a constant hunger instead of a growing one


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class Goals:
    def __init__(self, type_draws=()):
        self.over = Overhypothesis()                    # the prior over goal types (part 22)
        for t in type_draws:                            # warm-up: WIN types from other games
            self.over.add("goal", t)
        self.win_types = []                             # this game's draws
        self.on_type = {}                               # (kind, colour) -> "on-rare" | "on-common"
        self.on = {}                                    # (kind, colour) -> errors
        self.at = {}                                    # kind -> {state: errors}
        self.at_rest = {}                               # kind -> errors shared by every state never named
        self.joint, self.joint_rest = {}, 0             # joint state -> errors; errors of every joint state never named
        self.wins = 0

    def register(self, kinds, colours, static=None):
        if static is not None:                          # an "on" is rare where its colour covers little of the screen
            counts = dict(zip(*np.unique(static, return_counts=True)))
            for k in kinds:
                for c in colours:
                    self.on_type[(k, c)] = "on-rare" if counts.get(c, 0) <= RARE * len(k[1]) else "on-common"
        for k in kinds:
            if k not in self.at:
                self.at[k], self.at_rest[k] = {}, self.wins
            for c in colours:
                if (k, c) not in self.on:
                    self.on[(k, c)] = self.wins

    def type_prior(self, even=False):
        return {t: 1 / len(TYPES) for t in TYPES} if even else self.over.prior("goal", TYPES)

    def _weights(self, even=False):
        P = self.type_prior(even)
        n_type = {t: max(1, sum(1 for key in self.on if self.on_type.get(key, "on-common") == t)) for t in TYPES[:2]}
        n_k = max(len(self.at), 1)
        w_on = {key: math.log(P[self.on_type.get(key, "on-common")] / n_type[self.on_type.get(key, "on-common")])
                + LOG_EPS * e for key, e in self.on.items()}
        base = {k: math.log(P["at"] / n_k / AT_STATES) for k in self.at}
        self._bj = math.log(P["joint"] / JOINT_STATES)
        return w_on, base

    def mass(self, held):
        """Posterior probability that the WIN is caused by one of the hypotheses in `held`."""
        w_on, base = self._weights()
        bj = self._bj
        m = max(list(w_on.values()) + [b + LOG_EPS * self.at_rest[k] for k, b in base.items()] +
                [bj + LOG_EPS * self.joint_rest])
        z, num = 0.0, 0.0
        z += (JOINT_STATES - len(self.joint)) * math.exp(bj + LOG_EPS * self.joint_rest - m)
        for s, e in self.joint.items():
            z += math.exp(bj + LOG_EPS * e - m)
        for h in held:
            if h[0] == "joint":
                num += math.exp(bj + LOG_EPS * self.joint.get(h[1], self.joint_rest) - m)
        for key, w in w_on.items():
            v = math.exp(w - m)
            z += v
            if ("on",) + key in held:
                num += v
        for k, b in base.items():
            named = self.at[k]
            z += (AT_STATES - len(named)) * math.exp(b + LOG_EPS * self.at_rest[k] - m)
            for s, e in named.items():
                v = math.exp(b + LOG_EPS * e - m)
                z += v
            for h in held:
                if h[0] == "at" and h[1] == k:
                    e = named.get(h[2], self.at_rest[k])
                    num += math.exp(b + LOG_EPS * e - m)
        return num / z if z > 0 else 0.0

    def no_win(self, held):
        for h in held:
            if h[0] == "on" and (h[1], h[2]) in self.on:
                self.on[(h[1], h[2])] += 1
            elif h[0] == "at" and h[1] in self.at:
                self.at[h[1]][h[2]] = self.at[h[1]].get(h[2], self.at_rest[h[1]]) + 1
            elif h[0] == "joint":
                self.joint[h[1]] = self.joint.get(h[1], self.joint_rest) + 1

    def win_type(self, held):
        """The type of the most probable unrefuted hypothesis that held at the WIN, scored with even type priors."""
        w_on, base = self._weights(even=True)
        best = None
        for h in held:
            if h[0] == "on" and (h[1], h[2]) in w_on:
                w, t = w_on[(h[1], h[2])], self.on_type.get((h[1], h[2]), "on-common")
            elif h[0] == "at" and h[1] in base:
                w, t = base[h[1]] + LOG_EPS * self.at[h[1]].get(h[2], self.at_rest[h[1]]), "at"
            elif h[0] == "joint":
                w, t = self._bj + LOG_EPS * self.joint.get(h[1], self.joint_rest), "joint"
            else:
                continue
            if best is None or w > best[0]:
                best = (w, t)
        return None if best is None else best[1]

    def win(self, held):
        t = self.win_type(held)
        if t is not None:                               # one draw for the overhypothesis over goal types
            self.over.add("goal", t)
            self.win_types.append(t)
        for key in self.on:
            if ("on",) + key not in held:
                self.on[key] += 1
        for k in self.at:
            keep = {h[2] for h in held if h[0] == "at" and h[1] == k}
            for s in list(self.at[k]):
                if s not in keep:
                    self.at[k][s] += 1
            for s in keep:
                self.at[k].setdefault(s, self.at_rest[k])
            self.at_rest[k] += 1
        keep = {h[1] for h in held if h[0] == "joint"}
        for s in list(self.joint):
            if s not in keep:
                self.joint[s] += 1
        for s in keep:
            self.joint.setdefault(s, self.joint_rest)
        self.joint_rest += 1
        self.wins += 1


class Executive:
    def __init__(self, brain, type_draws=()):
        self.brain = brain
        self.goals = Goals(type_draws)
        self.since = 0                                  # actions since the level started or the last WIN
        self.plan = []                                  # [(action, expected joint state after it)]
        self.kind_of_plan = None
        self.start_grid = None                          # the level's first screen: where RESET returns
        self.can_reset = False

    # ---- what holds in a joint state -----------------------------------------------------------------------------
    def held(self, states, static):
        out = {("joint", tuple(sorted(states.items())))}
        cols = {c.cid: c for c in self.brain.planning()}
        for cid, s in states.items():
            c = cols.get(cid)
            if c is None:                               # a hidden-state column: only in the joint hypothesis
                continue
            k = c.kind
            out.add(("at", k, s))
            if c.frame.kind == "translation":
                p = c.frame.place(s)
                if np.all((p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)):
                    under = set(int(v) for v in static[p[:, 1], p[:, 0]])
                    if len(under) == 1:
                        out.add(("on", k, under.pop()))
        return out

    def register(self, static):
        kinds = [c.kind for c in self.brain.planning() if c.frame.kind == "translation"]
        colours = [int(v) for v in np.unique(static)]
        self.goals.register(kinds, colours, static)
        for c in self.brain.planning():
            if c.kind not in self.goals.at:
                self.goals.register([c.kind], [])

    def observe(self, win_states=None):
        """After a step without a WIN: what holds now is refuted. At a WIN: `win_states` is the predicted joint state
        the last action led to."""
        static = self.brain.static_scene()
        self.register(static)
        if win_states is not None:
            self.goals.win(self.held(win_states, static))
        else:
            self.goals.no_win(self.held(dict(self.brain.joint()), static))

    # ---- search over the model --------------------------------------------------------------------------------
    def actions(self, available):
        keys = [(a, None, None) for a in available if a not in (0, 6)]
        clicks = []
        if 6 in available:
            seen = set()
            for c in self.brain.planning():             # a click on each column, wherever it is
                x, y = c.frame.place(c.state)[0].tolist()
                if 0 <= x < W and 0 <= y < H:
                    clicks.append((6, int(x), int(y), ("click-col", c.cid)))
                    seen.add((int(x), int(y)))
            for x, y, kind in self.brain.m1.proposals(self.brain.v1.scene, self.brain.v1.ground, self.brain.claimed()):
                if (x, y) not in seen:
                    clicks.append((6, x, y, kind))
        return keys, clicks

    def reset_states(self):
        """Where each planning column is on the level's first screen (RESET's known effect)."""
        out = {}
        for c in self.brain.planning():
            s = c.state
            if c.frame.kind == "translation" and self.start_grid is not None:
                hits = c.frame.find_all(self.start_grid)
                s = min(hits, key=lambda h: abs(h[0]) + abs(h[1])) if hits else s
            elif c.frame.kind == "recolour" and self.start_grid is not None:
                s = c.frame.read(self.start_grid)
            out[c.cid] = s
        if self.brain.hidden is not None:               # part 28: hidden-state columns start again too
            out.update(self.brain.hidden.reset_states())
        return out

    def search(self, available):
        brain = self.brain
        brain.cache.clear()                             # the model has changed since the last search
        static = brain.static_scene()
        start = dict(brain.joint())
        keys, clicks = self.actions(available)
        root = [(a, ("key", a[0])) for a in keys] + [((6, x, y), k) for _, x, y, k in clicks]
        deep = [(a, k) for a, k in root if k[0] == "key" or k[0] == "click-col" or brain.m1.effect.get(k, (0, 0))[0] > 0]
        if self.can_reset and self.start_grid is not None:   # RESET: a known effect, so plans may use it
            restart = self.reset_states()
            root, deep = root + [((0, None, None), ("reset",))], deep + [((0, None, None), ("reset",))]
        held0 = self.held(start, static)
        out = []                                        # (path, pwin, info); a path is [(action, expected state)]
        q = deque([(start, [], held0, 0.0, 0.0)])
        seen = {tuple(sorted(start.items()))}
        n, t0 = 0, time.time()
        while q and n < MAX_NODES and time.time() - t0 < BUDGET:
            states, path, held, pwin, info = q.popleft()
            n += 1
            if len(path) >= DEPTH:
                continue
            for action, kind in (root if not path else deep):
                if action[0] == 0:
                    new, bits = {cid: restart.get(cid, s) for cid, s in states.items()}, 0.0
                else:
                    new, bits = brain.model_step(states, action, static)
                if new == states and bits == 0.0 and not path and kind[0] not in ("click-col", "reset") and not brain.m1.tried(action) and not brain.modelled(action):
                    bits = h2(brain.m1.p_effect(kind))  # no column has a model of it, and never tried here
                if bits >= THETA:                       # an uncertain outcome: a leaf worth its bits
                    out.append((path + [(action, None)], pwin, info + bits))
                    continue
                if new == states:
                    continue
                key = tuple(sorted(new.items()))
                if key in seen:
                    continue
                seen.add(key)
                h = self.held(new, static)
                fresh = h - held
                m = self.goals.mass(fresh) if fresh else 0.0
                node = (new, path + [(action, key)], held | h, pwin + (1 - pwin) * m, info + h2(m))
                out.append((node[1], node[3], node[4]))
                q.append(node)
        return out

    def decide(self, available):
        """(action, expected joint state after it), from the committed plan or a new one; None if there is no plan
        of any kind."""
        if self.plan:
            return self.plan.pop(0)
        cands = self.search(available)
        if not cands:
            return None
        hunger = FIXED_HUNGER if FIXED_HUNGER is not None else (self.since + 30) / 25
        u0 = max(c[2] for c in cands)

        def drive(c):                                   # part 17's DE17: deviations weighted by urgency
            path, pwin, info = c
            dp = 0.0 if pwin >= ALPHA else (1 - pwin) * hunger
            du = 0.0 if u0 < THETA else max(0.0, u0 - info - THETA) / u0
            de = len(path) / EFFORT
            return (dp ** 3 + du ** 3 + de ** 3) ** 0.5

        best = min(cands, key=lambda c: (round(drive(c), 12), len(c[0])))
        if best[1] <= 0 and best[2] < THETA:
            return None
        path = best[0]
        kind = "satisfice" if best[1] >= ALPHA else "explore" if best[2] >= THETA else "approach"
        self.plan, self.kind_of_plan = list(path), kind
        chosen = next(c for c in cands if c[0] is path)
        self.last_choice = dict(kind=kind, length=len(path), pwin=round(chosen[1], 4), info=round(chosen[2], 3),
                                candidates=len(cands), leaf=path[-1][1] is None)
        return self.plan.pop(0)

    def fallback(self, available):
        """No plan of any kind: the action whose outcome is most uncertain here, else an untried click, else None."""
        brain = self.brain
        static = brain.static_scene()
        start = dict(brain.joint())
        keys, clicks = self.actions(available)
        best = None
        for a in keys + [(6, x, y) for _, x, y, _ in clicks]:
            _, bits = brain.model_step(start, a, static)
            if a[0] == 6 and not brain.m1.tried(a):
                bits = max(bits, 1e-3)
            if bits > 0 and (best is None or bits > best[0]):
                best = (bits, a)
        return None if best is None else (best[1], None)

    def surprise(self, expected):
        """Drop the plan if the brain is not where the plan expected it to be."""
        if expected is not None and tuple(sorted(dict(self.brain.joint()).items())) != expected:
            self.plan = []
