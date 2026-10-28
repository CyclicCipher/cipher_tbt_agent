"""The executive (ARCHITECTURE.md section 6), a stand-in for the PFC: goal hypotheses, search over the brain's model,
and the choice of plan. Part 15b's lexicographic satisficing agent with part 17's commitment.

Goal hypotheses are about kinds (a column's content signature), so evidence carries across levels:
- ("on", kind, c): some column of this kind lies wholly on scene colour c;
- ("at", kind, state): some column of this kind is in this state.
Prior: half per family; "on" split evenly over (kind, colour) pairs; "at" split evenly over kinds, each kind's share
spread over AT_STATES states. A hypothesis that holds without a WIN gains an error; at a WIN, every hypothesis that
didn't hold gains one. Hypotheses first registered after WINs start charged with them (part 17).
"""
import math
from collections import deque

import numpy as np

from arc_agent.frames import H, W

ALPHA, THETA, STEP_BITS = 0.5, 0.05, 0.02
DEPTH, MAX_NODES, AT_STATES = 12, 3000, 4096
LOG_EPS = math.log(0.01)


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class Goals:
    def __init__(self):
        self.on = {}                                    # (kind, colour) -> errors
        self.at = {}                                    # kind -> {state: errors}
        self.at_rest = {}                               # kind -> errors shared by every state never named
        self.wins = 0

    def register(self, kinds, colours):
        for k in kinds:
            if k not in self.at:
                self.at[k], self.at_rest[k] = {}, self.wins
            for c in colours:
                if (k, c) not in self.on:
                    self.on[(k, c)] = self.wins

    def _weights(self):
        n_on = max(len(self.on), 1)
        n_k = max(len(self.at), 1)
        w_on = {key: math.log(0.5 / n_on) + LOG_EPS * e for key, e in self.on.items()}
        base = {k: math.log(0.5 / n_k / AT_STATES) for k in self.at}
        return w_on, base

    def mass(self, held):
        """Posterior probability that the WIN is caused by one of the hypotheses in `held`."""
        w_on, base = self._weights()
        m = max(list(w_on.values()) + [b + LOG_EPS * self.at_rest[k] for k, b in base.items()] + [-1e9])
        z, num = 0.0, 0.0
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

    def win(self, held):
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
        self.wins += 1


class Executive:
    def __init__(self, brain):
        self.brain = brain
        self.goals = Goals()
        self.plan = []                                  # [(action, expected joint state after it)]
        self.kind_of_plan = None

    # ---- what holds in a joint state -----------------------------------------------------------------------------
    def held(self, states, static):
        out = set()
        cols = {c.cid: c for c in self.brain.planning()}
        for cid, s in states.items():
            c = cols[cid]
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
        self.goals.register(kinds, colours)
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

    def search(self, available):
        brain = self.brain
        brain.cache.clear()                             # the model has changed since the last search
        static = brain.static_scene()
        start = dict(brain.joint())
        keys, clicks = self.actions(available)
        root = [(a, ("key", a[0])) for a in keys] + [((6, x, y), k) for _, x, y, k in clicks]
        deep = [(a, k) for a, k in root if k[0] == "key" or k[0] == "click-col" or brain.m1.effect.get(k, (0, 0))[0] > 0]
        held0 = self.held(start, static)
        out = []                                        # (path, pwin, info); a path is [(action, expected state)]
        q = deque([(start, [], held0, 0.0, 0.0)])
        seen = {tuple(sorted(start.items()))}
        n = 0
        while q and n < MAX_NODES:
            states, path, held, pwin, info = q.popleft()
            n += 1
            if len(path) >= DEPTH:
                continue
            for action, kind in (root if not path else deep):
                new, bits = brain.model_step(states, action, static)
                if new == states and bits == 0.0 and not path and kind[0] != "click-col" and                         not brain.m1.tried(action) and not brain.modelled(action):
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
        acc = [c for c in cands if c[1] >= ALPHA]
        if acc:
            path, kind = min(acc, key=lambda c: (len(c[0]), -c[1]))[0], "satisfice"
        else:
            best = max(cands, key=lambda c: (c[2] - STEP_BITS * len(c[0]), -len(c[0])))
            if best[2] - STEP_BITS * len(best[0]) >= THETA:
                path, kind = best[0], "explore"
            else:
                appr = [c for c in cands if c[1] > 1e-6]
                if not appr:
                    return None
                path, kind = max(appr, key=lambda c: (c[1], -len(c[0])))[0], "approach"
        self.plan, self.kind_of_plan = list(path), kind
        chosen = next(c for c in cands if c[0] is path)
        self.last_choice = dict(kind=kind, length=len(path), pwin=round(chosen[1], 4), info=round(chosen[2], 3),
                                candidates=len(cands), leaf=path[-1][1] is None)
        return self.plan.pop(0)

    def surprise(self, expected):
        """Drop the plan if the brain is not where the plan expected it to be."""
        if expected is not None and tuple(sorted(dict(self.brain.joint()).items())) != expected:
            self.plan = []
