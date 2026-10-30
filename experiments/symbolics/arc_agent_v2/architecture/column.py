"""A column: the coordinator of its four layers (ARCHITECTURE.md sections 1 and 4).

Each input arrives from the bus (bus.py), stamped, with its role for this column, and is taken in a fixed order:
1. L4 takes the input and compares it with what the column expected (its likelihood under the column's state);
2. the column's state is weighed by it (Bayes);
3. if the input moves the place (an efference copy, or a read symbol), L5 moves L6's place by it. A sensed input
   doesn't move it.

If L5 doesn't know the move, the column is lost until the next episode, and the brain falls back on the hippocampus.

**Part 5:** where the frame is factored, the column also tracks (phase, coordinate). An input moves the phase by its
transition and the coordinate by its shift in the phase's shift class (part 6; a phase left out of the coordinate has
none).

**Part 9: the column's state is a union of hypotheses, a posterior** (docs/NEURAL_COLUMN.md).
- **On each content input,** every hypothesis's weight is multiplied by how expected the input was there (Bayes'
  rule, with part 12's content model, information.py).
- **On each moving input,** every hypothesis moves. One whose phase has no transition by the input is dropped, and so
  is one whose transition has no shift (its phase is left out of the coordinate).
- **Bursting:** when no hypothesis has a transition by the input, the union falls back to every phase the input is
  made from, weighted by how often (`origins`). The coordinate moves on regardless: only the phase widens.
- **The prediction** comes from the union's posterior predictive, and from the place's content only where the union
  holds nothing.
- **`union=False`** keeps only the most probable hypothesis and never bursts (part 9's control).

**Part 12: objects, and today's memory.**
- **A hypothesis is (object, phase, coordinate),** or (object, None, place) where the frame isn't factored. What is at
  a state depends on which object is there; the moves don't, so the objects at a state move together.
- **The objects** are learned at sleep (objects.py), each with prior weight n, the episodes it explains.
- **Where an object holds nothing at a state,** the shared content is used: naming, then the phase's counts (parts 5
  and 9).
- **Today's memory is one more object,** always in the union: the Chinese restaurant's new table, with prior weight
  α = 1. Its content at a state is what came next there today (the hippocampus's location memory): nothing, the
  prior, before the day files anything there. It replaces part 8's override of the cortex by the day's memory, so the
  day wins only by evidence.
- **The cap:** at most MAX_HYPS states are kept, with every object at them (amendment 2).

The column learns its frame from consolidation (`install`), not online. Online memory is the hippocampus's job.
`stamp` is the bus's check that the column applied every message it was sent.
"""
import heapq
import math

import numpy as np

from arc_agent_v2.architecture.information import NEG_INF, Content, logsumexp, mixture_bits, mixture_scores
from arc_agent_v2.architecture.l23 import L23
from arc_agent_v2.architecture.l4 import L4
from arc_agent_v2.architecture.l5 import L5
from arc_agent_v2.architecture.l6 import L6

PRUNE = math.log(1e6)                                   # hypotheses this far below the best are dropped
MAX_HYPS = 64                                           # states kept (part 12: the objects at a state move together)
ALPHA = 1.0                                             # part 12: a new object's prior weight (the CRP's)
TODAY = "today"                                         # part 12: the object that is today's memory


def renaming(frame, coordinates=True):
    """The column's names for a frame's places: a coordinated orbit's places are named by their codes."""
    rename = {}
    if coordinates:
        for orbit in frame.coords:
            rename.update(orbit.codes)
    return rename


class Column:
    def __init__(self, d, w, seed=0, union=True):
        self.d, self.w = d, w
        self.l4, self.l5, self.l6, self.l23 = L4(w), L5(d, w, seed), L6(), L23(d, w)
        self.stamp = 0
        self.fac = None                                 # part 5: the frame's factorization, if any
        self.union = union                              # part 9
        self.bursts, self.burst_sizes = 0, []           # part 9: counted, for the harness
        self._at = {}                                   # part 10: a place's name -> [((phase, coordinate), times)]
        self.prior = Content(d, w)                      # part 12: what a cell with nothing in it predicts
        self.objects = {}                               # part 12: object id -> its prior weight n (episodes)
        self.cells = {}                                 # part 12: (object, state, sensor) -> Content
        self.shared = {}                                # part 12: (phase, sensor) -> Content (the phase's counts)
        self.known = {}                                 # part 12: sensor -> the inputs seen as its content
        self.day = None                                 # part 12: the hippocampus, whose day memory is an object
        self.today_on = True                            # part 12: today's memory as an object (arm c: off)
        self._place, self._named, self._stacks = {}, {}, {}

    def reset(self):
        self.l6.place = self.l6.start
        self.l6.hyps = self.start_union()[1]

    # ---- the union's likelihoods ----------------------------------------------------------------------------------
    def _prior(self):
        """Each object's log prior weight: n / (Σ n + α); today's, α / (Σ n + α)."""
        objs = dict(self.objects)
        if self.today_on and self.day is not None:
            objs[TODAY] = ALPHA
        z = math.log(sum(self.objects.values()) + ALPHA)
        return {o: math.log(n) - z for o, n in objs.items()}

    def _content(self, hyp, sensor):
        """What came next at a hypothesis (a Content)."""
        obj, phase, coord = hyp
        if obj == TODAY:
            c = self.day.today_content((phase, coord), sensor)
            return c if c is not None else self.prior
        c = self.cells.get((obj, (phase, coord), sensor))
        if c is not None:
            return c
        if self.fac is None:
            return self.prior
        kind = self.fac.content.get((phase, sensor))
        if kind is not None and kind[0] == "named":
            name = self.fac.named.get(coord)
            if name is not None:
                key = (phase, coord, sensor)
                if key not in self._named:
                    shared = self.shared.get((phase, sensor))
                    self._named[key] = Content(self.d, self.w, {name: shared.N if shared is not None else 1})
                return self._named[key]
        return self.shared.get((phase, sensor), self.prior)

    def _normalize(self, hyps):
        """Drop what is PRUNE below the best, keep at most MAX_HYPS states (with every object at them), and
        normalize."""
        if not hyps:
            return {}
        best = max(hyps.values())
        kept = {h: lw for h, lw in hyps.items() if lw >= best - PRUNE}
        states = {}
        for h, lw in kept.items():
            states.setdefault(h[1:], []).append(lw)
        if len(states) > MAX_HYPS:
            top = set(heapq.nlargest(MAX_HYPS, states, key=lambda st: logsumexp(states[st])))
            kept = {h: lw for h, lw in kept.items() if h[1:] in top}
        z = logsumexp(kept.values())
        return {h: lw - z for h, lw in kept.items()}

    def _stack(self, state, sensor):
        """The known objects' cells at a state, stacked: (object -> row, lo, T, memo), with log L(x) = (Σ_{i∈x}
        lo[row, i] + T[row]) / 2w, information.Content's likelihood for many objects at once."""
        key = (state, sensor)
        if key not in self._stacks:
            objs = [o for o in self.objects if (o, state, sensor) in self.cells]
            a = self.w / self.d
            lo = np.empty((len(objs), self.d))
            T = np.empty(len(objs))
            for r, o in enumerate(objs):
                c = self.cells[(o, state, sensor)]
                bits_, cnt, T[r] = c._stats()
                q = np.full(self.d, a / (c.N + 1.0))
                q[bits_] = (cnt + a) / (c.N + 1.0)
                lo[r] = np.log(q) - np.log1p(-q)
            self._stacks[key] = ({o: r for r, o in enumerate(objs)}, lo, T, {})
        return self._stacks[key]

    def _lls(self, hyps, x, sensor):
        """{hypothesis: log L(x) there}: the objects at a state taken together."""
        out, by = {}, {}
        for h in hyps:
            by.setdefault(h[1:], []).append(h)
        for state, hs in by.items():
            rows, v = {}, None
            if len(hs) > 4:
                rows, lo, T, memo = self._stack(state, sensor)
                v = memo.get(x)
                if v is None and rows:
                    v = memo[x] = (lo[:, x.idx].sum(1) + T) / (2 * self.w)
            for h in hs:
                r = rows.get(h[0])
                out[h] = float(v[r]) if r is not None else self._content(h, sensor).loglik(x)
        return out

    def _weigh(self, hyps, x, sensor):
        """Bayes' rule: (the union after x, each weight times how expected x was there; log of x's predictive density
        before it)."""
        lls = self._lls(hyps, x, sensor)
        joint = {h: lw + lls[h] for h, lw in hyps.items()}
        logp = logsumexp(joint.values()) - logsumexp(hyps.values())
        if len(hyps) == 1:                              # one hypothesis: normalized, its weight stays log 1 = 0
            return hyps, logp
        return self._normalize(joint), logp

    def _weighed(self, hyps, x, sensor):
        return self._weigh(hyps, x, sensor)[0]

    def _moved(self, hyps, x):
        """Every hypothesis moves by x; if none can, the union bursts to every phase x is made from. Returns (the
        union, its size if it burst, else None)."""
        f, new = self.fac, {}

        def step(phase, coord):
            """(whether the state has a transition by x, the state it leads to or None)."""
            if f is None:                               # an unfactored frame: the state is the place
                n = self.l5.move(coord, x)
                return n is not None, ((None, n) if n is not None else None)
            nph, delta = f.trans.get((phase, x)), f.shift.get((f.cls.get(phase), x))
            if nph is None:
                return False, None
            return True, ((nph, f.modules.bind(coord, delta)) if delta is not None else None)

        def add(obj, st, lw):
            k = (obj, *st)
            new[k] = logsumexp([new.get(k, NEG_INF), lw])

        hyps = hyps if self.union else dict([max(hyps.items(), key=lambda kv: kv[1])])
        steps, expected = {}, False                     # the objects at a state move together: one step per state
        for (obj, phase, coord), lw in hyps.items():
            s = steps.get((phase, coord))
            if s is None:
                s = steps[(phase, coord)] = step(phase, coord)
            expected |= s[0]
            if s[1] is not None:
                add(obj, s[1], lw)
        if not expected and self.union and f is not None:   # bursting: the coordinate moves on, the phase widens
            origins = f.origins.get(x, {})
            total = sum(origins.values())
            for (obj, _, coord), lw in hyps.items():
                for phase, n in origins.items():
                    s = step(phase, coord)
                    if s[1] is not None:
                        add(obj, s[1], lw + math.log(n / total))
            out = self._normalize(new)
            return out, (len(out) if out else None)
        if not self.union and new:
            new = dict([max(new.items(), key=lambda kv: kv[1])])
        return self._normalize(new), None

    def _seeded(self, place):
        """Part 10: the union from a known place: one hypothesis per frame place of that name and per object, (its
        phase, its coordinate), weighted by how often it was lived through and by the object's prior."""
        seeds, prior = {}, self._prior()
        for (phase, coord), n in self._at.get(place, ()):
            for obj, lp in prior.items():
                h = (obj, phase, coord)
                seeds[h] = logsumexp([seeds.get(h, NEG_INF), math.log(n) + lp])
        return self._normalize(seeds)

    # ---- the column's state, as a function (part 10: the thalamus simulates with it) --------------------------------
    def start_union(self):
        """(start place, the union at the start: every object at the start state, by its prior; not pruned)."""
        start = self.l6.start
        if not self.objects:
            return start, {}
        state = self.fac.start if self.fac is not None else ((None, start) if start is not None else None)
        if state is None:
            return start, {}
        return start, {(obj, *state): lp for obj, lp in self._prior().items()}

    def start_state(self):
        return self.start_union()

    def advance(self, place, hyps, x, sensor, moves, content):
        """(place, union) after input x, without changing the column. Returns (place, union, the union's size if it
        burst, else None)."""
        burst = None
        if content and hyps:
            hyps = self._weighed(hyps, x, sensor)
        if moves and place is not None:
            place = self.l5.move(place, x)
        if moves and hyps:
            hyps, burst = self._moved(hyps, x)
        if moves and self.fac is not None and not hyps and place is not None and self.union:
            hyps = self._seeded(place)                  # amendment 1: the place says where, the union in what context
        return place, hyps, burst

    def locations(self, hyps):
        """Part 12: where the column is, as [(state, probability)], the objects summed out: where the day's memory of
        an input is filed."""
        if not hyps:
            return []
        mass = {}
        for (_, phase, coord), lw in hyps.items():
            mass[(phase, coord)] = mass.get((phase, coord), 0.0) + math.exp(lw)
        z = sum(mass.values())
        return [(s, m / z) for s, m in mass.items()]

    def objects_posterior(self, hyps=None):
        """Part 12: P(object) under the union (the region's recognition), {object: probability}."""
        hyps = self.l6.hyps if hyps is None else hyps
        mass = {}
        for (obj, _, _), lw in hyps.items():
            mass[obj] = mass.get(obj, 0.0) + math.exp(lw)
        z = sum(mass.values())
        return {o: m / z for o, m in mass.items()} if z else {}

    def _place_content(self, place, sensor):
        key = (place, sensor)
        if key not in self._place:
            self._place[key] = Content(self.d, self.w, self.l23.counts(place, sensor))
        return self._place[key]

    def logp(self, place, hyps, x, sensor):
        """log of x's predictive density here (the components' mixture, as `components` chooses them)."""
        if hyps and any(self._content(h, sensor).N for h in hyps):
            return self._weigh(hyps, x, sensor)[1]
        c = self._place_content(place, sensor) if place is not None else None
        return (c if c is not None and c.N else self.prior).loglik(x)

    def components(self, place, hyps, sensor):
        """The prediction's mixture components [(log-weight, Content)]: the union's hypotheses where any has content,
        else the place's content (information.py)."""
        if hyps:
            comps = [(lw, self._content(h, sensor)) for h, lw in hyps.items()]
            if any(c.N for _, c in comps):
                return comps
        c = self._place_content(place, sensor) if place is not None else None
        return [(0.0, c)] if c is not None and c.N else []

    def name(self, place, hyps):
        """The symbol whose own position is the state's coordinate (part 5's naming), or None."""
        if self.fac is None:
            return None
        if hyps:
            return self.fac.named.get(max(hyps, key=hyps.get)[2])
        at = self._at.get(place)
        return self.fac.named.get(max(at, key=lambda hn: hn[1])[0][1]) if at else None

    # ---- the column's steps ---------------------------------------------------------------------------------------
    def predict_state(self, place, hyps, sensor="symbols"):
        comps = self.components(place, hyps, sensor)
        if not comps:
            return None
        if len(comps) == 1:                             # one component: its most likely input
            return comps[0][1].best()
        scores = mixture_scores(comps, self.known.get(sensor, ()))
        return max(scores, key=scores.get) if scores else None

    def predict(self, sensor="symbols"):
        return self.predict_state(self.l6.place, self.l6.hyps, sensor)

    def bits(self, target, sensor, known=()):
        """The code length of target under this column's prediction (information.py), or None if it predicts
        nothing (the same order as `predict`)."""
        comps = self.components(self.l6.place, self.l6.hyps, sensor)
        return mixture_bits(comps, target, tuple(self.known.get(sensor, ())) + tuple(known)) if comps else None

    def step(self, x, sensor, moves, stamp, content=True):
        """Sense x (unless it is an efference copy, which nothing here predicts) and, if it moves the place, move by
        it. Returns where the column was once x was weighed and before it moved (part 12: where the day's memory of x
        is filed)."""
        hyps = self.l6.hyps
        if content:
            self.l4.sense(x, self.logp(self.l6.place, hyps, x, sensor), self.prior.loglik(x))
            if hyps:
                hyps = self._weighed(hyps, x, sensor)
        where = self.locations(hyps) if content else []
        self.l6.place, self.l6.hyps, burst = self.advance(self.l6.place, hyps, x, sensor, moves, False)
        if burst is not None:
            self.bursts += 1
            self.burst_sizes.append(burst)
        self.stamp = stamp
        return where

    def route(self, goal=None):
        """The moves from the current place to goal (the start, by default)."""
        return self.l5.route(self.l6.place, self.l6.start if goal is None else goal)

    def install(self, frame, coordinates=True):
        """Take a frame from consolidation (frameLearning.Frame): its start, its moves, its content, its objects and
        (unless told not to) its coordinates. A coordinated orbit's places are renamed to their codes, and its moves
        become shifts. Consolidation replays everything the hippocampus holds, so the new frame replaces the old."""
        self.l5.clear()
        self.l23.clear()
        rename, shifted = renaming(frame, coordinates), set()
        if coordinates:
            for orbit in frame.coords:
                for x, delta in orbit.shifts.items():
                    self.l5.learn_shift(x, orbit.modules, delta)
                    shifted |= {(p, x) for p in orbit.codes}
        name = lambda p: rename.get(p, p)
        self.l6.start = name(frame.start)
        self.l5.learn_many([(name(place), x, name(nxt)) for place, x, nxt in frame.moves if (place, x) not in shifted])
        self.l23.learn_many([(name(place), sensor, x, n) for place, sensor, x, n in frame.content])
        if coordinates:
            for orbit in frame.coords:
                self.l5.build_words(orbit.modules)
        self.fac = frame.factored if coordinates else None
        self.l6.fac_start = self.fac.start if self.fac is not None else None
        self._place, self._named, self._stacks = {}, {}, {}
        known = {}
        for _, sensor, x, _ in frame.content:
            known.setdefault(sensor, {})[x] = None
        self.known = {s: tuple(xs) for s, xs in known.items()}
        self.shared = {}
        if self.fac is not None:
            self.shared = {k: Content(self.d, self.w, c) for k, c in self.fac.counts.items()}
        self.objects, self.cells = {}, {}
        for obj in frame.objects:
            self.objects[obj.sdr] = obj.n
            for (state, sensor), c in obj.cells.items():
                self.cells[(obj.sdr, state, sensor)] = Content(self.d, self.w, c)
        self._at = {}                                   # part 10: a place's name -> [((phase, coordinate), times)]
        if self.fac is not None:
            lived = {}
            for place, sensor, x, n in frame.content:
                lived[place] = lived.get(place, 0) + n
            for place, ph in self.fac.phase.items():
                if place in self.fac.coord:
                    self._at.setdefault(name(place), []).append(((ph, self.fac.coord[place]), lived.get(place, 0) + 1))
