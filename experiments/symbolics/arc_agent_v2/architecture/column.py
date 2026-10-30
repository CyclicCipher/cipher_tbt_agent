"""A column: the coordinator of its four layers (ARCHITECTURE.md sections 1 and 4).

Each input arrives from the bus (bus.py), stamped, with its role for this column, and is taken in a fixed order:
1. L2/3 predicts the next input of that sensor from the content at L6's place;
2. L4 takes the input and compares it with that prediction;
3. if the input moves the place (an efference copy, or a read symbol), L5 moves L6's place by it. A sensed input
   doesn't move it.

If L5 doesn't know the move, the column is lost until the next episode, and the brain falls back on the hippocampus.

**Part 5:** where the frame is factored, the column also tracks (phase, coordinate). An input moves the phase by its
transition and the coordinate by its shift in the phase's shift class (part 6; a phase left out of the coordinate has
none).

**Part 9: that factored state is a union of hypotheses, a posterior** (docs/NEURAL_COLUMN.md).
- **On each content input,** every hypothesis's weight is multiplied by how expected the input was there (Bayes'
  rule), by Laplace's rule, (n + 1) / (N + K).
  - **Where the counts come from:** the hypothesis's cell, the content of the frame's place with that phase and
    coordinate. Otherwise its phase's: where the phase names its coordinate, the name has all N, and otherwise the
    phase's counts.
  - **The cell, in HTM's terms:** the conjunction of the shared part (the coordinate's code, the minicolumn) and the
    context (the phase).
- **On each moving input,** every hypothesis moves. One whose phase has no transition by the input is dropped, and so
  is one whose transition has no shift (its phase is left out of the coordinate).
- **Bursting:** when no hypothesis has a transition by the input, the union falls back to every phase the input is
  made from, weighted by how often (`origins`). The coordinate moves on regardless: only the phase widens.
- **The prediction** comes from the union's posterior predictive, and from the place's content only where the union
  is empty. Codes are shared across rings, so a place's content mixes contexts that the union keeps apart.
- **`union=False`** keeps only the most probable hypothesis and never bursts (part 9's control).

The column learns its frame from consolidation (`install`), not online. Online memory is the hippocampus's job.
`stamp` is the bus's check that the column applied every message it was sent.
"""
import heapq
import math

from arc_agent_v2.architecture.information import NEG_INF, bits, laplace, logsumexp
from arc_agent_v2.architecture.l23 import L23
from arc_agent_v2.architecture.l4 import L4
from arc_agent_v2.architecture.l5 import L5
from arc_agent_v2.architecture.l6 import L6

PRUNE = math.log(1e6)                                   # hypotheses this far below the best are dropped
MAX_HYPS = 64


class Column:
    def __init__(self, d, w, seed=0, union=True):
        self.l4, self.l5, self.l6, self.l23 = L4(w), L5(d, w, seed), L6(), L23(d, w)
        self.stamp = 0
        self.fac = None                                 # part 5: the frame's factorization, if any
        self.union = union                              # part 9
        self.bursts, self.burst_sizes = 0, []           # part 9: counted, for the harness

    def reset(self):
        self.l6.reset()

    # ---- the union's likelihoods (part 9) -----------------------------------------------------------------------
    def _counts(self, hyp, sensor):
        """What came next at a hypothesis: ({input: n}, N). Where its phase names its coordinate, the name has all N."""
        phase, coord = hyp
        cell = self.fac.cells.get(hyp, {}).get(sensor)
        if cell:
            return cell, sum(cell.values())
        c = self.fac.counts.get((phase, sensor), {})
        N = sum(c.values())
        kind = self.fac.content.get((phase, sensor))
        if kind is not None and kind[0] == "named":
            name = self.fac.named.get(coord)
            if name is not None:
                return {name: N}, N
        return c, N

    def _scores(self, sensor):
        """The posterior predictive's log-probability of each input some hypothesis has seen next."""
        K = self.fac.vocab
        per = [(lw,) + self._counts(h, sensor) for h, lw in self.l6.hyps.items()]
        cands = list(dict.fromkeys(x for _, c, _ in per for x in c))
        return {x: logsumexp(lw + laplace(c.get(x, 0), N, K)[0] for lw, c, N in per) for x in cands}

    def _normalize(self, hyps):
        if not hyps:
            return {}
        best = max(hyps.values())
        kept = {h: lw for h, lw in hyps.items() if lw >= best - PRUNE}
        if len(kept) > MAX_HYPS:
            kept = dict(heapq.nlargest(MAX_HYPS, kept.items(), key=lambda kv: kv[1]))
        z = logsumexp(kept.values())
        return {h: lw - z for h, lw in kept.items()}

    def _weigh(self, x, sensor):
        """Bayes' rule: each hypothesis's weight times how expected x was there."""
        K = self.fac.vocab
        hyps = {}
        for h, lw in self.l6.hyps.items():
            c, N = self._counts(h, sensor)
            hyps[h] = lw + laplace(c.get(x, 0), N, K)[0]
        self.l6.hyps = self._normalize(hyps)

    def _move(self, x):
        """Every hypothesis moves by x; if none can, the union bursts to every phase x is made from."""
        f, new = self.fac, {}

        def go(phase, coord, lw):
            nxt, delta = f.trans.get((phase, x)), f.shift.get((f.cls.get(phase), x))
            if nxt is not None and delta is not None:
                key = (nxt, f.modules.bind(coord, delta))
                new[key] = logsumexp([new.get(key, NEG_INF), lw])
            return nxt is not None

        hyps = self.l6.hyps if self.union else dict([max(self.l6.hyps.items(), key=lambda kv: kv[1])])
        expected = [go(phase, coord, lw) for (phase, coord), lw in hyps.items()]
        if not any(expected) and self.union:            # bursting: the coordinate moves on, the phase widens
            origins = f.origins.get(x, {})
            total = sum(origins.values())
            for (_, coord), lw in hyps.items():
                for phase, n in origins.items():
                    go(phase, coord, lw + math.log(n / total))
            self.l6.hyps = self._normalize(new)
            if self.l6.hyps:
                self.bursts += 1
                self.burst_sizes.append(len(self.l6.hyps))
            return
        if not self.union and new:
            new = dict([max(new.items(), key=lambda kv: kv[1])])
        self.l6.hyps = self._normalize(new)

    # ---- the column's steps ---------------------------------------------------------------------------------------
    def predict(self, sensor="symbols"):
        if self.fac is not None and self.l6.hyps:
            scores = self._scores(sensor)
            if scores:
                return max(scores, key=scores.get)
        return self.l23.predict(self.l6.place, sensor)

    def bits(self, target, sensor, K):
        """The code length of target under this column's prediction (information.py), or None if it predicts
        nothing: the union's posterior predictive where it has one, else Laplace's rule over the place's content (the
        same order as `predict`)."""
        if self.fac is not None and self.l6.hyps:
            per = [(lw,) + self._counts(h, sensor) for h, lw in self.l6.hyps.items()]
            if any(N for _, _, N in per):
                parts = [laplace(c.get(target, 0), N, K) for _, c, N in per]
                return bits(logsumexp(lw + t for (lw, _, _), (t, _) in zip(per, parts)),
                            logsumexp(lw + r for (lw, _, _), (_, r) in zip(per, parts)))
        c = self.l23.counts(self.l6.place, sensor)
        if c:
            return bits(*laplace(c.get(target, 0), sum(c.values()), K))
        return None

    def step(self, x, sensor, moves, stamp, content=True):
        """Sense x (unless it is an efference copy, which nothing here predicts) and, if it moves the place, move by
        it."""
        if content:
            self.l4.sense(x, self.predict(sensor))
            if self.fac is not None and self.l6.hyps:
                self._weigh(x, sensor)
        if moves and not self.l6.lost():
            self.l6.place = self.l5.move(self.l6.place, x)
        if moves and self.fac is not None and self.l6.hyps:
            self._move(x)
        self.stamp = stamp

    def route(self, goal=None):
        """The moves from the current place to goal (the start, by default)."""
        return self.l5.route(self.l6.place, self.l6.start if goal is None else goal)

    def install(self, frame, coordinates=True):
        """Take a frame from consolidation (frameLearning.Frame): its start, its moves, its content, and (unless told
        not to) its coordinates. A coordinated orbit's places are renamed to their codes, and its moves become shifts.
        Consolidation replays everything the hippocampus holds, so the new frame replaces the old."""
        self.l5.clear()
        self.l23.clear()
        rename, shifted = {}, set()
        if coordinates:
            for orbit in frame.coords:
                rename.update(orbit.codes)
                for x, delta in orbit.shifts.items():
                    self.l5.learn_shift(x, orbit.modules, delta)
                    shifted |= {(p, x) for p in orbit.codes}
        name = lambda p: rename.get(p, p)
        self.l6.start = name(frame.start)
        for place, x, nxt in frame.moves:
            if (place, x) not in shifted:
                self.l5.learn(name(place), x, name(nxt))
        for place, sensor, x, n in frame.content:
            self.l23.learn(name(place), sensor, x, n)
        if coordinates:
            for orbit in frame.coords:
                self.l5.build_words(orbit.modules)
        self.fac = frame.factored if coordinates else None
        self.l6.fac_start = self.fac.start if self.fac is not None else None
