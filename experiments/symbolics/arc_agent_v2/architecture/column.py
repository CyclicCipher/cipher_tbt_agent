"""A column: the coordinator of its four layers (ARCHITECTURE.md sections 1 and 4).

Each input arrives from the bus (bus.py), stamped, with its role for this column, and is taken in a fixed order:
1. L2/3 predicts the next input of that sensor from the content at L6's place;
2. L4 takes the input and compares it with that prediction;
3. if the input moves the place (an efference copy, or a read symbol), L5 moves L6's place by it. A sensed input
   doesn't move it.

If L5 doesn't know the move, the column is lost until the next episode, and the brain falls back on the hippocampus.

**Part 5:** where the frame is factored, the column also tracks (phase, coordinate). An input moves the phase by its
transition and the coordinate by its shift in the phase's shift class (part 6; a phase left out of the coordinate has
none, and the column stops tracking). A prediction comes from the place where it has content, and otherwise from
the phase: one symbol for the whole phase, or the symbol whose own position is the coordinate (naming).
The column learns its frame from consolidation (`install`), not online. Online memory is the hippocampus's job.
`stamp` is the bus's check that the column applied every message it was sent.
"""
from arc_agent_v2.architecture.l23 import L23
from arc_agent_v2.architecture.l4 import L4
from arc_agent_v2.architecture.l5 import L5
from arc_agent_v2.architecture.l6 import L6


class Column:
    def __init__(self, d, w, seed=0):
        self.l4, self.l5, self.l6, self.l23 = L4(w), L5(d, w, seed), L6(), L23(d, w)
        self.stamp = 0
        self.fac = None                                 # part 5: the frame's factorization, if any

    def reset(self):
        self.l6.reset()

    def predict(self, sensor="symbols"):
        p = self.l23.predict(self.l6.place, sensor)
        if p is not None or self.fac is None or self.l6.fac is None:
            return p
        phase, coord = self.l6.fac
        kind = self.fac.content.get((phase, sensor))
        if kind is None:
            return None
        return kind[1] if kind[0] == "const" else self.fac.named.get(coord)

    def step(self, x, sensor, moves, stamp, content=True):
        """Sense x (unless it is an efference copy, which nothing here predicts) and, if it moves the place, move by
        it."""
        if content:
            self.l4.sense(x, self.predict(sensor))
        if moves and not self.l6.lost():
            self.l6.place = self.l5.move(self.l6.place, x)
        if moves and self.l6.fac is not None:
            phase, coord = self.l6.fac
            nxt, delta = self.fac.trans.get((phase, x)), self.fac.shift.get((self.fac.cls.get(phase), x))
            self.l6.fac = None if nxt is None or delta is None else (nxt, self.fac.modules.bind(coord, delta))
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
