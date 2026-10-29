"""A column: the coordinator of its four layers (ARCHITECTURE.md sections 1 and 4).

Each input arrives from the bus (bus.py), stamped, with its role for this column, and is taken in a fixed order:
1. L2/3 predicts the next input of that sensor from the content at L6's place;
2. L4 takes the input and compares it with that prediction;
3. if the input moves the place (an efference copy, or a read symbol), L5 moves L6's place by it. A sensed input
   doesn't move it.

If L5 doesn't know the move, the column is lost until the next episode, and the brain falls back on the hippocampus.
The column learns its frame from consolidation (`install`), not online. Online memory is the hippocampus's job.
`stamp` is the bus's check that the column applied every message it was sent.
"""
from arc_agent_v2.l23 import L23
from arc_agent_v2.l4 import L4
from arc_agent_v2.l5 import L5
from arc_agent_v2.l6 import L6


class Column:
    def __init__(self, d, w, seed=0):
        self.l4, self.l5, self.l6, self.l23 = L4(w), L5(d, w, seed), L6(), L23(d, w)
        self.stamp = 0

    def reset(self):
        self.l6.reset()

    def predict(self, sensor="symbols"):
        return self.l23.predict(self.l6.place, sensor)

    def step(self, x, sensor, moves, stamp, content=True):
        """Sense x (unless it is an efference copy, which nothing here predicts) and, if it moves the place, move by
        it."""
        if content:
            self.l4.sense(x, self.predict(sensor))
        if moves and not self.l6.lost():
            self.l6.place = self.l5.move(self.l6.place, x)
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
