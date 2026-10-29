"""A column: the coordinator of its four layers (ARCHITECTURE.md sections 1 and 4).

Each step, in a fixed order:
1. L2/3 predicts the next input from the content at L6's place;
2. L4 takes the input and compares it with that prediction;
3. L5 moves L6's place by the input.

If L5 doesn't know the move, the column is lost until the next episode, and the brain falls back on the hippocampus.
The column learns its frame from consolidation (`install`), not online. Online memory is the hippocampus's job.
"""
from arc_agent_v2.l23 import L23
from arc_agent_v2.l4 import L4
from arc_agent_v2.l5 import L5
from arc_agent_v2.l6 import L6


class Column:
    def __init__(self, d, w, seed=0):
        self.l4, self.l5, self.l6, self.l23 = L4(w), L5(d, w, seed), L6(), L23(d, w)

    def reset(self):
        self.l6.reset()

    def predict(self):
        return self.l23.predict(self.l6.place)

    def step(self, x):
        """Sense x and move by it. Returns True if the column still knows where it is."""
        self.l4.sense(x, self.predict())
        if not self.l6.lost():
            self.l6.place = self.l5.move(self.l6.place, x)
        return not self.l6.lost()

    def install(self, frame):
        """Take a frame from consolidation (frameLearning.Frame): its start, its moves, its content. Consolidation
        replays everything the hippocampus holds, so the new frame replaces the old."""
        self.l5.clear()
        self.l23.clear()
        self.l6.start = frame.start
        for place, x, nxt in frame.moves:
            self.l5.learn(place, x, nxt)
        for place, x, n in frame.content:
            self.l23.learn(place, x, n)
