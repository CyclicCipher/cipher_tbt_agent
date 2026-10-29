"""The hippocampus: fast, one-shot memory of episodes (ARCHITECTURE.md section 5, question 3: DG, CA3, CA1 on SDRs).

The neocortex is very bad at memorizing, so it delegates that here (DESIGN.md section 1a). The hippocampus remembers
what happened, exactly, after one exposure. It does not generalize: that is consolidation's job (frameLearning.py),
replaying these memories into the cortex's frame.

- **DG (pattern separation):** every situation (the current place, the input) gets a fresh place SDR, the conjunction
  of the two (HDClibrary.Conjunction: a random expansion and kwta). Situations that differ in anything get unrelated
  places, so episodes don't interfere. The same situation always gets the same place.
- **CA3:** two one-shot associative memories.
  - **What followed:** place -> the input that came next.
  - **The transitions:** next place -> (place, input), which replay reads back.
- **CA1:** maps a recalled place back into the cortex's places. It is not built yet, because nothing in A1 needs it.

Each episode starts at the same place, its origin (P1).
"""
import numpy as np

from arc_agent_v2.HDClibrary import SDR, Associations, Conjunction


class Hippocampus:
    def __init__(self, d, w, seed=0):
        self.dg = Conjunction(d, w, seed + 1)
        self.start = SDR.random(d, w, np.random.default_rng((seed, 31)))
        self.followed = Associations(d, theta=w // 2)   # CA3: place -> what input came next
        self.transitions = Associations(d, theta=w // 2)   # CA3: next place -> (place, input)
        self.place = self.start

    def begin(self):
        self.place = self.start

    def step(self, x, learn=True):
        """Take input x: remember it (if learning) and move to the situation's place."""
        nxt = self.dg(self.place, x)
        if learn:
            self.followed.store(self.place, x)
            self.transitions.store(nxt, (self.place, x))
        self.place = nxt

    def predict(self):
        """The input that most often came next here, if this place was ever lived through."""
        return self.followed.best(self.place)

    def replay(self):
        """Every remembered transition: [(place, input, next place, times seen)]."""
        return [(place, x, nxt, n) for nxt, came in self.transitions.items() for (place, x), n in came.items()]
