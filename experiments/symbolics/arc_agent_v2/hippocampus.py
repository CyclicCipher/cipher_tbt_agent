"""The hippocampus: fast, one-shot memory of episodes (ARCHITECTURE.md section 5, question 3: DG, CA3, CA1 on SDRs).

The neocortex is very bad at memorizing, so it delegates that here (DESIGN.md section 1a). The hippocampus remembers
what happened, exactly, after one exposure. It does not generalize: that is consolidation's job (frameLearning.py),
replaying these memories into the cortex's frame.

- **DG (pattern separation):** every situation (the current place, an input that moves it) gets a fresh place SDR, the
  conjunction of the two (HDClibrary.Conjunction: a random expansion and kwta). Situations that differ in anything
  get unrelated places, so episodes don't interfere. The same situation always gets the same place.
- **CA3:** two one-shot associative memories.
  - **What followed:** place -> each input that came next there, by sensor. Sensed inputs (part 4) are remembered
    here and don't move the place.
  - **The transitions:** next place -> (place, input), for the inputs that move it. Replay reads these back.
- **CA1:** maps a recalled place back into the cortex's places. It is not built yet, because nothing needs it.

Each episode starts at the same place, its origin (P1). The bus says, for each input, whether it moves the place.
"""
import numpy as np

from arc_agent_v2.HDClibrary import SDR, Associations, Conjunction


class Hippocampus:
    def __init__(self, d, w, seed=0):
        self.dg = Conjunction(d, w, seed + 1)
        self.start = SDR.random(d, w, np.random.default_rng((seed, 31)))
        self.followed = Associations(d, theta=w // 2)   # CA3: place -> (sensor, input) that came next
        self.transitions = Associations(d, theta=w // 2)   # CA3: next place -> (place, input)
        self.place = self.start

    def begin(self):
        self.place = self.start

    def step(self, x, sensor="symbols", moves=True, learn=True, content=True):
        """Take input x: remember it as content (if learning, and unless it is an efference copy) and, if it moves
        the place, move to the situation's place."""
        if learn and content:
            self.followed.store(self.place, (sensor, x))
        if moves:
            nxt = self.dg(self.place, x)
            if learn:
                self.transitions.store(nxt, (self.place, x))
            self.place = nxt

    def predict(self, sensor="symbols"):
        """The input of this sensor that most often came next here, if this place was ever lived through."""
        came = {x: n for (s, x), n in self.followed.recall(self.place).items() if s == sensor}
        return max(came, key=came.get) if came else None

    def replay(self):
        """Every remembered transition: [(place, input, next place, times seen)]."""
        return [(place, x, nxt, n) for nxt, came in self.transitions.items() for (place, x), n in came.items()]

    def contents(self):
        """Every remembered content: [(place, sensor, input, times seen)]."""
        return [(place, s, x, n) for place, came in self.followed.items() for (s, x), n in came.items()]
