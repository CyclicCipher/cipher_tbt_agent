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
- **The entorhinal input (part 8):** with each input, the bus gives the cortex's location (the place of a column that
  knows it). What came next is also kept under that location, exactly, for the day: until the next sleep, when the day
  is consolidated and these memories are cleared. A place the cortex knows, reached today by a new path, recalls what
  was seen there today (TEM: the hippocampus binds what is sensed to the position the cortex path-integrates).

Each episode starts at the same place, its origin (P1). The bus says, for each input, whether it moves the place.

**The end of an episode is observed (part 6):** on each sensor that gave nothing at the episode's last place, what came
next there was the end, one fixed SDR (`end`), as every episode's origin is one fixed place. Without it, a place where
episodes only ended has no recorded future, and so agrees with any place (P3's full form would merge them).
"""
import numpy as np

from collections import Counter

from arc_agent_v2.architecture.HDClibrary import SDR, Associations, Conjunction


class Hippocampus:
    def __init__(self, d, w, seed=0):
        self.dg = Conjunction(d, w, seed + 1)
        self.start = SDR.random(d, w, np.random.default_rng((seed, 31)))
        self.followed = Associations(d, theta=w // 2)   # CA3: place -> (sensor, input) that came next
        self.transitions = Associations(d, theta=w // 2)   # CA3: next place -> (place, input)
        self.end = SDR.random(d, w, np.random.default_rng((seed, 37)))
        self.place = self.start
        self.sensors, self.heard = set(), set()         # the content sensors; those heard at the current place
        self.today = {}                                 # part 8: cortex location -> Counter((sensor, input)), until sleep

    def begin(self):
        self.place, self.heard = self.start, set()

    def finish(self, learn=True):
        """The episode ended: on each content sensor not heard at this place, the end came next."""
        if learn:
            for s in sorted(self.sensors - self.heard):
                self.followed.store(self.place, (s, self.end))

    def step(self, x, sensor="symbols", moves=True, learn=True, content=True, where=None):
        """Take input x: remember it as content (if learning, and unless it is an efference copy), also under the
        cortex's location `where` if given, and, if it moves the place, move to the situation's place."""
        if learn and content:
            self.followed.store(self.place, (sensor, x))
            if where is not None:
                self.today.setdefault(where, Counter())[(sensor, x)] += 1
        if content:
            self.sensors.add(sensor)
            self.heard.add(sensor)
        if moves:
            self.heard = set()
            nxt = self.dg(self.place, x)
            if learn:
                self.transitions.store(nxt, (self.place, x))
            self.place = nxt

    def predict(self, sensor="symbols"):
        """The input of this sensor that most often came next here, if this place was ever lived through."""
        came = {x: n for (s, x), n in self.followed.recall(self.place).items() if s == sensor}
        return max(came, key=came.get) if came else None

    def recall_today(self, where, sensor="symbols"):
        """Part 8: the input of this sensor that most often came next today at the cortex's location `where`, or
        None."""
        came = {x: n for (s, x), n in self.today.get(where, {}).items() if s == sensor}
        return max(came, key=came.get) if came else None

    def consolidated(self):
        """Sleep has consolidated the day: its location memories are cleared (the episodes stay, for replay)."""
        self.today = {}

    def replay(self):
        """Every remembered transition: [(place, input, next place, times seen)]."""
        return [(place, x, nxt, n) for nxt, came in self.transitions.items() for (place, x), n in came.items()]

    def contents(self):
        """Every remembered content: [(place, sensor, input, times seen)]."""
        return [(place, s, x, n) for place, came in self.followed.items() for (s, x), n in came.items()]
