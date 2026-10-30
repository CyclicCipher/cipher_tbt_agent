"""The brain (ARCHITECTURE.md section 4): regions of columns built from a config, the hippocampus beside them, and the
thalamocortical bus between them and the world.

**Awake,** each input from a sensor is published on the bus (bus.py), which stamps it and delivers it exactly once:
- to the hippocampus, which remembers it in one shot;
- to every column whose region receives that sensor, which senses it and, if the input moves its place (an efference
  copy, or a read symbol), moves by it.

**Asleep (`sleep`),** the hippocampus replays what it remembers, and consolidation (frameLearning.py) turns it into a
frame, with coordinates where the frame's moves form a commutative group. Every column installs it.

**A prediction of the next input** from a sensor comes from the cortex while a column knows where it is, and from the
hippocampus otherwise. So before any sleep, and wherever the frame has no move, the brain can only recall what it has
lived through. Since part 8, what the hippocampus saw today at the cortex's location comes first: a day's memory at a
place overrides older knowledge there, until sleep consolidates it.

**Not built yet:** one frame per receptive field (regions that receive different sensors would need their own
consolidation). A1's and the walk worlds' configs have one region.
"""
from arc_agent_v2.architecture.bus import Bus
from arc_agent_v2.architecture.column import Column
from arc_agent_v2.config.config import ARITHMETIC
from arc_agent_v2.architecture.frameLearning import consolidate
from arc_agent_v2.architecture.hippocampus import Hippocampus
from arc_agent_v2.architecture.information import bits, laplace


class Brain:
    def __init__(self, config=ARITHMETIC, consolidation=True, regularity=True, coordinates=True, factoring=True,
                 relation=True, locations=True, recency=True, union=True):
        self.config, self.consolidation = config, consolidation
        self.regularity, self.coordinates, self.factoring = regularity, coordinates, factoring
        self.relation = relation                        # part 6: shift classes related by an automorphism
        self.recency = recency                          # part 8: the day's memory at a location comes first
        d, w, seed = config.sdr.d, config.sdr.w, config.seed
        self.regions = {r.name: [Column(d, w, seed + 97 * i, union) for i in range(r.columns)]   # part 9: union
                        for r in config.validate()}
        self.hippocampus = Hippocampus(d, w, seed)
        self.bus = Bus(config, self.regions, self.hippocampus, locations)
        self.frame = None

    def fed_by(self, sensor):
        return [c for c, _ in self.bus.routes.get(sensor, [])]

    def columns(self):
        return [c for cols in self.regions.values() for c in cols]

    # ---- awake --------------------------------------------------------------------------------------------------
    def begin(self):
        """A new episode: every column and the hippocampus go back to their start."""
        for c in self.columns():
            c.reset()
        self.hippocampus.begin()

    def step(self, x, sensor="symbols", learn=True):
        return self.bus.publish(sensor, x, learn)

    def predict(self, sensor="symbols"):
        """The next input from a sensor (part 8's order):
        1. what came next today at the cortex's location (a memory since the last sleep);
        2. else the first column fed by it that knows where it is and has content there;
        3. else the hippocampus's recall of the path lived through."""
        where = self.bus.where(sensor)
        today = self.hippocampus.recall_today(where, sensor) if where is not None else None
        if self.recency and today is not None:
            return today
        for c in self.fed_by(sensor):
            p = c.predict(sensor)
            if p is not None:
                return p
        return today if today is not None else self.hippocampus.predict(sensor)

    def bits(self, target, sensor="symbols"):
        """Part 9: the code length of target, in bits, under the distribution the prediction is read from (the same
        order as `predict`), by Laplace's rule over K, the number of inputs the brain has received (information.py)."""
        K = len(self.bus.inputs)
        count = lambda c: bits(*laplace(c.get(target, 0), sum(c.values()), K))
        where = self.bus.where(sensor)
        today = {x: n for (s, x), n in self.hippocampus.today.get(where, {}).items() if s == sensor}             if where is not None else {}
        if self.recency and today:
            return count(today)
        for c in self.fed_by(sensor):
            b = c.bits(target, sensor, K)
            if b is not None:
                return b
        if today:
            return count(today)
        return count({x: n for (s, x), n in self.hippocampus.followed.recall(self.hippocampus.place).items()
                      if s == sensor})

    def finish(self, learn=True):
        """The episode ended (the hippocampus records it)."""
        self.hippocampus.finish(learn)

    def episode(self, xs, sensor="symbols", learn=True):
        """An episode of inputs from one sensor, or of (sensor, input) pairs, to its end."""
        self.begin()
        for x in xs:
            s, x = x if isinstance(x, tuple) else (sensor, x)
            self.step(x, s, learn)
        self.finish(learn)

    def predict_after(self, prefix, sensor="symbols"):
        """Without learning: the prediction after this start of an episode."""
        self.episode(prefix, sensor, learn=False)
        return self.predict(sensor)

    # ---- asleep -------------------------------------------------------------------------------------------------
    def sleep(self):
        """Consolidation: replay the hippocampus into a frame, and give it to the columns."""
        if not self.consolidation:
            return None
        cfg = self.config.sdr
        self.frame = consolidate(self.hippocampus.replay(), self.hippocampus.contents(), self.hippocampus.start,
                                 self.regularity, (cfg.d, cfg.w, self.config.seed) if self.coordinates else None,
                                 self.factoring, self.hippocampus.end, self.relation)
        for c in self.columns():
            c.install(self.frame, self.coordinates)
        self.hippocampus.consolidated()
        return self.frame
