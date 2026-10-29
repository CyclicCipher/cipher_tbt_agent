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
lived through.

**Not built yet:** one frame per receptive field (regions that receive different sensors would need their own
consolidation). A1's and the walk worlds' configs have one region.
"""
from arc_agent_v2.architecture.bus import Bus
from arc_agent_v2.architecture.column import Column
from arc_agent_v2.config.config import ARITHMETIC
from arc_agent_v2.architecture.frameLearning import consolidate
from arc_agent_v2.architecture.hippocampus import Hippocampus


class Brain:
    def __init__(self, config=ARITHMETIC, consolidation=True, regularity=True, coordinates=True, factoring=True,
                 relation=True):
        self.config, self.consolidation = config, consolidation
        self.regularity, self.coordinates, self.factoring = regularity, coordinates, factoring
        self.relation = relation                        # part 6: shift classes related by an automorphism
        d, w, seed = config.sdr.d, config.sdr.w, config.seed
        self.regions = {r.name: [Column(d, w, seed + 97 * i) for i in range(r.columns)] for r in config.validate()}
        self.hippocampus = Hippocampus(d, w, seed)
        self.bus = Bus(config, self.regions, self.hippocampus)
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
        """The next input from a sensor: the first column fed by it that knows where it is and has content there; else
        the hippocampus."""
        for c in self.fed_by(sensor):
            p = c.predict(sensor)
            if p is not None:
                return p
        return self.hippocampus.predict(sensor)

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
        return self.frame
