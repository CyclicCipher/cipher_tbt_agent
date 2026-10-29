"""The brain (ARCHITECTURE.md section 4): regions of columns built from a config, and the hippocampus beside them.

**Awake,** each input does two things:
- the hippocampus remembers it in one shot;
- every column the input's sensor feeds senses it and moves by it.

**Asleep (`sleep`),** the hippocampus replays what it remembers, and consolidation (frameLearning.py) turns it into each
column's frame.

**A prediction of the next input** comes from the cortex while its column knows where it is, and from the hippocampus
otherwise. So before any sleep, and wherever the frame has no move, the brain can only recall what it has lived
through.

**Routing:** an input from a sensor goes to the regions whose feedforward includes that sensor. Links between regions
(bus.py) are not built yet, because nothing needs them in A1.
"""
from arc_agent_v2.column import Column
from arc_agent_v2.config import ARITHMETIC
from arc_agent_v2.frameLearning import consolidate
from arc_agent_v2.hippocampus import Hippocampus


class Brain:
    def __init__(self, config=ARITHMETIC, consolidation=True, regularity=True):
        self.config, self.consolidation, self.regularity = config, consolidation, regularity
        d, w, seed = config.sdr.d, config.sdr.w, config.seed
        self.regions = {r.name: [Column(d, w, seed + 97 * i) for i in range(r.columns)] for r in config.validate()}
        self.hippocampus = Hippocampus(d, w, seed)
        self.frame = None

    def fed_by(self, sensor):
        return [c for r in self.config.regions if sensor in r.feedforward for c in self.regions[r.name]]

    def columns(self):
        return [c for cols in self.regions.values() for c in cols]

    # ---- awake --------------------------------------------------------------------------------------------------
    def begin(self):
        """A new episode: every column and the hippocampus go back to their start."""
        for c in self.columns():
            c.reset()
        self.hippocampus.begin()

    def step(self, x, sensor="symbols", learn=True):
        self.hippocampus.step(x, learn)
        for c in self.fed_by(sensor):
            c.step(x)

    def predict(self, sensor="symbols"):
        """The next input: the first column fed by the sensor that knows where it is and has content there; else the
        hippocampus."""
        for c in self.fed_by(sensor):
            p = c.predict()
            if p is not None:
                return p
        return self.hippocampus.predict()

    def episode(self, xs, sensor="symbols", learn=True):
        self.begin()
        for x in xs:
            self.step(x, sensor, learn)

    def predict_after(self, prefix, sensor="symbols"):
        """Without learning: the prediction after this start of an episode."""
        self.begin()
        for x in prefix:
            self.step(x, sensor, learn=False)
        return self.predict(sensor)

    # ---- asleep -------------------------------------------------------------------------------------------------
    def sleep(self):
        """Consolidation: replay the hippocampus into a frame, and give it to the columns."""
        if not self.consolidation:
            return None
        self.frame = consolidate(self.hippocampus.replay(), self.hippocampus.start, self.regularity)
        for c in self.columns():
            c.install(self.frame)
        return self.frame
