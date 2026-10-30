"""The brain (ARCHITECTURE.md section 4): regions of columns built from a config, the hippocampus beside them, and the
thalamocortical bus between them and the world.

**Awake,** each input from a sensor is published on the bus (bus.py), which stamps it and delivers it exactly once:
- to every column whose region receives that sensor, which senses it and, if the input moves its place (an efference
  copy, or a read symbol), moves by it;
- then to the hippocampus, which remembers it in one shot, under the cortex's location.

**Asleep (`sleep`),** the hippocampus replays what it remembers. Each region consolidates its own receptive field
(part 12): the replay of its own sensors' inputs, into its own frame.
- **A learned frame** comes from consolidation (frameLearning.py), with coordinates where the frame's moves form a
  commutative group.
- **An innate frame** is given (innate.py, rule 5), and only its content is learned.
- **Then its objects** are learned from the episodes (objects.py), and every column of the region installs the frame.

**A prediction of the next input** from a sensor comes from the cortex while a column knows where it is, and from the
hippocampus otherwise. So before any sleep, and wherever the frame has no move, the brain can only recall what it has
lived through. What the hippocampus saw today at the cortex's location is one more object in the cortex's union (part
12, replacing part 8's override), so the day's memory wins where the evidence favours it.

**`today=False`** leaves today's memory out of the cortex's union (part 8's arm (c), "recency off" before part 12).
"""
from arc_agent_v2.architecture import innate, objects
from arc_agent_v2.architecture.bus import Bus
from arc_agent_v2.architecture.column import Column, renaming
from arc_agent_v2.architecture.frameLearning import consolidate
from arc_agent_v2.architecture.hippocampus import Hippocampus
from arc_agent_v2.architecture.information import Content, mixture_bits
from arc_agent_v2.architecture.thalamus import Thalamus
from arc_agent_v2.config.config import ARITHMETIC, retinotopy


class Brain:
    def __init__(self, config=ARITHMETIC, consolidation=True, regularity=True, coordinates=True, factoring=True,
                 relation=True, locations=True, today=True, union=True, thalamus=True):
        self.config, self.consolidation = config, consolidation
        self.regularity, self.coordinates, self.factoring = regularity, coordinates, factoring
        self.relation = relation                        # part 6: shift classes related by an automorphism
        d, w, seed = config.sdr.d, config.sdr.w, config.seed
        self.regions = {r.name: [Column(d, w, seed + 97 * i, union) for i in range(r.columns)]   # part 9: union
                        for r in config.validate()}
        self.hippocampus = Hippocampus(d, w, seed)
        for c in self.columns():                        # part 12: the day's memory is an object of the cortex's
            c.day, c.today_on = self.hippocampus, today
        self.bus = Bus(config, self.regions, self.hippocampus, locations)
        read = {s for s in self.bus.moving if s not in self.bus.efference}   # inputs read: content that moves
        first = self.columns()[0]
        field = {s for s, routes in self.bus.routes.items() if any(c is first for c, _ in routes)}
        self.thalamus = Thalamus(first, read & field, enabled=thalamus, field=field)   # part 10: working memory
        self.frames, self.frame = {}, None

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
        self.thalamus.begin()

    def step(self, x, sensor="symbols", learn=True):
        """One input: the thalamus branches, weighs and moves its thoughts by it (its estimate of where x came is the
        hippocampus's location), then the bus delivers it."""
        moves, content = sensor in self.bus.moving, sensor not in self.bus.efference
        where = Bus.OWN
        if self.thalamus.receives(sensor):
            w = self.thalamus.observe(x, sensor, moves, content, learn)
            if self.thalamus.active():
                where = w
            self.thalamus.advance(x, sensor, moves, content)
        return self.bus.publish(sensor, x, learn, where)

    def predict(self, sensor="symbols"):
        """The next input from a sensor:
        1. the thalamus's thoughts, while it holds any and the sensor reaches its column (part 10);
        2. else the first column fed by it that has content where it is (today's memory among its objects);
        3. else the hippocampus's recall of the path lived through."""
        if self.thalamus.active() and self.thalamus.receives(sensor):
            p = self.thalamus.predict(sensor)
            if p is not None:
                return p
        for c in self.fed_by(sensor):
            p = c.predict(sensor)
            if p is not None:
                return p
        return self.hippocampus.predict(sensor)

    def bits(self, target, sensor="symbols"):
        """Part 9: the code length of target, in bits, under the distribution the prediction is read from (the same
        order as `predict`), normalized over the inputs of that sensor (information.py)."""
        known = tuple(self.bus.inputs_by.get(sensor, ()))
        if self.thalamus.active() and self.thalamus.receives(sensor):
            b = self.thalamus.bits(target, sensor, known)
            if b is not None:
                return b
        for c in self.fed_by(sensor):
            b = c.bits(target, sensor, known)
            if b is not None:
                return b
        came = {x: n for (s, x), n in self.hippocampus.followed.recall(self.hippocampus.place).items() if s == sensor}
        d, w = self.config.sdr.d, self.config.sdr.w
        return mixture_bits([(0.0, Content(d, w, came))], target, known)

    def finish(self, learn=True):
        """The episode ended (the hippocampus records it; while learning, the thalamus's gate learns from it)."""
        self.hippocampus.finish(learn)
        self.thalamus.finish(learn)

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
    def _field(self, r):
        """A region's receptive field: (the inputs that move it, the sensors whose inputs are its content)."""
        moving = {x for s in r.moving() for x in self.bus.inputs_by.get(s, ())}
        sensed = {s for s in r.feedforward if s in self.config.sensors and s not in self.bus.efference}
        return moving, sensed

    def _observations(self, frame, sensed):
        """Part 12: each episode's content at the states of the region's frame, [(index, [(state, sensor, input)])]."""
        fac = frame.factored if self.coordinates else None
        rename = renaming(frame, self.coordinates)
        out = []
        for i, events in enumerate(self.hippocampus.episodes):
            obs = []
            for h, s, x in events:
                if s not in sensed:
                    continue
                u = frame.places.get(h)
                if u is None:
                    continue
                if fac is not None:
                    if u in fac.coord:
                        obs.append(((fac.phase[u], fac.coord[u]), s, x))
                else:
                    obs.append(((None, rename.get(u, u)), s, x))
            out.append((i, obs))
        return out

    def sleep(self):
        """Consolidation: replay the hippocampus into each region's frame and objects, and give them to its columns.
        Returns the first region's frame."""
        if not self.consolidation:
            return None
        cfg = self.config.sdr
        transitions, contents, start = self.hippocampus.replay(), self.hippocampus.contents(), self.hippocampus.start
        self.frames = {}
        for r in self.config.validate():
            moving, sensed = self._field(r)
            mine = [t for t in transitions if t[1] in moving]
            reach, nxt = {start}, {}
            for h, x, q, _ in mine:
                nxt.setdefault(h, []).append(q)
            stack = [start]
            while stack:                                # the places this region's own moves reach
                for q in nxt.get(stack.pop(), ()):
                    if q not in reach:
                        reach.add(q)
                        stack.append(q)
            seen = [c for c in contents if c[1] in sensed and c[0] in reach]
            if r.frame == "learned":
                f = consolidate(mine, seen, start, self.regularity,
                                (cfg.d, cfg.w, self.config.seed) if self.coordinates else None,
                                self.factoring, self.hippocampus.end, self.relation)
            else:
                f = innate.frame(r.frame, mine, seen, start, retinotopy(self.config, r.name))
            f.objects = objects.learn(self._observations(f, sensed), cfg.d, cfg.w, self.config.seed)
            self.frames[r.name] = f
            for c in self.regions[r.name]:
                c.install(f, self.coordinates)
        self.hippocampus.consolidated()
        self.frame = next(iter(self.frames.values()), None)
        return self.frame
