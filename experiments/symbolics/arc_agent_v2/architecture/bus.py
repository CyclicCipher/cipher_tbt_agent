"""The thalamocortical bus (ARCHITECTURE.md section 4): every input reaches the brain's parts through here, and only here.

**Why one router (the user, 2026-10-29):** efference copies went wrong before when several columns were in the mix,
"one column had the efference, the others did not". A move that reaches some columns and not others leaves their
places silently out of step, and nothing downstream can tell.

**What it does:**
- **Stamping:** each input from a sensor is stamped with the next step number.
- **Delivery:** in step order and exactly once, to the hippocampus (which hears everything) and to every column whose
  region receives that sensor (`config`: the region's feedforward sources). That is each column's receptive field.
- **Roles:** for each column, the bus says whether the input moves its place (an efference copy, or a symbol read in
  sequence: the region's `moves`) or is sensed at it, and whether it is content. An efference copy never is
  (`config.efference`): the agent's own action is not something a place predicts.

**Its guarantees:**
1. **All or nothing per step:** every subscriber gets a message before the next one is published.
2. **No silent drift:** each column records the stamp of the last message it applied. Before delivering, the bus
   checks that the column applied the last message it was sent. A column that missed one (some code path stepped it,
   or skipped it, outside the bus) is marked lost (its place is None until the next episode), and the fault is
   counted.
3. **Sources outside a column's receptive field never reach it,** so they never move it.

**What a copy means in each frame is learned per column:** an efference copy is the same SDR for every subscriber,
and what it does in each column's frame is that column's own move (L5). So a column whose frame is different learns
its own meaning for the same copy.

**Not built yet:**
- the thalamus's change of frame (egocentric to allocentric), which no region needs yet;
- a separate consolidation per receptive field (B1).

The efference copy's route up the hierarchy (L5a to a higher region, through the thalamus) is the same delivery,
once there are regions above.
"""


class Bus:
    def __init__(self, config, regions, hippocampus, locations=True):
        self.t, self.faults = 0, 0
        self.locations = locations                      # part 8: give the hippocampus the cortex's location
        self.inputs = set()                             # part 9: every distinct input received (K, for Laplace)
        self.hippocampus = hippocampus
        self.routes = {s: [] for s in config.sensors}   # sensor -> [(column, moves)]
        self.moving = set()                             # sensors that move some column (and so the hippocampus)
        for r in config.regions:
            for s in r.feedforward:
                if s in self.routes:
                    moves = s in r.moving()
                    self.moving |= {s} if moves else set()
                    self.routes[s] += [(c, moves) for c in regions[r.name]]
        self.sent = {}                                  # id(column) -> the stamp last sent to it
        self.efference = set(config.efference)

    def publish(self, sensor, x, learn=True):
        """Stamp x and deliver it. Returns the stamp."""
        self.t += 1
        self.inputs.add(x)
        content = sensor not in self.efference
        where = self.where(sensor) if self.locations else None
        self.hippocampus.step(x, sensor, sensor in self.moving, learn, content, where)
        for c, moves in self.routes[sensor]:
            if c.stamp != self.sent.get(id(c), 0):     # it missed a message, or applied one twice
                self.faults += 1
                c.l6.place = None
            c.step(x, sensor, moves, self.t, content)
            self.sent[id(c)] = self.t
        return self.t

    def where(self, sensor):
        """The cortex's location for this sensor (the entorhinal input, part 8): the place of the first column it
        reaches that knows its place, or None."""
        return next((c.l6.place for c, _ in self.routes.get(sensor, []) if not c.l6.lost()), None)
