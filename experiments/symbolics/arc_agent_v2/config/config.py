"""The brain's configuration (ARCHITECTURE.md section 3): its hyperparameters, and the regions of its neocortex.

A configuration is data. `Brain(config)` (brain.py) builds what it describes; changing the brain's shape, such as
adding a region, a link or a column, is an edit here, not in the code. Hyperparameters are added here as the parts
that use them are built (nothing is configured before it is used), so for now only the SDR's are.

**A region:** a sheet of columns sharing a kind of frame. Its fields:
- `columns`: how many columns it has;
- `frame`: the frame they use, innate ("innate:<kind>", given the way retinotopy is given) or "learned" (found by
  frameLearning.py from the data);
- `feedforward` and `feedback`: the regions or sensors it receives from. Feedforward links must not form a cycle;
  feedback links usually reverse feedforward ones.

All links run through the thalamocortical bus (bus.py). The hippocampus is not a region of the neocortex: it is
configured apart (hippocampus.py).

**Inputs that move, and inputs that are sensed (part 4):** `moves` lists the sources whose inputs move the region's
columns' place (P1): an efference copy of an action, or a symbol read in sequence. Every other source's inputs are
sensed at the place (content, P2) and don't move it. By default every source moves (P1 as A1 uses it). This is
wiring, set per region, as the efference copy reaches L5/L6 and sensation lands in L4.

**Efference copies (part 4):** `efference` lists the sensors that carry copies of the agent's own actions. Their inputs
move the place where a region lists them in `moves`, and they are never content: an action is what the agent does at
a place, not something the place predicts. A place's observed future is what the world shows for each action taken,
so an agent's own choices never count as a difference between two places. A read symbol (A1) comes from the world, so
it both moves the place and is content.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SDRConfig:
    d: int = 2048                       # bits per SDR (the user: 2048 for now)
    w: int = 40                         # active bits in a free SDR (2%; two random SDRs share 0.8 bits)
    blocks: int = 32                    # blocks in a block code: 32 blocks of 64 bits (an 8 x 8 torus for places)


@dataclass(frozen=True)
class Region:
    name: str
    columns: int
    frame: str                          # "innate:<kind>" or "learned"
    feedforward: tuple = ()             # sources: region or sensor names
    feedback: tuple = ()
    note: str = ""
    moves: tuple = None                 # the sources whose inputs move the place (None: every source)
    shape: tuple = None                 # part 12: an innate retinotopic sheet's positions (rows, columns)

    def moving(self):
        return tuple(self.feedforward) if self.moves is None else tuple(self.moves)


@dataclass(frozen=True)
class BrainConfig:
    sdr: SDRConfig = SDRConfig()
    sensors: tuple = ()                 # where data enters as SDRs (HDClibrary's encoders)
    regions: tuple = ()
    seed: int = 0
    efference: tuple = ()               # sensors that carry copies of the agent's own actions (part 4)

    def region(self, name):
        return next(r for r in self.regions if r.name == name)

    def validate(self):
        """Names unique, every link's source exists, the feedforward links acyclic. Returns the regions in feedforward
        order (each after its sources)."""
        names = [r.name for r in self.regions]
        if len(set(names)) != len(names) or set(names) & set(self.sensors):
            raise ValueError("region and sensor names must be unique")
        if not set(self.efference) <= set(self.sensors):
            raise ValueError("an efference copy must come from a sensor")
        known = set(names) | set(self.sensors)
        for r in self.regions:
            for src in r.feedforward + r.feedback:
                if src not in known:
                    raise ValueError("%s receives from unknown %s" % (r.name, src))
            if r.columns < 1 or not (r.frame == "learned" or r.frame.startswith("innate:")):
                raise ValueError("bad region %s" % r.name)
            if r.frame == "innate:retinotopic-2d" and (r.shape is None or len(r.shape) != 2):
                raise ValueError("%s: a retinotopic sheet needs its shape" % r.name)
            if not set(r.moving()) <= set(r.feedforward):
                raise ValueError("%s moves on a source it doesn't receive" % r.name)
        order, done = [], set(self.sensors)
        while len(order) < len(self.regions):
            ready = [r for r in self.regions if r.name not in done and set(r.feedforward) <= done]
            if not ready:
                raise ValueError("the feedforward links form a cycle")
            for r in ready:
                order.append(r)
                done.add(r.name)
        return order


# ---- presets ---------------------------------------------------------------------------------------------------------
ARITHMETIC = BrainConfig(
    sensors=("symbols",),
    regions=(Region("assoc", 1, "learned", ("symbols",),
                    note="milestone A: symbols are random SDRs; the frame they live on must be learned"),))

WALK = BrainConfig(
    sensors=("motor", "retina"), efference=("motor",),
    regions=(Region("assoc", 1, "learned", ("motor", "retina"), moves=("motor",),
                    note="part 4: actions (efference copies) move the place; observations are sensed at it"),))

MNIST = BrainConfig(
    sensors=("retina",),
    regions=(Region("V1", 16, "innate:retinotopic-2d", ("retina",), ("IT",), shape=(4, 4),
                    note="milestone B: each column senses a patch; its place is given by the innate 2-D frame"),
             Region("IT", 1, "learned", ("V1",),
                    note="the digit: content at places, in a frame learned from the data")))

HUMAN = BrainConfig(                    # the target layout: data only, not wired yet (scaled-down column counts)
    sensors=("retina", "proprioception"),
    regions=(
        Region("V1", 64, "innate:retinotopic-2d", ("retina",), ("V2",), shape=(8, 8)),
        Region("V2", 32, "innate:retinotopic-2d", ("V1",), ("V4", "MT"), shape=(4, 8)),
        Region("V4", 16, "learned", ("V2",), ("IT",), note="ventral stream: what"),
        Region("IT", 8, "learned", ("V4",), ("PFC",), note="objects in their own frames"),
        Region("MT", 16, "innate:retinotopic-2d", ("V2",), ("PPC",), note="dorsal stream: motion", shape=(4, 4)),
        Region("PPC", 8, "innate:egocentric-2d", ("MT", "V2"), ("PFC", "PMC"), note="where; space for action"),
        Region("EC", 4, "innate:allocentric-grid", ("IT", "PPC"), ("PFC",),
               note="entorhinal: the cortex's gateway to the hippocampus"),
        Region("PFC", 4, "learned", ("IT", "PPC", "EC"), (), note="task frames; routes between the others"),
        Region("PMC", 8, "learned", ("PFC", "PPC"), ("PFC",), note="action sequences"),
        Region("M1", 8, "innate:motor-map", ("PMC", "proprioception"), ("PMC",)),
    ))

JOINT = BrainConfig(                    # part 12: arithmetic and images in one brain, a region per sensor
    sensors=("symbols", "retina", "gaze"), efference=("gaze",),
    regions=(Region("S", 1, "learned", ("symbols",), note="read symbols: arithmetic"),
             Region("V", 1, "innate:retinotopic-2d", ("retina", "gaze"), moves=("gaze",), shape=(4, 4),
                    note="a retinal patch, at the place the gaze's innate shifts bring it to")))

PRESETS = {"arithmetic": ARITHMETIC, "walk": WALK, "mnist": MNIST, "human": HUMAN, "joint": JOINT}


def retinotopy(config, name):
    """Part 12: the innate retinotopic code of region `name` (HDClibrary.Retinotopic), the same for the brain's frame
    and for the body's gaze: one genome."""
    from arc_agent_v2.architecture.HDClibrary import Retinotopic
    names = [r.name for r in config.regions]
    r = config.region(name)
    return Retinotopic(config.sdr.d, config.sdr.w, r.shape, seed=config.seed * 1000 + 606 + names.index(name))
