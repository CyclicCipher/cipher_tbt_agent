"""The brain blueprint (ARCHITECTURE.md section 2): which areas the brain has, how many columns each holds, and which
reference frame they use. Innate structure lives here (priors are geometries); learned areas get only a menu of frame
types and find their own frames from the data. `Brain(blueprint)` builds the columns. Adding an area, a frame type or
(later) a PFC is an edit here, not in the brain.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Area:
    name: str
    role: str                     # "visual" | "motor" | "association" | "episodic" | "executive"
    columns: int                  # how many column slots
    frame: str                    # innate frame name, or "learned"
    frame_types: tuple = ()       # for learned areas: the frame types a column may take
    inputs: tuple = ()            # names of the areas that feed this one
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Blueprint:
    areas: tuple

    def area(self, name):
        return next(a for a in self.areas if a.name == name)

    def by_role(self, role):
        return [a for a in self.areas if a.role == role]


DEFAULT = Blueprint(areas=(
    Area("V1", "visual", 1, frame="retina", params=dict(height=64, width=64)),
    Area("M1", "motor", 1, frame="motor_map", inputs=("V1",)),
    Area("assoc", "association", 32, frame="learned", frame_types=("translation", "recolour", "bar"),
         inputs=("V1", "M1")),
    Area("HC", "episodic", 1, frame="product", inputs=("assoc",)),
    Area("PFC", "executive", 0, frame="router", inputs=("assoc", "HC")),     # a stand-in for now: executive.py
))
