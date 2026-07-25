"""region.py — a cortical REGION: a column plus the declared WIRING that says what it is.

`htm.py` argues that a LAYER is one `HTMLayer` plus a declared wiring — the three dendrite-zone inputs and the projection
target — because biology fixes a layer's role by CONNECTIVITY inside a uniform microcircuit, not by a different algorithm.
This is the same statement one level up: a REGION is one `Column` plus a declaration of where its inputs come from.

WHY THIS IS NOT A MODALITY. A `Modality` is a SENSE: it has a transducer at the periphery turning world into readings. The
neocortex does not add senses to represent events, rules or causes — there is no receptor for them. It adds LEVELS: a higher
region's proximal input is the OUTPUT OF OTHER REGIONS, arriving cortico-cortically (L2/3 IT) and transthalamically (L5PT →
higher-order thalamus). So "an event modality" is a category error, and the thing that was missing is a way to say *this
column is driven by that column* rather than by a transducer.

WHAT MAKES A REGION THE REGION IT IS, then, is entirely its wiring — Mountcastle; Douglas & Martin. PFC needs no special
case and no transducer: it is a region whose proximal input is the convergent output of many regions, whose apical input
carries value/goal feedback, and whose L5IT projects to the striatum (`Column.striatum`). Declaring that is all it takes;
the algorithm is the same one V1 runs.

Deliberately thin: this declares and resolves wiring, and holds no state of its own. The columns do the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Region:
    """One cortical region: a `Column` and where its dendrite zones are fed from.

    `proximal` names the SOURCE of its feedforward drive — a modality name for a peripheral region (a transducer supplies
    it) or another region's name for a higher one (that region's output supplies it). `frame` names what supplies its L6a
    location: the body's pose for a sensorimotor region, or another region's pose space for a compositional one. `target`
    is where its output goes, which biology fixes by cell-type identity rather than by input (`htm.py`)."""

    name: str
    column: object                       # the `Column` running this region's microcircuit
    proximal: Optional[str] = None       # source of the L4 feedforward drive: a modality, or another region
    frame: Optional[str] = None          # source of the L6a location code
    target: Optional[str] = None         # where its output projects (a region, or 'striatum' / 'motor')

    def is_peripheral(self, modalities) -> bool:
        """A region is PERIPHERAL iff a transducer feeds it. Everything else is fed by cortex, which is the only structural
        difference between a sensory area and an association area."""
        return self.proximal in modalities


class Hierarchy:
    """The declared set of regions and the edges between them — the heterarchy, as a thing that can be inspected and
    asserted about rather than an implicit pattern of method calls between specialised columns."""

    def __init__(self) -> None:
        self.regions: dict = {}

    def add(self, region: Region) -> Region:
        self.regions[region.name] = region
        return region

    def get(self, name: str):
        return self.regions.get(name)

    def edges(self) -> list:
        """Every CORTICAL edge `(source_region, target_region)` — a region whose proximal drive is another region's output.
        An empty list means there is no hierarchy, only parallel columns; that was the honest state until the sensory →
        compositional edge was built."""
        return [(r.proximal, name) for name, r in self.regions.items()
                if r.proximal is not None and r.proximal in self.regions]
