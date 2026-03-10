"""Live CTKG: accumulates a global KnowledgeGraph via sheaf_merge() on each
segment boundary detected by the MorphismGraph.

Architecture (BLUEPRINT.md §"Global structure: the CTKG as a sheaf"):
  - Each segment boundary emits a local chunk.
  - FCA on the chunk yields formal concepts (Galois connections = adjunctions).
  - sheaf_merge() integrates the local concept lattice into the global CTKG.
  - SheafViolation → two usages of the same symbol are structurally incompatible
    → sense_disambiguate() creates two distinct CTKG concepts.

The CTKG is built on experiments/ctkg/graph.py which is unchanged.
LiveCTKG is the bridge between the MorphismGraph (perceptual layer) and the
CTKG (conceptual layer).

Usage:
    topo = sequence_1d()
    mg   = MorphismGraph()
    ctkg = LiveCTKG(topo)
    mg.on_segment(ctkg.on_segment)   # wire callback
    mg.observe_sequence(data, topo)
    mg.flush()
    print(ctkg.global_kg)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to path so ctkg/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.ctkg.graph import (
    KnowledgeGraph, Concept, TypeDef, Prerequisite,
    Adjunction, Interface, SheafViolation,
)
try:
    from experiments.ctkg.parser import merge as ctkg_merge
except ImportError:
    ctkg_merge = None

from ..core.morphism import Atom, Composition, MorphismGraph
from ..core.topology import Topology
from .fca import concepts_from_chunk


class LiveCTKG:
    """Accumulates a global CTKG from per-chunk FCA results.

    Registered as a segment-boundary callback on a MorphismGraph.
    Each call to on_segment() processes one completed chunk.
    """

    def __init__(self, topology: Topology) -> None:
        self.topology   = topology
        self.global_kg  = KnowledgeGraph()
        self._n_merges  = 0   # total sheaf merges performed
        self._n_violations = 0  # total sense disambiguations triggered

    # ── Segment boundary callback ─────────────────────────────────────────────

    def on_segment(
        self, chunk: list[tuple[int, Optional[int]]], mg: MorphismGraph
    ) -> None:
        """Process one completed chunk: run FCA, build local KG, sheaf-merge.

        Called automatically by MorphismGraph._emit_chunk().
        """
        n_etypes = self.topology.n_edge_types()
        concepts = concepts_from_chunk(chunk, n_etypes)

        if not concepts:
            return

        local_kg = self._build_local_kg(chunk, concepts, mg)
        self._sheaf_merge(local_kg, mg)

    # ── Local KG construction ─────────────────────────────────────────────────

    def _build_local_kg(
        self,
        chunk: list[tuple[int, Optional[int]]],
        concepts: list[tuple[frozenset, frozenset]],
        mg: MorphismGraph,
    ) -> KnowledgeGraph:
        """Build a KnowledgeGraph from the formal concepts in a chunk.

        Each formal concept (symbol_set, edge_type_set) becomes:
          - A TypeDef with a name derived from the shared edge types.
          - A Concept for each symbol in the symbol_set.
          - An Interface listing the shared edge types as exported attributes.

        Composition symbols are also added as concepts with their rule as
        a Prerequisite (left →[etype]→ right).
        """
        kg = KnowledgeGraph()
        reg = self.topology.registry

        for obj_set, attr_set in concepts:
            if not obj_set or not attr_set:
                continue

            # TypeDef: name = sorted edge type names joined by '+'
            etype_names = sorted(reg.name(a) for a in attr_set)
            type_name   = "_".join(etype_names) + "_group"

            # Avoid duplicate type names
            if type_name not in {t.name for t in kg.types.values()}:
                td = TypeDef(
                    name=type_name,
                    constructor="tagged",
                    params=etype_names,
                    annotations=frozenset(),
                )
                kg.add_type(td)

            # Concept for each symbol in the set
            for sid in obj_set:
                sym = mg.symbols[sid]
                if isinstance(sym, Atom):
                    concept_name = f"atom_{sym.value!r}"
                    concept_type = type_name
                elif isinstance(sym, Composition):
                    concept_name = f"comp_{sid}"
                    concept_type = type_name
                else:
                    continue

                if concept_name not in kg.concepts:
                    c = Concept(
                        name=concept_name,
                        type_name=concept_type,
                        level=sym.level,
                    )
                    kg.add_concept(c)

            # Composition prerequisites: left req → right (via composition)
            for sid in obj_set:
                sym = mg.symbols[sid]
                if isinstance(sym, Composition):
                    left_sym  = mg.symbols[sym.left]
                    right_sym = mg.symbols[sym.right]
                    if isinstance(left_sym, Atom):
                        left_name = f"atom_{left_sym.value!r}"
                    else:
                        left_name = f"comp_{sym.left}"
                    if isinstance(right_sym, Atom):
                        right_name = f"atom_{right_sym.value!r}"
                    else:
                        right_name = f"comp_{sym.right}"
                    comp_name = f"comp_{sid}"

                    if (left_name in kg.concepts and
                            right_name in kg.concepts and
                            comp_name in kg.concepts):
                        p = Prerequisite(
                            source=left_name,
                            target=comp_name,
                            role=reg.name(sym.etype),
                            transfer_probability=1.0,
                        )
                        try:
                            kg.add_prerequisite(p)
                        except Exception:
                            pass  # duplicate or structural issue — skip silently

        return kg

    # ── Sheaf merge ───────────────────────────────────────────────────────────

    def _sheaf_merge(self, local_kg: KnowledgeGraph, mg: MorphismGraph) -> None:
        """Merge local_kg into global_kg, handling SheafViolations."""
        try:
            self.global_kg.sheaf_merge(local_kg)
            self._n_merges += 1
        except SheafViolation as v:
            self._n_violations += 1
            self._sense_disambiguate(v, local_kg, mg)
            # After disambiguation, try merge again with the updated local_kg
            try:
                self.global_kg.sheaf_merge(local_kg)
                self._n_merges += 1
            except SheafViolation:
                pass  # second attempt failed — log and continue

    def _sense_disambiguate(
        self,
        violation: SheafViolation,
        local_kg: KnowledgeGraph,
        mg: MorphismGraph,
    ) -> None:
        """On a SheafViolation, create two distinct concepts for the clashing symbol.

        The symbol appears in two incompatible structural roles (e.g. the Latin
        word 'est' appears as both a verb and a ligature).  We suffix the local
        concept with '_sense2' to distinguish the two usages.

        This is the automatic sense disambiguation mechanism from the BLUEPRINT.
        """
        # violation.message contains the conflicting concept name
        # Rename it in the local KG to avoid the clash
        msg = str(violation)
        # Simple heuristic: extract the concept name from the violation message
        # Full implementation would parse the violation details more carefully
        for cname in list(local_kg.concepts.keys()):
            if cname in msg:
                new_name = cname + "_sense2"
                concept  = local_kg.concepts.pop(cname)
                concept.name = new_name
                local_kg.concepts[new_name] = concept
                break

    # ── Inspection ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        kg = self.global_kg
        return (
            f"LiveCTKG("
            f"types={len(kg.types)}, "
            f"concepts={len(kg.concepts)}, "
            f"merges={self._n_merges}, "
            f"disambiguations={self._n_violations})"
        )

    def __repr__(self) -> str:
        return self.summary()
