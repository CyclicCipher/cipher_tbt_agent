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

import math

from experiments.ctkg.graph import (
    KnowledgeGraph, Concept, TypeDef, Prerequisite,
    Adjunction, Interface, SheafViolation, MasteryState,
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

        # Phase 19 L1 — Marcus type-token distinction:
        # Write FCA concept membership back to each atom so that type lookup is
        # O(1) at query time (atom.concept_ids) rather than re-running FCA.
        self._write_concept_ids(concepts, mg)

        local_kg = self._build_local_kg(chunk, concepts, mg)
        self._sheaf_merge(local_kg, mg)

    # ── Phase 19 L1: type-token write-back ───────────────────────────────────

    def _write_concept_ids(
        self,
        concepts: list[tuple[frozenset, frozenset]],
        mg: MorphismGraph,
    ) -> None:
        """Write FCA concept IDs back onto each atom in the chunk.

        For each formal concept (symbol_set, attr_set):
          1. Get or create a stable integer type ID from mg.fca_type_id(attr_set).
             The type ID is the same every time the same attr_set is seen, even
             across chunks — this gives Marcus-style stable type identity.
          2. For each atom in symbol_set, add the type ID to atom.concept_ids.

        After this runs, any atom that appeared in a formal concept satisfies
        Marcus's type-token distinction: the token (atom) explicitly carries
        references to its type(s) (concept_ids), not just implicit membership
        derivable by re-running FCA.

        Non-atom symbols (compositions) do not receive concept_ids here; they
        are handled by the CTKG prerequisite structure instead.
        """
        from ..core.morphism import Atom  # already imported at module top but safe
        for obj_set, attr_set in concepts:
            if not obj_set or not attr_set:
                continue
            type_id = mg.fca_type_id(attr_set)
            for sid in obj_set:
                sym = mg.symbols[sid]
                if isinstance(sym, Atom):
                    # frozenset union — creates a new frozenset (slots-safe)
                    sym.concept_ids = sym.concept_ids | {type_id}

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

        # ── Pass 1: add all TypeDefs and Concepts ─────────────────────────────
        # Must complete before Pass 2 so that prerequisites can reference any
        # concept regardless of which formal-concept group they came from.

        for obj_set, attr_set in concepts:
            if not obj_set or not attr_set:
                continue

            # TypeDef: name = sorted edge type names joined by '_'
            etype_names = sorted(reg.name(a) for a in attr_set)
            type_name   = "_".join(etype_names) + "_group"

            # Avoid duplicate type names
            if type_name not in {t.name for t in kg.types.values()}:
                td = TypeDef(
                    name=type_name,
                    constructor="tagged",
                    params=etype_names,
                    annotations=set(),
                    description=f"Morphism group with edge types: {etype_names}",
                )
                kg.add_type(td)

            # Concept for each symbol in the set
            for sid in obj_set:
                sym = mg.symbols[sid]
                if isinstance(sym, Atom):
                    concept_name = f"atom_{sym.value!r}"
                    desc = f"Atom '{sym.value}' (level {sym.level}, type group: {type_name})"
                elif isinstance(sym, Composition):
                    concept_name = f"comp_{sid}"
                    desc = f"Composition {sid} (level {sym.level}, type group: {type_name})"
                else:
                    continue

                if concept_name not in kg.concepts:
                    c = Concept(
                        name=concept_name,
                        description=desc,
                        domain="morphism",
                    )
                    kg.add_concept(c)

        # ── Pass 2: add Prerequisite edges for all Compositions ──────────────
        # Now that all concepts are registered, the source/target lookups will
        # succeed even when left/right atoms appear in a different FCA group
        # from the composition itself.

        for obj_set, attr_set in concepts:
            if not obj_set or not attr_set:
                continue
            for sid in obj_set:
                sym = mg.symbols[sid]
                if not isinstance(sym, Composition):
                    continue

                left_sym  = mg.symbols[sym.left]
                right_sym = mg.symbols[sym.right]
                left_name  = (f"atom_{left_sym.value!r}"
                              if isinstance(left_sym, Atom) else f"comp_{sym.left}")
                right_name = (f"atom_{right_sym.value!r}"
                              if isinstance(right_sym, Atom) else f"comp_{sym.right}")
                comp_name  = f"comp_{sid}"

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
                        pass  # duplicate — skip silently

        return kg

    # ── Sheaf merge ───────────────────────────────────────────────────────────

    def _sheaf_merge(self, local_kg: KnowledgeGraph, mg: MorphismGraph) -> None:
        """Merge local_kg into global_kg, handling SheafViolations.

        sheaf_merge() returns List[SheafViolation] — it never raises.
        An empty list means the merge succeeded and the graph was updated.
        A non-empty list means there were conflicts; the graph was NOT modified.
        """
        violations = self.global_kg.sheaf_merge(local_kg)
        if not violations:
            self._n_merges += 1
            return

        # At least one violation — sense-disambiguate using the first conflict,
        # then retry once with the renamed concept.
        self._n_violations += len(violations)
        self._sense_disambiguate(violations[0], local_kg, mg)
        violations2 = self.global_kg.sheaf_merge(local_kg)
        if not violations2:
            self._n_merges += 1
        # else: second attempt still conflicts — skip this chunk silently

    def _sense_disambiguate(
        self,
        violation: SheafViolation,
        local_kg: KnowledgeGraph,
        mg: MorphismGraph,
    ) -> None:
        """On a SheafViolation, create two distinct concepts for the clashing symbol.

        Two-level fix:
          1. CTKG level: rename the conflicting concept in local_kg with a
             "_sense2" suffix so the sheaf_merge retry sees no name clash.
          2. MorphismGraph level: call split_atom() to redistribute historical
             edges, ensuring future observations route to the correct atom ID.
             This step is best-effort and silently skipped on any failure.

        This is the full sense-disambiguation algorithm from BLUEPRINT.md §"FCA".
        """
        msg = str(violation)
        for cname in list(local_kg.concepts.keys()):
            if cname in msg:
                # Step 1: rename in local_kg (resolves CTKG-level clash)
                new_name = cname + "_sense2"
                concept  = local_kg.concepts.pop(cname)
                concept.name = new_name
                local_kg.concepts[new_name] = concept

                # Step 2: propagate split to MorphismGraph if possible
                if cname.startswith("atom_"):
                    self._try_split_atom(cname, concept, mg)
                break

    def _try_split_atom(
        self,
        concept_name: str,
        local_concept,
        mg: MorphismGraph,
    ) -> None:
        """Attempt to split the MorphismGraph atom that clashed in the CTKG.

        Extracts:
          - atom value from concept_name: "atom_'x'" → value = "x"
          - sense_b_etypes from local_concept.description:
              "Atom 'x' (level N, type group: next_group)" → etype "next"

        Calls mg.split_atom(atom_id, sense_a_etypes, sense_b_etypes).
        Silently returns on any parsing or lookup failure.
        """
        import ast as _ast

        # Recover atom value from concept_name = f"atom_{repr(value)}"
        try:
            value = _ast.literal_eval(concept_name[5:])
        except (ValueError, SyntaxError):
            return

        atom_id = mg.atoms.get(value)
        if atom_id is None:
            return

        # Parse type_name from description: "... type group: X_group"
        marker    = "type group: "
        desc      = local_concept.description
        idx       = desc.find(marker)
        if idx < 0:
            return
        type_name = desc[idx + len(marker):].rstrip(")")
        if not type_name.endswith("_group"):
            return

        # type_name = "_".join(sorted_etype_names) + "_group"
        # e.g. "next_group" → inner = "next" → etype_names = ["next"]
        inner       = type_name[:-6]   # strip "_group" suffix
        etype_names = inner.split("_")

        sense_b_etypes: set[int] = set()
        reg = self.topology.registry
        for name in etype_names:
            try:
                sense_b_etypes.add(reg.code(name))
            except Exception:
                pass

        if not sense_b_etypes:
            return

        all_etypes     = {reg.code(n) for n in reg.names()}
        sense_a_etypes = all_etypes - sense_b_etypes

        try:
            mg.split_atom(atom_id, sense_a_etypes, sense_b_etypes)
        except Exception:
            pass   # best-effort; CTKG-level rename already fixes the sheaf

    # ── Type-map extraction (Phase 10b) ──────────────────────────────────────

    def atom_type_map(self, mg: MorphismGraph) -> dict[int, str]:
        """Return {atom_id: type_name} from the global CTKG type assignments.

        For each atom in mg, looks up the matching CTKG concept (named
        ``atom_{repr(value)}``) and extracts the type-group annotation from
        its description string: "Atom 'x' (level N, type group: X_group)".

        Only atoms that have been registered in the global CTKG are returned;
        atoms seen too rarely to trigger a segment boundary are absent.

        Used by predict() as the FCA adjunction back-off: atoms sharing a
        type-group pool their outgoing edges for unseen-context prediction.
        """
        result: dict[int, str] = {}
        marker = "type group: "
        for value, atom_id in mg.atoms.items():
            concept_name = f"atom_{value!r}"
            concept = self.global_kg.concepts.get(concept_name)
            if concept is None:
                continue
            desc = concept.description
            idx  = desc.find(marker)
            if idx >= 0:
                # Extract type name: everything after marker up to closing ')'
                type_name = desc[idx + len(marker):].rstrip(")")
                result[atom_id] = type_name
        return result

    # ── Phase 14a: MasteryState ───────────────────────────────────────────────

    def mastery_state(self, mg: MorphismGraph) -> MasteryState:
        """Return a MasteryState tracking per-concept mastery from edge counts.

        Mastery levels are estimated from the MorphismGraph's edge counts:
          • Atoms  — mastery = 1.0 if observed (primitive symbols are trivially
                     mastered once seen; they carry no internal structure to learn).
          • Comps  — mastery = 1 - 1/(1 + count), where count = total outgoing
                     edges from the composition in mg._out.  The formula maps:
                     count=0 → 0.0, count=1 → 0.5, count=9 → 0.9, count=19 → 0.95.

        Only concepts that are registered in global_kg are updated; atoms or
        compositions that have not yet triggered a segment boundary remain at
        the initialised value of 0.0.

        Returns a fresh MasteryState each call (cheap: O(concepts)).
        """
        import ast as _ast

        ms = self.global_kg.mastery_state()   # initialises all levels to 0.0

        for concept_name in self.global_kg.concepts:
            if concept_name.startswith("atom_"):
                try:
                    value = _ast.literal_eval(concept_name[5:])
                except (ValueError, SyntaxError):
                    continue
                if value in mg.atoms:
                    ms.observe(concept_name, 1.0)   # atoms are primitive — trivially mastered

            elif concept_name.startswith("comp_"):
                try:
                    comp_id = int(concept_name[5:])
                except ValueError:
                    continue
                # Count total observations in which this composition acted as source
                out = mg._out.get(comp_id, {})
                total = sum(sum(v.values()) for v in out.values())
                mastery = 1.0 - 1.0 / (1.0 + total)
                ms.observe(concept_name, mastery)

        return ms

    def frontier(self, mg: MorphismGraph) -> set[str]:
        """Return the mastery frontier: concepts ready to learn but not yet mastered.

        Delegates to MasteryState.frontier() after computing mastery from mg.
        Frontier concepts are those whose prerequisites are mastered (≥ 0.8
        expected readiness) but which are not themselves mastered (< 0.95).

        Used by the active-inference planner to decide what to observe next.
        """
        return self.mastery_state(mg).frontier()

    # ── Phase 14b: Information flow ───────────────────────────────────────────

    def information_flow(self, mg: MorphismGraph) -> dict[str, float]:
        """Return per-edge information flow in bits, grounded in edge counts.

        Each composition C = (left →[etype]→ right) defines a structural
        dependency: atom *left* is a prerequisite for composition *C*.
        The information flow on that edge is a normalised log-count measure
        of how actively C has been observed as a prediction source:

            flow(left → comp_C) = log2(1 + count(C)) / log2(1 + max_count)

        where count(C) = total outgoing edges from comp_C in mg._out.
        The log scaling gives a [0, 1] range and prevents high-count
        compositions from completely swamping low-count ones.

        Source of truth: mg.rules (composition registry), NOT global_kg.prerequisites.
        Compositions are derived structures that do not appear in chunk
        observations, so global_kg.prerequisites is always empty.
        This method bypasses global_kg and reads mg.rules directly.

        Returns {} if no compositions exist or all composition counts are zero.
        """
        if not mg.rules:
            return {}

        # Count total outgoing edges for every composition
        comp_counts: dict[int, int] = {}
        for comp_id in mg.rules:
            out = mg._out.get(comp_id, {})
            total = sum(sum(v.values()) for v in out.values())
            comp_counts[comp_id] = total

        max_count = max(comp_counts.values(), default=0)
        if max_count == 0:
            return {}

        log_max = math.log2(1 + max_count)

        flows: dict[str, float] = {}
        for comp_id, (left, etype, right) in mg.rules.items():
            count = comp_counts.get(comp_id, 0)
            if count == 0:
                continue

            left_sym = mg.symbols[left]
            left_name = (f"atom_{left_sym.value!r}"
                         if isinstance(left_sym, Atom) else f"comp_{left}")
            comp_name = f"comp_{comp_id}"

            flow = math.log2(1 + count) / log_max
            flows[f"{left_name}->{comp_name}"] = flow

        return flows

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
