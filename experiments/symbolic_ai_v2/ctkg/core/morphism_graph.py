"""
Objects, morphisms, hom-sets with probability distributions.

The MorphismGraph is the CTKG's structural layer.  Nodes are CTKGObjects
(discovered concept types); directed edges are CTKGMorphisms (discovered
compositional relationships between types).

Design decisions:
  - Objects wrap DistributionalConcepts from the ConceptLattice.  They inherit
    the concept's centroid vector and support, so the type system is grounded in
    the distributional statistics of the data.
  - Morphisms carry a body (ordered list of obj_ids through the rule) and a
    log-confidence (the Phase 8 lens parameter P = ℝ for atomic morphisms).
  - composition(f, g): defined iff target(g) == source(f).  Returns a new
    composite morphism with body = body(g) + body(f)[1:] and confidence equal
    to min(confidence(f), confidence(g)) (weakest link heuristic, updated by
    the lens in Phase 8).
  - Identity morphisms: one per object, body = [obj_id], confidence = 0.0
    (log(1) = 0 = certainty).

See CTKG_ARCHITECTURE.md §Phase 3 and §Prediction for the full specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import (
    DistributionalConcept,
    ConceptId,
)


ObjectId = int
MorphId = int


@dataclass
class CTKGObject:
    """A node in the MorphismGraph — one discovered concept type.

    Parameters
    ----------
    obj_id:
        Unique integer identifier within this MorphismGraph.
    concept:
        The underlying DistributionalConcept from Phase 2 FCA.
        None for theory objects (Belief Layer, Stage 2).
    label:
        Human-readable label, auto-generated from the concept's top atoms.
        Not used for lookup — obj_id is the canonical key.
    active:
        Whether this object is currently active.  Inactive objects are
        deactivated (not deleted) when their support falls below the MDL
        threshold during the EM loop.  Use MorphismGraph.objects(active_only=True)
        to restrict iteration to active objects.
    is_theory:
        True for theory objects created by add_theory() (Belief Layer, Stage 2).
        Theory objects have no concept; their belief is stored as the weight on
        their identity self-loop.  Excluded from mg.objects() by default.
    """

    obj_id: ObjectId
    concept: Optional[DistributionalConcept] = None
    label: str = ""
    active: bool = True
    is_theory: bool = False

    def __post_init__(self) -> None:
        if not self.label:
            if self.concept is not None:
                top = self.concept.top_atoms(3)
                self.label = "/".join(a for a, _ in top)

    def __repr__(self) -> str:
        flag = "" if self.active else " INACTIVE"
        theory_flag = " THEORY" if self.is_theory else ""
        return f"Object(id={self.obj_id}, label={self.label!r}{flag}{theory_flag})"


@dataclass
class CTKGMorphism:
    """A directed edge in the MorphismGraph.

    Parameters
    ----------
    morph_id:
        Unique integer identifier.
    source:
        obj_id of the domain object.
    target:
        obj_id of the codomain object.
    body:
        Ordered list of obj_ids representing the full rule body (inclusive of
        source and target).  For a binary morphism f: A → B, body = [A, B].
        For a composite f = g∘h: A → M → B, body = [A, M, B].
    confidence:
        Log-confidence ∈ ℝ.  log(1) = 0.0 = perfect certainty.
        Negative = below-1 confidence (doubt).
        Updated by the Phase 8 lens backward pass.
    evidence_count:
        Number of times this morphism type was observed in the corpus.
    morph_type:
        Human-readable label for this morphism type, used in diagnostics
        (e.g. "SUCC_PATTERN", "DIGIT_CHAIN", "MORPH_0:len=0").
        Empty string for identity morphisms.
    morph_concept_id:
        ConceptId pointer into the ConceptLattice identifying the discovered
        morphism-type cluster that produced this morphism (from Phase 3 FCA
        on H_morph).  -1 when not yet set (e.g. composite morphisms created
        by compose() or identity morphisms).
    """

    morph_id: MorphId
    source: ObjectId
    target: ObjectId
    body: list[ObjectId]
    confidence: float = 0.0
    evidence_count: int = 1
    morph_type: str = ""
    morph_concept_id: int = -1
    weight: Optional[float] = None
    """Optional probability weight for theory-membership edges (Stage 2 Belief Layer)."""
    is_identity: bool = False
    """True iff this is an identity morphism (replaces __identity__ sentinel)."""
    payload: Optional[object] = field(default=None, compare=False, repr=False)
    """Arbitrary rule payload for typed morphisms (Stage 4).

    CHAIN_STEP:      (format: str, input_key: tuple[str,...], output_tokens: tuple[str,...])
                     format is "chain" (trace step/ans) or "eq" (eq-delimited).
    FOLD_RULE:       BinaryFoldRule — the NNO fold structure for this binary op.
    FC_EDGE:         (input_tup: tuple[str,...], output_tup: tuple[str,...])
    ADJ_EDGE:        preserved_position: Optional[int] — the preserved argument index.
    RELATION_RULE:   RelationRule — the arity-free role-based rule.
    KLEISLI_CHAIN:   (disc_role: NodeId, chains: dict[NodeId, list[RelationRule]])
    SUCC_EDGE:       None — the source→target pair IS the information.
    """

    def __repr__(self) -> str:
        return (
            f"Morphism(id={self.morph_id}, "
            f"{self.source}→{self.target}, "
            f"type={self.morph_type!r}, "
            f"conf={self.confidence:.2f}, ev={self.evidence_count})"
        )


class MorphismGraph:
    """The CTKG's structural layer: objects + morphisms with composition.

    All identifiers are stable (never reassigned once issued).  Deletions are
    not supported at this stage — morphisms with zero evidence are pruned by
    `mdl_prune.py` in the EM loop, but the ids remain reserved.
    """

    def __init__(self) -> None:
        self._objects: dict[ObjectId, CTKGObject] = {}
        self._morphisms: dict[MorphId, CTKGMorphism] = {}
        # Indices for fast lookup
        self._hom_index: dict[tuple[ObjectId, ObjectId], list[MorphId]] = {}
        self._source_index: dict[ObjectId, list[MorphId]] = {}
        self._target_index: dict[ObjectId, list[MorphId]] = {}
        # Composite cache: (f_id, g_id) → composite morph_id
        self._compose_cache: dict[tuple[MorphId, MorphId], MorphId] = {}
        self._next_obj_id: int = 0
        self._next_morph_id: int = 0
        # Reverse label index for fast object_by_label lookup (Stage 4)
        self._label_index: dict[str, ObjectId] = {}

    # ------------------------------------------------------------------
    # Adding objects
    # ------------------------------------------------------------------

    def add_object(self, concept: DistributionalConcept, label: str = "") -> CTKGObject:
        """Create a new CTKGObject wrapping `concept` and return it."""
        obj = CTKGObject(obj_id=self._next_obj_id, concept=concept, label=label)
        self._objects[obj.obj_id] = obj
        if obj.label:
            self._label_index[obj.label] = obj.obj_id
        self._next_obj_id += 1
        # Create the identity morphism immediately
        self._add_identity(obj.obj_id)
        return obj

    def _add_identity(self, obj_id: ObjectId) -> CTKGMorphism:
        m = CTKGMorphism(
            morph_id=self._next_morph_id,
            source=obj_id,
            target=obj_id,
            body=[obj_id],
            confidence=0.0,
            evidence_count=0,
            morph_type="",
            is_identity=True,
        )
        self._register_morphism(m)
        return m

    # ------------------------------------------------------------------
    # Adding morphisms
    # ------------------------------------------------------------------

    def add_morphism(
        self,
        source_id: ObjectId,
        target_id: ObjectId,
        body: Optional[list[ObjectId]] = None,
        evidence: int = 1,
        morph_type: str = "",
        confidence: float = 0.0,
        morph_concept_id: int = -1,
        payload: Optional[object] = None,
    ) -> CTKGMorphism:
        """Add a new morphism and return it.

        Parameters
        ----------
        source_id, target_id:
            Must already exist as objects.
        body:
            Full ordered path of obj_ids.  Defaults to [source_id, target_id].
        evidence:
            Initial evidence count.
        morph_type:
            Human-readable label for diagnostics.
        confidence:
            Initial log-confidence (0.0 = certainty).
        morph_concept_id:
            ConceptId pointer into the ConceptLattice for the discovered
            morphism-type cluster (Phase 3).  -1 if not applicable.
        """
        if source_id not in self._objects:
            raise KeyError(f"Source object {source_id} not found")
        if target_id not in self._objects:
            raise KeyError(f"Target object {target_id} not found")
        if body is None:
            body = [source_id, target_id]

        m = CTKGMorphism(
            morph_id=self._next_morph_id,
            source=source_id,
            target=target_id,
            body=list(body),
            confidence=confidence,
            evidence_count=evidence,
            morph_type=morph_type,
            morph_concept_id=morph_concept_id,
            payload=payload,
        )
        self._register_morphism(m)
        return m

    def _register_morphism(self, m: CTKGMorphism) -> None:
        self._morphisms[m.morph_id] = m
        key = (m.source, m.target)
        self._hom_index.setdefault(key, []).append(m.morph_id)
        self._source_index.setdefault(m.source, []).append(m.morph_id)
        self._target_index.setdefault(m.target, []).append(m.morph_id)
        self._next_morph_id += 1

    def observe(self, morph_id: MorphId, count: int = 1) -> None:
        """Increment evidence count for an existing morphism."""
        self._morphisms[morph_id].evidence_count += count

    def update_confidence(self, morph_id: MorphId, log_conf: float) -> None:
        """Set the log-confidence of a morphism (Phase 8 lens update)."""
        self._morphisms[morph_id].confidence = log_conf

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def object_by_id(self, obj_id: ObjectId) -> Optional[CTKGObject]:
        return self._objects.get(obj_id)

    def source_morphisms(
        self,
        obj_id: ObjectId,
        morph_type: Optional[str] = None,
        include_identity: bool = False,
    ) -> list[CTKGMorphism]:
        """Return all outgoing morphisms from obj_id.

        Parameters
        ----------
        obj_id:
            Source object ID.
        morph_type:
            If given, filter to morphisms with this exact morph_type string.
        include_identity:
            If False (default), identity morphisms are excluded.

        Used by Stage 4 CTKG path-find: iterate outgoing typed edges from the
        op node to dispatch on morph_type without knowing the op's name.
        """
        ids = self._source_index.get(obj_id, [])
        morphs = [self._morphisms[mid] for mid in ids if mid in self._morphisms]
        if not include_identity:
            morphs = [m for m in morphs if not m.is_identity]
        if morph_type is not None:
            morphs = [m for m in morphs if m.morph_type == morph_type]
        return morphs

    def object_by_label(self, label: str) -> Optional[CTKGObject]:
        """Look up an object by its label string (Stage 4 helper).

        Returns None if no object with the given label exists.
        """
        obj_id = self._label_index.get(label)
        return self._objects.get(obj_id) if obj_id is not None else None

    def get_or_create_object(self, label: str) -> CTKGObject:
        """Return an existing object by label, or create a new one.

        Used by _populate_succ_edges_to_mg (Stage 4) to register digit
        objects for SUCC_EDGE morphisms without requiring a DistributionalConcept.
        New objects are created with concept=None (theory objects) and
        is_theory=False so they don't clutter the theory layer.
        """
        existing = self.object_by_label(label)
        if existing is not None:
            return existing
        obj = CTKGObject(obj_id=self._next_obj_id, concept=None, label=label)
        self._objects[obj.obj_id] = obj
        self._label_index[label] = obj.obj_id
        self._next_obj_id += 1
        self._add_identity(obj.obj_id)
        return obj

    def morphism_by_id(self, morph_id: MorphId) -> Optional[CTKGMorphism]:
        return self._morphisms.get(morph_id)

    def objects(
        self,
        active_only: bool = False,
        include_theories: bool = False,
    ) -> list[CTKGObject]:
        """All concept objects, sorted by obj_id.

        Parameters
        ----------
        active_only:
            If True, return only objects with active=True (i.e. objects that
            have not been deactivated by the MDL pruning step).
        include_theories:
            If True, include theory objects (is_theory=True) in the result.
            Default False — theory objects are excluded to keep existing code
            that iterates objects and accesses obj.concept from breaking.
        """
        objs = sorted(self._objects.values(), key=lambda o: o.obj_id)
        if not include_theories:
            objs = [o for o in objs if not o.is_theory]
        if active_only:
            objs = [o for o in objs if o.active]
        return objs

    def theories(self) -> list[CTKGObject]:
        """Return all theory objects (Belief Layer, Stage 2), sorted by obj_id."""
        return sorted(
            [o for o in self._objects.values() if o.is_theory],
            key=lambda o: o.obj_id,
        )

    def morphisms(self, include_identity: bool = False) -> list[CTKGMorphism]:
        """All non-identity morphisms (or all if include_identity=True), sorted by morph_id."""
        ms = sorted(self._morphisms.values(), key=lambda m: m.morph_id)
        if not include_identity:
            ms = [m for m in ms if not m.is_identity]
        return ms

    def hom(
        self,
        source_id: ObjectId,
        target_id: ObjectId,
        include_identity: bool = True,
    ) -> list[CTKGMorphism]:
        """All morphisms from source_id to target_id."""
        ids = self._hom_index.get((source_id, target_id), [])
        ms = [self._morphisms[i] for i in ids]
        if not include_identity:
            ms = [m for m in ms if not m.is_identity]
        return ms

    def out_morphisms(self, obj_id: ObjectId, include_identity: bool = False) -> list[CTKGMorphism]:
        """All morphisms with source == obj_id."""
        ids = self._source_index.get(obj_id, [])
        ms = [self._morphisms[i] for i in ids]
        if not include_identity:
            ms = [m for m in ms if not m.is_identity]
        return ms

    def in_morphisms(self, obj_id: ObjectId, include_identity: bool = False) -> list[CTKGMorphism]:
        """All morphisms with target == obj_id."""
        ids = self._target_index.get(obj_id, [])
        ms = [self._morphisms[i] for i in ids]
        if not include_identity:
            ms = [m for m in ms if not m.is_identity]
        return ms

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, f_id: MorphId, g_id: MorphId) -> Optional[CTKGMorphism]:
        """Return f ∘ g (apply g first, then f).

        Valid iff target(g) == source(f).  Returns the existing composite if
        already registered; otherwise creates and registers a new one.

        Confidence: min(conf(f), conf(g)) — weakest-link heuristic.
        Body: body(g) + body(f)[1:]  (glue at the shared middle object).
        """
        if (f_id, g_id) in self._compose_cache:
            return self._morphisms.get(self._compose_cache[(f_id, g_id)])

        f = self._morphisms.get(f_id)
        g = self._morphisms.get(g_id)
        if f is None or g is None:
            return None
        if f.source != g.target:
            return None  # composition undefined

        # Compose body: g's body then f's tail (skip first element = shared node)
        new_body = list(g.body) + list(f.body[1:])
        new_conf = min(f.confidence, g.confidence)
        composite = self.add_morphism(
            source_id=g.source,
            target_id=f.target,
            body=new_body,
            evidence=0,
            morph_type=f"({g.morph_type})∘({f.morph_type})",
            confidence=new_conf,
        )
        self._compose_cache[(f_id, g_id)] = composite.morph_id
        return composite

    def identity(self, obj_id: ObjectId) -> Optional[CTKGMorphism]:
        """Return the identity morphism for obj_id."""
        for m_id in self._hom_index.get((obj_id, obj_id), []):
            m = self._morphisms[m_id]
            if m.is_identity:
                return m
        return None

    # ------------------------------------------------------------------
    # Belief Layer (Stage 2)
    # ------------------------------------------------------------------

    def add_theory(self, morphism_ids: list[MorphId]) -> ObjectId:
        """Register a new theory as a CTKGObject (Belief Layer, Stage 2).

        A theory is a consistent set of morphisms that together explain
        observations.  All state is stored as graph edges — no new Python dicts.

        Representation
        --------------
        - The theory object has is_theory=True and concept=None.
        - Belief is stored as weight on the theory's identity self-loop.
          Initial prior belief = 1.0 (unnormalized; call normalize_beliefs()
          after adding all theories to convert to a probability distribution).
        - Membership is stored as THEORY_MEMBER self-loops on the theory object,
          one per member morphism, with morph_concept_id = member morph_id.

        Parameters
        ----------
        morphism_ids:
            The morph_ids of morphisms that constitute this theory.
            Morphisms not present in the graph are silently skipped.

        Returns
        -------
        The theory's obj_id.
        """
        obj_id = self._next_obj_id
        obj = CTKGObject(
            obj_id=obj_id,
            concept=None,
            label=f"Theory_{obj_id}",
            is_theory=True,
        )
        self._objects[obj_id] = obj
        self._next_obj_id += 1

        # Identity self-loop doubles as belief carrier; initial belief = 1.0
        id_m = self._add_identity(obj_id)
        id_m.weight = 1.0

        # Theory-membership self-loops: one per member morphism.
        # morph_concept_id encodes which morphism (by MorphId) is a member.
        for m_id in morphism_ids:
            if m_id in self._morphisms:
                mem = CTKGMorphism(
                    morph_id=self._next_morph_id,
                    source=obj_id,
                    target=obj_id,
                    body=[obj_id],
                    morph_type="THEORY_MEMBER",
                    morph_concept_id=m_id,
                    weight=1.0,
                )
                self._register_morphism(mem)

        return obj_id

    def update_belief(self, theory_id: ObjectId, delta: float) -> None:
        """Add *delta* to the belief weight of *theory_id*.

        The belief is stored as weight on the theory's identity self-loop.
        Clamps to [0.0, ∞) — beliefs cannot go negative.

        Raises KeyError if *theory_id* has no identity morphism.
        """
        id_m = self.identity(theory_id)
        if id_m is None:
            raise KeyError(f"No identity morphism for theory {theory_id}")
        current = id_m.weight if id_m.weight is not None else 0.0
        id_m.weight = max(0.0, current + delta)

    def get_belief(self, theory_id: ObjectId) -> float:
        """Return the belief weight of *theory_id* (Belief Layer).

        Returns 0.0 if the theory has no identity morphism or its weight
        has not been set.
        """
        id_m = self.identity(theory_id)
        if id_m is None or id_m.weight is None:
            return 0.0
        return id_m.weight

    def normalize_beliefs(self) -> None:
        """Normalize all theory beliefs so they sum to 1.0.

        If no theories exist or the total belief is 0, this is a no-op.
        Operates in-place on the identity morphism weights.
        """
        theory_objs = self.theories()
        total = sum(self.get_belief(o.obj_id) for o in theory_objs)
        if total <= 0.0:
            return
        for o in theory_objs:
            id_m = self.identity(o.obj_id)
            if id_m is not None and id_m.weight is not None:
                id_m.weight /= total

    def theory_members(self, theory_id: ObjectId) -> list[MorphId]:
        """Return the morph_ids of morphisms that belong to *theory_id*.

        Recovered by reading THEORY_MEMBER self-loops on the theory object
        (morph_concept_id encodes the member morph_id).
        """
        result: list[MorphId] = []
        for m_id in self._hom_index.get((theory_id, theory_id), []):
            m = self._morphisms[m_id]
            if m.morph_type == "THEORY_MEMBER":
                result.append(m.morph_concept_id)
        return result

    def add_theory_member(self, theory_id: ObjectId, morph_id: MorphId) -> None:
        """Add a single morphism to an existing theory (idempotent).

        If *morph_id* is already a member of *theory_id* this is a no-op.
        Raises KeyError if *theory_id* is not a registered theory or
        *morph_id* is not present in the graph.
        """
        theory_obj = self._objects.get(theory_id)
        if theory_obj is None or not theory_obj.is_theory:
            raise KeyError(f"add_theory_member: {theory_id} is not a theory object")
        if morph_id not in self._morphisms:
            raise KeyError(f"add_theory_member: morphism {morph_id} not in graph")
        # Idempotent: skip if already a member
        if morph_id in self.theory_members(theory_id):
            return
        mem = CTKGMorphism(
            morph_id=self._next_morph_id,
            source=theory_id,
            target=theory_id,
            body=[theory_id],
            morph_type="THEORY_MEMBER",
            morph_concept_id=morph_id,
            weight=1.0,
        )
        self._register_morphism(mem)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_obj = len(self._objects)
        n_morph = len([m for m in self._morphisms.values() if not m.is_identity])
        n_id = len([m for m in self._morphisms.values() if m.is_identity])
        morph_types = {m.morph_type for m in self._morphisms.values()
                       if not m.is_identity}
        lines = [
            f"MorphismGraph",
            f"  objects    : {n_obj}",
            f"  morphisms  : {n_morph}  (+ {n_id} identities)",
            f"  morph types: {len(morph_types)}",
        ]
        for obj in self.objects():
            out = self.out_morphisms(obj.obj_id)
            lines.append(f"  {obj}  →  {len(out)} out-morphisms")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_obj = len(self._objects)
        n_morph = len([m for m in self._morphisms.values() if not m.is_identity])
        return f"MorphismGraph(objects={n_obj}, morphisms={n_morph})"
