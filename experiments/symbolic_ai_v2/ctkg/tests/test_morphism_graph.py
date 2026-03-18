"""Tests for ctkg/core/morphism_graph.py (Step 2.1)."""

from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph, CTKGObject, CTKGMorphism,
)
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import DistributionalConcept


def _make_concept(cid: int, top_atom: str) -> DistributionalConcept:
    centroid = np.zeros(5)
    centroid[cid % 5] = 1.0
    return DistributionalConcept(
        concept_id=cid,
        centroid_vector=centroid,
        extent_weights={f"ctx_{cid}": 1.0},
        intent_weights={top_atom: 0.9, "other": 0.1},
        support=10.0,
    )


@pytest.fixture
def mg_two_objects():
    mg = MorphismGraph()
    c0 = _make_concept(0, "digit")
    c1 = _make_concept(1, "operator")
    obj0 = mg.add_object(c0, label="DIGIT")
    obj1 = mg.add_object(c1, label="OPERATOR")
    return mg, obj0, obj1


class TestObjects:
    def test_add_object_returns_object(self):
        mg = MorphismGraph()
        c = _make_concept(0, "x")
        obj = mg.add_object(c)
        assert isinstance(obj, CTKGObject)
        assert obj.obj_id == 0

    def test_objects_list(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        objs = mg.objects()
        assert len(objs) == 2
        assert objs[0].obj_id == 0
        assert objs[1].obj_id == 1

    def test_label_auto_generated(self):
        mg = MorphismGraph()
        c = _make_concept(0, "myatom")
        obj = mg.add_object(c)
        assert "myatom" in obj.label

    def test_identity_created_on_add(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        id0 = mg.identity(obj0.obj_id)
        id1 = mg.identity(obj1.obj_id)
        assert id0 is not None and id0.is_identity
        assert id1 is not None and id1.is_identity
        assert id0.source == id0.target == obj0.obj_id
        assert id1.source == id1.target == obj1.obj_id


class TestMorphisms:
    def test_add_morphism(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        m = mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="TEST")
        assert isinstance(m, CTKGMorphism)
        assert m.source == obj0.obj_id
        assert m.target == obj1.obj_id
        assert m.morph_type == "TEST"

    def test_hom_lookup(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        m = mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="T")
        hom = mg.hom(obj0.obj_id, obj1.obj_id, include_identity=False)
        assert len(hom) == 1
        assert hom[0].morph_id == m.morph_id

    def test_hom_empty_if_no_morphism(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        # No morphism obj1 → obj0
        hom = mg.hom(obj1.obj_id, obj0.obj_id, include_identity=False)
        assert hom == []

    def test_default_body(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        m = mg.add_morphism(obj0.obj_id, obj1.obj_id)
        assert m.body == [obj0.obj_id, obj1.obj_id]

    def test_custom_body(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        m = mg.add_morphism(obj0.obj_id, obj1.obj_id,
                            body=[obj0.obj_id, 999, obj1.obj_id])
        assert m.body == [obj0.obj_id, 999, obj1.obj_id]

    def test_add_morphism_unknown_source_raises(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        with pytest.raises(KeyError):
            mg.add_morphism(999, obj1.obj_id)

    def test_morphisms_excludes_identity_by_default(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="T")
        ms = mg.morphisms(include_identity=False)
        assert all(not m.is_identity for m in ms)
        assert len(ms) == 1

    def test_evidence_update(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        m = mg.add_morphism(obj0.obj_id, obj1.obj_id, evidence=5)
        mg.observe(m.morph_id, 3)
        assert mg.morphism_by_id(m.morph_id).evidence_count == 8

    def test_confidence_update(self, mg_two_objects):
        mg, obj0, obj1 = mg_two_objects
        m = mg.add_morphism(obj0.obj_id, obj1.obj_id)
        mg.update_confidence(m.morph_id, -1.5)
        assert mg.morphism_by_id(m.morph_id).confidence == pytest.approx(-1.5)


class TestComposition:
    def test_compose_valid(self):
        mg = MorphismGraph()
        c0, c1, c2 = (_make_concept(i, f"a{i}") for i in range(3))
        obj0 = mg.add_object(c0)
        obj1 = mg.add_object(c1)
        obj2 = mg.add_object(c2)
        g = mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="G")
        f = mg.add_morphism(obj1.obj_id, obj2.obj_id, morph_type="F")
        # f ∘ g: apply g first (0→1), then f (1→2) → 0→2
        fg = mg.compose(f.morph_id, g.morph_id)
        assert fg is not None
        assert fg.source == obj0.obj_id
        assert fg.target == obj2.obj_id
        assert fg.body == [obj0.obj_id, obj1.obj_id, obj2.obj_id]

    def test_compose_invalid_mismatch(self):
        mg = MorphismGraph()
        c0, c1, c2 = (_make_concept(i, f"a{i}") for i in range(3))
        obj0 = mg.add_object(c0)
        obj1 = mg.add_object(c1)
        obj2 = mg.add_object(c2)
        f = mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="F")
        g = mg.add_morphism(obj0.obj_id, obj2.obj_id, morph_type="G")
        # f ∘ g where target(g)=2 ≠ source(f)=0 → None
        result = mg.compose(f.morph_id, g.morph_id)
        assert result is None

    def test_compose_cached(self):
        mg = MorphismGraph()
        c0, c1, c2 = (_make_concept(i, f"a{i}") for i in range(3))
        obj0 = mg.add_object(c0)
        obj1 = mg.add_object(c1)
        obj2 = mg.add_object(c2)
        g = mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="G")
        f = mg.add_morphism(obj1.obj_id, obj2.obj_id, morph_type="F")
        fg1 = mg.compose(f.morph_id, g.morph_id)
        fg2 = mg.compose(f.morph_id, g.morph_id)
        assert fg1.morph_id == fg2.morph_id

    def test_compose_confidence_weakest_link(self):
        mg = MorphismGraph()
        c0, c1, c2 = (_make_concept(i, f"a{i}") for i in range(3))
        obj0 = mg.add_object(c0)
        obj1 = mg.add_object(c1)
        obj2 = mg.add_object(c2)
        g = mg.add_morphism(obj0.obj_id, obj1.obj_id, confidence=-0.5)
        f = mg.add_morphism(obj1.obj_id, obj2.obj_id, confidence=-2.0)
        fg = mg.compose(f.morph_id, g.morph_id)
        assert fg.confidence == pytest.approx(-2.0)

    def test_identity_compose(self):
        mg = MorphismGraph()
        c0, c1 = (_make_concept(i, f"a{i}") for i in range(2))
        obj0 = mg.add_object(c0)
        obj1 = mg.add_object(c1)
        f = mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="F")
        id1 = mg.identity(obj1.obj_id)
        # id ∘ f: apply f then id → still 0→1
        result = mg.compose(id1.morph_id, f.morph_id)
        assert result is not None
        assert result.source == obj0.obj_id
        assert result.target == obj1.obj_id


class TestSummary:
    def test_summary_runs(self, mg_two_objects):
        mg, _, _ = mg_two_objects
        mg.add_morphism(0, 1, morph_type="TEST")
        s = mg.summary()
        assert "MorphismGraph" in s
        assert "objects" in s


class TestBeliefLayer:
    """Stage 2 — Belief Layer API."""

    def _mg_with_morphism(self):
        mg = MorphismGraph()
        c0 = _make_concept(0, "x")
        c1 = _make_concept(1, "y")
        obj0 = mg.add_object(c0)
        obj1 = mg.add_object(c1)
        m = mg.add_morphism(obj0.obj_id, obj1.obj_id, morph_type="T")
        return mg, m

    def test_add_theory_returns_obj_id(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id])
        assert isinstance(t_id, int)

    def test_theory_object_is_theory(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id])
        obj = mg.object_by_id(t_id)
        assert obj is not None
        assert obj.is_theory

    def test_theory_excluded_from_objects(self):
        mg, m = self._mg_with_morphism()
        mg.add_theory([m.morph_id])
        objs = mg.objects()
        assert all(not o.is_theory for o in objs)
        assert len(objs) == 2  # original two concept objects only

    def test_theories_method(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id])
        ts = mg.theories()
        assert len(ts) == 1
        assert ts[0].obj_id == t_id

    def test_initial_belief_is_one(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id])
        assert mg.get_belief(t_id) == pytest.approx(1.0)

    def test_update_belief(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id])
        mg.update_belief(t_id, 0.5)
        assert mg.get_belief(t_id) == pytest.approx(1.5)

    def test_update_belief_negative_clamps_to_zero(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id])
        mg.update_belief(t_id, -10.0)
        assert mg.get_belief(t_id) == pytest.approx(0.0)

    def test_normalize_beliefs(self):
        mg, m = self._mg_with_morphism()
        t1 = mg.add_theory([m.morph_id])
        t2 = mg.add_theory([m.morph_id])
        # initial: both = 1.0; after normalize: both = 0.5
        mg.normalize_beliefs()
        assert mg.get_belief(t1) == pytest.approx(0.5)
        assert mg.get_belief(t2) == pytest.approx(0.5)

    def test_normalize_beliefs_respects_asymmetry(self):
        mg, m = self._mg_with_morphism()
        t1 = mg.add_theory([m.morph_id])
        t2 = mg.add_theory([m.morph_id])
        mg.update_belief(t2, 3.0)   # t2 now has weight=4.0, t1=1.0 → total=5.0
        mg.normalize_beliefs()
        assert mg.get_belief(t1) == pytest.approx(0.2)
        assert mg.get_belief(t2) == pytest.approx(0.8)

    def test_normalize_beliefs_noop_if_no_theories(self):
        mg, _ = self._mg_with_morphism()
        mg.normalize_beliefs()   # must not raise

    def test_theory_members(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id])
        members = mg.theory_members(t_id)
        assert m.morph_id in members

    def test_theory_empty_members(self):
        mg, _ = self._mg_with_morphism()
        t_id = mg.add_theory([])
        assert mg.theory_members(t_id) == []

    def test_theory_skips_unknown_morphism(self):
        mg, m = self._mg_with_morphism()
        t_id = mg.add_theory([m.morph_id, 9999])  # 9999 doesn't exist
        members = mg.theory_members(t_id)
        assert m.morph_id in members
        assert 9999 not in members

    def test_two_theories_independent(self):
        mg, m = self._mg_with_morphism()
        t1 = mg.add_theory([m.morph_id])
        t2 = mg.add_theory([])
        mg.update_belief(t1, 2.0)
        assert mg.get_belief(t1) == pytest.approx(3.0)
        assert mg.get_belief(t2) == pytest.approx(1.0)  # unchanged
