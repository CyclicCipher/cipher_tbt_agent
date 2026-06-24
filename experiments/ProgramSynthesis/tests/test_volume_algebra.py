"""Phase-2 prototype — the algebra over box concepts (docs/phase2/VOLUME_CONCEPTS.md §1):
meet = AND / context-narrowing, entails = IS-A / specificity (the asymmetric relation points
cannot express). Dimensions are plain ints (e.g. hue=0, brightness=3)."""

from volume.box import BoxConcept
from volume.algebra import entails, meet, volume


def _box(dims, intervals):
    return BoxConcept(tuple(dims), tuple(lo for lo, _ in intervals), tuple(hi for _, hi in intervals))


def test_meet_conjoins_across_subspaces():
    # a concept over dim 0 (hue); a context over dim 3 (brightness) -> their conjunction lives over both
    warm = _box([0], [(0.2, 0.8)])
    bright = _box([3], [(0.6, 0.9)])
    m = meet(warm, bright)
    assert set(m.dims) == {0, 3}
    assert m.contains([0.5, 9, 9, 0.7])      # in on both relevant axes; dims 1,2 are don't-care
    assert not m.contains([0.5, 9, 9, 0.1])  # fails the brightness context


def test_meet_tightens_a_shared_dimension():
    m = meet(_box([1], [(0.2, 0.8)]), _box([1], [(0.5, 1.0)]))
    assert m.dims == (1,) and m.lo == (0.5,) and m.hi == (0.8,)


def test_meet_is_empty_when_disjoint():
    assert meet(_box([1], [(0.2, 0.4)]), _box([1], [(0.6, 0.8)])) is None


def test_meet_is_a_lower_bound():
    # the conjunction is at least as specific as each conjunct (greatest-lower-bound property)
    a = _box([0], [(0.2, 0.8)])
    b = _box([3], [(0.6, 0.9)])
    m = meet(a, b)
    assert entails(m, a) and entails(m, b)


def test_entails_is_asymmetric():
    narrow = _box([1], [(0.4, 0.6)])
    broad = _box([1], [(0.2, 0.8)])
    assert entails(narrow, broad)          # narrower IS-A broader
    assert not entails(broad, narrow)      # but not the reverse


def test_extra_constrained_dimension_is_more_specific():
    specific = _box([1, 4], [(0.4, 0.6), (0.6, 0.9)])
    general = _box([1], [(0.2, 0.8)])
    assert entails(specific, general)      # constraining an extra axis only narrows the concept
    assert not entails(general, specific)


def test_context_narrowing_yields_a_more_specific_concept():
    # the geometric form of "context narrows meaning": intersecting a concept with a context
    # produces a concept that ENTAILS the original and has no larger volume on the shared axis.
    concept = _box([0], [(0.0, 1.0)])
    context = _box([0], [(0.3, 0.6)])
    narrowed = meet(concept, context)
    assert entails(narrowed, concept)
    assert volume(narrowed) <= volume(concept)
