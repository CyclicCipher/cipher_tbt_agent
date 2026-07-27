"""test_lateral_voting.py — H1: two peer columns reach CONSENSUS over direct lateral cortico-cortical links.

H0 measured that one frame over the joint state does not factorise (`test_h0_factorisation`), so more than one column is
needed. H1 is the step the legacy `HETERARCHY_PLAN` puts before any task column: prove inter-column messaging works at all —
two identical columns sensing the SAME object from different vantage points, each ambiguous alone, agreeing after a vote.

THE PLAN'S SUBSTRATE WAS WRONG, AND CORRECTING IT IS PART OF THE STEP. It says to route the hypotheses "through the
thalamus" and call `L23.vote`. Per TBP's long-range-connections paper (arXiv:2507.05888, `reference_long_range_connections`)
voting is DIRECT LATERAL cortico-cortical — "cells in L3 in different columns are associatively linked via a simple Hebbian
learning rule" — and the thalamus carries a different, HIERARCHICAL route (the L5a efference copy going up, and the
transthalamic compositional path `Agent.place_object` uses, which was always right). We had recorded that locus error on
2026-07-22 as debt and left the thalamic version in place; this is the cut-over. `L23.vote` did not exist either — it was a
legacy name — so the mechanism is built where the corrected anatomy puts it, on `ColumnPooler`'s lateral synapses.

WHY THE LINK MUST BE LEARNED, which is the part that makes this more than message-passing: two columns MINT THEIR OWN random
identity SDRs for the same object, so nothing makes column A's code for an object resemble column B's. What they share is the
world. Co-experience — attending the same thing at the same time — is what turns that into a correspondence, by the same
Hebbian rule the feedforward synapses use. Columns that have never attended anything together have nothing to say to each
other, and that is a property of the mechanism rather than a limitation bolted onto it.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402

# Three objects over a shared feature vocabulary, arranged so that NO single feature identifies an object but the PAIR of
# glances does. Feature 1 belongs to X and Y; feature 3 belongs to Y and Z; the intersection is Y and nothing else.
X = [1, 2]
Y = [1, 3]
Z = [4, 3]


def _column(seed: int) -> Agent:
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=seed)
    for x in range(0, 9):                                    # the "E" step, so object-relative path integration works
        a.learn_move("E", (x, 0), (x + 1, 0))
    return a


def _study(a: Agent, feats) -> None:
    """LEARN one object as an episode: onset → buffer each fixation → commit (the R5 learning path)."""
    a.start_object()
    a.sense_sweep(feats[0])
    for f in feats[1:]:
        a.path_integrate("E")
        a.sense_sweep(f)
    a.commit()


def _glance(a: Agent, feature) -> int:
    """ONE fixation of an object whose identity is not yet determined — the ambiguous vantage point."""
    a.start_object()
    return a.perceive(feature)


def _pair(passes: int = 4):
    """Two peer columns that have learned the same three objects and attended them TOGETHER, so their private identity codes
    are laterally associated. Different seeds: they mint genuinely different SDRs, which is the point."""
    a, b = _column(0), _column(7)
    ca, cb = a._nav_col(), b._nav_col()
    ca.link(cb)
    for _ in range(passes):
        for obj in (X, Y, Z):
            _study(a, obj)
            _study(b, obj)
    for _ in range(passes):                                  # CO-EXPERIENCE: both settle on the same object, then link
        for obj in (X, Y, Z):
            _sweep_settle(a, obj)
            _sweep_settle(b, obj)
            ca.learn_lateral()
            cb.learn_lateral()
    return a, b, ca, cb


def _sweep_settle(a: Agent, feats) -> int:
    """INFER a whole object (unambiguous once the arrangement is seen) — how a column arrives at an opinion to share."""
    a.start_object()
    label = a.perceive(feats[0])
    for f in feats[1:]:
        a.path_integrate("E")
        label = a.perceive(f)
    return label


def test_one_glance_leaves_each_column_honestly_ambiguous():
    """The precondition, and it must be a real ambiguity rather than a contrived one: a single shared feature is consistent
    with two objects, and the column reports nothing rather than guessing. Recognition returning −1 here is the mechanism
    being honest, not failing."""
    a, b, _, _ = _pair()
    assert _glance(a, 1) == -1, "feature 1 belongs to both X and Y — one glance cannot separate them"
    assert _glance(b, 3) == -1, "feature 3 belongs to both Y and Z"


def test_two_ambiguous_columns_agree_after_one_vote():
    """H1'S RESULT. Column A glances a feature consistent with {X, Y}; column B, at a different vantage point on the same
    object, glances one consistent with {Y, Z}. Alone each is stuck. One lateral round and both settle on Y — the object
    consistent with BOTH — with no column ever seeing enough on its own to name it.

    Nothing here intersects sets. Each column re-ranks its OWN live identities by how strongly the peers' opinion
    depolarises them through learned lateral synapses; the intersection is what that computation happens to yield."""
    a, b, ca, cb = _pair()
    _glance(a, 1)
    _glance(b, 3)
    ca.receive_votes()
    cb.receive_votes()
    label_a, label_b = ca.label_of(ca.pooler.active), cb.label_of(cb.pooler.active)
    assert label_a != -1 and label_b != -1, "after voting neither column may still be ambiguous"
    assert ca.label_of(_identity(ca, Y)) == label_a, "and the consensus must be Y, the object consistent with both glances"
    assert cb.label_of(_identity(cb, Y)) == label_b


def _identity(col, feats) -> frozenset:
    """The identity a column holds for an object, found by sweeping the whole of it — used only to NAME the expected answer,
    never by the mechanism under test."""
    from tbt.encoders import CategoryEncoder
    enc = CategoryEncoder(range(16), w=8, capacity=16)
    col.start_object()
    out = col.perceive(enc.encode(feats[0]))
    for f in feats[1:]:
        col.path_integrate("E")
        out = col.perceive(enc.encode(f))
    return out


def test_a_column_cannot_be_voted_into_an_object_its_own_senses_refute():
    """THE GUARD THAT KEEPS VOTING FROM BEING AN ECHO. The vote is MODULATORY: it re-ranks identities the column already
    holds live and can never add one. So a column with an unambiguous view keeps it however loudly its peer disagrees —
    consensus pools evidence, it does not overwrite it. Without this, two columns would converge on whichever spoke first,
    which looks like agreement and carries no information."""
    a, b, ca, cb = _pair()
    alone = _sweep_settle(a, X)                              # A sees the WHOLE of X — no ambiguity to re-rank
    assert alone != -1
    _glance(b, 3)                                            # B meanwhile is stuck between Y and Z
    ca.receive_votes()
    assert ca.label_of(ca.pooler.active) == alone, "a settled column must not be talked out of what it saw"


def test_voting_reaches_the_answer_in_fewer_fixations_than_looking_alone():
    """The CMP speed-up the plan asks for, measured. Alone, a column needs a second fixation to break the ambiguity; with a
    peer it is done after the first. That is the whole point of many columns — evidence gathered in PARALLEL across the
    sensor array rather than serially by one sensor, "often in a single visual fixation"."""
    a, b, ca, cb = _pair()
    alone = 0                                                # how many fixations one column needs on its own
    a.start_object()
    for f in Y:
        if alone:
            a.path_integrate("E")
        alone += 1
        if a.perceive(f) != -1:
            break
    assert alone == 2, f"alone, the shared first feature forces a second fixation (got {alone})"

    a2, b2, ca2, cb2 = _pair()
    _glance(a2, Y[0])                                        # ONE fixation each, at different vantage points
    _glance(b2, Y[1])
    ca2.receive_votes()
    assert ca2.label_of(ca2.pooler.active) != -1, "with a peer, one fixation plus one vote suffices"


def test_columns_that_never_attended_anything_together_have_nothing_to_say():
    """The honest limit, and a property of the mechanism rather than a caveat: the correspondence between two columns'
    private codes is LEARNED from co-experience, so peers that have only ever looked at things separately cannot help each
    other. Silence is not a vote — the column stays exactly as ambiguous as it was, rather than being pushed somewhere
    arbitrary."""
    a, b = _column(0), _column(7)
    ca, cb = a._nav_col(), b._nav_col()
    ca.link(cb)
    for _ in range(4):                                       # both LEARN the objects — but never attend them together
        for obj in (X, Y, Z):
            _study(a, obj)
            _study(b, obj)
    _glance(a, 1)
    _glance(b, 3)
    ca.receive_votes()
    assert ca.label_of(ca.pooler.active) == -1, "with no learned lateral links there is no consensus to reach"
