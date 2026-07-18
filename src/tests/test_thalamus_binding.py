"""End-to-end test of the THALAMUS content⊗location BINDING (ARCHITECTURE.md §3; ROADMAP Phase 5) — the conjunctive register
for PLACE-VALUE and cross-column CMP VOTING, driven through `Agent.vote_consensus`.

The thalamus binds content to location (the Smolensky/VSA tensor product), so a bound thing is STRUCTURE-PRESERVING —
"digit 4 at the tens place", "object A at pose P" — reversible by `read`. Two uses, one mechanism:
  * PLACE-VALUE: each place holds one digit; reading a place recovers exactly its digit (min_support = 1, exact recall).
  * VOTING: many columns bind their (object, pose) vote into a shared register; reading with min_support = k keeps only the
    content that ≥ k columns AGREE on, filtering a single column's distractor (`reference_tbt_layers_4_23`: the CMP vote is
    content bound to location, not a bag of features).
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                    # noqa: E402
from tbt.encoders import CategoryEncoder, SDR  # noqa: E402

DIGIT = CategoryEncoder(range(10), w=8, capacity=10)
PLACE = CategoryEncoder(range(4), w=8, capacity=4)                 # 0=ones, 1=tens, 2=hundreds, 3=thousands
OBJ = CategoryEncoder(range(6), w=8, capacity=6)
POSE = CategoryEncoder(range(6), w=8, capacity=6)


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _decode(enc: CategoryEncoder, bits):
    return enc.decode(SDR(enc.n, sorted(bits))) if bits else None


def test_place_value_is_bound_and_read_back_exactly():
    """A multi-digit number = digits bound to places, bundled into one register. Reading each place recovers exactly its
    digit — the roundtrip a positional number needs, and which a bag-of-digits (no binding) could not give."""
    agent = _fresh()
    digits = {2: 4, 1: 2, 0: 5}                                    # the number 4·100 + 2·10 + 5 = "425"
    votes = [(DIGIT.encode(d).active, PLACE.encode(p).active) for p, d in digits.items()]
    for p, d in digits.items():
        got = agent.vote_consensus(votes, PLACE.encode(p).active)
        assert _decode(DIGIT, got) == d, f"place {p} must read back digit {d}, got {_decode(DIGIT, got)}"


def test_the_binding_is_STRUCTURE_PRESERVING():
    """Content is bound TO its location, so the SAME digit at different places, and different digits at the same place, do not
    leak: reading a place returns ITS digit, never a neighbour's. This is why it is a structure (a number), not a set."""
    agent = _fresh()
    votes = [(DIGIT.encode(7).active, PLACE.encode(2).active),     # 7 at hundreds
             (DIGIT.encode(7).active, PLACE.encode(0).active),     # 7 again, at ones (same content, other place)
             (DIGIT.encode(3).active, PLACE.encode(1).active)]     # 3 at tens
    assert _decode(DIGIT, agent.vote_consensus(votes, PLACE.encode(2).active)) == 7, "hundreds is 7"
    assert _decode(DIGIT, agent.vote_consensus(votes, PLACE.encode(1).active)) == 3, "tens is 3 (not 7 — no leak)"
    assert _decode(DIGIT, agent.vote_consensus(votes, PLACE.encode(0).active)) == 7, "ones is 7"
    assert agent.vote_consensus(votes, PLACE.encode(3).active) == frozenset(), "an empty place reads back nothing"


def test_cross_column_voting_keeps_the_CONSENSUS_filters_the_distractor():
    """Three columns vote at the same pose: two agree on object A, one dissents with B. At min_support=2 the register returns
    the CONSENSUS (A) and filters the lone distractor (B); at min_support=1 it returns the union (both). Support = agreement."""
    agent = _fresh()
    votes = [(OBJ.encode(0).active, POSE.encode(1).active),        # col 1: object A
             (OBJ.encode(0).active, POSE.encode(1).active),        # col 2: object A (agrees)
             (OBJ.encode(1).active, POSE.encode(1).active)]        # col 3: object B (distractor)
    consensus = agent.vote_consensus(votes, POSE.encode(1).active, min_support=2)
    assert _decode(OBJ, consensus) == 0, f"≥2 columns agree on A ⇒ consensus A, got {_decode(OBJ, consensus)}"
    both = agent.vote_consensus(votes, POSE.encode(1).active, min_support=1)
    assert set(OBJ.encode(0).active) <= both and set(OBJ.encode(1).active) <= both, "min_support=1 keeps every vote (A and B)"


def test_a_split_vote_reaches_NO_consensus():
    """When columns DISAGREE (different objects, same pose, one each), there is no majority — the honest answer at
    min_support=2 is nothing, not a forced pick."""
    agent = _fresh()
    votes = [(OBJ.encode(3).active, POSE.encode(2).active),        # col 1: C
             (OBJ.encode(4).active, POSE.encode(2).active)]        # col 2: D — a tie, no agreement
    assert agent.vote_consensus(votes, POSE.encode(2).active, min_support=2) == frozenset(), "a split vote has no consensus"


if __name__ == "__main__":
    ag = _fresh()
    number = {2: 4, 1: 2, 0: 5}
    votes = [(DIGIT.encode(d).active, PLACE.encode(p).active) for p, d in number.items()]
    print("place-value read-back of 4_2_5:",
          {n: _decode(DIGIT, ag.vote_consensus(votes, PLACE.encode(p).active)) for p, n in [(2, "H"), (1, "T"), (0, "O")]})
    v = [(OBJ.encode(0).active, POSE.encode(1).active)] * 2 + [(OBJ.encode(1).active, POSE.encode(1).active)]
    print("voting at pose 1 (A,A,B): consensus(≥2) =",
          _decode(OBJ, ag.vote_consensus(v, POSE.encode(1).active, min_support=2)))
