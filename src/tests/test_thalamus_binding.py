"""End-to-end test of the THALAMUS content⊗location BINDING (ARCHITECTURE.md §3; ROADMAP Phase 5) — the conjunctive
register, driven through `Agent.read_register`.

The thalamus binds content to location (the Smolensky/VSA tensor product), so a bound thing is STRUCTURE-PRESERVING —
"digit 4 at the tens place", "feature 6 at cell (3,1)" — and reversible by `read`. That is what makes a number a number
rather than a bag of digits, and it is what `_sense_frame` writes the L4 surface into.

WHAT WAS CUT, 2026-07-27. This file used to also test cross-column CMP VOTING here — many columns binding their (object,
pose) votes into the shared register and reading with `min_support=k` to keep the majority. Per TBP's long-range-connections
paper (arXiv:2507.05888, `reference_long_range_connections`) that is the wrong locus: consensus between PEER columns travels
on direct LATERAL cortico-cortical links, not through a relay. The thalamus's own long-range job is HIERARCHICAL (`project`,
the transthalamic route carrying a recognised object UP). Voting moved to where the anatomy puts it — `test_lateral_voting`
— and the tests that asserted it here are gone rather than kept alongside, because two mechanisms for one job is the failure
this codebase is disciplined against. The register itself is untouched and still earns its place below.
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


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _decode(enc: CategoryEncoder, bits):
    return enc.decode(SDR(enc.n, sorted(bits))) if bits else None


def test_place_value_is_bound_and_read_back_exactly():
    """A multi-digit number = digits bound to places, bundled into one register. Reading each place recovers exactly its
    digit — the roundtrip a positional number needs, and which a bag-of-digits (no binding) could not give."""
    agent = _fresh()
    digits = {2: 4, 1: 2, 0: 5}                                    # the number 4·100 + 2·10 + 5 = "425"
    entries = [(DIGIT.encode(d).active, PLACE.encode(p).active) for p, d in digits.items()]
    for p, d in digits.items():
        got = agent.read_register(entries, PLACE.encode(p).active)
        assert _decode(DIGIT, got) == d, f"place {p} must read back digit {d}, got {_decode(DIGIT, got)}"


def test_the_binding_is_STRUCTURE_PRESERVING():
    """Content is bound TO its location, so the SAME digit at different places, and different digits at the same place, do not
    leak: reading a place returns ITS digit, never a neighbour's. This is why it is a structure (a number), not a set."""
    agent = _fresh()
    entries = [(DIGIT.encode(7).active, PLACE.encode(2).active),   # 7 at hundreds
               (DIGIT.encode(7).active, PLACE.encode(0).active),   # 7 again, at ones (same content, other place)
               (DIGIT.encode(3).active, PLACE.encode(1).active)]   # 3 at tens
    assert _decode(DIGIT, agent.read_register(entries, PLACE.encode(2).active)) == 7, "hundreds is 7"
    assert _decode(DIGIT, agent.read_register(entries, PLACE.encode(1).active)) == 3, "tens is 3 (not 7 — no leak)"
    assert _decode(DIGIT, agent.read_register(entries, PLACE.encode(0).active)) == 7, "ones is 7"
    assert agent.read_register(entries, PLACE.encode(3).active) == frozenset(), "an empty place reads back nothing"


def test_support_counts_how_many_writers_asserted_a_conjunction():
    """The register grades agreement: a conjunction written twice carries support 2, so `min_support` can filter content only
    one writer asserted. Kept because `_sense_frame` relies on the register being a graded store rather than a set — but this
    is a property of the STORE, and it is no longer claimed to be how columns vote."""
    agent = _fresh()
    entries = [(DIGIT.encode(4).active, PLACE.encode(1).active),
               (DIGIT.encode(4).active, PLACE.encode(1).active),   # the same fact asserted twice
               (DIGIT.encode(9).active, PLACE.encode(1).active)]   # a different one, asserted once
    strong = agent.read_register(entries, PLACE.encode(1).active, min_support=2)
    assert _decode(DIGIT, strong) == 4, f"only the twice-written conjunction survives, got {_decode(DIGIT, strong)}"
    both = agent.read_register(entries, PLACE.encode(1).active, min_support=1)
    assert set(DIGIT.encode(4).active) <= both and set(DIGIT.encode(9).active) <= both, "min_support=1 keeps everything"


if __name__ == "__main__":
    ag = _fresh()
    number = {2: 4, 1: 2, 0: 5}
    entries = [(DIGIT.encode(d).active, PLACE.encode(p).active) for p, d in number.items()]
    print("place-value read-back of 4_2_5:",
          {n: _decode(DIGIT, ag.read_register(entries, PLACE.encode(p).active)) for p, n in [(2, "H"), (1, "T"), (0, "O")]})
