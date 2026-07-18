"""test_ca3.py — the CA3 autoassociative attractor: one-shot store + pattern completion (hippocampus/ca3.py; DESIGN §2/§3, slice 3).

The maze-wall case one region up from the column: a partial cue completes to the whole stored pattern, an ambiguous cue stays
ambiguous (no confabulation — §3½), a novel cue recalls nothing. Plus the wired episodic path: remember a multi-object scene,
then recall the whole of it from a glimpse of ONE object.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                    # noqa: E402
from tbt.hippocampus.ca3 import CA3            # noqa: E402
from tbt.operator import eye                   # noqa: E402


def test_completes_a_strict_subset_cue():
    """One-shot store, then recall the WHOLE pattern from a strict subset — the attractor pulls back even the bits the cue
    omitted (absence filled by the model, never a new partial pattern)."""
    ca3 = CA3()
    P = {0, 1, 2, 3, 4}
    ca3.store(P)
    assert ca3.complete({0, 1}) == P, "a partial cue must complete to the whole stored pattern"
    assert ca3.complete({3}) == P, "even a single-bit cue completes (the whole basin pulls it back)"


def test_completes_a_noisy_cue():
    """A cue with a WRONG bit still settles to the clean pattern — the noise co-occurs with nothing stored, so it drops out."""
    ca3 = CA3()
    P = {0, 1, 2, 3, 4}
    ca3.store(P)
    assert ca3.complete({0, 1, 99}) == P, "noise (a bit that co-occurs with nothing) must drop out during the settle"


def test_capacity_several_patterns_no_crosstalk():
    """Store several well-separated patterns; each partial cue recalls ITS pattern, not a blend — capacity holds when the
    patterns don't overlap (which is exactly what DG pattern separation, slice 4, guarantees for real memories)."""
    ca3 = CA3()
    patterns = [set(range(4 * k, 4 * k + 4)) for k in range(5)]     # {0..3},{4..7},...,{16..19}
    for p in patterns:
        ca3.store(p)
    for p in patterns:
        cue = set(sorted(p)[:2])
        assert ca3.complete(cue) == p, f"cue {cue} must recall exactly its own pattern {p}, got {ca3.complete(cue)}"


def test_an_ambiguous_cue_stays_ambiguous():
    """§3½ at the scene level: a cue that fits TWO stored patterns equally must not confabulate one — it settles to the UNION
    of both, overlapping each. A cue with a pattern's OWN distinctive bits resolves cleanly to that pattern."""
    ca3 = CA3()
    P = {1, 2, 3, 4, 5}
    Q = {4, 5, 6, 7, 8}                                            # shares {4,5} with P
    ca3.store(P)
    ca3.store(Q)
    clear = ca3.complete({1, 2, 3})                               # P's distinctive bits → resolve to P
    assert clear == P, f"a distinctive cue must resolve cleanly, got {clear}"
    ambiguous = ca3.complete({4, 5})                             # only the shared bits → fits both
    assert {1, 2, 3} <= ambiguous and {6, 7, 8} <= ambiguous, (
        f"an ambiguous cue must stay ambiguous (the union of both), not confabulate one, got {ambiguous}")


def test_a_novel_cue_recalls_nothing():
    """A cue of bits that were never stored collapses to nothing — the 'no memory here' signal CA1 will read as novelty."""
    ca3 = CA3()
    ca3.store({0, 1, 2, 3})
    assert ca3.complete({50, 51}) == set(), "a wholly novel cue must recall nothing"


def test_agent_recalls_a_whole_scene_from_a_glimpse():
    """The wired episodic path: remember a three-object scene, then recall ALL of it from a glimpse of ONE object — the
    maze-wall / partial-scene completion, end to end through the agent."""
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a.place_object("A", ((5.0, 5.0), eye(2)))
    a.place_object("B", ((10.0, 20.0), eye(2)))
    a.place_object("C", ((30.0, 2.0), eye(2)))
    full = a.scene_tokens()
    a.remember_scene()
    recalled = a.recall_scene({("A", (5, 5))})
    assert recalled == set(full), f"a glimpse of one object must recall the whole remembered scene, got {recalled}"


if __name__ == "__main__":
    one = CA3()                                                   # ONE well-separated pattern → clean completion
    one.store({1, 2, 3, 4, 5})
    print(f"partial cue   {{1,2}}    → {sorted(one.complete({1, 2}))}")
    print(f"noisy cue     {{1,2,99}} → {sorted(one.complete({1, 2, 99}))}  (99 dropped)")
    print(f"novel cue     {{40,41}}  → {sorted(one.complete({40, 41}))}")
    two = CA3()                                                   # two OVERLAPPING patterns (sharing {4,5}) — DG's job to separate
    two.store({1, 2, 3, 4, 5}); two.store({4, 5, 6, 7, 8})
    print(f"distinctive cue {{1,2,3}} → {sorted(two.complete({1, 2, 3}))}  (resolves to one)")
    print(f"ambiguous cue   {{4,5}}   → {sorted(two.complete({4, 5}))}  (union of both = ambiguous, no confabulation)")
