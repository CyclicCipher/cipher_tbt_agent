"""test_dg.py — dentate gyrus PATTERN SEPARATION → chart keys (hippocampus/dg.py; DESIGN §2/§3, slice 4).

DG turns an environment signature into a sparse, orthogonalized chart key: the SAME environment returns the SAME key
(deterministic), DISTINCT environments get well-separated keys, and the separation is GRADED (more-similar signatures keep
more key overlap). Its payoff is the fix for CA3's one honest seam (slice 3): overlapping raw memories cross-talk in CA3, but
their DG-separated keys are decorrelated enough that CA3 stores and recalls each cleanly.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                    # noqa: E402
from tbt.hippocampus.ca3 import CA3            # noqa: E402
from tbt.hippocampus.dg import DG              # noqa: E402


def _ov(a, b) -> int:
    return len(a & b)


def test_the_same_signature_gives_the_same_key():
    """Deterministic separation — a returning agent recognises the same environment as the same chart."""
    dg = DG(n_inputs=512, seed=0)
    sig = set(range(0, 30))
    assert dg.separate(sig) == dg.separate(sig), "the same environment signature must yield the same chart key"


def test_distinct_environments_get_separated_keys():
    """Disjoint signatures → near-disjoint keys, so their memories will not interfere in CA3."""
    dg = DG(n_inputs=512, seed=0)
    kA = dg.separate(set(range(0, 30)))
    kB = dg.separate(set(range(200, 230)))
    assert _ov(kA, kB) < 0.25 * len(kA), f"distinct environments must separate (low key overlap), got {_ov(kA, kB)}/{len(kA)}"


def test_separation_is_graded():
    """A different VIEW of the same environment (mostly-shared signature) keeps more key overlap than a wholly distinct one —
    the monotonic property remapping relies on to tell 'same place, new view' from 'new place'."""
    dg = DG(n_inputs=512, seed=0)
    kA = dg.separate(set(range(0, 30)))
    kSimilar = dg.separate(set(range(0, 25)) | set(range(30, 35)))     # shares 25/30 input bits with A
    kDistinct = dg.separate(set(range(200, 230)))
    assert _ov(kA, kSimilar) > _ov(kA, kDistinct), (
        f"similar env must share more key than a distinct one, got similar={_ov(kA, kSimilar)} distinct={_ov(kA, kDistinct)}")


def test_dg_separation_resolves_ca3_crosstalk():
    """The slice-3 seam, closed: two OVERLAPPING raw signatures cross-talk when stored directly in CA3 (a shared-bit cue
    recalls the union), but their DG keys are decorrelated below the input overlap and CA3 recalls each cleanly."""
    dg = DG(n_inputs=512, seed=0)
    S1, S2 = set(range(0, 30)), set(range(15, 45))                    # 50% input overlap → would cross-talk
    K1, K2 = dg.separate(S1), dg.separate(S2)
    assert _ov(K1, K2) < 0.4 * len(K1), f"DG must decorrelate below the input overlap, got {_ov(K1, K2)}/{len(K1)}"

    raw = CA3()
    raw.store(S1); raw.store(S2)
    bled = raw.complete(set(sorted(S1 & S2)[:5]))                     # a cue from the SHARED bits
    assert S1 <= bled and S2 <= bled, f"raw overlapping signatures must cross-talk in CA3 (the union), got {sorted(bled)}"

    charts = CA3()
    charts.store(K1); charts.store(K2)
    cue = set(sorted(K1 - K2)[:8])                                    # a glimpse of K1's distinctive key bits
    assert charts.complete(cue) == K1, "DG-separated keys must be recalled cleanly by CA3 — no cross-talk"


def test_agent_produces_separated_chart_keys():
    """Wired: the agent turns an environment signature into a separated chart key — deterministic, and distinct environments
    separate more than similar ones (the base for remapping, slice 5)."""
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    sigA = set(range(0, 30))
    sigSimilar = set(range(0, 25)) | set(range(30, 35))
    sigB = set(range(200, 230))
    kA = a.chart_key(sigA)
    assert a.chart_key(sigA) == kA, "same environment signature → same chart key (deterministic)"
    assert _ov(kA, a.chart_key(sigB)) < _ov(kA, a.chart_key(sigSimilar)), \
        "distinct environments must separate more than similar ones"


if __name__ == "__main__":
    d = DG(n_inputs=512, seed=0)
    kA = d.separate(set(range(0, 30)))
    kSim = d.separate(set(range(0, 25)) | set(range(30, 35)))
    kDist = d.separate(set(range(200, 230)))
    print(f"|key| = {len(kA)}")
    print(f"key overlap  A vs similar-view : {_ov(kA, kSim)}")
    print(f"key overlap  A vs distinct env : {_ov(kA, kDist)}  (separation)")
