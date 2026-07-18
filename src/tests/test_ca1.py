"""test_ca1.py — the CA1 comparator + multi-chart REMAPPING (hippocampus/ca1.py; DESIGN §2/§3, slice 5).

CA1 compares CA3's recall against the observed input: a recall MATCHES iff it explains every observed bit (observed ⊆
recalled). That is the §3½ invariant one region up — a PARTIAL view matches (absence ≠ novelty), a CONTRADICTED view (a bit
the recall lacks) mismatches (novelty). Remapping composes CA3 (recall) + CA1 (compare): revisit → recall the chart; a novel
or CHANGED environment → mint a new one. The scene-level analogue of the column's mint-on-refutation.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                          # noqa: E402
from tbt.hippocampus.ca1 import CA1, Remapper        # noqa: E402

A = {"a1", "a2", "a3", "a4"}
B = {"b1", "b2", "b3", "b4"}


# ── the CA1 comparator: the §3½ rule (absence ≠ contradiction) ──────────────────────────────────────────────────────
def test_ca1_partial_view_matches():
    """A partial view of a chart matches it — the unobserved landmarks are absence, not evidence against."""
    r = CA1().compare(observed={"a1", "a2"}, recalled=A)
    assert r.matched and r.novelty == 0.0, f"observed ⊆ recalled must MATCH, got {r}"


def test_ca1_contradiction_mismatches():
    """An observed bit the recall does NOT contain is a contradiction → mismatch (novelty)."""
    r = CA1().compare(observed={"a1", "a2", "X"}, recalled=A)
    assert not r.matched and r.novelty > 0.0 and "X" in r.unexplained, f"a contradiction must MISMATCH, got {r}"


def test_ca1_nothing_recalled_is_novelty():
    """When CA3 recalls nothing, the whole observation is unexplained → maximal novelty (a wholly new place)."""
    r = CA1().compare(observed={"a1", "a2"}, recalled=set())
    assert not r.matched and r.novelty == 1.0, f"an empty recall must be full novelty, got {r}"


# ── remapping: recall a known chart, mint a new one ─────────────────────────────────────────────────────────────────
def test_remapper_mints_distinct_charts_and_recalls_them():
    """Distinct environments get distinct charts; revisiting one recalls it (not a new mint)."""
    rm = Remapper()
    ida, ra = rm.visit(A)
    idb, rb = rm.visit(B)
    assert not ra.matched and not rb.matched, "first sight of each environment is novel"
    assert idb != ida, "a distinct environment gets its OWN chart"
    ida2, ra2 = rm.visit(A)
    assert ida2 == ida and ra2.matched, "revisiting an environment RECALLS its chart"
    assert len(rm.charts) == 2, "revisiting must not mint a duplicate"


def test_remapper_recalls_from_a_partial_view():
    """A glimpse of a known environment recalls its chart — you are not thrown into a new chart for seeing less (§3½)."""
    rm = Remapper()
    ida, _ = rm.visit(A)
    idp, rp = rm.visit({"a1", "a2"})
    assert idp == ida and rp.matched, "a partial view must RECALL the known chart"
    assert len(rm.charts) == 1, "a partial view must not mint a new chart"


def test_remapper_remaps_on_a_contradicted_view():
    """A landmark the chart lacks (the environment CHANGED) is a CA1 mismatch → remap to a new chart."""
    rm = Remapper()
    ida, _ = rm.visit(A)
    idc, rc = rm.visit({"a1", "a2", "a3", "X"})
    assert not rc.matched, "a contradicting landmark must MISMATCH the known chart"
    assert idc != ida and len(rm.charts) == 2, "a contradicted (changed) environment must remap to a new chart"


def test_agent_remaps_environments():
    """Wired: the agent recalls a known environment from a partial view and separates a distinct one."""
    ag = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    ida, _ = ag.visit_environment(A)
    idb, _ = ag.visit_environment(B)
    ida2, r = ag.visit_environment({"a1", "a2"})
    assert idb != ida, "distinct environments separate"
    assert ida2 == ida and r.matched, "a partial view recalls the known chart"


if __name__ == "__main__":
    rm = Remapper()
    print(f"visit A            → chart {rm.visit(A)[0]} (minted)")
    print(f"visit B            → chart {rm.visit(B)[0]} (minted, distinct)")
    print(f"revisit A          → chart {rm.visit(A)[0]} (recalled)")
    print(f"glimpse {{a1,a2}}     → chart {rm.visit({'a1', 'a2'})[0]} (recalled from a partial view)")
    cid, res = rm.visit({"a1", "a2", "a3", "X"})
    print(f"A but with X       → chart {cid} (remapped; novelty={res.novelty:.2f})")
