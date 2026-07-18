"""hippocampus/ca1.py — the CA1 COMPARATOR + multi-chart REMAPPING (DESIGN §2/§3, slice 5).

CA1 (Lisman; Hasselmo): the hippocampus's match/novelty detector. It receives CA3's RECALL (via the Schaffer collaterals) and
the current EC INPUT (via the direct perforant path) and COMPARES them — a match means the memory predicted reality, a
mismatch is NOVELTY (the situation is new, or a known one has CHANGED), which gates storing a new memory and drives exploration.

The comparison IS the §3½ invariant (`reference_recognition_under_occlusion`) one region up from the column's ART reset: a
recall MATCHES the observation iff it EXPLAINS every observed bit (observed ⊆ recalled). Bits of the recalled memory you have
not observed are just not-yet-seen (ABSENCE — a partial view still matches); an observed bit the recall does NOT contain is a
CONTRADICTION (something is genuinely different — mismatch). Mint on REFUTATION, never on incompleteness.

REMAPPING composes CA3 (recall) + CA1 (compare): on entering an environment, CA3 completes the observed CONTENT to a stored
chart and CA1 confirms; a mismatch recruits a NEW chart (global remapping). The comparison runs on CONTENT tokens, NOT DG keys
— the §3½ rule needs the SUBSET relation, and DG's k-WTA does not preserve it (a partial signature is not a subset key). DG's
role (slice 4) is separating DISTINCT full signatures at the chart-INDEX layer; the two layers compose in the orchestrator
(slice 6). Pure stdlib.
"""

from __future__ import annotations

from collections import namedtuple

from .ca3 import CA3

CA1Result = namedtuple("CA1Result", ["matched", "novelty", "unexplained"])


class CA1:
    """The comparator: `compare(observed, recalled)` → a `CA1Result`. `matched` iff the recall explains every observed bit
    (observed ⊆ recalled); `novelty` is the fraction it fails to explain; `unexplained` are the contradicting bits."""

    def compare(self, observed, recalled) -> CA1Result:
        observed, recalled = set(observed), set(recalled)
        unexplained = observed - recalled                     # observed bits the recall fails to explain = CONTRADICTION
        novelty = len(unexplained) / max(1, len(observed))
        return CA1Result(matched=not unexplained, novelty=novelty, unexplained=frozenset(unexplained))


class Remapper:
    """Multi-chart REMAPPING: hold a DISTINCT chart per environment and, on entering one, RECALL its chart or MINT a new one.
    CA3 completes the observed content to a stored chart, CA1 confirms the match; on a CA1 MISMATCH — a novel environment, or a
    known one that has CHANGED (a contradicting landmark) — a new chart is recruited. A PARTIAL view of a known environment
    still matches (absence ≠ novelty), so seeing LESS does not throw you into a new chart. Overlapping full signatures are kept
    separable by DG (slice 4) at the index layer — composed in the orchestrator; here the recall/compare on content is the
    mechanism. (Graded remap — UPDATE a chart on a small change vs MINT on a large one — is a policy over `result.novelty`,
    a refinement; this mints on any contradiction.)"""

    def __init__(self) -> None:
        self.ca3 = CA3()                                      # the chart store (one CA3 mechanism, used for charts)
        self.ca1 = CA1()
        self.charts: list = []                               # chart id (index) -> its stored content tokens

    def visit(self, observed):
        """Enter an environment described by `observed` content tokens. Returns `(chart_id, CA1Result)`: the RECALLED chart if
        a stored one explains the observation, else a freshly MINTED chart (with `result.matched` False)."""
        observed = frozenset(observed)
        recalled = self.ca3.complete(observed)
        result = self.ca1.compare(observed, recalled)
        if recalled and result.matched:
            cid = self._nearest(recalled)
            if cid is not None:
                return cid, result                            # RECALL a known chart
        self.ca3.store(observed)                              # MINT: novelty (or a changed known env) → a new chart
        self.charts.append(observed)
        return len(self.charts) - 1, result

    def _nearest(self, recalled):
        """The stored chart whose tokens best match a CA3 recall (highest overlap); None if there are none."""
        best, best_ov = None, 0
        for cid, toks in enumerate(self.charts):
            ov = len(toks & recalled)
            if ov > best_ov:
                best, best_ov = cid, ov
        return best
