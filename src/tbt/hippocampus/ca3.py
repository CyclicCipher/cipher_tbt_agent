"""hippocampus/ca3.py — the CA3 autoassociative ATTRACTOR: one-shot storage + pattern completion (DESIGN §2/§3, slice 3).

Marr/Treves/Rolls: CA3's recurrent collaterals form a SINGLE attractor network doing TWO jobs Rolls shows are one — the
one-trial EPISODIC store AND pattern COMPLETION ("rapid, one-trial associations... completion of the whole memory during
recall from any part"). `store` binds a pattern into the recurrent weights in a SINGLE pass (Hebbian: co-active bits
strengthen — fast, complementary to the slow cortex); `complete` settles from a PARTIAL or NOISY cue back to the whole
stored pattern.

This is the maze-wall / partial-scene case of `reference_recognition_under_occlusion`, one region up from the column: a local
glimpse completes to the whole, because recall is pattern-COMPLETION, not whole-pattern matching. It carries the §3½ invariant
too — a partial cue COMPLETES (the model fills the absence), an AMBIGUOUS cue stays ambiguous (it settles to the UNION of the
patterns it fits, never a confabulated single answer), and a wholly NOVEL cue recalls nothing (the novelty CA1 will read).

A sparse Hopfield / Kanerva SDM over SDRs — patterns are sets of HASHABLE bits (ints, or (object, cell) episode tokens). The
settle uses BINARY co-occurrence overlap (robust to weight magnitude); the stored COUNT is kept for future reliability
weighting. Pure stdlib. DG pattern separation (slice 4) keeps distinct memories from colliding here; without it, heavily
overlapping patterns will cross-talk — that is DG's job, not a flaw of the attractor.
"""

from __future__ import annotations

from collections import Counter, defaultdict


class CA3:
    """The recurrent autoassociator: `store(pattern)` one-shot, `complete(cue)` by settling. `theta` is the vigilance of
    completion — a bit joins the recalled pattern iff it co-occurred with at least `theta` of the currently-active bits."""

    def __init__(self, theta: float = 0.5, iters: int = 10) -> None:
        self.co: dict = defaultdict(dict)   # co[i][j] = times bits i,j were co-stored — the recurrent weight
        self.theta = float(theta)
        self.iters = int(iters)
        self.n_stored = 0

    def store(self, pattern) -> None:
        """One-shot Hebbian storage: strengthen the recurrent weight between every pair of co-active bits. One pass = one
        trial — the fast episodic write the cortex cannot do."""
        bits = list(set(pattern))
        for a in bits:
            row = self.co[a]
            for b in bits:
                if a != b:
                    row[b] = row.get(b, 0) + 1
        self.n_stored += 1

    def complete(self, cue):
        """Pattern-complete from a partial/noisy `cue`: GROW the active set by repeatedly adding bits that co-occurred with at
        least `theta` of it, then PRUNE bits the settled set does not support. A SUBSET cue fills in (even a 1-bit cue recovers
        the whole — a strict recompute would OSCILLATE on a 2-bit pattern, each bit pulling the other but dropping itself);
        NOISE drops out (a cue bit that belongs to no stored pattern fails the final prune); an AMBIGUOUS cue settles to the
        UNION of the patterns it fits (no confabulation); a wholly NOVEL cue collapses to nothing. Bounded by `iters`.

        The GROW pass is monotonic (never removes mid-settle), which is what makes it converge for small patterns; the single
        PRUNE at the end applies the settled-set threshold once, so noise and stray cue bits are removed without re-opening
        the oscillation."""
        active = set(cue)
        for _ in range(self.iters):
            count = self._support(active)
            thresh = self.theta * len(active) if active else 0.0
            add = {j for j, c in count.items() if c >= thresh}
            if add <= active:
                break                                        # nothing new co-occurs strongly enough — settled
            active |= add                                    # GROW (monotonic): recover the whole pattern from any cue
        count = self._support(active)                        # PRUNE once: keep only what the settled set supports (drops noise)
        thresh = self.theta * len(active) if active else 0.0
        return {j for j in active if count.get(j, 0) >= thresh}

    def _support(self, active) -> Counter:
        """For each bit, the number of ACTIVE bits it co-occurred with in storage (binary overlap) — the recurrent drive."""
        count: Counter = Counter()
        for i in active:
            for j in self.co.get(i, ()):
                count[j] += 1
        return count
