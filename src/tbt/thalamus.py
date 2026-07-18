"""The thalamus — the inter-column ROUTER / GATE (ARCHITECTURE.md §3, the loop's relay/gate arm).

Two roles in the loop: it **relays** a cortical column's percept into the shared currency the selector reads, and it
**gates** the basal ganglia's winner back out to the motor (disinhibition — one selection drives behaviour, the rest are
suppressed). It is how "many columns" and "the selector" and "the motor" are joined without fusing them into one fatter
unit (the 2^K conjunctive explosion; ARCHITECTURE §2). The thalamus itself is a stateless router.

SCOPE. `relay` + `gate` = the cortex→BG→motor path of the decision loop. `project` = the transthalamic relay of a recognised
object UP to a compositional column (the override slice). **Phase 5 adds the CONTENT ⊗ LOCATION binding** (`bind`/`bundle`/
`read`) — the conjunctive register for place-value (a digit AT a place) and cross-column CMP VOTING (an object bound to a
pose), the Smolensky/VSA tensor product. Grounded: Sherman & Guillery (transthalamic relay); `reference_tbt_layers_4_23` (the
CMP vote is STRUCTURE-PRESERVING — content bound to location, not a bag of features); reference_htm_pooling_recall_heterarchy.
"""

from __future__ import annotations

from collections import Counter


class Thalamus:
    """The relay/gate fabric of the cortico-basal-ganglia-thalamic loop, plus the content⊗location BINDING register. Stateless:
    every method is a pure function of its inputs (a register is returned, not held)."""

    def relay(self, percept_cells) -> frozenset:
        """Relay a column's L4 active-cell list (its PERCEPT) into a stable, hashable CONTEXT the basal ganglia keys on —
        the cortex→BG relay. Order-independent + deterministic, so the same percept always maps to the same selector
        context (a frozen sensory column gives a stable percept per stimulus)."""
        return frozenset(percept_cells)

    def gate(self, selection, n_channels: int = 0):
        """DEFAULT-OFF disinhibition of the BG winner: the resting state is all channels INHIBITED, and selection opens ONLY
        the winner's channel (the competitors stay closed). `selection is None` ⇒ nothing is disinhibited ⇒ nothing enacted
        (the true resting default). Returns the enacted channel (the winner), or None. `n_channels` is accepted for callers
        that want to assert the winner is a valid channel; the suppression of the rest is implicit in returning only the one."""
        if selection is None or (n_channels and not (0 <= selection < n_channels)):
            return None
        return selection

    # ── CONTENT ⊗ LOCATION binding (Phase 5): the conjunctive register for place-value + cross-column voting ─────────
    def bind(self, content, location) -> frozenset:
        """The tensor-product BIND (Smolensky/VSA): content ⊗ location = every `(content_bit, location_bit)` conjunction. It
        is REVERSIBLE — `read` recovers the content bound at a location — and STRUCTURE-PRESERVING: "digit 4 at the tens
        place" or "object A at pose P" is one bound thing, distinct from the same content at another location (the CMP vote,
        `reference_tbt_layers_4_23`). `content`/`location` are iterables of active bits."""
        location = list(location)
        return frozenset((c, l) for c in content for l in location)

    def bundle(self, *bounds) -> Counter:
        """Superpose bound pairs into one REGISTER — a support Counter over conjunctions — the shared register columns write
        their votes into. OVERLAP = AGREEMENT: a `(content, location)` conjunction bound by k votes accrues support k, so the
        count IS the cross-column vote tally (the substrate for consensus)."""
        register: Counter = Counter()
        for b in bounds:
            register.update(b)
        return register

    def read(self, register, location, min_support: int = 1) -> frozenset:
        """UNBIND: the content bits bound at `location` with at least `min_support` — a content bit `c` is "at `location`" to
        the extent it is bound with EVERY bit of `location` (its support = the weakest of those conjunction counts). So
        `min_support=1` is exact recall (place-value: recover the digit at a place); `min_support=k` is the VOTING threshold
        (only content ≥ k columns agree on survives, filtering a single column's distractor). Empty location ⇒ nothing."""
        loc = list(location)
        if not loc:
            return frozenset()
        return frozenset(c for c in {c for c, _l in register}
                         if min(register.get((c, l), 0) for l in loc) >= min_support)

    def project(self, content, location):
        """The transthalamic RELAY of a recognised object to a HIGHER region: carry its CONTENT (object-id) bound with its
        LOCATION (pose) upward, so a compositional column can treat the pair as one feature-at-location (Sherman & Guillery;
        `reference_tbt_layers_4_23`: "object id as a FEATURE → compositional objects"). This is the content ⊗ location binding
        the first slice deferred "until a task needs it (a multi-object scene)" — the context-gated override is that task.
        Stateless: the binding is the pairing itself, and the higher column does the modelling."""
        return content, location
