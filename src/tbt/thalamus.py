"""The thalamus — the inter-column ROUTER / GATE (ARCHITECTURE.md §3, the loop's relay/gate arm).

Two roles in the loop: it **relays** a cortical column's percept into the shared currency the selector reads, and it
**gates** the basal ganglia's winner back out to the motor (disinhibition — one selection drives behaviour, the rest are
suppressed). It is how "many columns" and "the selector" and "the motor" are joined without fusing them into one fatter
unit (the 2^K conjunctive explosion; ARCHITECTURE §2). The thalamus itself is a stateless router.

SCOPE (2026-07-10, first slice): `relay` + `gate` — the cortex→BG→motor path of the decision loop. The RICHER thalamic
role — binding **content ⊗ location** across two columns into a conjunctive register for place-value / cross-column VOTING
(Smolensky/VSA tensor product; the legacy `Thalamus.bind/read`) — is added with a task that needs it (a place-value number,
a multi-object scene). Grounded: Sherman & Guillery (transthalamic relay); reference_htm_pooling_recall_heterarchy (voting).
"""

from __future__ import annotations


class Thalamus:
    """The relay/gate fabric of the cortico-basal-ganglia-thalamic loop."""

    def relay(self, percept_cells) -> frozenset:
        """Relay a column's L4 active-cell list (its PERCEPT) into a stable, hashable CONTEXT the basal ganglia keys on —
        the cortex→BG relay. Order-independent + deterministic, so the same percept always maps to the same selector
        context (a frozen sensory column gives a stable percept per stimulus)."""
        return frozenset(percept_cells)

    def gate(self, selection):
        """Gate the basal ganglia's SELECTION back out to the motor: default-closed disinhibition — the one selected
        action is enacted, the competitors suppressed. (In this discrete agent the gate is the identity relay of the
        winner; the suppression is implicit in 'only the winner is enacted'.)"""
        return selection

    def project(self, content, location):
        """The transthalamic RELAY of a recognised object to a HIGHER region: carry its CONTENT (object-id) bound with its
        LOCATION (pose) upward, so a compositional column can treat the pair as one feature-at-location (Sherman & Guillery;
        `reference_tbt_layers_4_23`: "object id as a FEATURE → compositional objects"). This is the content ⊗ location binding
        the first slice deferred "until a task needs it (a multi-object scene)" — the context-gated override is that task.
        Stateless: the binding is the pairing itself, and the higher column does the modelling."""
        return content, location
