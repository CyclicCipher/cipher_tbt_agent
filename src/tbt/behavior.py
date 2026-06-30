"""Interaction-learned object BEHAVIOUR -- the dynamical model the column attaches to each RECOGNISED object.

An object's appearance (a connected component -- the Gestalt bias) does not tell you how it BEHAVES: a wall and a
walk-through painting look alike; you learn the difference only by INTERACTING (the JEPA lesson -- affordance comes
from dynamics, not co-occurring pixels). So this faculty records the OBSERVED EFFECT of the agent acting on an object,
keyed on the object's RECOGNISED identity (recognize.py) so it GENERALISES across instances and poses, and REVISES it
on contradiction. Nothing is assumed -- a thing is not a barrier until a bump shows it (a painting passes) -- and no
KIND is enumerated; only the outcome is learned (the bitter lesson).

This is possible now because the cortical column is the ENGINE: it owns path integration (so we know where the agent
TRIED to go) and prediction error (the bump = the move that didn't advance). Before the reset the column was a
decoration, so wall learning could not be done correctly -- this is the corrected attempt.

First effect modelled: BARRIER-ness -- 'moving into this object leaves the agent in place'. That IS the object's
contribution to reachability (a wall = reshaped reachability, generalised by recognition rather than memorised per
cell). Pushable / collectible / autonomous effects extend the same pattern (a learned outcome per identity, revised).
Pure stdlib; the identity comes from L2/3's object recognition (`column.recognize_object`).
"""

from __future__ import annotations


def contact_outcome(prev_pos, agent_pos, attempted_delta, object_cells):
    """Attribute a move: which object (identity) the agent moved INTO, and whether it ADVANCED. The agent intended to
    go from `prev_pos` to target = prev_pos + attempted_delta; `object_cells` = {identity: cell-set}. Returns
    (identity, advanced) for the object occupying the target, or None if the target held no object (empty space /
    off-board -- no interaction). advanced = the agent actually reached the target (it PASSED onto the object); else
    it was BLOCKED (the bump). This is the prediction-error the column owns, turned into one labelled observation."""
    tx, ty = round(prev_pos[0] + attempted_delta[0]), round(prev_pos[1] + attempted_delta[1])
    hit = next((ident for ident, cells in object_cells.items() if (tx, ty) in cells), None)
    if hit is None:
        return None
    advanced = (round(agent_pos[0]), round(agent_pos[1])) == (tx, ty)
    return hit, advanced


class ObjectBehaviour:
    """Per RECOGNISED identity, the learned outcome of the agent moving INTO it -- revisable, no assumption, no KINDS.

    `blocks(identity)` is a running estimate in [0,1] of 'moving into it leaves the agent in place'. Default 0.5 =
    UNKNOWN, so an un-probed object is NOT assumed a barrier (the agent will try it). Each `observe_move` nudges the
    estimate by EWMA toward the observed outcome, so a contradiction (a 'barrier' that turns out to pass) REVISES it."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.block: dict = {}                                # identity -> P(blocked) in [0,1]

    def observe_move(self, identity, advanced: bool) -> None:
        """One interaction: the agent moved into `identity` and ADVANCED (passed) or not (blocked). EWMA toward the
        outcome -- so repeated contradiction decays a stale belief (revision)."""
        prior = self.block.get(identity, 0.5)               # unknown -> 0.5 (no assumption either way)
        self.block[identity] = (1.0 - self.alpha) * prior + self.alpha * (0.0 if advanced else 1.0)

    def blocks(self, identity) -> float:
        """The learned probability that moving into `identity` is blocked (0.5 = unknown -> the agent will try it)."""
        return self.block.get(identity, 0.5)

    def is_barrier(self, identity, thresh: float = 0.5) -> bool:
        """A learned, REVISABLE barrier (above threshold) -- predicted for a recognised instance WITHOUT re-bumping it.
        This is the reachability cost the planner reads: route around a predicted barrier instead of walking into it."""
        return self.blocks(identity) > thresh
