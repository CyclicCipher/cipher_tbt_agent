"""
Force-directed embedding — structure from position and feedback.

Every token is just a token. No roles, no labels, no operand/operator
distinction. Structure emerges from two signals only:

1. **Position**: tokens close together in an observation exert stronger
   forces on each other than tokens far apart. Force ∝ 1/distance.

2. **Feedback**: when the system acts and receives feedback, the
   (context_token, action_token) pair gets an attractive force if
   CORRECT, repulsive if WRONG. The strength depends on how close
   the context token was to the action position.

The equilibrium of these forces IS the embedding. Tokens that
consistently appear near correct actions cluster together. Tokens
that appear near wrong actions are pushed apart.

One example per unique pair. Redundant data biases the forces.
"""
from __future__ import annotations

import math
import random as _random
from collections import defaultdict
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Extract feedback-signed forces from observation history
# ---------------------------------------------------------------------------

def _extract_forces(
    hippo: Hippocampus,
    kg: KnowledgeGraph,
    since_index: int = 0,
) -> list[tuple[NodeId, NodeId, float, float]]:
    """Extract (token_a, token_b, sign, distance_weight) from observations.

    Scans observation triplets: context → action → feedback.
    For each context token, creates a force to the action token:
    - sign = +1 if feedback contains a "preferred" token (correct)
    - sign = -1 if feedback contains a "dispreferred" token (wrong)
    - distance_weight = 1 / positional_distance (closer = stronger)

    Also: within each observation, adjacent tokens attract each other
    (co-occurrence force, always positive, distance-weighted).

    Returns list of (nid_a, nid_b, sign, weight).
    Each unique (a, b, sign) appears at most once.
    """
    obs_list = hippo.all_observations()
    recent = obs_list[max(0, since_index):]

    preferred = kg.preferred_nodes()  # {nid: level}
    preferred_pos = {nid for nid, lv in preferred.items() if lv > 0}
    preferred_neg = {nid for nid, lv in preferred.items() if lv < 0}

    forces: list[tuple[NodeId, NodeId, float, float]] = []
    seen: set[tuple[NodeId, NodeId, int]] = set()  # dedup key: (a, b, sign_bucket)

    # --- Feedback forces: context → action, signed by feedback ---
    # Pattern: obs[i] = context, obs[i+1] = action (1 token), obs[i+2] = feedback
    for i in range(len(recent) - 2):
        ctx = recent[i]
        act = recent[i + 1]
        fb = recent[i + 2]

        if len(act.token_nids) != 1:
            continue
        action_nid = act.token_nids[0]

        # Determine feedback sign from preferred tokens in feedback obs.
        fb_set = set(fb.token_nids)
        is_correct = bool(fb_set & preferred_pos)
        is_wrong = bool(fb_set & preferred_neg)

        if not is_correct and not is_wrong:
            continue

        sign = 1.0 if is_correct else -1.0

        # Create force from each context token to the action token,
        # weighted by 1/distance from the action position.
        action_pos = len(ctx.token_nids)  # action would be at the next position
        for pos, ctx_nid in enumerate(ctx.token_nids):
            if ctx_nid == action_nid:
                continue
            dist = action_pos - pos
            if dist < 1:
                dist = 1
            weight = 1.0 / dist

            sign_bucket = 1 if sign > 0 else -1
            key = (ctx_nid, action_nid, sign_bucket)
            if key not in seen:
                seen.add(key)
                forces.append((ctx_nid, action_nid, sign, weight))

    # --- Co-occurrence forces: adjacent tokens within observations ---
    for obs in recent:
        nids = obs.token_nids
        for j in range(len(nids) - 1):
            a, b = nids[j], nids[j + 1]
            if a == b:
                continue
            key = (a, b, 1)  # co-occurrence is always attractive
            if key not in seen:
                seen.add(key)
                forces.append((a, b, 1.0, 0.5))  # moderate attraction

    return forces


# ---------------------------------------------------------------------------
# Force simulation
# ---------------------------------------------------------------------------

def compute_force_embedding(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    n_dims: int = 3,
    iterations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute force-directed embedding from position and feedback.

    Low-dimensional (default 3D). Forces:
    - Feedback-signed: attract (correct) or repel (wrong) between
      context tokens and action tokens, weighted by 1/distance.
    - Co-occurrence: adjacent tokens attract weakly.
    - Global repulsion: all tokens repel each other slightly to
      prevent collapse to a single point.

    Stores result as kg._force_embeddings: {NodeId: list[float]}.
    """
    forces = _extract_forces(hippo, kg, since_index)

    if not forces:
        kg._force_embeddings = {}
        kg._force_operators = []
        return {"force_pairs": 0, "tokens_embedded": 0, "dims": 0}

    # Collect all tokens involved.
    all_tokens: set[NodeId] = set()
    for a, b, s, w in forces:
        all_tokens.add(a)
        all_tokens.add(b)

    if len(all_tokens) < 2:
        kg._force_embeddings = {}
        kg._force_operators = []
        return {"force_pairs": len(forces), "tokens_embedded": 0, "dims": 0}

    # Initialize random positions.
    rng = _random.Random(seed)
    positions: dict[NodeId, list[float]] = {
        nid: [rng.gauss(0, 0.5) for _ in range(n_dims)]
        for nid in all_tokens
    }

    token_list = sorted(all_tokens)
    n_tokens = len(token_list)
    REPEL_STRENGTH = 0.01  # global repulsion

    for iteration in range(iterations):
        damping = 1.0 / (1.0 + iteration * 0.02)

        # Accumulate forces.
        f: dict[NodeId, list[float]] = {
            nid: [0.0] * n_dims for nid in all_tokens
        }

        # Feedback + co-occurrence forces.
        for a, b, sign, weight in forces:
            pa = positions[a]
            pb = positions[b]
            for d in range(n_dims):
                diff = pb[d] - pa[d]
                # Attractive (sign > 0): pull a toward b.
                # Repulsive (sign < 0): push a away from b.
                force = sign * weight * diff * 0.1
                f[a][d] += force * damping
                f[b][d] -= force * damping

        # Global repulsion: all pairs push apart slightly.
        # Use a random subset to keep O(n) not O(n²).
        n_repel = min(n_tokens * 3, n_tokens * (n_tokens - 1) // 2)
        for _ in range(n_repel):
            i = rng.randint(0, n_tokens - 1)
            j = rng.randint(0, n_tokens - 2)
            if j >= i:
                j += 1
            a = token_list[i]
            b = token_list[j]
            pa = positions[a]
            pb = positions[b]
            dist_sq = sum((pa[d] - pb[d]) ** 2 for d in range(n_dims))
            dist = math.sqrt(dist_sq) + 0.01
            for d in range(n_dims):
                direction = (pa[d] - pb[d]) / dist
                repel = REPEL_STRENGTH / (dist * dist) * damping
                f[a][d] += direction * repel
                f[b][d] -= direction * repel

        # Apply forces.
        for nid in all_tokens:
            p = positions[nid]
            for d in range(n_dims):
                p[d] += f[nid][d]

    kg._force_embeddings = positions
    kg._force_operators = []  # no operator dimensions — just a metric space

    return {
        "force_pairs": len(forces),
        "tokens_embedded": len(positions),
        "dims": n_dims,
    }


# ---------------------------------------------------------------------------
# Navigate: find nearest token in the embedding
# ---------------------------------------------------------------------------

def force_navigate(
    kg: KnowledgeGraph,
    query_nid: NodeId,
    operator_nid: NodeId,
    candidates: list[NodeId],
) -> NodeId | None:
    """Find the candidate nearest to the query in force embedding space,
    excluding the query itself.

    The operator_nid is currently unused — the force embedding doesn't
    have per-operator dimensions. Navigation is purely by proximity.
    This will change when operator-specific structure is discovered.
    """
    if not kg._force_embeddings:
        return None

    eq = kg._force_embeddings.get(query_nid)
    if eq is None:
        return None

    best_nid = None
    best_dist = float('inf')
    for cand in candidates:
        if cand == query_nid:
            continue
        ec = kg._force_embeddings.get(cand)
        if ec is None:
            continue
        dist = math.sqrt(sum((eq[d] - ec[d]) ** 2 for d in range(len(eq))))
        if dist < best_dist:
            best_dist = dist
            best_nid = cand

    return best_nid
