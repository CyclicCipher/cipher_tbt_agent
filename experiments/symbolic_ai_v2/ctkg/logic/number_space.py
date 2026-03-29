"""
Number space — spatial navigation for multi-token sequences.

Each observed multi-token sequence (like [9, 9] for 99) is a point
in a low-dimensional space. The successor operation is a displacement
vector between adjacent points. The topology of the space (wrapping,
carry) emerges from the data via force-directed positioning.

Three operations:
1. **Encode**: map a token sequence to its point in the space.
2. **Navigate**: move from a point by applying a displacement (successor).
3. **Decode**: find the token sequence nearest to a destination point.

Error correction: when the system predicts wrong, weaken forces along
the path taken and strengthen forces toward the correct destination.
This deforms the space so correct paths become easier.

The space is shaped by counting data (0, 1, 2, ..., 99). Each step
in counting is one attractive force between adjacent number-points.
The periodicity of the units digit (wrapping every 10) and the
incrementing of the tens digit (carry) create the toroidal topology
automatically — no carry rule is coded, it's a property of the
geometry.
"""
from __future__ import annotations

import math
import random as _random
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Number point: a multi-token sequence as a single entity
# ---------------------------------------------------------------------------

# A number is identified by its token tuple, e.g., (nid_9, nid_9) for "99".
# We store positions for these tuples in a separate space from the
# single-token KnowledgeGraph embedding.

NumberKey = tuple[NodeId, ...]  # e.g., (nid_for_9, nid_for_9)


class NumberSpace:
    """A metric space of multi-token sequences.

    Each point is a NumberKey (tuple of NodeIds). Points have positions
    in R^d. Forces between points shape the geometry.
    """

    def __init__(self, n_dims: int = 2, seed: int = 42):
        self.n_dims = n_dims
        self._rng = _random.Random(seed)
        # Positions: NumberKey → list[float]
        self.positions: dict[NumberKey, list[float]] = {}
        # Successor pairs: (from_key, to_key) — for displacement computation
        self.successor_pairs: list[tuple[NumberKey, NumberKey]] = []
        # Error history: (predicted_key, correct_key, context_key) — for correction
        self.errors: list[tuple[NumberKey, NumberKey, NumberKey]] = []

    def _ensure_point(self, key: NumberKey) -> list[float]:
        """Ensure a point exists for this key. Initialize randomly if new."""
        if key not in self.positions:
            self.positions[key] = [
                self._rng.gauss(0, 0.3) for _ in range(self.n_dims)
            ]
        return self.positions[key]

    # -------------------------------------------------------------------
    # Learn from counting: record successor pairs
    # -------------------------------------------------------------------

    def observe_successor(self, from_key: NumberKey, to_key: NumberKey) -> None:
        """Record that to_key is the successor of from_key.

        This creates an attractive force: from and to should be adjacent
        in the space, separated by exactly the successor displacement.
        """
        self._ensure_point(from_key)
        self._ensure_point(to_key)
        pair = (from_key, to_key)
        if pair not in self.successor_pairs:
            self.successor_pairs.append(pair)

    # -------------------------------------------------------------------
    # Error correction: backprop through the space
    # -------------------------------------------------------------------

    def record_error(
        self,
        query_key: NumberKey,
        predicted_key: NumberKey,
        correct_key: NumberKey,
    ) -> None:
        """Record a prediction error for space deformation.

        The system navigated from query to predicted, but the answer was
        correct. Weaken the path to predicted, strengthen toward correct.
        """
        self._ensure_point(predicted_key)
        self._ensure_point(correct_key)
        self.errors.append((predicted_key, correct_key, query_key))

    # -------------------------------------------------------------------
    # Settle: run force simulation to reshape the space
    # -------------------------------------------------------------------

    def settle(self, iterations: int = 300, learning_rate: float = 0.05) -> dict[str, Any]:
        """Run force simulation to position all points.

        Forces:
        1. Successor attraction: adjacent numbers should be one step apart.
           The direction of "one step" is consistent across ALL pairs.
        2. Global repulsion: all points push apart slightly to prevent collapse.
        3. Error correction: push predicted away from query along the path
           that was taken; pull correct closer.

        The successor displacement vector is NOT predefined. It emerges
        from the force equilibrium. All successor pairs want to be the
        SAME displacement apart — this consensus defines the direction.
        """
        if len(self.positions) < 2:
            return {"settled": False, "points": len(self.positions)}

        keys = list(self.positions.keys())
        n = len(keys)

        for iteration in range(iterations):
            damping = learning_rate / (1.0 + iteration * 0.01)

            # --- Compute consensus successor displacement ---
            # Average displacement across all known successor pairs.
            if self.successor_pairs:
                avg_disp = [0.0] * self.n_dims
                count = 0
                for from_k, to_k in self.successor_pairs:
                    pf = self.positions[from_k]
                    pt = self.positions[to_k]
                    for d in range(self.n_dims):
                        avg_disp[d] += pt[d] - pf[d]
                    count += 1
                if count > 0:
                    for d in range(self.n_dims):
                        avg_disp[d] /= count
            else:
                avg_disp = [1.0] + [0.0] * (self.n_dims - 1)

            # --- Force 1: Successor attraction ---
            # Each pair wants: to_pos = from_pos + avg_disp.
            for from_k, to_k in self.successor_pairs:
                pf = self.positions[from_k]
                pt = self.positions[to_k]
                for d in range(self.n_dims):
                    target = pf[d] + avg_disp[d]
                    error = target - pt[d]
                    # Pull to_k toward the target.
                    pt[d] += error * damping
                    # Push from_k slightly in the opposite direction.
                    pf[d] -= error * damping * 0.3

            # --- Force 2: Global repulsion ---
            # Sample random pairs to keep O(n) per iteration.
            n_repel = min(n * 2, n * (n - 1) // 2)
            for _ in range(n_repel):
                i = self._rng.randint(0, n - 1)
                j = self._rng.randint(0, n - 2)
                if j >= i:
                    j += 1
                pa = self.positions[keys[i]]
                pb = self.positions[keys[j]]
                dist_sq = sum((pa[d] - pb[d]) ** 2 for d in range(self.n_dims))
                dist = math.sqrt(dist_sq) + 0.001
                repel = 0.001 / (dist * dist)
                for d in range(self.n_dims):
                    direction = (pa[d] - pb[d]) / dist
                    pa[d] += direction * repel * damping
                    pb[d] -= direction * repel * damping

            # --- Force 3: Error correction ---
            for predicted_k, correct_k, query_k in self.errors:
                pq = self.positions.get(query_k)
                pp = self.positions.get(predicted_k)
                pc = self.positions.get(correct_k)
                if pq is None or pp is None or pc is None:
                    continue
                for d in range(self.n_dims):
                    # Push predicted AWAY from where it was (wrong path).
                    pp[d] += (pp[d] - pq[d]) * damping * 0.5
                    # Pull correct TOWARD where it should be
                    # (one step from query along avg_disp).
                    target = pq[d] + avg_disp[d]
                    pc[d] += (target - pc[d]) * damping

        # Compute final successor displacement.
        final_disp = [0.0] * self.n_dims
        if self.successor_pairs:
            for from_k, to_k in self.successor_pairs:
                pf = self.positions[from_k]
                pt = self.positions[to_k]
                for d in range(self.n_dims):
                    final_disp[d] += pt[d] - pf[d]
            for d in range(self.n_dims):
                final_disp[d] /= len(self.successor_pairs)

        self._successor_displacement = final_disp

        return {
            "settled": True,
            "points": len(self.positions),
            "pairs": len(self.successor_pairs),
            "errors_applied": len(self.errors),
            "displacement_magnitude": math.sqrt(
                sum(v * v for v in final_disp)
            ),
        }

    # -------------------------------------------------------------------
    # Encode: token sequence → point
    # -------------------------------------------------------------------

    def encode(self, token_nids: list[NodeId]) -> NumberKey:
        """Convert a token sequence to a NumberKey."""
        return tuple(token_nids)

    def get_position(self, key: NumberKey) -> list[float] | None:
        """Get the position of a number in the space."""
        return self.positions.get(key)

    # -------------------------------------------------------------------
    # Navigate: move from a point by the successor displacement
    # -------------------------------------------------------------------

    def successor_position(self, key: NumberKey) -> list[float] | None:
        """Compute where the successor of this number SHOULD be.

        Returns the predicted position: current_pos + displacement.
        """
        pos = self.positions.get(key)
        if pos is None or not hasattr(self, '_successor_displacement'):
            return None
        disp = self._successor_displacement
        return [pos[d] + disp[d] for d in range(self.n_dims)]

    # -------------------------------------------------------------------
    # Decode: find the nearest known number to a position
    # -------------------------------------------------------------------

    def nearest_number(
        self,
        target_pos: list[float],
        exclude: NumberKey | None = None,
    ) -> NumberKey | None:
        """Find the NumberKey whose position is closest to target_pos."""
        best_key = None
        best_dist = float('inf')
        for key, pos in self.positions.items():
            if key == exclude:
                continue
            dist = math.sqrt(
                sum((target_pos[d] - pos[d]) ** 2 for d in range(self.n_dims))
            )
            if dist < best_dist:
                best_dist = dist
                best_key = key
        return best_key

    def predict_successor(self, key: NumberKey) -> NumberKey | None:
        """Predict the successor by navigating + decoding.

        Encode → navigate (add displacement) → decode (nearest neighbor).
        """
        target = self.successor_position(key)
        if target is None:
            return None
        return self.nearest_number(target, exclude=key)

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def displacement(self) -> list[float]:
        """Return the learned successor displacement vector."""
        if hasattr(self, '_successor_displacement'):
            return list(self._successor_displacement)
        return [0.0] * self.n_dims
