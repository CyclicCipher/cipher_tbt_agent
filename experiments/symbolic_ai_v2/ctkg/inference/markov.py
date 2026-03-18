"""
Markov category operations: d-separation, reachability, morphism influence.

The MorphismGraph is treated as a directed graphical model:
  nodes  = CTKGObjects (discovered concept types)
  edges  = CTKGMorphisms (directed causal relationships)

d-separation
------------
Uses the Bayes-ball algorithm (Shachter 1988).  A path from X to Y is *active*
given conditioning set Z if no node on the path blocks the ball:

  Non-collider V (arrows flow through V, not meeting at V):
    - Ball passes through iff V ∉ Z.

  Collider V (two arrows meet at V: X → V ← Y):
    - Ball passes through iff V ∈ Z OR some descendant of V is in Z.

X and Y are *d-separated* given Z iff no ball can reach Y from X.

Implementation uses directed Bayes-ball traversal with states (node, direction):
  direction='up'   — ball travelling toward ancestors (received from a child)
  direction='down' — ball travelling toward descendants (received from a parent)

Propagation rules:
  (V, 'up'),  V ∉ Z → spread (parent, 'up') for each parent of V,
                       spread (child,  'down') for each child of V.
  (V, 'down'), V ∉ Z → spread (child, 'down') for each child of V.
  (V, 'down'), V ∈ ancestors_of_Z → spread (parent, 'up') for each parent of V
                                     (collider activation).

See CTKG_ARCHITECTURE.md §Markov for the full specification.
"""

from __future__ import annotations

import math
from collections import deque

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph


# ---------------------------------------------------------------------------
# d-separation
# ---------------------------------------------------------------------------

def d_separated(
    mg: MorphismGraph,
    x_id: int,
    y_id: int,
    given_ids: set[int],
) -> bool:
    """Test d-separation of objects x and y given the conditioning set.

    Parameters
    ----------
    mg:
        The MorphismGraph (directed graphical model).
    x_id, y_id:
        Object IDs to test.
    given_ids:
        Set of object IDs in the conditioning set Z.

    Returns
    -------
    True  — x and y are d-separated given Z (no active path exists).
    False — x and y are d-connected given Z (at least one active path exists).
    """
    if x_id == y_id:
        return False  # a variable is always connected to itself

    given = frozenset(given_ids)

    # Compute ancestors of the given set (for collider activation).
    # A collider V is activated iff V ∈ Z or some descendant of V is in Z,
    # equivalently iff V is an ancestor of (or equal to) some node in Z.
    ancestors_of_given = set(given)
    queue: deque[int] = deque(given)
    while queue:
        v = queue.popleft()
        for m in mg.in_morphisms(v, include_identity=False):
            if m.source not in ancestors_of_given:
                ancestors_of_given.add(m.source)
                queue.append(m.source)

    # Bayes-ball traversal
    visited: set[tuple[int, str]] = set()
    bq: deque[tuple[int, str]] = deque()
    # Start from x in both directions
    bq.append((x_id, "up"))
    bq.append((x_id, "down"))

    while bq:
        v, direction = bq.popleft()
        state = (v, direction)
        if state in visited:
            continue
        visited.add(state)

        if v == y_id:
            return False  # y is reachable → NOT d-separated

        if direction == "up" and v not in given:
            # Non-collider going up: spread upward to parents and downward to children
            for m in mg.in_morphisms(v, include_identity=False):
                bq.append((m.source, "up"))
            for m in mg.out_morphisms(v, include_identity=False):
                bq.append((m.target, "down"))

        if direction == "down" and v not in given:
            # Non-collider going down: spread downward to children only
            for m in mg.out_morphisms(v, include_identity=False):
                bq.append((m.target, "down"))

        if direction == "down" and v in ancestors_of_given:
            # Collider activated (v is an ancestor of Z): spread upward to parents
            for m in mg.in_morphisms(v, include_identity=False):
                bq.append((m.source, "up"))

    return True  # y was never reached


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

def reachable(
    mg: MorphismGraph,
    source_id: int,
    max_hops: int = 10,
) -> set[int]:
    """BFS following directed morphisms from source_id.

    Parameters
    ----------
    mg:
        The MorphismGraph.
    source_id:
        Starting object ID.
    max_hops:
        Maximum number of edge traversals.

    Returns
    -------
    Set of all object IDs reachable from source_id (including source_id itself).
    """
    seen: set[int] = {source_id}
    frontier: list[int] = [source_id]

    for _ in range(max_hops):
        next_frontier: list[int] = []
        for v in frontier:
            for m in mg.out_morphisms(v, include_identity=False):
                if m.target not in seen:
                    seen.add(m.target)
                    next_frontier.append(m.target)
        if not next_frontier:
            break
        frontier = next_frontier

    return seen


# ---------------------------------------------------------------------------
# Morphism influence
# ---------------------------------------------------------------------------

def morphism_influence(
    mg: MorphismGraph,
    source_id: int,
    target_id: int,
) -> float:
    """Total causal influence from source_id to target_id.

    Defined as the sum of exp(confidence) over all directed paths from
    source to target in the MorphismGraph.  Returns 0.0 if no path exists.

    A direct morphism A → B contributes exp(confidence(A→B)).
    A two-hop path A → M → B contributes exp(min(conf(A→M), conf(M→B)))
    (weakest-link, consistent with `MorphismGraph.compose`).

    Parameters
    ----------
    mg:
        The MorphismGraph.
    source_id, target_id:
        Object IDs.

    Returns
    -------
    Total influence ∈ [0, ∞).  Returns 0.0 if no directed path exists.
    """
    # BFS over paths; state = (current_node, min_confidence_so_far)
    # Accumulate exp(min_conf) for each path that reaches target.
    total = 0.0
    # (current_node, accumulated_min_conf, visited_nodes_on_path)
    stack: list[tuple[int, float, frozenset[int]]] = [
        (source_id, 0.0, frozenset({source_id}))
    ]

    MAX_PATHS = 1000  # guard against combinatorial explosion
    n_paths_explored = 0

    while stack and n_paths_explored < MAX_PATHS:
        v, min_conf, path_visited = stack.pop()
        n_paths_explored += 1

        for m in mg.out_morphisms(v, include_identity=False):
            w = m.target
            if w in path_visited:
                continue  # no cycles
            edge_conf = m.confidence
            new_min = min(min_conf, edge_conf)
            if w == target_id:
                total += math.exp(new_min)
            else:
                stack.append((w, new_min, path_visited | {w}))

    return total
