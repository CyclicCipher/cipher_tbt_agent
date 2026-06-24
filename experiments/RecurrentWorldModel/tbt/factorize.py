"""Disentanglement — discover the factorization of an entangled stream into independent factors.

Higgins et al. 2018 ("Towards a Definition of Disentangled Representations"): a representation is disentangled
w.r.t. a group decomposition G = G1 x G2 x ... when the space splits so each subgroup Gi acts on its own
subspace and trivially on the others. You CANNOT read that group off static observations (Locatello 2019:
unsupervised disentanglement is impossible without an inductive bias) — only off how states TRANSFORM under
ACTION. So we read it off the agent's own actions, which is exactly what a sensorimotor column has:

  - each action generates a cyclic subgroup whose ORBITS partition the states;
  - actions whose orbit-partitions coincide (an action and its inverse, or two actions of the same factor)
    generate the SAME factor;
  - two factors are INDEPENDENT (a direct product) when their partitions are TRANSVERSE — every pair of
    classes meets in exactly one state and the class-counts multiply to |S|.

A factor's COORDINATE of a state = its class under all the OTHER factors' partitions: moving along a factor
changes that coordinate and leaves the rest fixed — the Higgins condition, made operational. Each factor then
becomes its own column over those coordinate-classes; the thalamus binds the per-factor codes into the joint
state. Pure stdlib (operates on the observed transition graph: {state: {action: next_state}})."""

from __future__ import annotations


def orbit_partition(graph, action):
    """state -> orbit id under repeated `action` (the cosets of the cyclic subgroup <action>)."""
    part, oid = {}, 0
    for s in graph:
        if s in part:
            continue
        cur = s
        while cur not in part:
            part[cur] = oid
            cur = graph.get(cur, {}).get(action, cur)        # apply the action (self-loop if undefined)
        oid += 1
    return part


def _same_grouping(p, q):
    """True if partitions p and q induce the SAME grouping of states (their orbit sets coincide)."""
    fwd, bwd = {}, {}
    for s in p:
        a, b = p[s], q[s]
        if fwd.setdefault(a, b) != b or bwd.setdefault(b, a) != a:
            return False
    return True


def discover_factors(graph, actions):
    """Cluster actions by orbit-partition into factors; give each state its per-factor coordinate.

    Returns (factors, is_product). Each factor = {'actions': [...], 'coord': {state: int}, 'n': int}.
    is_product is True iff the factors form a direct product (transverse partitions, sizes multiply to |S|)."""
    parts = {a: orbit_partition(graph, a) for a in actions}
    groups = []                                              # [{'actions': [...], 'part': {...}}]
    for a in actions:
        for g in groups:
            if _same_grouping(parts[a], g["part"]):
                g["actions"].append(a)
                break
        else:
            groups.append({"actions": [a], "part": parts[a]})
    states = list(graph)
    factors = []
    for i, g in enumerate(groups):
        others = [h["part"] for j, h in enumerate(groups) if j != i]   # this factor's coord = class under the OTHERS
        raw = {s: tuple(p[s] for p in others) for s in states}
        ids = {c: k for k, c in enumerate(sorted(set(raw.values())))}
        factors.append({"actions": g["actions"], "coord": {s: ids[raw[s]] for s in states}, "n": len(ids)})
    joint = {tuple(f["coord"][s] for f in factors) for s in states}     # direct product?  bijection + sizes multiply
    size = 1
    for f in factors:
        size *= f["n"]
    return factors, (len(joint) == len(states) == size)


def cyclic_coords(graph, succ, factor):
    """Coordinates from RAW transitions for a CYCLIC (place-value-like) COUPLED structure — the disentangle →
    residual bridge for the coupled case. The `factor` action's orbit COUNT is the base b; the `succ` action's
    global cycle gives each state a position p; the factored coordinates are (p mod b, p // b). (The clean
    direct-product case gets its coordinates straight from discover_factors' per-factor `coord`.) Returns
    (coords, b)."""
    b = len(set(orbit_partition(graph, factor).values()))
    order, s, p = {}, next(iter(graph)), 0
    while s not in order:                                    # follow succ: it generates the whole cycle
        order[s] = p
        p += 1
        s = graph[s].get(succ, s)
    return {s: (order[s] % b, order[s] // b) for s in graph}, b


def factor_graph(graph, factor):
    """The transition graph of ONE factor over its coordinate-classes — what a column learns to model it.

    An edge appears wherever one of the factor's actions MOVES its coordinate (leaving the other factors
    fixed); the result is the factor's own small structure (a ring / line of `factor['n']` classes)."""
    coord, g = factor["coord"], {}
    for s in graph:
        for a in factor["actions"]:
            s2 = graph[s].get(a)
            if s2 is not None and coord[s2] != coord[s]:
                g.setdefault(coord[s], {})[a] = coord[s2]
    return g
