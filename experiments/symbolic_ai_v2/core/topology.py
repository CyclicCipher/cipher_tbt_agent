"""Graph topology: edge type registry and built-in topologies.

A Topology converts raw input data into a stream of observations.  Two formats:

  stream_tokens(data) → Iterator[(value, Optional[int])]
      Sequential (1D) data.  First element has edge_type_int = None.
      Fed one token at a time to MorphismGraph.observe().

  stream_edges(data) → Iterator[(src_value, edge_type_int, tgt_value)]
      2D+ data where a sequential scan would introduce raster artefacts at
      dimension boundaries.  Fed to MorphismGraph.observe_edge() [future].

Edge types are small integers inside the MorphismGraph; human-readable names
are registered once at Topology creation.  This removes string hashing from
the inner observation loop.  Max 256 types (u8-compatible for Rust rewrite).

Worst perceptual case: 3D 26-connected + time = 28 edge types (5 bits).
All built-in topologies fit comfortably within uint8.
See DATA_FORMATS.md for the full topology table and bit-layout analysis.
"""

from __future__ import annotations
from typing import Any, Iterator, Optional


# ── Edge type registry ────────────────────────────────────────────────────────

class EdgeTypeRegistry:
    """Maps human-readable edge type names to small integers.

    Names are registered once when the Topology is created.  Afterwards all
    internal code uses integer codes.  The registry is serialised as part of
    the checkpoint JSON header so checkpoints are self-describing.

    Maximum 256 types (fits in uint8 for the Rust rewrite).  The worst
    practical case — 3D 26-connected + time — uses 28 types.  If a future
    use case requires more than 256 types (e.g. raw Wikidata with ~10 000
    property types), apply a CTKG-level functor first to compress them;
    see DATA_FORMATS.md §"Known limits and upgrade paths".
    """

    def __init__(self) -> None:
        self._name_to_int: dict[str, int] = {}
        self._int_to_name: list[str] = []

    def register(self, name: str) -> int:
        """Register a name and return its integer code.  Idempotent."""
        if name in self._name_to_int:
            return self._name_to_int[name]
        if len(self._int_to_name) >= 256:
            raise ValueError(
                f"Edge type limit (256) exceeded registering '{name}'. "
                "See DATA_FORMATS.md for guidance."
            )
        code = len(self._int_to_name)
        self._name_to_int[name] = code
        self._int_to_name.append(name)
        return code

    def code(self, name: str) -> int:
        """O(1) lookup of integer code for a registered name."""
        try:
            return self._name_to_int[name]
        except KeyError:
            raise KeyError(
                f"Edge type '{name}' not registered. "
                f"Known types: {self._int_to_name}"
            ) from None

    def name(self, code: int) -> str:
        """O(1) lookup of name for an integer code."""
        return self._int_to_name[code]

    def names(self) -> list[str]:
        return list(self._int_to_name)

    def __len__(self) -> int:
        return len(self._int_to_name)

    def __repr__(self) -> str:
        return f"EdgeTypeRegistry({self._int_to_name})"


# ── Abstract topology ─────────────────────────────────────────────────────────

class Topology:
    """Abstract base for input topologies.

    A Topology has a name, an EdgeTypeRegistry, and either stream_tokens()
    (for 1D sequential data) or stream_edges() (for 2D+ data).

    Built-in factories: sequence_1d(), cycle_1d(), grid_2d(), grid_3d().
    """

    def __init__(self, name: str, registry: EdgeTypeRegistry) -> None:
        self.name = name
        self.registry = registry

    def stream_tokens(self, data: Any) -> Iterator[tuple[Any, Optional[int]]]:
        """Yield (value, edge_type_int) for each element in sequential data.

        The first element always yields edge_type_int = None (no incoming edge).
        Subsequent elements yield the integer code of the edge from the previous
        observation to this one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stream_tokens(). "
            "Use stream_edges() for non-sequential topologies."
        )

    def stream_edges(self, data: Any) -> Iterator[tuple[Any, int, Any]]:
        """Yield (src_value, edge_type_int, tgt_value) for all edges.

        For 2D+ topologies.  All directed edges are emitted (both directions
        for each undirected spatial pair).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stream_edges()."
        )

    def n_edge_types(self) -> int:
        return len(self.registry)

    def __repr__(self) -> str:
        return f"Topology({self.name!r}, edge_types={self.registry.names()})"


# ── Built-in topologies ───────────────────────────────────────────────────────

def sequence_1d() -> Topology:
    """1D path graph: 'next' (0) and 'prev' (1) edge types.

    For text, arithmetic sequences, audio samples, or any ordered 1D data.
    stream_tokens() yields (value, 0) for each element after the first.
    The 'prev' type exists in the registry for back-reference queries but
    is not emitted during forward streaming.
    """
    reg = EdgeTypeRegistry()
    reg.register("next")   # code 0
    reg.register("prev")   # code 1

    class _Seq1D(Topology):
        def stream_tokens(self, data: Any) -> Iterator[tuple[Any, Optional[int]]]:
            next_code = self.registry.code("next")
            first = True
            for value in data:
                if first:
                    yield value, None
                    first = False
                else:
                    yield value, next_code

    return _Seq1D("sequence_1d", reg)


def cycle_1d() -> Topology:
    """1D cycle graph: 'next' (0) and 'prev' (1) edge types, last wraps to first.

    stream_tokens() emits the linear part only.  After processing, call
    mg.observe_edge(last_value, registry.code('next'), first_value) to record
    the wrap-around edge.
    """
    reg = EdgeTypeRegistry()
    reg.register("next")
    reg.register("prev")

    class _Cycle1D(Topology):
        def stream_tokens(self, data: Any) -> Iterator[tuple[Any, Optional[int]]]:
            next_code = self.registry.code("next")
            first = True
            for value in data:
                if first:
                    yield value, None
                    first = False
                else:
                    yield value, next_code
            # Wrap-around edge: call mg.observe_edge(last, next_code, first) separately.

    return _Cycle1D("cycle_1d", reg)


def grid_2d(diagonal: bool = False) -> Topology:
    """2D image grid.

    4-connected (default): 'right' (0), 'left' (1), 'down' (2), 'up' (3).
    8-connected (diagonal=True): adds 'dr' (4), 'dl' (5), 'ur' (6), 'ul' (7).

    Uses stream_edges(), not stream_tokens().  Raster-order streaming is
    intentionally absent: the jump from row end to row start is not a grid
    edge and would produce spurious edge pairs.

    data passed to stream_edges() must be a 2D sequence: list[list[value]]
    or any object supporting grid[row][col] and len(grid) / len(grid[0]).
    """
    reg = EdgeTypeRegistry()
    for name in ("right", "left", "down", "up"):
        reg.register(name)
    if diagonal:
        for name in ("dr", "dl", "ur", "ul"):
            reg.register(name)

    _diag = diagonal

    class _Grid2D(Topology):
        def stream_edges(self, grid: Any) -> Iterator[tuple[Any, int, Any]]:
            rows = len(grid)
            if rows == 0:
                return
            cols = len(grid[0])
            r = self.registry
            rc, lc = r.code("right"), r.code("left")
            dc, uc = r.code("down"),  r.code("up")
            for row in range(rows):
                for col in range(cols):
                    v = grid[row][col]
                    if col + 1 < cols:
                        nv = grid[row][col + 1]
                        yield v, rc, nv
                        yield nv, lc, v
                    if row + 1 < rows:
                        bv = grid[row + 1][col]
                        yield v, dc, bv
                        yield bv, uc, v
                    if _diag:
                        drc, dlc = r.code("dr"), r.code("dl")
                        urc, ulc = r.code("ur"), r.code("ul")
                        if row + 1 < rows and col + 1 < cols:
                            dv = grid[row + 1][col + 1]
                            yield v,  drc, dv
                            yield dv, ulc, v
                        if row + 1 < rows and col - 1 >= 0:
                            dv = grid[row + 1][col - 1]
                            yield v,  dlc, dv
                            yield dv, urc, v

    tag = f"grid_2d{'_diagonal' if diagonal else ''}"
    return _Grid2D(tag, reg)


def grid_3d(connectivity: int = 6, time_edges: bool = False) -> Topology:
    """3D volume grid for CAD / volumetric imaging / physics simulations.

    connectivity: 6 (face neighbors only), 18 (face + edge), or 26 (full 3×3×3−1).
    time_edges: if True, adds 'later' (forward) and 'earlier' (backward) edges.

    Edge type counts by configuration — all fit in uint8:
      6-connected:              6 types (3 bits)
      6-connected + time:       8 types (3 bits)
      18-connected:            18 types (5 bits)
      18-connected + time:     20 types (5 bits)
      26-connected:            26 types (5 bits)
      26-connected + time:     28 types (5 bits)  ← worst perceptual case

    stream_edges() is stubbed; implement by analogy with grid_2d.stream_edges()
    when 3D data support is needed.  The edge type registry is fully populated
    so the Topology can be used for checkpoint metadata even before stream_edges
    is implemented.
    """
    if connectivity not in (6, 18, 26):
        raise ValueError(f"connectivity must be 6, 18, or 26; got {connectivity}")
    reg = EdgeTypeRegistry()
    for fwd, rev in (("+x", "-x"), ("+y", "-y"), ("+z", "-z")):
        reg.register(fwd); reg.register(rev)
    if connectivity >= 18:
        for fwd, rev in (
            ("+x+y", "-x-y"), ("+x-y", "-x+y"),
            ("+x+z", "-x-z"), ("+x-z", "-x+z"),
            ("+y+z", "-y-z"), ("+y-z", "-y+z"),
        ):
            reg.register(fwd); reg.register(rev)
    if connectivity >= 26:
        for fwd, rev in (
            ("+x+y+z", "-x-y-z"), ("+x+y-z", "-x-y+z"),
            ("+x-y+z", "-x+y-z"), ("+x-y-z", "-x+y+z"),
        ):
            reg.register(fwd); reg.register(rev)
    if time_edges:
        reg.register("later")
        reg.register("earlier")

    tag = f"grid_3d_{connectivity}conn{'_time' if time_edges else ''}"
    return Topology(tag, reg)   # stream_edges() raises NotImplementedError until filled in


# ── Math topology ─────────────────────────────────────────────────────────────

# Hard-coded vocabulary sets for the math benchmark corpus.
# Each set defines which tokens belong to that syntactic category.
# Unknown tokens default to 'op' (operator).

_MATH_OPS: frozenset[str] = frozenset({
    'add', 'sub', 'mul', 'div', 'pow', 'sq', 'sqrt',
    'd', 'int', 'succ', 'pred',
    'conserve', 'bernoulli', 'ke', 'pe', 'ftc',
    'eval', 'at',
})
_MATH_EQ:  frozenset[str] = frozenset({'eq', 'and'})
_MATH_VAR: frozenset[str] = frozenset({'x', 'C', 'dx', 'half', 'third'})


def math_topology() -> Topology:
    """Topology for mathematical formula sequences.

    Four edge type codes encoding the syntactic role of the TARGET token:

      op  (0) — operator token  (add, mul, d, int, succ, ...)
      num (1) — numeric literal (0..99, decimal strings)
      var (2) — variable / special symbol (x, C, dx, half, third)
      eq  (3) — equality / connective (eq, and)

    FCA on these four edge types discovers meaningful structural types:
      - All numeric literals cluster together (observed on 'num' edges)
      - All operators cluster together (observed on 'op' edges)
      - Type back-off then generalises: "this context expects a number"
        even for numeric values never seen before in this context.

    With sequence_1d (single 'next' etype), FCA degenerates to one global
    group (all atoms get type 'next_group').  math_topology() gives FCA
    four structurally distinct edge types, enabling meaningful adjunctions.
    """
    reg = EdgeTypeRegistry()
    reg.register('op')   # code 0
    reg.register('num')  # code 1
    reg.register('var')  # code 2
    reg.register('eq')   # code 3

    class _MathTopo(Topology):
        def stream_tokens(self, data: Any) -> Iterator[tuple[Any, Optional[int]]]:
            op_e  = self.registry.code('op')
            num_e = self.registry.code('num')
            var_e = self.registry.code('var')
            eq_e  = self.registry.code('eq')
            first = True
            for token in data:
                t = str(token)
                if first:
                    yield t, None
                    first = False
                    continue
                if t in _MATH_OPS:
                    etype = op_e
                elif t in _MATH_EQ:
                    etype = eq_e
                elif t in _MATH_VAR:
                    etype = var_e
                else:
                    try:
                        float(t)
                        etype = num_e
                    except ValueError:
                        etype = op_e   # unknown token → treat as operator
                yield t, etype

    return _MathTopo('math', reg)


def agent_topology() -> Topology:
    """Topology for agent environments with structural intero/extero distinction.

    Three edge type codes:

      extero (0) — sequential exteroceptive observation (world state tokens)
      intero (1) — sequential interoceptive observation (agent-body tokens)
      action (2) — action token: the agent's chosen action, fed by AgentLoop
                   between the current observation and the next one.

    This makes the self/world boundary structural, not a naming convention.
    The MorphismGraph learns separate composition hierarchies for each stream:
      - extero-extero: room contents, object properties, exits
      - intero-intero: hunger, health, inventory
      - action-intero: action → change in agent state  (affordance: what I gain)
      - action-extero: action → change in world state  (affordance: what changes)

    The AgentLoop feeds action tokens automatically, so every Environment that
    uses agent_topology() gets the structural intero/extero/action distinction
    without any per-experiment boilerplate.

    stream_tokens() is NOT used for agent environments; tokens are fed one at
    a time via mg.observe(value, etype) from Environment.observe().
    """
    reg = EdgeTypeRegistry()
    reg.register("extero")   # code 0
    reg.register("intero")   # code 1
    reg.register("action")   # code 2

    return Topology("agent", reg)
