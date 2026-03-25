# Optimization Research — CTKG Performance

Research on algorithms and techniques for optimizing the CTKG graph
operations: spread, learn, edge iteration, and consolidation.

---

## Current Bottlenecks

The CTKG graph operations are O(E) per spread/learn call, where E is the
total number of edges. The main bottleneck is in `spread()` and `learn()`,
which iterate over ALL edges in the graph to find relevant ones. With the
math classroom running 1000+ observe/act cycles, each touching ~20 tokens
with ~50 edges per token, this becomes expensive.

Specific hot paths:
1. `spread()`: iterates `self._edges.items()` to group edges by source
2. `learn()`: iterates `self._edges.items()` to find contributing sources
3. `edges_from()`: iterates ALL edges to find those with matching source
4. `select_action()`: calls `transition_distribution()` and `edges_from()`
   for EACH candidate action

---

## Adjacency List / Outgoing Edge Index

### The Fix

Replace the flat `dict[(src, tgt), Edge]` with an adjacency list index:
`dict[NodeId, list[Edge]]` for outgoing edges. This changes `edges_from()`
from O(E) to O(degree(node)), and `spread()` from O(E) to O(active * avg_degree).

### Implementation

```python
# Current: flat edge dict, O(E) iteration
self._edges: dict[tuple[NodeId, NodeId], Edge] = {}

# Better: adjacency list + edge dict
self._edges: dict[tuple[NodeId, NodeId], Edge] = {}
self._outgoing: dict[NodeId, list[Edge]] = defaultdict(list)
```

On edge creation: `self._outgoing[src].append(edge)`.
On `edges_from(nid)`: `return self._outgoing.get(nid, [])`.
On `spread()`: iterate only over active sources' outgoing edges.

### Sources

- [Unlocking Graph Algorithm Efficiency with Adjacency List](https://www.numberanalytics.com/blog/adjacency-list-implementation-and-optimization) — adjacency lists are the default for sparse graphs, O(V+E) traversal
- [Adjacency list - Wikipedia](https://en.wikipedia.org/wiki/Adjacency_list) — comparison of representations, space/time tradeoffs
- [Python defaultdict for graph adjacency](https://dev.to/nithinbharathwaj/7-powerful-python-performance-optimization-techniques-for-faster-code-51ii) — defaultdict(list) is the standard Python pattern

---

## Vectorized Spreading Activation

### The Opportunity

Spreading activation on a sparse graph is structurally identical to sparse
matrix-vector multiplication: `v_next = A @ v_current` where A is the
adjacency matrix and v is the activation vector. This can be vectorized
with NumPy/SciPy sparse matrices for 10-100x speedup over pure Python loops.

### Trade-offs

- Requires converting the dynamic graph to a static sparse matrix format
  (CSR) periodically, which is expensive if the graph changes every step.
- Works best for read-heavy phases (consolidation replay) where the graph
  is stable.
- Pure Python dict iteration may be faster for small graphs (<1000 edges)
  due to overhead of sparse matrix construction.

### Sources

- [Vectorizing Graph Algorithms (Modern Descartes)](https://www.moderndescartes.com/essays/vectorized_pagerank/) — vectorizing PageRank (analogous to spreading activation) in Python. Achieved 60x speedup, with another 5x possible via Numba JIT. Key insight: adjacency lists have ragged shapes that don't vectorize well; need to reshape into rectangular form.
- [SciPy Compressed Sparse Graph Routines](https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html) — industry-standard sparse graph operations including shortest path, connected components, minimum spanning tree
- [Accelerating Sparse GNNs with Tensor Core Optimization (arXiv:2412.12218, 2024)](https://arxiv.org/html/2412.12218v2) — sparse GNN acceleration on GPU, but the aggregation pattern (scatter/gather on neighbor lists) is identical to spreading activation

---

## Hash Map / Dict Optimization

### Python Dict Internals

Python dicts are hash tables with O(1) average lookup. Key performance
considerations:
- Hash collisions degrade to O(n) linked list traversal per bucket
- Rehashing occurs when load factor exceeds ~2/3, causing O(n) spike
- Iterating `dict.items()` is O(n) but has good cache locality in CPython 3.7+
  (dicts maintain insertion order via a compact array)

### Recommendations for the CTKG

- Use `dict[NodeId, list[Edge]]` (adjacency list) instead of iterating
  `dict[(src, tgt), Edge].items()` — avoids scanning irrelevant edges
- Use `dict[NodeId, Node]` for node lookup (already done)
- Consider `__slots__` on Node and Edge dataclasses to reduce memory per
  object (~40% reduction)
- Profile with cProfile before optimizing — the bottleneck may not be
  where expected

### Sources

- [Python Performance Optimization: Profiling and Speedup](https://dasroot.net/posts/2026/03/python-performance-optimization-profiling-speedup/) — cProfile workflow, identifying real bottlenecks
- [Python Performance Tips (Wiki)](https://wiki.python.org/moin/PythonSpeed/PerformanceTips) — canonical reference for Python optimization
- [Comprehensive C++ HashMap Benchmarks 2022](https://martin.ankerl.com/2022/08/27/hashmap-bench-01/) — not Python, but useful context: `tsl::sparse_map` for small objects, memory-optimized alternatives

---

## Consolidation-Specific Optimization

### Replay Performance

Replay iterates over all snapshot pairs and all outgoing edges from each
active node. For 200 snapshots with 50 active nodes each, that's
200 * 50 * avg_degree edge operations per replay pass.

Optimizations:
- **Batched replay**: Instead of replaying one pair at a time, batch all
  snapshot pairs and do one pass over edges. For each edge, check all
  snapshot pairs where the source was active — amortizes the edge iteration.
- **Selective replay**: Only replay snapshot pairs that contributed to
  high-surprise observations. Low-surprise pairs are already well-learned.
- **Incremental replay**: Track which edges were created/modified since
  last consolidation. Only replay snapshots touching those edges.

### Colimit Performance

Finding co-activation cliques is O(S * N^2) where S = snapshots and
N = max active nodes per snapshot. For large graphs this is expensive.

Optimizations:
- **Streaming co-activation counts**: Maintain a running counter of
  pairwise co-activations instead of scanning all snapshots each time.
- **Locality-sensitive hashing**: Use MinHash to find candidate cliques
  without full pairwise comparison.

### Sources

- [MapReduce Spreading Activation](https://www.researchgate.net/publication/258107439_A_MapReduce_Implementation_of_the_Spreading_Activation_Algorithm_for_Processing_Large_Knowledge_Bases_Based_on_Semantic_Networks) — scalable spreading activation on large knowledge bases using MapReduce. Relevant for future parallelization.
- [Fast Training of Sparse GNNs on Dense Hardware (arXiv:1906.11786)](https://arxiv.org/pdf/1906.11786) — batching and caching strategies for sparse graph operations

---

## Immediate Priority: Adjacency Index

The highest-impact, lowest-risk optimization is adding an outgoing edge
index (`_outgoing: dict[NodeId, list[Edge]]`). This:
- Eliminates all O(E) scans in `spread()`, `learn()`, `edges_from()`
- Requires minimal code changes (add index on edge creation, use in queries)
- No algorithmic complexity change — just better constant factors
- Expected 5-20x speedup for graphs with >500 edges

This should be implemented before any other optimization.
