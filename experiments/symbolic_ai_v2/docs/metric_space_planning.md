# Planning: Metric Space over Category Structure

## Status: STUB — for future discussion

## The Biological Inspiration

Grid cells in the entorhinal cortex were originally discovered as spatial
navigation cells — they fire in hexagonal patterns as an animal moves through
physical space. The assumption was that they were navigation-specific.

Behrens, Whittington et al. (UCL) showed this was wrong. Grid-cell-like
representations appear whenever subjects navigate ABSTRACT relational spaces:
task structures, social hierarchies, conceptual similarity spaces. The brain
uses the same representational machinery for abstract structure that it uses
for physical space.

Whittington's Tolman-Eichenbaum Machine formalised this: the hippocampal-
entorhinal system implements something close to graph traversal over relational
structure, with grid cells encoding the METRIC of that abstract space.

## The Key Insight

The brain doesn't just store categorical relationships — it builds a METRIC
over category space. Structural distance between two concepts is measured in
this metric. Distant analogies (electromagnetism ↔ gravity) are recognised as
CLOSE in the metric even though they're far apart in surface semantics.

This is significant because:
- Functor discovery requires finding structure-preserving maps between
  subgraphs. If we have a metric, "structurally similar subgraphs" are
  NEARBY in the metric — reducing search from all-pairs to neighborhood lookup.
- The Poincare Principle (consolidation discovering cross-domain isomorphisms)
  becomes tractable: instead of comparing all subgraph pairs, compare only
  subgraphs that are metrically close.
- Grid cell hexagonal tiling provides a MULTI-SCALE representation: coarse
  grids find approximate matches, fine grids refine them. This is exactly
  the ω-category level hierarchy: level-3 functors are coarse matches,
  level-1 morphisms are fine-grained.

## What This Would Look Like in the CTKG

Each node in the KG gets a position in a metric space (not just graph distance
— a learned embedding that captures structural similarity). The metric is
updated during consolidation based on:
- Edge structure similarity (nodes with similar neighborhoods are close)
- Functor relationships (nodes mapped by the same functor are close)
- Co-activation patterns (nodes that consistently co-activate are close)

The metric enables:
- Fast approximate nearest-neighbor lookup for functor discovery
- Multi-scale attention (attend to metrically nearby nodes first)
- Analogy detection (find the closest structural match across domains)

## Open Questions

1. What is the right metric? Graph Laplacian eigenvectors? Learned embeddings
   from spread dynamics? Something derived from the categorical structure?
2. How does the metric interact with the ω-category levels? Is it one metric
   per level, or one metric that spans all levels?
3. How does the hexagonal grid tiling map onto discrete graph structure?
4. What is the RAM cost of storing positions for every node?
5. How does the metric update during learning? Online (fast path) or only
   during consolidation (slow path)?

## References

- Whittington et al. — The Tolman-Eichenbaum Machine
- Behrens et al. — Grid cells for abstract relational structure
- Bronstein et al. — Geometric Deep Learning (metric spaces on graphs)
- Moser & Moser — Nobel Prize work on grid cells and place cells

## Relation to Current Architecture

The current system has graph distance (hop count) but no learned metric.
The attention mechanism uses co-occurrence edge weights as a form of similarity,
but this is local (per-edge) not global (embedding in a space). Adding a
metric would complement the categorical structure: categories provide the
algebraic structure, the metric provides the geometric structure. Together
they form a "Riemannian category" — a category enriched over metric spaces.

This is a significant architectural addition and should be planned carefully
after the core categorical structure (products, NNO, adjunctions) is working.
