# Manifold Redesign — From Lookup Tables to Geometric Constraints

## The Three Failures

1. **Fixed modules per OOM:** Toroidal modules at scales 1, 10, 100
   hardcode three orders of magnitude. Can't handle decimals, large
   numbers, or arbitrary precision. Need a single scale-free mechanism.

2. **10,000 examples = memorization:** The shared triple field is a
   1000-entry lookup table filled by brute force. NOT manifold learning.
   The manifold approach promised 3 points define a plane. Must deliver.

3. **No integration:** The addition system is a standalone object, not
   a subgraph in the main graph. Not connected to ANS. No cortical
   column structure.

## What Must Change

### Representation: from discrete modules to continuous distance

The torus modules discretize the number line into fixed-scale digits.
This creates carries (an artifact of discretization) and limits
precision to the number of modules.

The brain doesn't discretize. It represents numbers as POSITIONS on
a continuous line. The distance between positions IS the number.
Addition is: the distance from a to c equals the distance from 0 to b.
One rule, scale-free, no carries.

For CipherNet: represent numbers as positions in the graph with
DISTANCES between them, not as phase vectors in discrete tori.
The graph can still use toroidal modules for efficient encoding,
but the MANIFOLD operates on distances, not on per-module phases.

### Learning: from counting co-occurrences to fitting constraints

The triple field counts how often (a_phase, b_phase, c_phase)
co-occur. This is memorization. With 3 examples, you get 3 entries
in a 1000-element table — no generalization.

Instead: from 3 examples, extract the CONSTRAINT that relates the
three positions. For addition:
  Example (1, 2, 3): distance(a, c) = 2 = distance(0, b)
  Example (8, 7, 15): distance(a, c) = 7 = distance(0, b)
  Example (20, 13, 33): distance(a, c) = 13 = distance(0, b)

The invariant: distance(a, c) = distance(0, b). Always. This is
the co-termination rule from the earlier invariant_extraction work.

The constraint IS the manifold. It's not a table — it's a RULE
about distances. It applies at any scale, to any numbers, because
distances are scale-free.

### Computing answers: from table lookup to graph navigation

Given the constraint distance(a, c) = distance(0, b), computing
3 + 4 = ? means: find c where distance(3, c) = distance(0, 4) = 4.
Answer: c = 3 + 4 = 7. This is walking 4 steps from position 3.

The walk is performed on the graph itself. No lookup table. The
graph's spatial edges define the metric. Walking = following edges.

For large numbers: the walk is long but the RULE is the same.
307 + 456 = ? means walk 456 steps from position 307. Or: walk
distance(0, 456) from position 307. The graph doesn't need to
explicitly contain position 307 — it needs the RULE for walking,
which is: follow successor edges.

For decimals: the walk can be fractional. Position 3.5 is halfway
between 3 and 4 on the graph. Walking 0.5 steps = moving to the
midpoint. The metric is continuous even if the graph is discrete.

### Integration: manifold as cortical column

The addition manifold lives in the main graph as a CORTICAL COLUMN
subgraph:

```
Layer 4 (input):    receives bindings from stimulus tokens
                    '3', '+', '4' bind to input nodes
Layer 2/3 (lateral): processes the constraint
                     distance(a, c) = distance(0, b)
                     walks the graph to find c
Layer 5 (output):   sends the result to other columns
                    c = 7 binds to output nodes
Layer 6 (feedback): sends prediction back to input
                    for error correction
```

This column connects to the ANS (for magnitude comparison), to the
number line subgraph (for the metric), and eventually to the PFC
(for multi-step sequencing).

## Example Counts Per Operation

### Addition: 3 examples minimum

With 3 non-degenerate examples of addition:
1. Extract the invariant: distance(a, c) co-terminates with distance(0, b)
2. Store the constraint (not a table)
3. Apply the constraint to ANY new (a, b) pair by walking the graph

This works because:
- 3 points define a plane in 3D (the addition plane c = a + b)
- The plane = the constraint distance(a, c) = b
- The constraint applies at any scale, to any numbers

### Multiplication: ~6-10 examples

Multiplication is c = a * b — a hyperbolic paraboloid (saddle surface).
NOT a plane. A general degree-2 surface has 6 parameters:
  c = k1*a^2 + k2*b^2 + k3*a*b + k4*a + k5*b + k6
So 6 non-degenerate examples are needed to determine the surface.

The spatial constraint for multiplication: distance(0, c) decomposes
into b segments of length a (repeated walking). This is a single
rule, but more complex than the addition walk. It requires understanding
"walk a steps, b times" which itself builds on addition.

Examples MUST include negative numbers to reveal the saddle shape:
  3 * 4 = 12     (positive × positive = positive)
  3 * (-4) = -12 (positive × negative = negative)
  (-3) * 4 = -12 (negative × positive = negative)
  (-3) * (-4) = 12 (negative × negative = positive)

Without negative examples, the system can't discover that the
sign rule is part of the manifold.

### Negative numbers

The number line extends in both directions from 0. Positions
include ..., -3, -2, -1, 0, 1, 2, 3, ...

For addition: distance is SIGNED (directed).
  3 + (-5) = -2 means: walk -5 steps from position 3 → arrive at -2.
  The constraint distance(a, c) = b still holds: distance(3, -2) = -5 = b.

For multiplication: the sign of the product depends on the signs of
both operands. This is NOT a distance constraint — it's a structural
property of the manifold (the saddle has four quadrants with alternating
signs). The system must discover this from examples that span all four
quadrants.

Minimum examples for multiplication with full sign coverage:
  Positive: 3 * 4 = 12, 2 * 5 = 10
  Mixed:    3 * (-2) = -6, (-4) * 3 = -12
  Negative: (-3) * (-4) = 12, (-2) * (-5) = 10
  Zero:     0 * 7 = 0, 5 * 0 = 0
Total: ~8-10 examples for robust manifold fitting.

## Implementation Plan

### Step 1: Distance-based constraint representation

A manifold is stored as a set of DISTANCE CONSTRAINTS between axes.
For addition: "distance(axis_a, axis_c) = position(axis_b)".
This is one data structure — not 3000 floats in a table.

### Step 2: Constraint extraction from 3-10 examples

Reuse the invariant extraction code (invariant_extraction.py).
Given examples, compute all pairwise distances, find which
distance relationships are invariant across all examples.
The invariant IS the constraint.

### Step 3: Graph navigation for inference

Given known values, apply the constraint by walking the graph.
Walking = following spatial edges the right number of steps.
The torus modules can STILL be used for efficient encoding,
but the computation is a walk, not a table lookup.

### Step 4: Cortical column structure

Wrap the constraint in a column subgraph with input/processing/output
layers. Connect to the main graph. Connect to ANS and number line.

### Step 5: Validate

- 3 examples of addition → generalize to 307 + 456 = 763
- 3 examples of multiplication → generalize to 6 * 7 = 42
- All within the main graph, connected to ANS
- No lookup tables, no OOM limits, no memorization

## Open Questions

1. How does the constraint representation handle carries for
   multi-digit output? If the walk is continuous, there are no
   carries. But the OUTPUT needs to be decoded into digits eventually
   (for display). Where does the digit decomposition happen?

2. How does multiplication work as a constraint? "b segments of
   length a" is a nested walk — more complex than a simple
   distance equality. Can this be represented as a single constraint?

3. How do we handle the 3-example case where the plane is
   underdetermined at higher orders? 3 examples of small additions
   (1+2=3, 8+7=15, 20+13=33) define the same plane as 3 examples
   of large additions. The plane IS scale-invariant. But is this
   true for all operations?

4. Where exactly in the cortical column does the "walking" happen?
   Is it layer 2/3 lateral processing? Is it a recurrent loop?
