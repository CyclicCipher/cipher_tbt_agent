# Graph-Native Generation: Spread, Settle, Read

## The Problem (revised)

The original problem was autoregressive left-to-right generation failing
on carry propagation. But the deeper problem is: ANY generation scheme
that assumes a 1D token sequence is wrong. Reality is multi-dimensional
and multi-modal:

- **Vision**: 2D grid of pixels with spatial adjacency
- **Audio**: 1D waveform with temporal adjacency
- **Motor output**: keyboard (2D spatial layout), mouse (2D + buttons)
- **Text**: 1D sequence of characters/words
- **Math**: digit sequences with positional dependencies (carry)

A universal generation mechanism must handle ALL of these with the same
code. Autoregressive handles 1D but not 2D. Diffusion handles images but
requires separate architectures for text. We need ONE mechanism.

## The Insight: The Graph IS the Spatial Structure

Vision Transformers fail because they FLATTEN the 2D image to a 1D sequence
and then try to recover spatial relationships with positional embeddings.
This destroys the intrinsic topology and forces attention to re-learn it.

Our system doesn't have this problem because the spatial structure IS the
graph topology. A pixel node connected to its 8 neighbors preserves the
2D structure without any flattening. An audio sample connected to its
temporal neighbors preserves the 1D waveform structure. A keyboard key
connected to its physical neighbors preserves the spatial layout.

The graph preserves topology by construction. No flattening. No positional
embeddings. The edges ARE the positions.

## The Mechanism: Spread → Settle → Read

There is one mechanism for all generation:

1. **Activate** the input nodes (observation from any modality)
2. **Spread** activation through the graph along weighted edges
3. **Settle** — iterate spread until activations converge
4. **Read** the output from the output modality's nodes

This is the SAME mechanism as perception (spread from sensory input),
prediction (spread produces expected activations), and learning (compare
spread prediction with actual observation). Generation is just spread
where the output nodes are motor/action nodes instead of sensory nodes.

### Why This Handles Everything

**Single-token selection** (game action menu): Input activates observation
nodes. Spread reaches action candidate nodes. The highest-activated
candidate wins. Convergence in one iteration.

**Multi-digit math**: Input digit nodes + operator node activate. Spread
reaches output digit nodes through product projections. Carry constraints
propagate through inter-position edges. Multiple iterations until carry
settles. Read off all output digits at convergence.

**Image generation**: Prompt/context nodes activate. Spread reaches pixel
nodes through learned content→visual edges. Pixel-to-pixel edges enforce
spatial coherence (neighbors should be similar). Multiple iterations until
the pixel field converges. Read off the image.

**Game action with mouse**: Visual observation activates screen nodes.
Spread reaches both keyboard and mouse output nodes simultaneously.
Cross-modal edges (trained from gameplay: "when this visual pattern is
active, this mouse position is correct") determine the action. Keyboard
and mouse outputs are read simultaneously.

**Audio + visual**: Both modality subgraphs activate simultaneously from
their respective inputs. Cross-modal edges (learned from experience:
"this sound co-occurs with this visual pattern") propagate between them.
A prediction in one modality constrains the other.

### One Mechanism, Different Convergence Times

The difference between "autoregressive" and "parallel refinement" is just
convergence time:

- **No interdependencies** (single action selection): converges in 1 iteration
- **Local dependencies** (spatial coherence in images): converges in O(diameter) iterations
- **Long-range dependencies** (carry propagation in 999→1000): converges in O(chain length) iterations
- **No convergence** (novel/uncertain situation): maximum iterations reached, read off the best guess

The system doesn't need to know which case it's in. It just runs spread
until convergence or timeout. The problem structure determines the number
of iterations, not a mode switch.

## Multi-Modal Graph Architecture

### Per-Modality Subgraphs

Each input/output modality is a subgraph with intrinsic topology:

```
Visual field:
  Nodes: one per pixel (or patch)
  Edges: 8-connected grid (diagonal + orthogonal neighbors)
  Topology: 2D

Audio stream:
  Nodes: one per sample (or frame)
  Edges: temporal adjacency (previous + next)
  Topology: 1D chain

Keyboard:
  Nodes: one per key
  Edges: physical adjacency (WASD cluster, number row, etc.)
  Topology: 2D irregular grid

Mouse:
  Nodes: position (discretised 2D grid) + buttons
  Edges: spatial adjacency for position, co-occurrence for button combos
  Topology: 2D grid + discrete

Text output:
  Nodes: character/token vocabulary
  Edges: co-occurrence (learned bigram structure)
  Topology: fully connected (any token can follow any token)

Math output:
  Nodes: digit vocabulary (0-9)
  Edges: product projections (position-specific), carry edges
  Topology: position-structured product
```

### Cross-Modal Edges

Learned from experience. Examples:

- **Visual → motor**: "when the health bar (visual pattern) is low,
  press the healing key" — edge from health-bar pixel cluster to healing-key node
- **Audio → attention**: "when a specific sound plays, look at a specific
  screen region" — edge from audio pattern node to visual region nodes
- **Visual → text**: "when this text appears on screen, the meaning is..."
  — edge from character-recognition pixel patterns to semantic nodes
- **Motor → visual**: efference copy — "when I press this key, this visual
  change should happen" — edge from key node to predicted visual change

These cross-modal edges are created by the normal co-occurrence and
transition learning. When two modality nodes are co-active across time,
transition edges form. When they're co-active within the same timestep,
co-occurrence edges form. No special cross-modal code.

### Simultaneity

The brain processes vision, audio, and motor output concurrently. In our
system, all modality subgraphs are part of ONE graph. A single spread
step propagates across ALL modalities:

```
One spread step:
  For each active node (in ANY modality):
    For each outgoing edge (to ANY modality):
      Activate the target proportional to edge weight

  Cross-modal spread happens automatically because the edges cross
  modality boundaries. No special routing code.
```

The agent sees, hears, and acts in the same spread operation. Motor
planning happens simultaneously with visual processing and audio
processing — not sequentially.

## Generalised Attention

Current attention: for each context token k and each candidate c,
compute `logit(c) += activation(k) * normalised_weight(k, c)`.

This already works on any graph. The "normalisation per source" makes
each source node's outgoing weights sum to 1 (stochastic kernel). The
graph topology determines which nodes each source can reach.

For 2D visual attention: a pixel node's outgoing edges go to its 8
neighbors. The normalised weights determine how much activation flows
to each neighbor. This IS spatial attention — it's just expressed as
graph spread instead of as a 2D convolution kernel.

For cross-modal attention: a visual node's edges to motor nodes carry
the attention weight for "this visual pattern recommends this action."
Normalisation ensures the visual node's total influence is bounded.

The only extension needed: **multi-hop spread for attention.** Current
attention is one-hop (each source directly activates its neighbors).
For visual patterns spanning multiple pixels, the spread needs multiple
hops to propagate the pattern. This is what multi-hop spread (step 7a,
already implemented) provides.

So generalised attention = multi-hop spread on the graph. Already in
the system. No new mechanism needed for multi-modal or 2D.

## Connection to Existing Architecture

| Existing piece | Role in graph-native generation |
|---|---|
| Product projections | Per-position morphism tables for structured output |
| Universal NT | Extend projections to unseen positions/modalities |
| Equalizers | Discover conditional rules (carry, agreement) |
| Attention (spread) | Propagate activation through graph topology |
| Prospective configuration | Iterative settling to consistent state |
| Hippocampus | Store convergence trajectories for learning |
| Consolidation | Discover cross-modal structure (functors between modalities) |

## Implementation Plan

### Step 1: Replace autoregressive with spread-settle-read

Instead of the digit loop in `loop.act()`, implement:
1. Create temporary output nodes (one per expected output position)
2. Initialize from product projections (parallel draft)
3. Run spread for N iterations (settling)
4. Read off the highest-activated candidate at each output position
5. Emit the full sequence

Test: non-carry succession still 100%. Carry cases improve.

### Step 2: Carry propagation as graph edges

The equalizer discovers carry edges between output positions.
These are TRANSITION edges within the output subgraph: "if position P
settled to the wrap target, position P+1 should increment."

Carry edges participate in the normal spread. When position 0 settles
to `0` (wrap), the carry edge propagates to position 1, changing its
activation from identity to successor. Multiple spread iterations
propagate carry across multiple positions.

Test: `succ(999) = 1000`, `succ(9999) = 10000`.

### Step 3: Visual subgraph prototype

Create a minimal visual input modality: a small grid of pixel nodes
with 8-connected edges. Feed a simple pattern (e.g., digit image from
MNIST). Verify that spread through the grid preserves spatial structure.

This is the foundation for TiTS/Danganronpa/Persona 5 visual input.

### Step 4: Multi-modal integration

Connect visual subgraph to the existing token graph via learned
cross-modal edges. The agent observes both text tokens AND visual
pixels from the game. Cross-modal consolidation discovers functors
between the visual and textual representations of the same concepts.

### Step 5: Game adapter

Wire the full multi-modal graph to a game environment. The agent
receives visual frames + text + audio, all as subgraph activations.
It produces keyboard + mouse actions as output subgraph activations.
One spread-settle-read cycle per game frame.

## Research References

### Neuroscience
- Preplanning activates all sequence elements simultaneously (J. Neuroscience, 2025)
- Feature separation during planning → integration during execution (J. Neuroscience, 2023)
- Spacetime attractors in PFC for planning (bioRxiv, 2025)
- ECoG evidence for diffusion-like iterative refinement in speech (OpenReview, 2025)

### Machine Learning
- Dream 7B: diffusion LLMs excel at constraint satisfaction (arXiv, 2025)
- DDPMs as continuous content-addressable memories (xcorr.net, 2023)
- Vision Transformers flatten 2D → 1D, losing spatial structure
  Our approach: graph preserves topology, no flattening needed

### Key Insight
The graph-native approach unifies:
- Perception (spread from sensory input)
- Prediction (spread produces expected state)
- Generation (spread settles output nodes)
- Planning (multi-iteration spread for constraint satisfaction)
- Multi-modal processing (cross-subgraph spread)

All are instances of: activate → spread → settle → read.
