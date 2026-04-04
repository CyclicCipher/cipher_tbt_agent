# BDH — Relevance to Our Symbolic AGI Project

## Why BDH Matters

BDH solves several problems that our CTKG system has struggled with:

1. **In-context learning:** BDH rewires synaptic weights during inference via Hebbian plasticity. Our CTKG's edge strengthening/weakening is the same principle, but BDH does it efficiently in a GPU-friendly tensor formulation.

2. **Non-autoregressive reasoning:** BDH solves Sudoku (97.4% on Extreme) without generating tokens left-to-right. It uses iterative layer-by-layer constraint propagation in a large sparse latent space. This is exactly what the calendar task needs — all answer digits computed simultaneously, not sequentially.

3. **Sparse monosemantic representations:** BDH's ReLU-enforced sparsity (~3-5% active neurons) produces monosemantic features without any regularization. Our system tried to achieve this with three-layer cortical columns and FCA — BDH gets it for free from the architecture.

4. **Compositional structure:** The modus ponens framing (belief + rule → conclusion) IS compositional morphism application. sigma(i,j) is a learned rule; X(i) is a belief; A(j) is the conclusion. This maps directly to CTKG edges.

5. **Scale-free graph emergence:** Without being forced, trained BDH-GPU matrices develop heavy-tailed degree distributions, high modularity, core-periphery structure. The graph structure we've been trying to hand-design EMERGES from training.

---

## What BDH Uses Backprop For (and What We Could Replace)

BDH has two kinds of learning:

### 1. Fixed weights (trained with backprop)
- **encoder** (d → n): projects embedding to neuron space
- **encoder_v** (d → n): projects attention output to neuron space
- **decoder** (n → d): projects neuron space back to embedding
- **Dx, Dy**: input/output transformations
- These are the "hardware" — the fixed circuitry

### 2. Dynamic weights (Hebbian, no backprop)
- **sigma(i,j)**: synaptic state, updated during inference
- `sigma(i,j) += Y(i) * X(j)` — neurons that fire together wire together
- This is the "software" — the runtime state that adapts to context

**Our opportunity:** Replace the backprop-trained fixed weights with structure discovered from data. The CTKG's FCA, algebraic skeleton, initial algebra discovery, and sheaf consistency could provide the encoder/decoder matrices. The Hebbian inference-time learning is already what our system does.

---

## Three Research Directions

### Direction 1: Structure Discovery as Backprop Replacement
If we can discover the encoder/decoder matrices from data (using FCA for type hierarchy, PMI for specific associations, initial algebra for algebraic structure), we have a learning algorithm that:
- Explicitly discovers compositional structure
- Doesn't need gradients
- Could work on CPU (no GPU required for the discovery phase)
- Produces interpretable representations by construction

### Direction 2: Spiking Neural Network Implementation
BDH-particle IS a spiking neural network with Hebbian learning. If we implement BDH-particle (the graph version, not the GPU version), we get:
- Native sparsity (only active neurons compute)
- Event-driven computation (no wasted cycles on inactive neurons)
- CPU-friendly (sparse graph operations, not dense matrix multiplies)
- Biological plausibility for free

### Direction 3: BDH-GPU with Our Learning Algorithm
Keep BDH-GPU's architecture (it works on GPU, matches transformer performance) but replace backprop with our structure discovery for the fixed weights. This gives:
- GPU acceleration for inference
- Interpretable, compositionally-structured weights
- Hebbian in-context learning (already built in)
- The best of both worlds

---

## Key Architectural Parallels

| Our CTKG | BDH | Mapping |
|----------|-----|---------|
| Identity nodes (layer 0) | Neuron X activations | What is active |
| Context nodes (layer 1) | Synaptic state sigma | Where/when context |
| Displacement nodes (layer 2) | Y activations | What fires as output |
| Edge weight | sigma(i,j) | Connection strength |
| Hebbian strengthen/weaken | Y(i)*X(j) → sigma(i,j) | Same principle |
| Spread (prediction) | Modus ponens: X(i),sigma(i,j)→A(j) | Same operation |
| Co-occurrence edges | Encoder matrix columns | What co-activates |
| Transition edges | Attention weights | What follows what |
| FCA concepts | Hub neurons in scale-free graph | Categorical structure |
| Consolidation | Training (backprop phase) | Slow learning |
| Online learning | Hebbian inference-time update | Fast learning |

---

## What BDH Lacks That We Have

1. **Explicit compositional structure** — BDH discovers it implicitly (scale-free graphs emerge). We discover it explicitly (FCA, algebra, initial algebra, sheaf).

2. **Categorical semantics** — BDH has no formal type system, no functors, no natural transformations. Our CTKG has all of these.

3. **Constraint solving** — BDH does iterative refinement (works for sudoku). Our sheaf Laplacian gives a principled energy to minimize. These could combine: BDH's iterative layers + sheaf energy as the objective.

4. **Interpretability by design** — BDH's monosemanticity is emergent and empirical. Our type hierarchy is interpretable by construction.

---

## Open Questions

1. Can FCA + PMI + initial algebra discovery produce encoder/decoder matrices that work as well as backprop-trained ones?

2. Does BDH-particle (the graph version) actually run efficiently on CPU for our scale (thousands of neurons, not millions)?

3. Can we use sheaf energy as a layer-wise objective for BDH's iterative refinement, replacing the cross-entropy training loss?

4. The 97.4% sudoku result isn't in the open-source code. Can we reproduce it? What's the actual architecture for that?

5. BDH's linear attention is O(N) but loses the ability to attend to specific positions. Does this matter for our tasks?
