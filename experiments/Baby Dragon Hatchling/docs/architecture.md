# Baby Dragon Hatchling (BDH) — Architecture Reference

**Paper:** "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"
**Authors:** A. Kosowski, P. Uznanski, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz (Pathway)
**arXiv:** 2509.26507 (2025)
**Code:** github.com/pathwaycom/bdh (MIT license)
**Educational impl:** github.com/krychu/bdh

---

## 1. Neuron Particles

BDH models computation as a network of **n locally-interacting neuron particles**. Each neuron i has:

- **X(i)** — excitatory activation state (positive, sparse ~20% active)
- **Y(i)** — output/inhibitory activation state (positive, extremely sparse ~3-5% active)
- **sigma(i,j)** — synaptic state on the edge between neurons i and j (dynamic, updated during inference)

The system defines an **interaction kernel** (Definition 1 in the paper):

```
q'_k := (1 - d_k) * q_k + sum_{i,j} r_{ijk} * q_i * q_j
```

This is chemical-reaction-network-style dynamics with z species. The **edge-reweighting kernel** (Definition 2) specializes this to graph-based dynamics where nodes interact via edges.

---

## 2. The Two Fundamental Equations

**Equation (1) — Modus ponens (inference):**
```
X(i), sigma(i,j) --> A(j)
```
If neuron i has belief X(i) and rule sigma(i,j) exists, then belief propagates to neuron j. This is compositional reasoning: belief + rule = conclusion.

**Equation (2) — Hebbian learning:**
```
Y(i), X(j) --> sigma(i,j)
```
If neuron Y(i) fires and neuron X(j) fires, the connection sigma(i,j) strengthens proportionally to Y(i) * X(j). Classic "neurons that fire together wire together." The synaptic state sigma updates continuously during inference — this IS the in-context learning mechanism.

**State-space update:**
```
sigma(t+1) := A(M, sigma(t), a_t)
```
where M is the model, sigma(t) is the current synaptic state, and a_t is the current input token.

---

## 3. BDH-GPU: The Tensor-Friendly Formulation

BDH-GPU replaces direct graph wiring with **mean-field communication** (radio network instead of wired network). Three key parameter matrices:

- **E** (N x D) — the edge/connection matrix (evolving attention state, synaptic weights)
- **Dx** (H x D x Nh) — input transformation to neuron dimension n (fixed learned parameters)
- **Dy** (H x D x Nh) — output transformation from neuron dimension n (fixed learned parameters)

These compose into two circuits:
- **Causal Circuit: Gx = E @ Dx** — propagates signals from previous neurons (y) to current neurons (x). Implements probabilistic if-then reasoning.
- **Output Circuit: Gy = Dy @ E** — determines which neurons fire based on attention-weighted context.

**Parameter count:** (3 + o(1)) * n * d, where n is the neuron count and d is a small dimension satisfying log(n) < d << n (d=256 in practice).

---

## 4. Forward Pass (per layer)

From the reference implementation:

```python
# 1. Project to sparse neuron space
x_sparse = ReLU(x @ encoder)           # (B, n_head, T, N)

# 2. Linear attention (Q=K tied, causal mask, RoPE)
yKV = linear_attn(Q=x_sparse, K=x_sparse, V=x)

# 3. Project attention output to sparse space
y_sparse = ReLU(yKV @ encoder_v)

# 4. Multiplicative gating (the Hebbian interaction)
xy_sparse = x_sparse * y_sparse         # elementwise Hadamard product

# 5. Decode back to embedding dimension
output = xy_sparse @ decoder

# 6. Residual + LayerNorm
x = LayerNorm(x + output)
```

Repeated L times (6-12 layers depending on configuration).

---

## 5. Attention: BDH vs Transformers

BDH uses **linear attention** instead of softmax attention:

```python
scores = QR @ KR.mT     # no softmax, just dot product
scores = scores.tril(-1) # causal mask
output = scores @ V
```

Key differences from transformers:
- **Linear complexity O(N)** instead of quadratic O(N^2)
- **Positive orthant:** Keys mapped to positive space via Locality-Sensitive Hashing (LSH)
- **Q = K tied:** Query and key projections are shared
- **RoPE** positional encoding applied
- Attention operates in the **neuron dimension n** (e.g. 2048-8000+) not the embedding dimension d

Memory is NOT in a KV-cache. Memory lives in the synaptic state sigma — the dynamic edge weights that update via Hebbian learning during inference. This gives theoretically unbounded context.

---

## 6. Sparsity

Sparsity is **inherent by construction**, not learned or imposed:

- **ReLU activation** on neuron projections guarantees positive, sparse activations
- **x activations:** ~20% active neurons
- **y activations:** ~3-5% active neurons (extremely sparse)
- All activation vectors are **strictly positive** (no negative values)

Consequence: prevents **polysemantic superposition** (where a single neuron encodes multiple unrelated concepts). BDH neurons tend to be **monosemantic** — individual synapses correlate strongly with distinct concepts across multiple prompts.

---

## 7. Sudoku Without Autoregressive Generation

BDH achieves **97.4% accuracy on Sudoku Extreme** (~250,000 of the hardest puzzles), while leading LLMs (o3-mini, DeepSeek-R1, Claude 3.7 Sonnet) score approximately 0%.

The mechanism is NOT autoregressive token-by-token generation. Instead:
- Large latent reasoning space holds multiple candidate solutions simultaneously
- Iterative layer-by-layer refinement acts as **constraint propagation** — each layer refines the solution by propagating constraints through the neuron graph
- Hebbian synaptic state allows **dynamic working memory** that tracks constraint satisfaction
- No chain-of-thought, no backtracking, no external tools required

**Caveat:** The 97.4% result is from Pathway's internal implementation, NOT the open-source repo. The open-source code does not reproduce this benchmark.

---

## 8. In-Context Learning

In-context learning in BDH is **synaptic plasticity during inference:**

- Synaptic weights sigma(i,j) update via the Hebbian rule (Equation 2) at timescales of hundreds of tokens
- The model literally rewires itself as it reads context
- No backpropagation through time needed
- Monosemantic synapses: the same synapse consistently strengthens for the same concept across different prompts
- The multiplicative gating `x_sparse * y_sparse` in the forward pass IS the Hebbian interaction — when both pre-synaptic (x) and post-synaptic (y) neurons fire, their product strengthens the signal path

Fundamentally different from transformers, where in-context learning happens through attention patterns over a static KV-cache.

---

## 9. BDH-Particle vs BDH-GPU

| Aspect | BDH-Particle | BDH-GPU |
|--------|-------------|---------|
| Communication | Direct graph wiring ("by wire") | Mean-field ("radio network") |
| State location | On edges (sigma on each edge) | Localized at neurons |
| Topology | Explicit sparse graph (Gx, Gy) | Dense tensor operations |
| Efficiency | Graph-native, biologically faithful | GPU-optimized, tensor-friendly |
| Equivalence | General form | Special case (Claims 3-4) |

The paper proves (Claims 3-4) that BDH-GPU is a special case of BDH-particle: the matrices Dx, Dy, E can be expressed as graphs Gx, Gy while preserving parameter count at O(nd). Trained BDH-GPU models spontaneously develop **scale-free, heavy-tailed degree distributions** — biological network topology emerges from dense matrices.

---

## 10. Theoretical Results

- **Claim 1:** BDH is an attention-based state-space sequence learning architecture expressible as a distributed system of n particles with local rulesets. It can emulate spiking neural networks with excitatory/inhibitory circuits and Hebbian learning.
- **Claim 2:** BDH-GPU has (3+o(1))nd parameters with a mean-field/particle state-space interpretation.
- **Claims 3-4:** Graph expressibility — the graph and tensor formulations are equivalent. Attention sparsification on graphs maintains this equivalence.
- **Scaling laws:** BDH-GPU empirically matches GPT-2 architecture transformer performance across 10M to 1B parameters on language and translation tasks, with comparable FLOP counts.
- **Thermodynamic limit:** The paper defines a scale-free model family {M_n ~ P_A(n)} with a limit object P_A := lim(n->inf) P_A(n), arguing this enables "foreseeable AI" through asymptotic property characterization.
- **Model merging:** BDH-GPU's uniform parameterization allows direct concatenation — two models compose into a single model with parameter count equal to their sum, preserving architecture. Validated on translation tasks.

No formal universality theorem (Turing completeness) is stated — expressiveness is argued empirically plus a RASP framework connection in the appendix.

---

## 11. Codebase

**Official repo (pathwaycom/bdh):**
- License: MIT (Copyright 2025 Pathway Technology, Inc.)
- Dependencies: torch, numpy, requests (no version pinning)
- Files: bdh.py (model), train.py (training on tiny Shakespeare), requirements.txt
- Very small codebase — essentially two Python files
- Stars: ~3.4k, Forks: ~213
- Training: AdamW (lr=1e-3, weight_decay=0.1), mixed precision (bfloat16/float16), torch.compile, 3000 iterations

**Educational repo (krychu/bdh):**
- Implements BDH on a pathfinding task (shortest path on NxN grids with obstacles)
- Files: bdh.py (model), boardpath.py (task + training + visualization), utils/
- Config: 2048 neurons, 64-dim embeddings, 12 layers, 4 heads, RoPE
- Produces animated GIFs showing layer-by-layer prediction refinement and neuron dynamics
- Demonstrates emergent hub-and-spoke topology from random initialization

**Community ports:** MLX port (severian42/bdh), Burn framework port (mosure/burn_dragon_hatchling), dynamic vocabulary variant (adamskrodzki/bdh).
