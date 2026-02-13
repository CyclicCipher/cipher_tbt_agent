# Mamba3 Memory Improvement Options

**Date:** 2026-02-13
**Context:** SSM memory is the primary bottleneck. With d_state=32-64, Mamba3 effectively retrieves from only ~1-2 recent positions (see `algorithmic_changes.md` Where task diagnosis). This limits all tasks requiring longer-range associative recall.

---

## 1. Increase d_state

**What:** Increase the SSM state dimension from 64 to 128-256.

**Why it might help:** d_state directly controls how many past tokens the SSM can retain. Current d_state=64 gives ~2 position effective memory. Doubling should extend this, though the relationship is sublinear due to the exponential decay of SSM memory.

**Cost:** Linear increase in parameters and FLOPS for the SSD scan. May push against 4GB VRAM limit at d_state=256.

**Status:** Not tested.

---

## 2. Block Tensor-Train (BTT) Compression

**What:** Replace dense d×d projections in Mamba3 with structured BTT decomposition. Dense: d² params. BTT with rank r: 2r × d^(3/2) params.

**Why it might help:** 8-15× FLOPS reduction on projections at d_model=256-1024. Frees VRAM budget for larger d_state or more layers. In ePC with T iterations, savings multiply (T forward passes + T backward passes).

**Cost:** Implementation complexity. Overhead for small d_model (<256). Requires structure-aware learning rates (κ_L = √d/(2r), κ_R = √d/2).

**Reference:** `experiments/ePC_Mamba/NOTES.md` §BTT Analysis; `docs/research/BTT paper.pdf`

**Status:** Documented, not implemented. Phase 4 optimization.

---

## 3. Multi-Head Latent Attention (MLA) / Hybrid Attention-SSM

**What:** Add sparse attention heads alongside SSM layers. Attention provides O(1) random-access lookup while SSM handles sequential processing.

**Why it might help:** SSM memory decays exponentially with distance. Attention doesn't — it can directly retrieve any position. A hybrid could give the best of both: SSM for sequential patterns, attention for arbitrary-distance recall.

**Cost:** Quadratic cost in seq_len for attention. Mitigated by sparse/sliding-window attention or only using it every N layers.

**Status:** Not explored. Standard approach in modern architectures (Jamba, etc.).

---

## 4. Warm-Starting Error Vectors

**What:** Instead of initializing errors to zero every batch, use a learned initialization network that predicts initial errors from the input.

**Why it might help:** Zero-initialized errors need T iterations to reach informative values. Warm starting could reach useful errors in fewer iterations (T=2 or T=1), saving compute AND potentially giving larger error magnitudes (which directly determines local gradient strength).

**Cost:** Additional small network. Doesn't work for shuffled batches with sample-level persistence (previous error at position k is noise for new sample).

**Reference:** `experiments/ePC_Mamba/NOTES.md` §Warm-Starting Errors

**Status:** Documented as future work. Not implemented.

---

## 5. PoPE Positional Embeddings (DONE)

**What:** Polar Positional Embeddings replacing RoPE. Encodes positions as phase angles on the unit circle with learnable frequencies.

**Why it helps:** 93.7% improvement on content matching (What task). Better retrieval fidelity at short distances (68.8% vs 20.6% at distance 1). Same memory depth but vastly better use of what IS remembered.

**Status:** IMPLEMENTED. Default in Mamba3Config.

---

## 6. Sparse Connectivity (RigL)

**What:** Dynamic sparse training — maintain sparse weight masks that evolve during training (grow useful connections, prune useless ones).

**Why it might help:** Could allow larger effective d_model within the same VRAM budget by keeping only ~10% of connections active.

**Cost:** GPU sparse ops currently slower than dense for training. Needs >90% sparsity for wall-clock speedup. Don't know which connections matter until training converges.

**Reference:** `experiments/ePC_Mamba/NOTES.md` §Sparse Connectivity

**Status:** LOW PRIORITY. Revisit when NVIDIA 2:4 sparsity hardware support matures.

---

## 7. PPCA Curvature Tracking for Error Optimization

**What:** Low-rank plus diagonal (PPCA) approximation to the Hessian for error optimization. Instead of SGD, use a warm-startable approximate Newton step.

**Why it might help:** Could reduce T from 20 to 5 or fewer iterations. The PPCA form Y = URU^T + s(I - UU^T) tracks curvature cheaply at O(dp²) cost.

**Caveat:** Mistake #33 established that SGD beats Newton/CG for error optimization. However, that was rank-1 Newton. PPCA is a more principled curvature approximation that might not share the same pathologies.

**Reference:** `experiments/ePC_Mamba/NOTES.md` §LRPD for Riccati; `docs/research/LRPD for riccati-like matrix DEs.pdf`

**Status:** Documented, not implemented.

---

## 8. Trapezoidal → Higher-Order Discretization

**What:** Replace Mamba3's trapezoidal (2nd-order) SSM discretization with higher-order methods: ETD with ψ₁ correction, Adams-Bashforth 3, or data-dependent multi-step.

**Why it might help:** Higher-order discretization reduces temporal approximation error, potentially improving long-range memory fidelity without increasing d_state.

**Cost:** Tighter temporal coupling. More complex implementation. Data-dependent multi-step is most expressive but requires learning integration coefficients.

**Reference:** `experiments/ePC_Mamba/NOTES.md` §Discretization

**Status:** Documented, not implemented.

---

## 9. Deeper Networks (More Layers)

**What:** Scale from 4 to 8-16 Mamba3 layers.

**Why it might help:** More layers = more representational capacity, deeper feature extraction. Each layer can specialize on different aspects of the input.

**Cost:** Linear increase in params and compute. ePC error optimization scales linearly with layers. Precision weighting becomes important (geometric precision ratio grows exponentially with depth).

**Status:** Current default is 4 layers. Not tested with more.

---

## Priority Ranking

For improving Mamba3's memory on sequence tasks:

1. **Increase d_state** — simplest, most direct impact on SSM memory
2. **Hybrid attention-SSM** — addresses the fundamental limitation (exponential decay)
3. **More layers** — standard scaling, known to help
4. **Warm-starting errors** — reduces ePC overhead, enables larger models in same VRAM
5. **Higher-order discretization** — improves memory fidelity without parameter cost
6. **BTT compression** — frees VRAM for options 1-3
7. **PPCA curvature** — faster error convergence
8. **Sparse connectivity** — hardware-dependent, premature
