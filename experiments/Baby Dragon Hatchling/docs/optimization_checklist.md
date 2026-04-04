# BDH Optimization Checklist

## Current state

The open-source BDH-GPU implementation uses dense tensor operations throughout.
ReLU produces ~80% zeros in x_sparse and ~95% zeros in y_sparse, but no
computation is skipped — the GPU multiplies all the zeros. The paper describes
persistent synaptic state (Equation 2) but the code doesn't implement it.

## Phase 2: Block-sparse Triton kernels

BDH's sparsity is real (~5% active in xy_sparse) but scattered across
individual neurons. GPUs can't skip individual zeros efficiently — they
need contiguous blocks. The path to real savings:

- [ ] **Block-sparse neuron groups**: Partition the N neurons per head into
  blocks of 32 (NVIDIA warp size). After ReLU/top-k, zero out entire
  blocks where no neuron exceeds a threshold. This converts scattered
  sparsity to block sparsity that hardware can exploit.

- [ ] **Triton block-sparse matmul**: Write a Triton kernel for the key
  operations (`x_sparse @ K^T`, `xy_sparse @ decoder`) that skips
  zero blocks entirely. A block-sparse (B, nh, T, N) times dense
  (N, D) only touches the nonzero blocks. With 95% sparsity in
  xy_sparse, ~95% of blocks are skippable.

- [ ] **Triton fused sparse attention**: Fuse the RoPE + sparse QK^T +
  causal mask + matmul-with-V into a single kernel. Avoid materializing
  the full (T, T) attention matrix — only compute entries where both
  Q_t and K_s have nonzero overlap in their active blocks.

## Phase 3: Training improvements

- [ ] **torch.compile**: Enable Triton-based kernel fusion for the remaining
  dense operations. ~1.5-2x speedup with zero code changes.

- [ ] **Gradient checkpointing**: Trade compute for memory. Recompute
  intermediate activations during backward instead of storing them.
  Allows larger models or batch sizes within 4GB VRAM.

## Phase 4: Structure discovery (our unique angle)

- [ ] **Replace backprop for encoder/decoder**: Use FCA, PMI, or algebraic
  discovery to construct weight matrices from data instead of gradients.

- [ ] **Hebbian weight update as learning rule**: Extend Phase 1's persistent
  sigma into an actual cross-sequence learning rule. The sigma accumulated
  during inference becomes a gradient-free update signal for the encoder/
  decoder matrices.

## Key references

- Kosowski et al. "The Dragon Hatchling" (arXiv:2509.26507) — BDH paper.
  Equation 2: Y(i), X(j) -> sigma(i,j). BDH-particle never implemented.
- Nawrot et al. "The Sparse Frontier" (arXiv:2504.17768) — larger sparse
  models beat smaller dense ones at the same compute budget.
- DeepSeek "Native Sparse Attention" (arXiv:2502.11089) — hardware-aligned
  block-sparse attention, 11.6x decoding speedup.
