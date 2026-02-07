# eBPC-ResNet: Error-based Bayesian Predictive Coding with ResNet

## Phase 1: Core Implementation (do first)

- [ ] **Diagonal eBPC layer**: Replace full V (in×in) and Ψ (out×out) matrices with diagonal vectors. Closed-form Hebbian updates still work — matrix inverse becomes element-wise reciprocal. ~6x parameter reduction on MLP, essential for ResNet scaling.

- [ ] **ResNet architecture with ePC skip connections**: Implement ResNet-18 blocks with skip connections compatible with ePC error reparameterization. Reference: `PCESkipConnection` from the ePC GitHub repo (https://github.com/cgoemaere/error_based_PC, cifar branch). Skip connections: output = F(x) + x, with `SaveIdentity`/`AddIdentity` modules passing (activity, identity) tuples.

- [ ] **bfloat16 mixed precision**: Use PyTorch AMP for the ePC forward/backward passes. Keep natural parameters in float32 for Hebbian update precision. Standard `torch.cuda.amp.autocast(dtype=torch.bfloat16)` pattern.

- [ ] **Adaptive T (early stopping)**: Monitor error energy during inference; break early if energy reduction falls below threshold. Easy samples may need T=1-2, hard samples get full T=5. Track average T for diagnostics.

- [ ] **CIFAR-10 training script**: Train eBPC-ResNet-18 on CIFAR-10. Baseline targets: ePC ResNet-18 92.17%, backprop ResNet-18 92.36%.

- [ ] **Validate on MNIST first**: Before CIFAR-10, verify diagonal eBPC matches full eBPC on MNIST (~95.7%) to confirm diagonalization doesn't hurt.

## Phase 2: Research Avenues (explore later)

- [ ] **Recovering off-diagonal benefits cheaply**: The full V matrix captures weight correlations that matter for uncertainty calibration, OOD detection, and active learning. Research options:
  - Low-rank V: V = diag(d) + UU^T with rank-k U (captures top-k correlation modes, O(k·n) instead of O(n²))
  - Block-diagonal V: Group correlated weights (e.g., spatial neighbors in conv filters), full covariance within blocks
  - Kronecker-factored V: V ≈ A ⊗ B (common in natural gradient methods like K-FAC)
  - Periodic full-rank snapshots: Run full V every N batches, diagonal between — amortized cost

- [ ] **Manifold hyperconnections (DeepSeek)**: Generalize skip connections from output = F(x) + x to multi-stream mixing with learned coefficients. Potential benefits: multiple error propagation pathways, richer Bayesian structure. Test AFTER standard ResNet skip connections are validated.

- [ ] **Error-gated sparsity**: Use error magnitudes to gate computation:
  - Skip Hebbian weight updates for layers with ||ε_i|| < threshold
  - Neuron-level sparsity: zero out small error components to reduce backprop cost
  - Per-sample early stopping: samples that converge fast exit inference early (needs custom batching)
  - Research question: does this hurt posterior quality or just save compute?

- [ ] **int8 weight quantization for inference**: After extracting M from natural params in float32, quantize to int8 for the ePC forward pass. Natural params stay float32 for Hebbian updates. Could further halve inference memory vs bfloat16.

- [ ] **Structured pruning**: After training, identify and remove low-importance neurons (small weights + high uncertainty). The Bayesian posterior provides a principled pruning criterion — prune weights where the posterior is wide relative to the mean.

- [ ] **Uncertainty-aware active learning**: Use the Bayesian posterior to select the most informative training samples. The diagonal posterior underestimates uncertainty — how much does this matter for sample selection?

- [ ] **Scaling beyond ResNet-18**: ResNet-34, ResNet-50, and eventually vision transformers. The ePC paper only tested up to ResNet-18. Deeper networks may need additional tricks (gradient checkpointing, memory-efficient attention for transformers).

## References

- BPC: Tschantz et al. 2025, arXiv:2503.24016 (Algorithm 1, Equation 7, Appendix F.1)
- ePC: Goemaere et al. 2025, arXiv:2505.20137 (Algorithm 2, Theorem C.7/C.9)
- ePC code: https://github.com/cgoemaere/error_based_PC (Apache 2.0)
- eBPC (ours): experiments/eBPC/ (MNIST baseline: 95.74% test, 3 epochs)
