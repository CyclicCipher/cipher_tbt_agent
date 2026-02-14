# ePC-Mamba3 Learning Dynamics

## Summary

ePC-Mamba3's learning dynamics depend critically on two architectural choices:

1. **Error optimizer**: SGD (simple gradient descent) is optimal. Newton and CG
   introduced unnecessary complexity and a pathological "phase transition" plateau.
2. **Error coverage**: N errors for N layers (every block gets an error node) +
   next-step prediction (every position gets a loss signal). Sparse coverage fails.

With SGD + complete error coverage, ePC-Mamba3 learns smoothly and fast:
- Task 1b (multi-rule regime change): **96% test accuracy on epoch 1**
- Final accuracy: 97% (matching the theoretical ceiling for next-step prediction)
- No plateau, no phase transition, no initialization sensitivity

---

## Historical Context: The Newton Phase Transition (Resolved)

### The Phenomenon (copy task, Newton error optimizer)

ePC-Mamba3 with Newton error optimization exhibited a phase transition: accuracy
plateaued at 10-15% for 19-28 epochs, then explosively converged to 99%+ in ~8
epochs. This was extensively analyzed and multiple "fixes" were attempted.

Results (copy task, 4-layer Mamba3, d_model=128, Newton T=2):
- ePC seed=0: plateau 7-17% for 19 epochs -> 99.03% at epoch 27
- ePC no seed (run 1): plateau for 28 epochs -> 99.3% at epoch 44
- ePC no seed (run 2): plateau for ~28 epochs -> 99.2% at epoch 36
- Backprop seed=42: smooth 7% -> 38% -> 99.2% at epoch 38

### What We Thought Was Happening

The original analysis identified a circular dependency:

```
Newton effectiveness --> error magnitude --> E_local gradient quality
        ^                                              |
        \____ Jacobian structure <-- weight updates <--+
```

Newton's rank-1 Hessian approximation required the Jacobian dout/de to reach a
specific quality threshold before errors could grow. But the Jacobian quality
depended on weight learning, which depended on error magnitude, which depended
on Newton finding errors, which depended on the Jacobian...

The escape mechanism was CE-driven embedding changes: the output CE loss in
E_local provided gradients to the embedding/output projection (weight-tied)
regardless of error magnitude. These slow changes eventually reshaped the
Jacobian enough for Newton to exploit, triggering explosive convergence.

### What Actually Happened

**The phase transition was Newton-specific, not ePC-specific.**

When we replaced Newton with SGD (lr=0.001, T=5), the phase transition vanished
entirely. On task 1b:
- **SGD**: 96% test accuracy on epoch 1. Final: 97.01%.
- **Newton**: 97.10% final, but slower to converge (passed 96% at epoch 3).
- **CG**: 96.96% final. Slowest, most expensive per batch.

SGD doesn't have a "Jacobian quality threshold." It simply follows the gradient
of the energy E with respect to errors. There's no rank-1 approximation, no
damping, no Woodbury formula. The gradient always points in a useful direction,
even with random weights. No circular dependency, no deadlock, no plateau.

### Failed Attempts to Fix Newton (All Unnecessary)

| Attempt | Mistake # | What Happened |
|---------|-----------|---------------|
| Init scale 2x | #27 | Blew up activations, loss 4846, stuck at 7% |
| mHC (hyperconnections) | #28 | 3x slower, wildly unstable, 32% final |
| muPC (Depth-muP) | #29 | Crushed Jacobian (alpha=0.031), 7% forever |
| Adaptive damping | #30 | Complete regression 99.2% -> 7.75% |
| CG optimizer | #32 | Autograd HVP through CE = NaN |
| AdaWoodbury | #22 | Rank-1 curvature correction, no benefit |

Every one of these was solving a problem that doesn't exist with SGD.

---

## The Real Breakthrough: Complete Error Coverage

### Principle

**Every error node and every position must receive a gradient signal.** Sparse
coverage creates "dead zones" where parameters can't learn. This principle
manifests in two dimensions:

1. **Layer dimension**: N errors for N blocks (not N-1)
2. **Temporal dimension**: next-step prediction at every position (not masked)

### Layer coverage: N-1 vs N errors

Early ePC-Mamba3 placed errors BETWEEN blocks only (N-1 errors for N blocks).
The last block had no error node, and therefore no local learning signal:

```
N-1 errors + no precision:  7% accuracy (random chance)
N-1 errors + geo precision: 38% (learning but starved)
N errors + geo precision:   99.3% (full coverage!)
```

The fix: place an error node AFTER every block, including the last one. This
ensures every block gets a local prediction error signal in E_local.

### Temporal coverage: masked vs next-step prediction

JEPA experiments revealed the same principle in the temporal dimension:

```
Stage 1b (multi-rule regime change):
  Masked prediction:    18.6% (positions with no causal context get no signal)
  Next-step prediction: 97.05% (every position predicts its successor)
```

Mamba is causal: position t only sees 0..t-1. Masking early positions leaves the
predictor with zero context. Next-step prediction gives every position a loss
signal using the maximum available causal context.

### Why 98.5% is the accuracy ceiling

Next-step prediction evaluates ALL positions, including position 0 (which
predicts position 1 with only 1 token of context). For structured sequences
with vocab_size=16, position 0's prediction is essentially random:

```
Expected accuracy = (1/16 + 62 * 1.0) / 63 ≈ 98.5%
```

Observed: 98.53% (Stage 1a). The model works PERFECTLY where prediction is
deterministic. The ~1.5% "error" is the unavoidable cost of evaluating
context-poor early positions.

### The unified principle

Both findings are instances of the same principle: **Mamba's dt parameter
(discretization step) needs gradient signal at every critical transition point.**

- Temporal: dt controls how Mamba's state transitions respond to each input
  token. Without loss signal at a position, dt can't learn that position's
  transition dynamics.
- Layer-wise: without an error node after a block, that block's contribution
  to dt has no local gradient signal.

Sparse signals leave "dead zones" where dt can't learn the transitions it needs.

---

## Current Architecture: SGD + Complete Coverage

### Error optimization (inference phase)

```python
# Simple SGD on error nodes
optim = torch.optim.SGD(errors, lr=0.001)
for t in range(T):  # T=5 iterations
    optim.zero_grad()
    E = pce.E(x, y, output_proj)  # 0.5 * sum(pi * ||e_i||^2) + CE
    E.backward()
    optim.step()
```

No Woodbury formula, no Hessian-vector products, no damping. The energy
landscape through Mamba's SSD computation is smooth enough that gradient descent
converges in 5 steps.

### Energy scaling

SGD with T=5 and lr=0.001 produces small errors. To compensate:

```python
energy_scale = min(1.0, e_lr * iters)  # = 0.005 for SGD
E_local_normalized = E_local / (batch_size * energy_scale)
```

This ensures weight gradients have appropriate magnitude despite small errors.

### Precision weighting (Salvatori et al. 2025)

Geometric precision: earlier layers get exponentially higher precision. With
base=3.0 and 4 layers, normalized precisions are approximately [3.7, 1.2, 0.4, 0.1].

This amplifies early-layer error contributions in E_local, preventing the
gradient imbalance where later layers (closer to the CE loss) dominate learning.

### iPC (incremental PC)

Interleaves error steps and weight steps: each SGD step on errors is immediately
followed by a weight update via E_local. This increases the rate of weight change
by T times, helping escape any early deadlock.

```
Standard ePC: T error steps -> 1 weight step
iPC:          T x (1 error step -> 1 weight step)
```

---

## ePC vs Backprop Comparison

### Where ePC wins

1. **Local credit assignment**: each layer learns from its own prediction error,
   not from a gradient traversing the entire network
2. **No vanishing gradients**: Layer 1's learning signal doesn't pass through
   layers 2-4 (E_local detaches between layers)
3. **Reduced interference**: layer changes don't propagate through the gradient
   path to disrupt other layers (Song et al. 2024, prospective configuration)
4. **Biological plausibility**: local Hebbian-like learning rules

### Where backprop wins

1. **Simpler**: no inference phase, no error nodes, no energy scaling
2. **Faster per-batch**: no T iterations of error optimization
3. **Smoother convergence**: monotonic improvement from epoch 1

### On the copy task

Both reach ~99% accuracy. Backprop converges smoothly. ePC with SGD also
converges smoothly (the Newton phase transition is gone). The remaining
question is whether ePC's local credit assignment provides advantages on
harder tasks where backprop's global gradient path breaks down.

---

## Open Questions

1. **Does ePC's local credit assignment help on genuinely hard tasks?** The
   copy task and structured sequence tasks are solvable by both ePC and backprop.
   Tasks requiring deep compositional reasoning might expose differences.

2. **What is the optimal T (error iterations)?** T=5 works well, but the
   relationship between T, e_lr, and convergence quality hasn't been
   systematically explored with SGD.

3. **Does iPC consistently outperform standard ePC with SGD?** The T-fold
   increase in weight updates per batch may not always help — it depends on
   whether error quality at step 1 is sufficient for useful weight gradients.

4. **Can error coverage insights improve other PC architectures?** The
   "complete coverage" principle (N errors for N layers, loss at every position)
   may generalize beyond Mamba3 to other ePC applications.

5. **How does ePC-Mamba3 with SGD scale to deeper networks?** With 4 layers
   and geometric precision, convergence is fast. With 8-16 layers, the precision
   ratio between first and last layer grows exponentially — does this remain
   stable?

---

## References

- Goemaere et al. 2025, arXiv:2505.20137 -- ePC framework
- Salvatori et al. 2025, arXiv:2506.23800 -- precision weighting for deeper PC
- Song et al. 2024, Nature Neuroscience -- prospective configuration
- Innocenti et al. 2025, arXiv:2505.13124 -- muPC (Depth-muP for PC)
- ARM (ICLR 2025) -- autoregressive Mamba, next-step prediction alignment
