# Long-Run Training Dynamics Analysis

**Date:** 2026-01-24
**Session:** claude/learn-project-docs-TnvRZ

## User's Observations (400-Iteration Run)

```
Iteration 0:   error=920.9
Iteration 30:  error=61.3   (first valley)
Iteration 100: error=31.4   (deeper)
Iteration 190: error=457.6  (climbs back!)
Iteration 285: error=1.5    (near-perfect!)
Iteration 320: error=0.16   (BEST)
Iteration 400: error=78.4   (slight drift)

Weight shrinkage: 86-98% over 400 iterations
Final reduction: 91.5%
```

**Non-monotonic convergence** with extreme oscillations, but eventually finds excellent solution.

## Answers to User's Questions

### Q1: Do we use enough inference iterations?

**Test Results:**
- With zero initialization, inference error: 35.9 → 0.054 over 200 iterations
- At iteration 50 (our current setting): error = 2.03
- Never fully "converges" - continues improving slowly

**Conclusion:**
- **50 iterations is reasonable but not optimal**
- Error is still 2.03 vs final 0.054 (38x higher)
- Could benefit from 100-200 iterations for better convergence
- However, diminishing returns after 50

**Tradeoff:**
- More iterations = better inference = slower training
- Current 50 is a pragmatic compromise

**Recommendation:**
- Keep 50 for now
- Consider adaptive: start with 100, reduce to 50 as training progresses

### Q2: Should we exit inference early when equilibrium reached?

**Test Results:**
- Inference never truly "converges" (within 0.1% threshold) even after 200 iterations
- Asymptotically approaches minimum like gradient descent with fixed LR

**Analysis:**

| Approach | Pros | Cons |
|----------|------|------|
| **Fixed iterations** | Simple, predictable, no overhead | May waste compute |
| **Early stopping** | Efficient, adaptive | Overhead of checking, complexity |

**Recommendation:**
- **For TRAINING:** Use fixed iterations (current approach) ✓
- **For DEPLOYMENT:** Use early stopping for efficiency
- Early stopping overhead likely not worth it for training

### Q3: Current network architecture

**Structure:**
```
Input (1000) → Layer0 (100) → Layer1 (100) → Layer2 (100) → Layer3 (100)
```

**Connectivity:**
- Fully connected between adjacent layers
- Each layer has:
  - W_basal (bottom-up): connections from layer below
  - W_apical (top-down): connections from layer above

**Weight Shapes:**
- Layer 0: W_basal (100, 1000), W_apical (100, 100)
- Layer 1-3: W_basal (100, 100), W_apical (100, 100)

**Total parameters:** 170,000

**Task:** Autoencoding 1000-dimensional input through 100-dimensional bottleneck

### Q4: Should we try deeper networks?

**Test Results (50 iterations):**

| Architecture | Parameters | Min Error | Final Error | Reduction |
|--------------|-----------|-----------|-------------|-----------|
| **5×100 (current)** | 170,000 | 50.6 | 111.6 | 88.1% |
| 10×50 | 92,500 | 129.3 | 129.3 | 86.8% |
| 20×25 | 48,125 | 393.4 | 393.4 | 60.5% |

**Observations:**
1. **Shallower & wider wins** for this task
2. Deeper networks learn slower (need more iterations)
3. Very deep (20 layers) struggles significantly

**Why shallow works better here:**
- Task is simple autoencoding (compression → reconstruction)
- Doesn't need hierarchical feature learning
- Bottleneck is at capacity, not depth

**When to use deeper networks:**
- Vision tasks (spatial hierarchies)
- Language (semantic hierarchies)
- Complex structured data
- With skip connections to help gradient flow

**Recommendation:** Keep 5 layers for current task

## Critical Issue: Long-Term Instability

### The Problem

All weight decay values show the same pattern:
1. Find good minimum (error ~50-85) at iteration 30-40
2. Then diverge (error climbs to 400-2700) by iteration 200

**Test Results:**

| Weight Decay | Min Error (iter) | Final Error | Weight Change | Oscillations |
|--------------|------------------|-------------|---------------|--------------|
| 0.01 | 53.3 (34) | 415.6 | -36.0% | 124 |
| 0.005 | 71.0 (33) | 647.1 | -23.9% | 119 |
| 0.001 | 84.8 (33) | 2766.7 | +31.3% | 149 |
| Adaptive | 50.4 (41) | 1670.3 | -9.3% | 142 |

**All decay values fail similarly!** This suggests the problem isn't just weight decay magnitude.

### Possible Root Causes

1. **Fixed learning rate is too high**
   - After finding minimum, same LR causes oscillations
   - Solution: Learning rate scheduling (decay LR over time)

2. **No momentum**
   - Plain gradient descent oscillates in narrow valleys
   - Solution: Adam optimizer (combines momentum + adaptive LR)

3. **Inference-learning mismatch**
   - Inference uses one LR (0.1), weights use another (0.0005)
   - These might be fighting each other
   - Solution: Tune inference_lr

4. **Insufficient inference convergence**
   - At iter 50, inference error still 2.03 (vs 0.054 final)
   - Weights updated based on non-equilibrium states
   - Solution: More inference iterations or adaptive stopping

### Connection to Optimizers (User's Insight)

You mentioned: "A lot of ideas from machine learning like gradient descent are applicable to predictive coding. Particularly, optimizers like Adam and Muon."

**This is exactly right!** Current implementation uses:
```python
W += lr * error * input - weight_decay * W  # Plain gradient descent + L2
```

**Adam would help because:**
- **Momentum:** Dampens oscillations
- **Adaptive learning rates:** Different LR per parameter
- **Better in non-convex landscapes:** Which predictive coding likely has

**PyTorch Adam analogy:**
```python
# Current (manual):
W += lr * gradient - weight_decay * W

# Adam equivalent:
# m = β1 * m + (1-β1) * gradient        # First moment
# v = β2 * v + (1-β2) * gradient²       # Second moment
# W += lr * m / (sqrt(v) + ε)           # Adaptive update
```

## Recommendations for 400+ Iteration Runs

### Immediate Fixes:

1. **Use Adam optimizer** for weight updates:
   ```python
   optimizer = torch.optim.Adam(network.parameters(), lr=0.0005, weight_decay=0.01)
   ```

2. **Learning rate scheduling:**
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
   ```

3. **Increase inference iterations** to 100-200 for better equilibrium

4. **Gradient clipping** to prevent sudden jumps:
   ```python
   torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
   ```

### Why User's 400-Iteration Run Eventually Succeeded

Despite oscillations, error reached 0.16 at iteration 320! Possible reasons:

1. **Lucky trajectory:** Oscillations eventually landed in good basin
2. **Weight decay helped:** Shrinking weights forced efficient representations
3. **Long enough time:** 400 iterations allowed exploration

But this is **inefficient and unreliable**. With proper optimizers, should converge much faster.

## Summary Table

| Question | Current State | Issue | Recommendation |
|----------|--------------|-------|----------------|
| **Inference iters** | 50 | Error still 2.03 vs 0.054 | Increase to 100 or use adaptive |
| **Early stopping** | No | N/A | Keep fixed for training |
| **Architecture** | 5×100 (170k params) | Good for task | Keep current |
| **Deeper networks** | Not tested | Worse for autoencoding | Use for vision/language |
| **Weight decay** | 0.01 | Too aggressive + plain GD unstable | Lower to 0.005 + use Adam |
| **Optimizer** | Manual GD | Oscillates after finding minimum | **Switch to Adam!** |
| **LR scheduling** | None | Same LR throughout | Add decay or plateau detection |

## Next Steps

**Priority 1: Implement Adam optimizer**
- Replace manual weight updates with Adam
- Should dramatically improve long-term stability

**Priority 2: Learning rate scheduling**
- Reduce LR when error plateaus
- Prevents oscillations after finding good minimum

**Priority 3: Test longer inference**
- Try 100-200 iterations to see if better equilibrium helps

**Priority 4: Consider Muon optimizer** (user mentioned)
- Newer optimizer designed for neural networks
- May be even better than Adam for predictive coding

## Code Example: Adding Adam

```python
# In backbone.py __init__:
self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=0.01)

# Replace update_weights():
def update_weights(self):
    """Update weights using Adam optimizer."""
    # Compute gradients manually (since we're not using autograd)
    for i, layer in enumerate(self.layers):
        # Get error and inputs (same as before)
        layer_error = ...
        input_from_below = ...
        input_from_above = ...

        # Store gradients in .grad attribute
        error_col = layer_error.unsqueeze(1)
        layer.neurons.W_basal.grad = -(error_col @ input_from_below.unsqueeze(0))
        layer.neurons.W_apical.grad = -(error_col @ input_from_above.unsqueeze(0))

    # Adam step (handles momentum, adaptive LR, weight decay automatically)
    self.optimizer.step()
    self.optimizer.zero_grad()
```

This should eliminate the oscillations and make 400-iteration runs converge smoothly!
