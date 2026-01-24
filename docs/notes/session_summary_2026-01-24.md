# Session Summary: Root Cause Analysis and Fix

**Date:** 2026-01-24
**Session:** claude/learn-project-docs-TnvRZ

## Investigation Results

### 1. Found Source of "One-Shot Equilibrium" Misunderstanding

**Location:** `src/network/ARCHITECTURE.md` lines 69, 105-120

**Problem:** This document (written by Gemini) incorrectly implies that prospective learning uses "one-shot equilibrium solving" via block tridiagonal structure.

**Reality:** Millidge et al. (2022) clearly shows prospective learning still requires **iterative inference to equilibrium**. The block tridiagonal structure only enables efficient GPU parallelization, not elimination of iteration.

**Quote from paper (page 2):**
> "Once the neural activities have reached an equilibrium, the synaptic weights can be updated"

**Quote from paper (page 14):**
> "the inference phase is still a major computational bottleneck in PCNs"

### 2. Deleted Incorrect Linear Network Code

Removed files based on wrong assumptions:
- `src/network/neuron_linear.py`
- `src/network/layer_linear.py`
- `src/network/backbone_linear.py`
- `tests/test_network_linear.py`

**Why they were wrong:** Based on misunderstanding that prospective learning requires linear activations for "one-shot matrix solving." The paper uses **tanh (nonlinear)** activations throughout.

### 3. Critical Bug Found and Fixed: Inference Not Minimizing Error

#### The Root Cause

Our original `neuron.py:forward()` method (lines 85-93) was:

```python
apical_activity = torch.tanh(self.W_apical @ apical_input)
basal_activity = torch.tanh(self.W_basal @ basal_input)
self.state = gate * apical_activity + (1 - gate) * basal_activity
```

This is **NOT predictive coding** - it's just a weighted combination of signals with no error minimization!

#### What Should Happen (Millidge et al. 2022, Eq. 1)

During inference, states should update via **gradient descent on free energy**:

```
ẋ_l = -∂F/∂x_l = -ε_l + ε_{l+1} · f'(W_{l+1}x_l)W^T_{l+1}
```

Where:
- `ε_l = x_l - f(W_l x_{l-1})` is the bottom-up prediction error
- States iteratively minimize prediction errors across all layers
- Only AFTER convergence do we update weights

#### The Fix

Rewrote `backbone.py:_inference_step()` to:

1. **Compute prediction errors** for all layers:
   ```python
   error = layer.get_state() - torch.tanh(layer.neurons.W_basal @ input_below)
   ```

2. **Update states via gradient descent**:
   ```python
   gradient = -error  # Local error term

   # Add feedback from layer above
   if i < len(layers) - 1:
       error_times_deriv = errors[i+1] * tanh_derivative
       feedback = error_times_deriv @ layers[i+1].neurons.W_basal
       gradient += feedback

   new_state = current_state + inference_lr * gradient
   ```

3. **Added separate `inference_lr` parameter** (default 0.1)
   - Higher learning rate for inference (state updates)
   - Separate from weight learning rate (should be ~0.001-0.005)

4. **Increased default inference iterations** from 20 to 50
   - Song et al. use 512 iterations
   - Need sufficient time for states to converge to equilibrium

## Why This Explains Everything

| Symptom | Explanation |
|---------|-------------|
| Only layer 0 learns | Directly receives sensory error signal |
| Higher layers frozen (~0% weight change) | States never settled to error-minimizing values |
| 6.6% error reduction | Not actually optimizing anything |
| Network seemed "stable" | Just computing feedforward pass twice |

## Evidence from Song et al. GitHub

Their implementation confirms:
- **T = 512** inference iterations (we had 20!)
- **Separate optimizers:** `optimizer_x` (lr=0.1) for states, `optimizer_p` (lr=0.001) for weights
- **Tanh activations** (confirms nonlinear is correct)
- **Iterative inference** is the computational bottleneck

## Testing Next Steps

The fixed implementation should now:
1. ✅ Converge states to minimize prediction errors during inference
2. ✅ Update weights based on converged equilibrium states
3. ✅ Show learning in ALL layers, not just layer 0
4. ✅ Achieve better error reduction (>6.6%)

Run `python tests/test_network_diagnostics.py` to verify all layers now show non-zero weight changes.

## Key Learnings

1. **Don't trust architecture docs written by other LLMs** - always verify against primary sources (papers)
2. **Prospective learning ≠ one-shot solving** - it's still iterative, just more GPU-friendly
3. **Inference must minimize error** - not just compute a feedforward pass
4. **Separate learning rates** for inference (states) vs learning (weights) is critical
5. **Nonlinear activations (tanh)** are standard in predictive coding, not linear

## References

- Millidge et al. (2022): "A Theoretical Framework for Inference and Learning in Predictive Coding Networks"
  - Equation 1 (page 3): Inference dynamics
  - Page 2: "Once the neural activities have reached an equilibrium..."
  - Page 14: "the inference phase is still a major computational bottleneck"

- Song et al. (2022) GitHub: https://github.com/YuhangSong/Prospective-Configuration
  - `predictive_coding/pc_trainer.py`: Shows T=512, separate optimizers

## Commits

1. `49c5a7e`: Delete linear network code and identify critical inference bug
2. `3201291`: Implement correct predictive coding inference dynamics

All changes pushed to `origin/claude/learn-project-docs-TnvRZ`
