# Critical Bug Found: Inference Not Implementing Gradient Descent

**Created:** 2026-01-24
**Status:** Root cause identified

## The Problem

Higher layers frozen (~0% weight change) because **inference doesn't minimize prediction error**.

## Root Cause

### What We're Doing (WRONG)

In `neuron.py:85-93`, during inference we compute:

```python
apical_activity = torch.tanh(self.W_apical @ apical_input)
basal_activity = torch.tanh(self.W_basal @ basal_input)
self.state = gate * apical_activity + (1 - gate) * basal_activity
```

This is just a **weighted combination** of top-down and bottom-up signals. No error minimization.

### What We Should Be Doing (CORRECT)

From Millidge et al. (2022), page 3, Equation 1:

```
Inference dynamics:
x˙_l = -∂F/∂x_l = -ε_l + ε_{l+1} · f'(W_{l+1}x_l)W^T_{l+1}

Where:
- ε_l = x_l - f(W_l x_{l-1})  (prediction error)
- ε_{l+1} = x_{l+1} - f(W_{l+1} x_l)  (error from layer above)
```

**Inference should update states via gradient descent on prediction errors.**

## Evidence

1. **From Millidge paper:**
   - Page 2: "first letting the neural activities update towards the configuration that minimizes the sum of prediction errors"
   - Page 3, Equation 1: Shows explicit gradient descent dynamics
   - Page 14: "the inference phase is still a major computational bottleneck" (because it requires many iterations)

2. **From Song et al. GitHub:**
   - Uses T=512 inference iterations (we use 20)
   - Separate optimizer for states during inference
   - States must converge BEFORE weight update

3. **Our symptoms:**
   - Only layer 0 learns (directly receives error from sensory input)
   - Higher layers frozen (states don't settle to error-minimizing values)
   - Minimal error reduction (6.6%)

## The Fix

Implement proper predictive coding inference:

```python
def _inference_step(self):
    """Update states via gradient descent on free energy."""
    for i, layer in enumerate(self.layers):
        # Compute prediction errors
        if i == 0:
            # Bottom error: sensory input - prediction from layer
            bottom_error = self.input_buffer - layer.compute_prediction_for_below()
        else:
            # Bottom error: state of layer below - prediction from this layer
            bottom_error = self.layers[i-1].get_state() - layer.compute_prediction_for_below()

        if i == len(self.layers) - 1:
            # Top layer: no layer above, so top error = 0
            top_error = 0
        else:
            # Top error: this layer's state - prediction from layer above
            top_error = layer.get_state() - self.layers[i+1].compute_prediction_for_below()

        # Gradient descent: x_l += lr_inference * (-bottom_error + top_error * f'(...) * W^T)
        # Update layer state
```

## What This Means

1. **The architecture doc was wrong** (written by Gemini, confabulated)
2. **Linear code was based on wrong assumptions** (correctly deleted)
3. **Current implementation isn't doing predictive coding properly** - it's just a two-stream feedforward network
4. **Need to rewrite inference to actually minimize prediction error**

## References

- Millidge et al. (2022), Section 2, Equations 1-2
- Song et al. (2024) GitHub: https://github.com/YuhangSong/Prospective-Configuration
- Our `docs/planning/millidge paper.pdf`, pages 2-3

## Next Steps

1. Implement correct inference dynamics
2. Use separate learning rates for inference (states) vs learning (weights)
3. Increase inference iterations (50-100 minimum, possibly 512 like Song et al.)
4. Re-test with proper predictive coding
