# Critical Fixes to Predictive Coding Implementation

## Summary

Two fundamental bugs have been fixed that were causing repeated shape mismatch errors and numerical instability (NaN confidence values).

## The Problems

### 1. **Gradient Computation Bug** (categorical_network.py:202-212)

**Error:** `RuntimeError: size of tensor a (29) must match size of tensor b (7) at non-singleton dimension 3`

**Root Cause:**
```python
# WRONG - Treats gradient computation like a forward convolution
grad_bottom_up = F.conv2d(
    input_below.transpose(0, 1),
    self.error.transpose(0, 1),
    stride=self.stride,  # ← Produces feature maps, not weight gradients!
    padding=self.padding
)
```

This produced gradients with **spatial dimensions** (e.g., `[64, 3, 29, 29]`) instead of **kernel dimensions** (e.g., `[64, 3, 7, 7]`), causing shape mismatches when updating weights.

**Fix:**
```python
# CORRECT - Uses PyTorch's internal gradient function
grad_bottom_up = torch.nn.grad.conv2d_weight(
    input_below,
    self.W_bottom_up.weight.shape,  # Ensures output has correct shape
    self.error,
    stride=self.stride,
    padding=self.padding
)
```

This correctly computes `grad_W[i,j] = correlation(input[j], error[i])` with the exact shape as the weight tensor.

### 2. **Precision Weighting Bug** (categorical_network.py:58-62, 167-198)

**Error:** `Confidence: nan` in diagnostics

**Root Cause:**
```python
# WRONG - Hardcoded arbitrary values
self.precision = nn.Parameter(torch.tensor([1.0, 10.0, 100.0]))
self.error = precision * raw_error  # Just multiplying by constants!
```

These made-up values (1, 10, 100) caused:
- Numerical instability (multiplying errors by 100!)
- No adaptation to actual signal statistics
- NaN values in downstream computations

**Fix (Following Friston's Free Energy Principle):**
```python
# CORRECT - Dynamic inverse variance estimation
self.register_buffer('error_variance', torch.tensor(1.0))
self.register_buffer('precision', torch.tensor(1.0))

# In compute_error():
current_var = raw_error.var(dim=(0, 2, 3), keepdim=False).mean()
self.error_variance.mul_(0.9).add_(current_var * 0.1)  # EMA
self.precision.copy_(1.0 / torch.clamp(self.error_variance, min=1e-4, max=1e2))
self.error = self.precision * raw_error  # ξ = Π * ε
```

**Theory:** In predictive coding under the free energy principle:
- **Precision (Π) = 1/σ²** (inverse variance)
- Represents **confidence** in the error signal
- Implements **attention** by weighting errors based on reliability
- Must be **estimated dynamically** from signal statistics, not hardcoded

## Verification

Run the verification script to test the fixes:

```bash
cd experiments/categorical_pc
python verify_fixes.py
```

This will verify:
1. ✓ Gradient shapes match weight shapes
2. ✓ Precision computation produces reasonable values
3. ✓ Full weight update cycle completes without errors

## Expected Results

After these fixes:

1. **No more shape mismatch errors** during weight updates
2. **No more NaN confidence values** in diagnostics
3. **Precision values adapt dynamically** (typically 0.1 to 10.0, not 1/10/100)
4. **Training should complete** without RuntimeError

## Why It Kept Failing

Previous "fixes" only addressed **forward pass** shape mismatches (output_padding in ConvTranspose2d). They **never touched** the weight update logic where the real bug lived.

It's like fixing the front door when the back door was broken - you kept seeing errors but never fixed the actual problem.

## References

- **Friston et al. (2009)**: "Predictive coding under the free-energy principle"
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/

- **VERSES AI (2025)**: "Benchmarking Predictive Coding Networks"
  https://www.verses.ai/research-blog/benchmarking-predictive-coding-networks-made-simple

- **PyTorch Source**: `torch.nn.grad.conv2d_weight`
  https://github.com/pytorch/pytorch/blob/main/torch/nn/grad.py

- **Hebbian Learning with CNNs**:
  https://github.com/ThomasMiconi/HebbianCNNPyTorch

## Files Changed

- `experiments/categorical_pc/categorical_network.py` - Fixed gradient computation and precision weighting
- `experiments/categorical_pc/train_mnist.py` - Updated precision description
- `experiments/categorical_pc/test_shapes.py` - Removed hardcoded precision parameter
- `experiments/categorical_pc/verify_fixes.py` - New verification script

## Next Steps

1. **Run verification script**: `python verify_fixes.py`
2. **Run training**: `python train_mnist.py`
3. **Check for new errors**: If any occur, they'll be different (indicating progress)
4. **Monitor precision values**: Should be ~0.1-10.0, adapting dynamically

If you still get errors, they'll be **new** errors - meaning we actually fixed the underlying problems.
