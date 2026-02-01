# Testing Notes - Uncertainty Term Fix

## Changes Made

### 1. Uncertainty Term Structure Fix (bayesian_pc_layer.py:208-210)
```python
# Changed from:
x_norm_sq = (x ** 2).sum()  # One scalar for entire batch
uncertainty_term = 0.5 * d_y * avg_V_entry * x_norm_sq

# To:
x_norm_sq_per_sample = (x ** 2).sum(dim=1)  # [batch_size] - per-sample norms
uncertainty_term = 0.5 * d_y * avg_V_entry * x_norm_sq_per_sample.sum()
```

**Note:** Numerically equivalent for this specific formula, but structurally correct for:
- Proper gradient flow per sample
- Variable batch sizes
- Future extensions (per-sample weighting, masking)

### 2. Restored Paper's Hyperparameters
- `prior_Psi_scale`: 1.0 → 1000.0 (back to paper's default)
- Removed arrogant "causes 258k precision!" comment

## Expected Results

With Ψ=1000 and ν=258, the prior precision will be:
- **Average diagonal precision**: 258,000 (ν × Ψ_scale)
- **Total trace**: 66,048,000 (258 × 1000 × 256 dimensions)

This means energies will be much higher than baseline PC (which uses precision=1).

### Diagnostic Output to Check

Run: `python experiments/BayesianPC/diagnose_bpc.py`

**Look for:**
1. **Energy scale**: Still large (~millions), but should be reasonable for high precision
2. **Inference convergence**: ΔF should be > 0 (not diverging)
3. **Natural params**: Should still be buffers (requires_grad=False) ✓
4. **Precision**: Average diagonal ~258,000

## Remaining Question

**Why does the paper's code work with Ψ=1000 but mine struggles?**

Possible reasons:
1. **Inference LR scaling**: May need to scale learning rate with precision
   - With 258,000x precision, errors weighted 258,000x more heavily
   - Gradients will be 258,000x larger
   - May need inference_lr = 0.01 / 258,000 = 3.87e-8?

2. **Energy normalization**: Paper may normalize by precision somehow

3. **Wishart parameterization**: Double-check if I'm using correct convention
   - Current: Σ^{-1} ~ Wishart(Ψ, ν) implies E[Σ^{-1}] = ν·Ψ
   - Alternative: Σ ~ Inv-Wishart(Ψ, ν) implies E[Σ] = Ψ/(ν-d-1)
   - Need to verify which the paper uses

4. **Architecture differences**: Paper uses 4 layers, 128 units per layer
   - My implementation: 7 layers, 256 units
   - Larger dimensions = larger energy scale

5. **Initialization**: Paper may initialize value nodes differently
   - Current: Initialize at E[W]·x + bias
   - Alternative: Initialize closer to supervision signal?

## Next Steps

1. **Test current fix**: Run training and see if it works with Ψ=1000
2. **If still failing**, try:
   - Match paper architecture exactly (4 layers, 128 units, T=10)
   - Scale inference LR: `inference_lr = 0.01 * (1.0 / Ψ_scale)`
   - Check Wishart parameterization in paper
3. **If working**, document what made the difference

## To Run Tests

```bash
# Pull latest changes
git pull origin claude/understand-current-files-VkmbS

# Run diagnostic
python experiments/BayesianPC/diagnose_bpc.py

# Run training (if diagnostic looks reasonable)
python experiments/BayesianPC/train_mnist_bayesian.py
```
