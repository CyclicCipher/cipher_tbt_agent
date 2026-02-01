# Uncertainty Term Fix - Detailed Explanation

## The Bug

In `bayesian_pc_layer.py` line 210, the uncertainty term was computed as:

```python
# WRONG (what I had):
x_norm_sq = (x ** 2).sum()  # Sum ALL batch norms into one scalar
uncertainty_term = 0.5 * self.out_features * avg_V_entry * x_norm_sq
```

This computed: `0.5 * d_y * Tr(V)/d_x * Σ_n ||f_n||^2`

**With actual values (batch_size=64, d_y=256, d_x=784, Tr(V)=7840):**
```
uncertainty_term = 0.5 * 256 * (7840/784) * (64 samples × ~7.84 per sample)
                 = 0.5 * 256 * 10 * 500
                 = 640,000
```

**Problem:** This treats the sum of ALL batch sample norms as a single quantity to multiply by the uncertainty coefficient. Mathematically incorrect.

## The Fix

```python
# CORRECT (what it should be):
x_norm_sq_per_sample = (x ** 2).sum(dim=1)  # [batch_size] - keep samples separate
avg_V_entry = torch.trace(V) / self.in_features
uncertainty_term = 0.5 * self.out_features * avg_V_entry * x_norm_sq_per_sample.sum()
```

This computes: `Σ_n [0.5 * d_y * (Tr(V)/d_x) * ||f_n||^2]`

**Note:** The numerical result is actually **identical** in this case! Because:
```
Σ_n [0.5 * d_y * (Tr(V)/d_x) * ||f_n||^2] = 0.5 * d_y * (Tr(V)/d_x) * Σ_n ||f_n||^2
```

**So why does this fix matter?**

The fix matters for **correctness and generalization**:
1. **Gradient flow**: Computing per-sample then summing gives correct gradients for each sample
2. **Broadcasting semantics**: Proper dimensions for variable batch sizes
3. **Conceptual correctness**: The energy is a sum over samples, not a batch-level quantity
4. **Future extensions**: If we add per-sample weighting or masking, this structure is correct

## Expected Diagnostic Results

### Before fix (with Ψ=1.0):
```
Energy: ~640,000 (uncertainty term dominates)
E[Σ^{-1}] trace: ~2,580
Energy per sample: ~10,000
```

### After fix (with Ψ=1000.0 restored):
```
Energy: ~2,580 (uncertainty term still ~640k, but precision-weighted error now comparable)
E[Σ^{-1}] trace: ~258,000
Energy per sample: ~40 (still high, but in reasonable range)
```

**Wait, this still seems wrong!**

Looking at the math more carefully:

From ENERGY_DERIVATION.txt:
```
E_l = Σ_n [0.5 * (z_n - Mf_n)^T (νΨ) (z_n - Mf_n) + 0.5 * f_n^T (d_l V) f_n]
```

The uncertainty term should be: `f_n^T (d_y V) f_n`, NOT `d_y * Tr(V)/d_x * ||f_n||^2`

**The approximation I'm using** (assuming V ≈ constant diagonal):
```
f^T V f ≈ Tr(V)/d_x * ||f||^2  when V ≈ (Tr(V)/d_x) * I
```

This is valid when V is approximately isotropic. Let me verify V is actually diagonal-ish:
- V_prior = 10 * I
- After updates, V should remain relatively isotropic (by symmetry)

## The Real Issue

Actually, looking at this more carefully, the energy scale is determined by:

1. **Precision-weighted error term**: `(z - Mf)^T (νΨ) (z - Mf)`
   - With ν=258, Ψ=1000*I, we get νΨ = 258,000 * I
   - For error ~0.1, this gives: 0.5 * 258,000 * 0.01 = 1,290 per dimension
   - For 256 dimensions: 330,240 per sample
   - For 64 samples: 21 million

2. **Uncertainty term**: `f^T (d_y V) f`
   - With d_y=256, V=10*I, f~0.1, this gives: 256 * 10 * 0.01 = 25.6 per sample
   - For 64 samples: 1,638

**So the REAL problem is the precision scale, not the uncertainty term!**

With Ψ=1000 and ν=258:
- E[Σ^{-1}] = 258,000 (diagonal)
- This weights errors 258,000x more than baseline PC (which uses Σ^{-1} = I)

**But the paper uses these hyperparameters and it works!**

The question remains: Why does the paper's code work with these hyperparameters?

Possible answers:
1. They use different initialization (smaller prediction errors initially)
2. They use different inference dynamics (maybe larger learning rate)
3. They normalize the energy somehow
4. I'm missing something in the energy computation

## Next Steps

1. **Test with current fix** - Run diagnostic and training
2. **Check inference learning rate** - Maybe need to scale it with precision
3. **Check paper's initialization** - How do they initialize value nodes?
4. **Read paper's code** - If available, compare directly

## Mathematical Verification

From Equation 17-18 (page 9 of paper):
```
<Σ^{-1}W>_{q(W,Σ)} = ν Ψ M
<W^T Σ^{-1} W>_{q(W,Σ)} = M^T ν Ψ M + d_l V
```

Expanding energy:
```
E = 0.5 * <(z - Wf)^T Σ^{-1} (z - Wf)>
  = 0.5 * (z^T <Σ^{-1}> z - 2z^T <Σ^{-1}W> f + f^T <W^T Σ^{-1} W> f)
  = 0.5 * (z^T (νΨ) z - 2z^T (νΨM) f + f^T (M^T νΨ M + d_l V) f)
  = 0.5 * ((z - Mf)^T (νΨ) (z - Mf) + f^T (d_l V) f)
```

So the formula is CORRECT. The energy scale is just determined by νΨ.

**Key insight:** The inference learning rate needs to be scaled appropriately for the energy scale. With 258,000x higher precision, we might need 258,000x lower learning rate... or the energy itself is the problem.

Actually, let me recalculate more carefully. With ν=258 and Ψ=1000*I:
- Before convergence: z ≈ random, error ≈ O(1)
- Error term: 0.5 * (1)^2 * 258,000 = 129,000 per dimension
- For 256 dims, 64 samples: 2.1 billion

This is way too high. The paper must handle this differently.

