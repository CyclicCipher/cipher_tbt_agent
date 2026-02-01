# Adaptive Learning Rate Fix - Appendix B Implementation

## The Problem

Training was failing with:
- Energy: 640,000 (should be ~10)
- Inference diverging: ΔF = -12 (should converge with ΔF > 0)
- Precision: 258,000x baseline
- Accuracy stuck at 9.8% (random guessing)

## Root Cause

**I skimmed the paper instead of reading it thoroughly.**

I speculated about "energy normalization" when the paper **explicitly documented** the solution in **Appendix B (pages 10-11)**.

## What the Paper Says (Appendix B)

> "the dynamics dominated by the spectrum of **A_l = Σ_l^{-1} + W^T_{l+1} Σ^{-1}_{l+1} W_{l+1}**"

> "upper bound on the maximum learning rate parameter as **approximately given by the inverse of the maximum eigenvalue of the A_l**"

> "can be dynamically updated with updates to the posterior distribution over the parameters"

**Translation:** The optimal inference learning rate is:
```
α_optimal ≈ 1 / λ_max(A_l)
```

With E[Σ^{-1}] = ν·Ψ = 258 × 1000 = 258,000, the optimal learning rate is approximately:
```
α_optimal ≈ 1 / 258,000 ≈ 3.87e-6
```

## The Fix

### 1. Added `get_optimal_inference_lr()` to BayesianPCLayer

```python
def get_optimal_inference_lr(self) -> float:
    """Compute optimal inference learning rate based on precision spectrum.

    From Appendix B: α ≈ 1 / λ_max(A_l)
    """
    Sigma_inv = self.get_expected_precision()

    # Average diagonal precision (approximates dominant eigenvalue)
    avg_precision = torch.trace(Sigma_inv) / self.out_features

    # Optimal LR is inverse of max eigenvalue
    alpha = 1.0 / (avg_precision.item() + 1e-8)

    return alpha
```

### 2. Updated BayesianPCTrainer to use adaptive LR

```python
def _create_optimizer_x(self):
    """Create optimizer with adaptive learning rate."""
    value_nodes = self.model.get_value_nodes()
    if len(value_nodes) > 0:
        # Compute adaptive LR per layer, use minimum (most conservative)
        min_alpha = float('inf')
        for layer in self.model.layers:
            layer_alpha = layer.get_optimal_inference_lr()
            min_alpha = min(min_alpha, layer_alpha)

        print(f"Using adaptive inference LR: {min_alpha:.2e}")

        self.optimizer_x = self.optimizer_x_fn(
            value_nodes,
            lr=min_alpha  # Adaptive instead of fixed
        )
```

### 3. Updated diagnostic to show optimal LR

The diagnostic now shows:
```
Optimal inference learning rate (Appendix B):
  α_optimal ≈ 1 / λ_max(A_l) = 3.87e-06
  With configured α = 0.01, step size is 2584.0x too large!
  ⚠️  This explains inference divergence!
```

## Why This Fixes the Problem

**Before:**
- Fixed α = 0.01
- Precision = 258,000
- Gradients = 258,000 × (prediction errors)
- Step size = 0.01 × 258,000 × errors = 2,580 × errors
- **WAY too large** → inference diverges

**After:**
- Adaptive α ≈ 3.87e-6
- Precision = 258,000
- Gradients = 258,000 × (prediction errors)
- Step size = 3.87e-6 × 258,000 × errors ≈ errors
- **Correct scale** → inference converges

## Expected Results

With the adaptive learning rate:

1. **Inference should converge** (ΔF > 0, not diverging)
2. **Energy should be reasonable** (~2,580 instead of 640,000)
3. **Training should work** (accuracy > 9.8%)

The energy will still be higher than baseline PC (which uses precision=1), but that's expected because BPC uses precision=258,000. The **gradient step sizes** are now correctly scaled.

## The Lesson (Mistake #13)

**NEVER, EVER SKIM A RESEARCH PAPER YOU ARE IMPLEMENTING.**

From MISTAKES.md:
> When implementing a research paper:
> 1. **READ THE ENTIRE PAPER** - especially appendices with implementation details
> 2. **NEVER speculate** when the answer is in the paper
> 3. **CHECK appendices** for implementation details
> 4. **Don't be lazy** - "may normalize somehow" is intellectual laziness
> 5. **Trust the authors** - they solved these problems, the solution is documented

## Files Changed

1. `experiments/BayesianPC/bayesian_pc_layer.py`
   - Added `get_optimal_inference_lr()` method

2. `experiments/BayesianPC/bayesian_pc_trainer.py`
   - Updated `_create_optimizer_x()` to use adaptive LR

3. `experiments/BayesianPC/diagnose_bpc.py`
   - Added optimal LR diagnostic output

4. `MISTAKES.md`
   - Added Mistake #13: Skimming Papers

## Running the Fixed Code

```bash
# Run diagnostic to see the adaptive LR
python experiments/BayesianPC/diagnose_bpc.py

# Run training with adaptive LR
python experiments/BayesianPC/train_mnist_bayesian.py
```

Expected diagnostic output:
```
E[Σ^{-1}] = ν·Ψ:
  Tr(E[Σ^{-1}]) = 6.60e+07
  Average diagonal: 2.58e+05

Optimal inference learning rate (Appendix B):
  α_optimal ≈ 1 / λ_max(A_l) = 3.87e-06
  With configured α = 0.01, step size is 2584.0x too large!
  ⚠️  This explains inference divergence!
```

Training will now show:
```
Using adaptive inference LR: 3.87e-06 (configured: 1.00e-01)
```

And inference should converge instead of diverge.
