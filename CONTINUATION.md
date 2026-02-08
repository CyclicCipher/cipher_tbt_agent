# Continuation Notes — Phase 5 Low-Rank eBPC

## Where We Are

**Best result so far:** 82.59% test accuracy (3 epochs) with Gershgorin inflation + proportional k.
**Target:** 95.74% (full eBPC baseline).

The code currently has the FITC correction committed (d4a5e82) which NaN'd. It needs to be reverted or replaced with a hybrid approach.

## The Core Problem

We need to approximate η1 (weight precision matrix, PSD) as diag(d) + U·U^T for computational savings. When we truncate to rank-k, the residual R = η1 - (diag(d_base) + U·U^T) must be absorbed into the diagonal d somehow.

Three approaches tried:

| Approach | Layer 1 (in=785) | Layers 2-4 (in=129) | Result |
|----------|-------------------|----------------------|--------|
| **Spectral norm** (d += λ_max(R) uniform) | Way too conservative (d=633 uniform for all 785 dims) | Fine | 80.44% (k=20), 84.11% (k=50) |
| **Gershgorin** (d_i += Σ_j \|R_ij\|) | Still too conservative (Layer 1 |M|=0.03 vs 0.05 needed) | Fine | 82.59% (k=98, Psi=1.0) |
| **FITC** (d_i += R_ii only) | Perfect — residual is prior-dominated | **FATAL** — V explodes, M→1e17, NaN at batch 15 | 9.80% (NaN) |

## Why FITC Failed for Layers 2-4

FITC diag(R) doesn't guarantee η1_approx ≥ η1_true in PSD sense. It works only when the residual is approximately diagonal (prior-dominated).

- Layer 1 (in=785, k=98): data rank ≤ 128 (batch size). k=98 captures most data directions. Residual is mostly diagonal prior. diag(R) ≈ R. **Safe.**
- Layers 2-4 (in=129, k=20): data rank ≤ 128 ≈ 129. Data can fill the entire space! k=20 drops 109 data-driven eigenvalues with strong off-diagonal coupling. diag(R) << R. **Explodes.**

The Phi clamp at 1.0 didn't help because the explosion happens in M = η2 · V, not in the precision pathway.

## What to Try Next

### Option A: Hybrid Approach (Most Promising)
Use FITC for layers where the residual is prior-dominated, and Gershgorin (or spectral norm) for layers where it isn't.

Detection criterion: if `k >= batch_size` or `k >= in_features * 0.5`, FITC is safe. Otherwise use Gershgorin.

For current settings: Layer 1 (k=98, batch=128) → FITC. Layers 2-4 (k=20, batch=128) → Gershgorin.

This should give Layer 1 the freedom it needs (FITC's tight correction) while keeping Layers 2-4 stable (Gershgorin's PSD guarantee).

### Option B: Increase k for Layers 2-4
Use ratio=1 (k=in_features) for small layers. For in=129, this means k=129 — effectively full rank, no truncation, no residual. Only use low-rank for Layer 1 where it actually saves computation.

Downside: no compression for small layers. But they're small anyway (129² = 16K params).

### Option C: Full η1 for Small Layers
If in_features ≤ threshold (e.g., 256), store full η1 and use exact inversion. Low-rank only for layers where in_features is large enough to benefit.

Requires refactoring to support both full and low-rank layers in the same network.

### Option D: Interpolation
d_correction = diag(R) + α · (gershgorin(R) - diag(R)) with α ∈ (0,1). Tuning α is ugly but could work.

### Option E: Something from the LRPD Papers
The "Beyond Low Rank" paper (Yeon & Anitescu 2025) proposes an alternating algorithm. The Bach 2024 paper uses tangent space projection. One of these might give a tighter-than-Gershgorin but still-safe diagonal correction.

## Other Observations from Debugging

### T=5 vs T=20 Comparison
- T=20 gave WORSE results (79.47% vs 82.59% at T=5)
- More inference with wrong precision → worse, not better
- Train accuracy DEGRADED across epochs with T=20 (88.41% → 81.92%)
- Incomplete inference acts as regularization against misspecified precision
- **Fix the inflation first, then T sensitivity should vanish**

### Frozen Phi Problem
- With prior_Psi_iw_scale=1000: η3_prior ≈ 130,000, Phi stuck at prior
- With prior_Psi_iw_scale=1.0: η3_prior ≈ 130, ss3 ≈ 40 is 30% of prior
- Current setting is 1.0, which should let Phi evolve once inflation is fixed

### Layer 1 Is the Bottleneck
- With Gershgorin: Layer 1 max|M| = 0.03 (vs full eBPC's 0.05)
- Weight posterior uncertainty for Layer 1 is ~10x higher than other layers
- Layers 2-4 are fine with current k=20 and Gershgorin

## Key Files

- `experiments/eBPC_ResNet/ebpc_lowrank_trainer.py` — _update_eta1_lowrank (where the correction happens)
- `experiments/eBPC_ResNet/ebpc_lowrank_layer.py` — LowRankeBPCLayer, Woodbury, Schur complement, Phi clamp
- `experiments/eBPC_ResNet/validate_lowrank_mnist.py` — main validation script
- `experiments/eBPC_ResNet/diagnose_lowrank.py` — diagnostic tests (v3)
- `experiments/eBPC/ebpc_trainer.py` — WORKING full-rank eBPC trainer (reference)
- `experiments/eBPC/ebpc_layer.py` — WORKING full-rank eBPC layer (reference)
- `MISTAKES.md` — 19 mistakes, #18 (quadratic constraint) and #19 (FITC failure) are most recent

## Research Papers (in docs/research/)
- Quiñonero-Candela & Rasmussen 2005: FITC/DTC/PITC unifying view for sparse GPs
- Snelson & Ghahramani 2006: Sparse GP pseudo-inputs (FITC origin)
- Bonnabel/Bach 2024: LRPD for Riccati-like matrix DEs
- Yeon & Anitescu 2025: Spectral LRPD alternating algorithm with convergence proof
- LoRA/continual learning papers (for future use)
- Epperly blog: https://www.ethanepperly.com/index.php/2022/10/11/low-rank-approximation-toolbox-nystrom-approximation/

## Performance Progression

| Config | Test Acc | Notes |
|--------|----------|-------|
| Diagonal eBPC | 9.80% | NaN at batch 5 |
| Low-rank k=20, diag(R) | NaN | batch 12 |
| Low-rank k=128, diag(R) | NaN | batch 12 |
| Low-rank k=20, spectral norm | **80.44%** | stable, Layer 1 frozen |
| Low-rank k=50, spectral norm | **84.11%** | still climbing |
| Low-rank k=98, Gershgorin, Psi=1.0 | **82.59%** | stable, Layer 1 still frozen |
| Low-rank k=98, FITC, Psi=1.0 | NaN | batch 15 (Layers 2-4 explode) |

## Git State

Branch: `claude/review-project-context-CJ79R`
Last commit: `d4a5e82` (FITC correction — NaN'd, needs fix)
The FITC code is committed. Next step: implement hybrid or revert to Gershgorin and fix differently.

## User's Broader Vision

The user sees LRPD math as universally applicable beyond just BPC:
1. Second-order optimizers (LRPD Hessian/Fisher)
2. JEPA inference with uncertainty
3. Mamba state transitions
4. LoRA improvements
5. Matrix inversion at runtime

Keep file/class structure modular so LRPD tools can be reused across applications.
