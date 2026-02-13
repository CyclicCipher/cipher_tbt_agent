# CLAUDE.md — Project Memory

## What This Project Is

A biologically-inspired AI system built on predictive coding (PC), targeting the Danganronpa visual novel as an evaluation environment. The core research question: can energy-based local learning (ePC) replace backprop while enabling reasoning through energy minimization?

Current focus: **ePC-JEPA** — combining energy-based predictive coding with JEPA-style latent prediction on a Mamba3 backbone. This is the `experiments/energy_reasoning/` directory.

## Critical Reference

**ALWAYS read `MISTAKES.md` before making changes.** It has 35 documented mistakes with root causes. The most relevant active ones:

- **#35 (Reduction mismatch):** Error penalty must use mean reduction, matching output losses. Sum reduction crushes errors to zero. Fixed in all variants 2026-02-13.
- **#33 (SGD wins):** Don't use Newton or CG for error optimization. SGD is fastest, simplest, same accuracy.
- **#34 (Next-step prediction):** Causal models (Mamba) need next-step prediction, not masked prediction.
- **#13 (Read the paper):** Never skim a research paper you're implementing. Read every appendix.

## Architecture Overview

### ePC-JEPA (experiments/energy_reasoning/)

Two-phase training per batch (Goemaere et al. 2025, "Energy-based Predictive Coding"):

1. **Error phase (Phase 1):** Freeze weights. Initialize errors to zero. Run T steps of SGD on errors to minimize `E = ½Σ mean(ε_i²) + L(output)`. Errors are the free variables.

2. **Weight phase (Phase 2):** Freeze errors. Compute `E_local` with detached errors between layers. Each layer gets gradient only from its local MSE `½||f(s) - (f(s)+ε)||²`. Predictor/decoder get gradient from JEPA + decode CE + VICReg.

Key files:
- `epc_jepa_model.py` — Model with E(), E_local(), minimize_error_energy()
- `train_epc_jepa.py` — Training loop, data generation, diagnostics
- `jepa_model.py` — Backprop JEPA baseline (for comparison)
- `data_gen.py` — Synthetic sequence generation (stages 1a/1b/2)

### Other ePC Variants

| Directory | Architecture | Task | Status |
|-----------|-------------|------|--------|
| `experiments/Mamba3/` | Mamba3 + ePC | Copy/sequence | Working (99%+) |
| `experiments/ePC_ResNet/` | ResNet-18 + ePC | CIFAR-10 | Target: 92.17% |
| `experiments/ePC_Mamba/` | Mamba2 + ePC | Synthetic | Archived (superseded by Mamba3) |
| `experiments/eBPC/` | MLP + eBPC | MNIST | Working (95.74%) |
| `experiments/eBPC_ResNet/` | MLP + eBPC + low-rank V | MNIST | Debugging (NaN from PSD violations) |

### Backprop Baselines

- `experiments/energy_reasoning/jepa_model.py` + `train_jepa.py` — JEPA without ePC
- `experiments/Mamba3/train_epc.py --backprop` — Standard backprop Mamba3

## Hyperparameter Defaults (ePC-JEPA)

```
T = 20          # Error optimization iterations
e_lr = 0.1      # Error learning rate (SGD)
precision = none # Uniform (no layer weighting)
reduction = mean # ALL energy terms use mean reduction
ipc = false     # Standard ePC (not interleaved)
```

## Key Design Rules

1. **Reduction consistency:** Every term in E and E_local must use the same reduction (mean). Sum reduction on errors with mean on losses creates ~1Mx scale mismatch (Mistake #35).

2. **E and E_local must optimize the same objective:** E (error phase) includes JEPA + decode CE + VICReg. E_local (weight phase) includes the same terms. Errors are optimized against the full loss so they carry informative gradients.

3. **No precision weighting:** The paper uses uniform `½||ε||²`. Geometric precision (e.g., 2.7, 0.9, 0.3, 0.1) suppresses deep-layer errors — the exact pathology ePC fixes (Mistake #4).

4. **SGD for errors, Adam for weights:** SGD is optimal for error optimization. Adam handles per-element scale variation in weight gradients. Don't use Newton, CG, or other second-order methods for errors (Mistake #33).

5. **Standard ePC, not iPC:** T error steps to convergence, then one weight update. Interleaved PC (iPC) prevents error convergence (paper Algorithm 4).

6. **Next-step prediction for causal models:** Mamba is causal — use next-step prediction, not masked prediction (Mistake #34).

## Directory Structure

```
predictive-coding-agent/
├── CLAUDE.md              # This file
├── MISTAKES.md            # 35 documented mistakes (ALWAYS READ)
├── experiments/
│   ├── energy_reasoning/  # ePC-JEPA (ACTIVE DEVELOPMENT)
│   ├── Mamba3/            # ePC-Mamba3 (working, 99%+)
│   ├── ePC_ResNet/        # ePC ResNet-18 for CIFAR-10
│   ├── ePC_Mamba/         # ePC-Mamba2 (archived)
│   ├── eBPC/              # Error-based Bayesian PC
│   ├── eBPC_ResNet/       # eBPC with low-rank V (debugging)
│   ├── BayesianPC/        # Original BPC (architecturally wrong, see #12)
│   └── archived_kronos/   # KFAC optimizer (abandoned, see #20)
├── src/
│   ├── network/           # Baseline PC (95.14% MNIST)
│   ├── wrapper/           # Sensorimotor wrapper for Danganronpa
│   └── ...
├── lrpd/                  # Low-Rank Plus Diagonal library
└── tests/
```

## Known Issues & Gotchas

- **ePC_Mamba/** is the OLD Mamba2 experiment. **Mamba3/** is the current one.
- **BayesianPC/** has the wrong Bayesian treatment (posteriors over value nodes instead of weights). See Mistake #12. Don't use it.
- **eBPC_ResNet/** has unresolved NaN from low-rank V approximation violating PSD constraints. See Mistakes #16-19.
- **energy_scale** was a hack compensating for sum reduction. It's been removed everywhere. If you see it, it's a bug.
- Errors must be fp32 even under AMP. fp16 rounds small Newton corrections to zero.
- `forward(targets=None)` resets `pce.errors` to scalar `[0.0]`. Collect diagnostics BEFORE calling forward without targets (Mistake #23).

## Hardware

- Development: NVIDIA RTX 3050 Ti Laptop (4GB VRAM)
- All models designed to fit in 4GB VRAM
- Mixed precision (fp16 autocast + GradScaler) used everywhere

## Testing

```bash
# ePC-JEPA (energy_reasoning)
python experiments/energy_reasoning/train_epc_jepa.py --stage 1b --epochs 10

# Mamba3
python experiments/Mamba3/train_epc.py

# ePC ResNet MNIST validation
python experiments/ePC_ResNet/train_mnist.py

# ePC ResNet CIFAR-10
python experiments/ePC_ResNet/train_cifar10.py
```

## Research Papers Implemented

1. **Goemaere et al. 2025** — "Energy-based Predictive Coding" (ePC). Algorithm 4. T error steps → 1 weight step. Local learning via E_local.
2. **Tschantz et al. 2025** — "Bayesian Predictive Coding" (BPC). Matrix Normal Wishart weight posteriors. Hebbian closed-form updates.
3. **Assran et al. 2023** — I-JEPA. Latent prediction with EMA target encoder. VICReg regularization.
4. **Dao & Gu 2024** — Mamba2/Mamba3. State Space Duality (SSD) for efficient sequence modeling.
