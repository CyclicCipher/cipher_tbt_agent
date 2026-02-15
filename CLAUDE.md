# CLAUDE.md — Project Memory

## What This Project Is

A biologically-inspired AI system targeting the Danganronpa visual novel as an evaluation environment. Originally built on predictive coding (PC), now pivoting to backprop with potential modifications for local-learning-like benefits.

Current focus: **Naja** — A hybrid architecture combining Mamba3's continuous-time SSM dynamics with the delta rule's targeted write/erase memory, MIMO, PoPE orthogonal pairs, per-channel decay, and surprise gating. This is the `experiments/Naja/` directory.

**Phase 5a+5b (per-channel decay WY chunkwise) is COMPLETE and numerically verified** (max diff ~2e-6 vs naive reference). The A matrix stays CxC because per-channel decay is absorbed into K_pos/K_neg weighting before contracting over d_state. Eight test cases pass including per-channel multi-chunk.

**Next priority: Phase 5c — PoPE pair (B2) in WY via virtual token expansion, then Phase 5d — ablation testing.** See `CONTINUATION.md`.

Previous focus: **JEPA** — JEPA-style latent prediction on Mamba3 backbone (still in `experiments/energy_reasoning/`).

**ePC (energy-based predictive coding) has been archived.** After fixing all known bugs, ePC was 15x slower than backprop with zero accuracy benefit. See Mistake #38.

## Critical Reference

**ALWAYS read `MISTAKES.md` before making changes.** It has 40 documented mistakes with root causes. The most relevant active ones:

- **#40 (WY chunkwise bugs):** Three independent bugs in the initial WY implementation — decay convention, inter-chunk state formula, pseudo-key decay factor. All fixed. Key lesson: single-chunk tests are necessary but not sufficient; always test multi-chunk.
- **#39 (Fake Phase 5):** The OLD `delta_recurrence_chunkwise()` was just gradient checkpointing. Now replaced by real `delta_recurrence_wy()` which is numerically correct.
- **#38 (ePC archived):** ePC is 15x slower than backprop for identical accuracy. Don't resurrect it without a qualitatively new argument.
- **#34 (Next-step prediction):** Causal models (Mamba) need next-step prediction, not masked prediction.
- **#13 (Read the paper):** Never skim a research paper you're implementing. Read every appendix.
- **#36 (Don't run training):** Never run full training loops on Claude's CPU machine. Commit, push, let the user test on GPU.

## Architecture Overview

### JEPA (experiments/energy_reasoning/)

Standard backprop JEPA with Mamba3 backbone:

- **Encoder** (online): Mamba3 blocks, processes input sequences
- **Target encoder** (EMA): Exponential moving average of online encoder
- **Predictor**: Mamba3 blocks, predicts target encoder representations from online encoder + optional rule vector z
- **Decoder**: Predicts next token from online encoder representations

Training: JEPA latent prediction loss + decode cross-entropy + VICReg regularization, all via standard backprop.

Key files:
- `jepa_model.py` — JEPA model with encoder, predictor, decoder
- `train_jepa.py` — Training loop, data generation, diagnostics
- `data_gen.py` — Synthetic sequence generation (stages 1a/1b/2)

### Naja (experiments/Naja/) — ACTIVE PRIORITY

Hybrid Mamba3 + Gated DeltaNet architecture with backprop training:

- **Delta rule**: Householder erase before write (targeted memory management)
- **PoPE orthogonal pair**: Two Householder reflections compose into rotation (B₁, B₂)
- **Per-channel decay**: KDA-style diagonal α_t replaces scalar exp(Δ·A)
- **MIMO**: Rank-r B, C, X projections for hardware efficiency
- **Surprise gating**: β modulated by cross-entropy surprise (Phase 4)

Key files:
- `naja.py` — Full model (NajaLM, NajaMixer, delta_recurrence, delta_recurrence_wy, KLSurpriseTracker)
- `train_naja.py` — Training loop with preset ablation configs
- `tasks.py` — Ablation task generators (associative recall, parity, etc.)
- `diagnose.py` — Diagnostic suite (timing, correctness, memory)
- `test_wy_minimal.py` — Standalone WY correctness test (5 test cases, all passing ~1e-6)
- `DESIGN.md` — Complete architecture specification

**WY chunkwise status:** `delta_recurrence_wy()` is numerically verified (Phase 5a+5b complete). Per-channel decay fully supported. Remaining simplifications: single Householder (no PoPE pair), SISO (r=1). Phase 5c will add PoPE pair via virtual token expansion.

### Mamba3 Backbone (experiments/Mamba3/)

- `mamba3_block.py` — Mamba3 block implementation (SSD-based, reference for chunkwise pattern)

### Archived ePC Variants

All ePC code has been moved to `archived_epc/` subdirectories. These are retained for reference but are not actively developed.

| Directory | Architecture | Task | Status |
|-----------|-------------|------|--------|
| `experiments/energy_reasoning/archived_epc/` | Mamba3 + ePC-JEPA | Sequence prediction | Archived (15x slower, no benefit) |
| `experiments/Mamba3/archived_epc/` | Mamba3 + ePC | Copy/sequence | Archived (15x slower, no benefit) |
| `experiments/ePC_ResNet/` | ResNet-18 + ePC | CIFAR-10 | Archived |
| `experiments/ePC_Mamba/` | Mamba2 + ePC | Synthetic | Archived (superseded by Mamba3) |
| `experiments/eBPC/` | MLP + eBPC | MNIST | Working (95.74%) but not actively developed |
| `experiments/eBPC_ResNet/` | MLP + eBPC + low-rank V | MNIST | Abandoned (NaN from PSD violations) |

## Directory Structure

```
predictive-coding-agent/
├── CLAUDE.md              # This file
├── MISTAKES.md            # 39 documented mistakes (ALWAYS READ)
├── CONTINUATION.md        # WY chunkwise implementation plan (ACTIVE)
├── experiments/
│   ├── energy_reasoning/  # JEPA backprop (ACTIVE DEVELOPMENT)
│   │   ├── archived_epc/  # ePC-JEPA (archived 2026-02-14)
│   │   ├── jepa_model.py  # Active JEPA model
│   │   ├── train_jepa.py  # Active training script
│   │   └── data_gen.py    # Synthetic data generation
│   ├── Mamba3/            # Mamba3 block + archived ePC
│   │   ├── archived_epc/  # ePC-Mamba3 (archived 2026-02-14)
│   │   └── mamba3_block.py # Mamba3 block (shared dependency)
│   ├── ePC_ResNet/        # ePC ResNet-18 (archived)
│   ├── ePC_Mamba/         # ePC-Mamba2 (archived)
│   ├── eBPC/              # Error-based Bayesian PC (reference)
│   ├── eBPC_ResNet/       # eBPC with low-rank V (abandoned)
│   ├── BayesianPC/        # Original BPC (wrong, see #12)
│   └── archived_kronos/   # KFAC optimizer (abandoned, see #20)
├── src/
│   ├── network/           # Baseline PC (95.14% MNIST)
│   ├── wrapper/           # Sensorimotor wrapper for Danganronpa
│   └── ...
├── lrpd/                  # Low-Rank Plus Diagonal library
└── tests/
```

## Known Issues & Gotchas

- **BayesianPC/** has the wrong Bayesian treatment (posteriors over value nodes instead of weights). See Mistake #12. Don't use it.
- **energy_scale** was a hack compensating for sum reduction in ePC. It's been removed everywhere. If you see it, it's a bug.

## Hardware

- Development: NVIDIA RTX 3050 Ti Laptop (4GB VRAM)
- All models designed to fit in 4GB VRAM
- Mixed precision (fp16 autocast + GradScaler) used everywhere

## Testing

```bash
# JEPA backprop — Stage 1b (default, recommended starting point)
python experiments/energy_reasoning/train_jepa.py --stage 1b --epochs 10

# JEPA backprop — Stage 1b with compile + AMP (GPU, fastest)
python experiments/energy_reasoning/train_jepa.py --stage 1b --epochs 10 --compile

# JEPA backprop — Stage 2 (pattern induction, auto-defaults to 50 epochs)
python experiments/energy_reasoning/train_jepa.py --stage 2

# JEPA backprop — Profile (5 epochs, timing breakdown)
python experiments/energy_reasoning/train_jepa.py --stage 1b --profile
```

## Stage 2 Status & Key Findings

- **Stage 1 (1a, 1b, 1c):** PASSED. Single-rule tasks generalize immediately (~97% train ≈ ~97% test).
- **Stage 2 (pattern induction, 5 rules):** FAILING TO GENERALIZE. 99% train, ~25% test.
- **Oracle z ignored:** Providing the correct rule vector doesn't help. Predictor doesn't condition on z.
- **Langevin gap negative:** Energy minimization over z actively hurts (~-5%).
- **Hypothesis:** Model interprets 5 simple rules as 1 complex rule. See `docs/hypotheses/generalization_vs_memorization.md`.

## Next Direction: Backprop With Local-Learning-Like Benefits

ePC's original goal was to achieve local learning as a path to:
1. **Catastrophic forgetting resistance** — local updates don't overwrite unrelated circuits
2. **Modular neural circuits** — each layer learns from its own local error signal
3. **Energy-based reasoning** — inference-time optimization over latent variables

These goals remain valid. Possible backprop-compatible approaches to explore:
- **Gradient isolation / stop-gradient techniques** — selective detaching to create semi-local learning
- **Auxiliary local losses** — per-layer prediction losses alongside global backprop
- **EWC / SI / PackNet** — established continual learning methods for catastrophic forgetting
- **Mixture of experts / modular networks** — architectural modularity without local learning
- **Progressive training** — stage-wise freezing and expansion

The Stage 2 generalization problem (5-rule induction) is the immediate priority. This is a representation problem, not a gradient problem.

## Research Papers Implemented

1. **Goemaere et al. 2025** — "Energy-based Predictive Coding" (ePC). Algorithm 4. ARCHIVED — 15x slower, no benefit over backprop.
2. **Tschantz et al. 2025** — "Bayesian Predictive Coding" (BPC). Matrix Normal Wishart weight posteriors. ARCHIVED — fundamentally wrong implementation (#12).
3. **Assran et al. 2023** — I-JEPA. Latent prediction with EMA target encoder. VICReg regularization. ACTIVE.
4. **Dao & Gu 2024** — Mamba2/Mamba3. State Space Duality (SSD) for efficient sequence modeling. ACTIVE.
5. **Bardes et al. 2022** — VICReg. Variance-Invariance-Covariance regularization (used in JEPA training). ACTIVE.

## Research Papers Referenced (Naja Architecture)

6. **Yang et al. 2024** — "Parallelizing Linear Transformers with the Delta Rule" (DeltaNet). WY chunkwise algorithm for Householder recurrence. arXiv:2406.06484. CRITICAL for Phase 5 implementation.
7. **Yang et al. 2025** — "Gated Delta Networks" (Gated DeltaNet, ICLR 2025). Adds data-dependent decay to delta rule. arXiv:2412.06464. Direct ancestor of Naja's gated delta recurrence.
8. **Siems et al. 2025** — "DeltaProduct" (NeurIPS 2025). Multiple Householder reflections per token via virtual token expansion. arXiv:2502.10297. Relevant: Naja's PoPE pair = DeltaProduct with n_h=2.
9. **Gopalakrishnan et al. 2024** — PoPE (Polar Positional Embeddings). Decouples content from position.
10. **Kimi Team (Moonshot AI) 2025** — "Kimi Linear: An Expressive, Efficient Attention Architecture" (KDA). arXiv:2510.26692. Per-channel diagonal decay with `a=b=k` DPLR constraint eliminates secondary chunking. FLA-style state update and decay-weighted pseudo-keys are the ground truth for WY correctness. Our Phase 5a bugs were found by comparing against KDA/FLA conventions.

## Research Papers Referenced (Generalization/Grokking)

See `docs/research/` for PDFs, `docs/research/important research links.txt` for URLs.

6. **Michaud et al. 2023** — "The Quantization Model of Neural Scaling". Skills learned as discrete quanta.
7. **Power et al. 2022** — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets". Original grokking paper.
8. **Liu et al. 2022** — "Towards Understanding Grokking". Representation learning theory of grokking.
9. **Wang et al. 2024** — "Grokked Transformers are Implicit Reasoners". Memorizing→generalizing circuit transition.
10. **Fan et al. 2024** — "Deep Grokking". Multi-stage grokking in deep networks.
11. **deMoss et al. 2024** — "The Complexity Dynamics of Grokking". Complexity rises then falls at generalization.

## Hypotheses & Research Notes

- `docs/hypotheses/generalization_vs_memorization.md` — Multi-rule collapse hypothesis, empirical evidence, reasoning chain, supporting literature.
