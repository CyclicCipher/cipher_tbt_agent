# CLAUDE.md — Project Memory

## What This Project Is

A biologically-inspired AI system targeting the Danganronpa visual novel as an evaluation environment. Originally built on predictive coding (PC), now pivoting to backprop with potential modifications for local-learning-like benefits.

Current focus: **Compositional Arithmetic Curriculum** — Testing whether a staged curriculum teaching composable sub-skills (mixed counting → single-digit +/- → two-digit±single-digit → two-digit±two-digit) enables generalization that direct training cannot. Uses the Mamba3 backbone. See `CONTINUATION.md`.

**Naja** (WY chunkwise) is complete and numerically verified (Phase 5a+5b+5c). Ablation testing (Phase 5d) showed all benchmarks are memorization, not generalization (Mistake #42). This motivated the compositionality pivot.

**Active priority: Compositional arithmetic curriculum in `experiments/Mamba3/`.** See `CONTINUATION.md`.

Previous focus: **JEPA** — JEPA-style latent prediction on Mamba3 backbone (still in `experiments/energy_reasoning/`).

**ePC (energy-based predictive coding) has been archived.** After fixing all known bugs, ePC was 15x slower than backprop with zero accuracy benefit. See Mistake #38.

## Critical Reference

**ALWAYS read `MISTAKES.md` before making changes.** It has 42 documented mistakes with root causes. The most relevant active ones:

- **#42 (Ablation benchmarks are memorization):** 5K samples + 1.26M params + 50 epochs = memorization, not generalization. All non-trivial tasks show 100% train / chance test. This motivated the compositionality curriculum pivot.
- **#41 (Ablation eval leak):** Answer token at seqs[:, -1] was visible to logits[:, -1]. All tasks scored ~100% via trivial copy. Fixed: use logits[:, -2] which genuinely predicts the answer.
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

### Naja (experiments/Naja/) — WY COMPLETE, ABLATIONS PAUSED

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
- `test_wy_minimal.py` — Standalone WY correctness test (11 test cases, all passing ~1e-6)
- `run_ablations.py` — Phase 5d ablation runner (preset × task grid)
- `DESIGN.md` — Complete architecture specification

**WY chunkwise status:** `delta_recurrence_wy()` is numerically verified (Phase 5a+5b+5c complete). Per-channel decay and PoPE pair (B₂ via virtual token expansion) fully supported. Remaining simplification: SISO (r=1).

### Mamba3 Backbone (experiments/Mamba3/) — ACTIVE PRIORITY

- `mamba3_block.py` — Mamba3 block implementation (SSD-based, backbone for arithmetic curriculum)
- `arithmetic_tasks.py` — Task generators for revised 4-stage curriculum (mixed counting, single-digit +/-, bridge, two-digit +/-)
- `train_arithmetic.py` — Curriculum training script (stages 1-4, curriculum/direct modes, per-token diagnostics)

### Archived ePC Variants

ePC code within active experiments has been moved to `archived_epc/` subdirectories within `experiments/Mamba3/` and `experiments/energy_reasoning/`. All standalone archived experiments (ePC_ResNet, ePC_Mamba, eBPC, eBPC_ResNet, BayesianPC, archived_kronos) have been deleted.

## Directory Structure

```
predictive-coding-agent/
├── CLAUDE.md              # This file
├── MISTAKES.md            # 42 documented mistakes (ALWAYS READ)
├── CONTINUATION.md        # Compositional arithmetic curriculum plan (ACTIVE)
├── experiments/
│   ├── energy_reasoning/  # JEPA backprop (paused)
│   │   ├── archived_epc/  # ePC-JEPA (archived 2026-02-14)
│   │   ├── jepa_model.py  # Active JEPA model
│   │   ├── train_jepa.py  # Active training script
│   │   └── data_gen.py    # Synthetic data generation
│   ├── Mamba3/            # Mamba3 backbone + arithmetic curriculum (ACTIVE)
│   │   ├── mamba3_block.py       # Mamba3 block (backbone model)
│   │   ├── arithmetic_tasks.py   # NEW: Compositional arithmetic task generators
│   │   ├── train_arithmetic.py   # NEW: Curriculum training script
│   │   └── archived_epc/        # ePC-Mamba3 (archived 2026-02-14)
│   └── Naja/              # Hybrid Mamba3 + Gated DeltaNet (WY complete, ablations paused)
│       ├── naja.py        # Full model
│       ├── train_naja.py  # Training loop
│       ├── tasks.py       # Ablation task generators
│       └── DESIGN.md      # Architecture specification
├── src/
│   ├── network/           # Baseline PC (95.14% MNIST)
│   ├── wrapper/           # Sensorimotor wrapper for Danganronpa
│   └── ...
└── lrpd/                  # Low-Rank Plus Diagonal library
```

## Known Issues & Gotchas

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

## Next Direction: Compositional Curriculum Learning

The core generalization problem persists across all architectures (JEPA, Naja, Mamba3): models memorize composite tasks instead of learning algorithmic structure. Ablation benchmarks (Mistake #42) confirmed this — 100% train / chance test on every non-trivial task.

The original 12-stage curriculum failed — provided no advantage over direct training. Root causes: early stages trivially memorizable (9-10 problems each), interleaved counting gave the answer away, composition test too small. See `CONTINUATION.md` for full post-mortem.

**Revised experiment:** 4-stage curriculum on Mamba3 (see `CONTINUATION.md`):
1. Mixed counting (shuffled DOT/TEN tokens, no interleaving, 100 combinations)
2. Single-digit +/- (155 problems, 2-digit output)
3. Two-digit ± single-digit (1,800 problems, bridge to multi-digit)
4. Two-digit ± two-digit (~12,195 problems, composition test)

**Key test:** Does curriculum training (stages 1→4) produce better generalization on Stage 4 than direct training on Stage 4 alone?

**Curriculum rules:** Stages advance only on ≥95% test accuracy. If a stage fails after max epochs, the curriculum halts. Per-token accuracy diagnostics show which result position is the bottleneck for multi-token stages.

Previous goals (catastrophic forgetting, modular circuits, energy-based reasoning) remain valid but are secondary until the generalization problem is understood.

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
