# CLAUDE.md — Project Memory

## What This Project Is

A biologically-inspired AI system targeting the Danganronpa visual novel as an evaluation environment. Originally built on predictive coding (PC), now pivoting to backprop with potential modifications for local-learning-like benefits.

Current focus: **Compositional Arithmetic Curriculum** — Testing whether a prerequisite-complete curriculum (counting → ordinality → comparison → counting-based addition/subtraction → multi-digit column arithmetic) enables generalization that direct training cannot. Key insight (Mistake #44): single-digit addition is NOT a fact stage — it decomposes into counting-up via the successor function. Uses the Mamba3 backbone. See `CONTINUATION.md`.

**Naja** (WY chunkwise) is complete and numerically verified (Phase 5a+5b+5c). Ablation testing (Phase 5d) showed all benchmarks are memorization, not generalization (Mistake #42). This motivated the compositionality pivot.

**Active priority: Compositional arithmetic curriculum in `experiments/Mamba3/`.** See `CONTINUATION.md`.

Previous focus: **JEPA** — JEPA-style latent prediction on Mamba3 backbone (still in `experiments/energy_reasoning/`).

**ePC (energy-based predictive coding) has been archived.** After fixing all known bugs, ePC was 15x slower than backprop with zero accuracy benefit. See Mistake #38.

## Critical Reference

**ALWAYS read `MISTAKES.md` before making changes.** It has 44 documented mistakes with root causes. The most relevant active ones:

- **#44 (Missing prerequisites):** Single-digit addition was treated as a fact stage (155 entries to memorize). But addition decomposes into counting-up via successor. The curriculum jumped from counting to arithmetic without teaching ordinality, comparison, or place value. Category theory constraint: every composition requires its constituent objects to be established.
- **#43 (Query in output, not input):** Scratchpad work tokens must be deterministically derivable from input. Randomly chosen tokens in the output are unpredictable. Fixed.
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

### Scratchpad Framework (experiments/scratchpad/) — NEW

Model-agnostic framework for generating problems with structured work areas:

- `framework.py` — Vocab (dynamic token registry), Problem (question + steps), Step (named graded tokens), Grader (per-step/per-token scoring), ProblemGenerator (abstract base), split_problems (train/test by held-out specs)
- `generators/counting.py` — QueryCountingGenerator (Stage 1, NOTE-based query), CombinedCountingGenerator (Stage 2, count-up process with STOP token)
- `generators/ordinality.py` — SuccessorGenerator (Stage 3), ComparisonGenerator (Stage 4)
- `generators/arithmetic.py` — SingleDigitArithmeticGenerator (Stage 3, supports reverse problems via `reverse_fraction`), TwoDigitSingleArithmeticGenerator (Stage 4), TwoDigitArithmeticGenerator (Stage 5)

Reverse problems (Stage 3): `? + 4 = 0 7 WORK 0 3` — given result + one operand, find missing operand. Forces induction (understanding the inverse), not just mechanical forward application. Mixed with forward problems within the same stage. Ref: Alemi (2025) factorization order.

### CTKG (experiments/ctkg/) — IMPLEMENTED + DSL + UNIVERSAL TYPES

Category Theory Knowledge Graph — a directed acyclic graph where nodes are concepts/skills and edges are prerequisite relationships. Built on a universal type system of primitives (`symbol`, `seq`, `tuple`, `tagged` + structure annotations) that compose into any domain-specific type. Includes functors (structure-preserving maps between domains) and adjunctions (forward/inverse pairs).

1. **Curriculum compiler** (IMPLEMENTED) — topological sort = valid curriculum, type checking catches missing prerequisites before training. `validate()` checks 7 error types including `UndefinedType`.
2. **Structured training data** (IMPLEMENTED) — `generate_curriculum()` produces ordered stages with replay policies from the graph structure.
3. **DSL parser** (IMPLEMENTED) — `.ctkg` indentation-based file format for declarative graph definition. Supports type, concept, functor, adjunction blocks. `parse_file()` returns KnowledgeGraph.
4. **Universal type system** (IMPLEMENTED) — `TypeDef` with constructors (`symbol`, `nat`, `seq`, `tuple`, `tagged`, `expr`, `proposition`), structure annotations (`ordered`, `metric`, `invertible`, etc.), and 4 builtin types. Validation catches undefined type references.
5. **External knowledge store** (deferred) — knowledge in system RAM, structure-aware retrieval, compensates for 4GB VRAM limit
6. **Computational aid** (deferred) — deterministic solver for multi-step problems

**Arithmetic domain:** 9 concepts, 13 custom types, 12 prerequisites, 1 adjunction (add/sub). All types defined with universal primitives (e.g., `type digit = symbol(0..9) ordered`).

**Target calculations:** (1) Analytically solve ODEs up to 3rd order, (2) Solve ODEs via Laplace transform, (3) Derive the impulse response of a damped harmonic oscillator — 47 stages from counting to solution. (4) Logic from propositional to natural transformations in category theory — 21 stages.

**Three curriculum patterns:** Process (composition/functors — step-by-step execution), Relationship (adjunctions/natural transformations — paired forward/inverse computations), Constraint (limits/pullbacks — multi-constraint satisfaction).

**Three levels of process primitives:** Level 1 (Computation: succ, pred, compare, fold, scan, emit, lookup, if), Level 2 (Logic: equal, and, or, not, implies, forall, exists), Level 3 (Transform: quote, match, substitute, rewrite, decompose, compose). Level 1 does arithmetic. Level 2 proves properties. Level 3 improves algorithms.

**Commercial architecture:** Layer 1 (Graph data — `.ctkg` files, customer provides), Layer 2 (Computation rules — declarative, customer provides), Layer 3 (Engine — we provide: parser, validator, curriculum generator, trainer).

Key files:
- `DESIGN.md` — Architecture, universal type system, DSL grammar, categorical structure assessment, curriculum patterns, commercial architecture
- `graph.py` — `TypeDef`, `Concept`, `Prerequisite`, `Functor`, `Adjunction`, `KnowledgeGraph` (validate, topological_sort, generate_curriculum, ancestors, descendants, frontier, missing_for)
- `parser.py` — DSL parser: `parse(text)`, `parse_file(path)`, `merge(target, source)`. Handles type/concept/functor/adjunction blocks, comments, multi-line process blocks.
- `domains/arithmetic.py` — Thin wrapper loading from arithmetic.ctkg
- `domains/arithmetic.ctkg` — Full arithmetic domain (9 concepts, 13 types, 12 prereqs, 1 adjunction)
- `test_parser.py` — 6 tests covering type parsing, domain loading, validation, errors, curriculum generation

### Mamba3 Backbone (experiments/Mamba3/) — ACTIVE PRIORITY

- `mamba3_block.py` — Mamba3 block (SSD-based, PoPE, trapezoidal discretization, `--stable_ssm` option, `--use_triton` option, `--mhc` option with `HyperConnection` class and `sinkhorn_normalize`)
- `triton_ssd.py` — Triton-accelerated SSD kernels (graceful fallback to PyTorch when Triton unavailable)
- `arithmetic_tasks.py` — Old task generators (superseded by scratchpad framework)
- `train_arithmetic.py` — Curriculum training script (uses scratchpad framework, stages 1-5, curriculum/direct modes, per-token diagnostics with train+test breakdown, epiplexity tracking per stage, `--reverse_fraction` for reverse problem mixing, `--stable_ssm` for StableSSM A-matrix, `--mhc` / `--mhc_n_streams` for manifold-constrained hyperconnections)

### BTT-Mamba3 (experiments/BTT_Mamba3/) — DESIGN PHASE

Block Tensor-Train compressed Mamba3. Replaces dense nn.Linear layers with BTT-decomposed versions for dramatically reduced parameter count at equivalent model width. Enables training d=1024+ models within 4GB VRAM.

Key files:
- `DESIGN.md` — Full analysis: layer inventory, BTT candidates, small-scale solutions, μP scaling, implementation plan

**Paper:** "Compute Better Spent: Replacing Dense Layers with Structured Matrices" (docs/research/BTT paper.pdf). c=2, r=4 matches GPT-2 Small at 2.7× fewer FLOPs.

### Archived ePC Variants

ePC code within active experiments has been moved to `archived_epc/` subdirectories within `experiments/Mamba3/` and `experiments/energy_reasoning/`. All standalone archived experiments (ePC_ResNet, ePC_Mamba, eBPC, eBPC_ResNet, BayesianPC, archived_kronos) have been deleted.

## Directory Structure

```
predictive-coding-agent/
├── CLAUDE.md              # This file
├── MISTAKES.md            # 44 documented mistakes (ALWAYS READ)
├── CONTINUATION.md        # Compositional arithmetic curriculum plan (ACTIVE)
├── experiments/
│   ├── ctkg/              # Category Theory Knowledge Graph (IMPLEMENTED + DSL)
│   │   ├── DESIGN.md      # Architecture, DSL grammar, curriculum patterns, commercial arch
│   │   ├── graph.py       # Concept, Prerequisite, Functor, Adjunction, KnowledgeGraph
│   │   ├── parser.py      # DSL parser: parse(), parse_file(), merge()
│   │   ├── test_parser.py # Parser verification test
│   │   └── domains/       # arithmetic.py/.ctkg (9 concepts, 13 types, 1 adjunction)
│   ├── scratchpad/        # Scratchpad framework (model-agnostic)
│   │   ├── framework.py   # Vocab, Problem, Step, Grader, ProblemGenerator
│   │   ├── DESIGN_GUIDE.md # Curriculum design principles (category theory, prerequisites)
│   │   └── generators/    # counting.py (S1-S2), ordinality.py (S3-S4), arithmetic.py (S5-S6), multi_digit.py (S7-S8)
│   ├── energy_reasoning/  # JEPA backprop (paused)
│   │   ├── archived_epc/  # ePC-JEPA (archived 2026-02-14)
│   │   ├── jepa_model.py  # Active JEPA model
│   │   ├── train_jepa.py  # Active training script
│   │   └── data_gen.py    # Synthetic data generation
│   ├── Mamba3/            # Mamba3 backbone + arithmetic curriculum (ACTIVE)
│   │   ├── mamba3_block.py       # Mamba3 block (backbone model)
│   │   ├── arithmetic_tasks.py   # Old task generators (superseded)
│   │   ├── train_arithmetic.py   # Curriculum training (uses scratchpad)
│   │   └── archived_epc/        # ePC-Mamba3 (archived 2026-02-14)
│   ├── BTT_Mamba3/        # BTT-compressed Mamba3 (DESIGN PHASE)
│   │   └── DESIGN.md      # Layer inventory, BTT candidates, implementation plan
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

The original 12-stage curriculum failed — provided no advantage over direct training. The 4-stage revision also failed — skipped sub-skill scaffolding, removed process supervision, autoregressive output asymmetry caused memorization. See `CONTINUATION.md` for full post-mortem.

**Current experiment:** 8-stage prerequisite-complete curriculum on Mamba3 (see `CONTINUATION.md`):
1. Query counting — "how many DOTs/TENs?" with confounders (sub-skill)
2. Combined counting — DOT 1..d STOP TEN 1..t STOP count-up process (composition)
3. Successor/predecessor — digit ordinality (SUCC(4)=5, PRED(7)=6)
4. Comparison — digit ordering (GT/LT/EQ)
5. Counting-based addition — `3 + 4 WORK 4 5 6 7 = 0 7` (reduces to counting-up)
6. Counting-based subtraction — `7 - 3 WORK 6 5 4 = 0 4` (reduces to counting-down)
7. Two-digit ± single-digit (bridge to multi-digit, column scratchpad)
8. Two-digit ± two-digit (~12,195 problems, composition test)

**Key insight (Mistake #44):** Single-digit addition is NOT a fact stage. It decomposes into counting-up via the successor function. Category theory constraint: every composition requires its prerequisite objects to be established.

**Key test:** Does curriculum training (stages 1→8) produce better generalization on Stage 8 than direct training on Stage 8 alone?

**Curriculum rules:** Stages advance only on ≥95% test accuracy. If a stage fails after max epochs, the curriculum halts. Per-token accuracy diagnostics show both train and test breakdowns for multi-token stages.

**Epiplexity tracking:** Each stage logs S_preq = sum(l_i - l_final), measuring structural information extracted during training. High S_preq = rich structure learned. Low S_preq = trivially memorizable (warning: model may pass without learning reusable circuits). Ref: Alemi (2025) "Epiplexity and the Solomonoff Prior."

**Reverse problems:** `--reverse_fraction 0.3` mixes 30% reverse problems into Stage 3 (single-digit arithmetic). Reverse format: `? + 4 = 0 7 WORK 0 3` — find missing operand. Forces induction (understanding the inverse) rather than mechanical forward application. Both forward and reverse maintain n_result=2 for batch compatibility.

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
11. **Wang & Li 2024** — "StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization" (ICML 2024). arXiv:2311.14495. Best reparameterization `f(w) = 1 - 1/(w² + 0.5)` minimizes gradient-to-weight ratio, enabling larger learning rates and better long-range dependency learning. Implemented as `--stable_ssm` option in Mamba3.

## Research Papers Referenced (Architecture Improvements)

13. **Xiao et al. 2025** — "Manifold-Constrained Hyperconnections" (mHC). Replaces standard residual connections with multi-stream gated communication. Column-stochastic gating matrices (enforced via Sinkhorn-Knopp) preserve the "residual manifold", preventing norm drift. ~6.7% compute overhead, negligible parameter overhead. Implemented as `--mhc` / `--mhc_n_streams` in Mamba3. Ref: `docs/research/mHC manifold constrained hyperconnections.pdf`.

## Research Papers Referenced (Curriculum Design / Information Theory)

12. **Alemi 2025** — "Epiplexity and the Solomonoff Prior". Prequential coding as measure of structural information. Key results applied: (a) epiplexity (S_preq) as per-stage diagnostic, (b) factorization order affects representation quality (chess experiment — reverse order forces induction, produces richer circuits), (c) "too easy" trap (low S_preq stages may pass without teaching reusable structure). Directly motivates epiplexity tracking in `train_arithmetic.py` and reverse problem design.

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
