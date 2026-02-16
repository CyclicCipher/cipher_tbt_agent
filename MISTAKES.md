# Mistakes File

## Purpose & Usage Instructions

This file catalogues all mistakes made during development — both code bugs and architectural/theoretical errors. **ALWAYS** consult this file before writing new code, designing architectures, or making design decisions.

---

## Active Mistakes (Current Relevance)

### #13. Never Skim Research Papers (CRITICAL PROCESS RULE)

**Rule:** When implementing a research paper, READ THE ENTIRE PAPER — every section, every appendix, every supplementary material. Never speculate about implementation details when the answer is in the paper.

**What happened (twice):**
1. Made lazy speculations about "energy normalization" when the paper covered adaptive learning rate in Appendix B. User: *"Stop. Skimming. The. Paper."*
2. Implemented theoretical dynamics from Appendix B instead of practical hyperparameters from Appendix F. Confused theoretical bounds with implementation details.

**Process:**
1. Read METHODS for algorithm
2. Read APPENDICES for theoretical analysis
3. Read EXPERIMENTAL DETAILS for actual hyperparameters
4. Match the paper's setup EXACTLY before making changes
5. Only speculate AFTER confirming the answer isn't in the paper

---

### #34. Causal Models Need Next-Step Prediction, Not Masked Prediction (CRITICAL)

Masked prediction on causal Mamba: **18.6%**. Next-step prediction: **97.05%**. A 78pp improvement.

**Why:** Mamba is causal — position t only sees 0..t-1. Masking early positions gives the predictor zero context. Next-step prediction gives EVERY position a loss signal with maximum available context.

**Principle:** Match prediction mode to model's inductive bias. Dense error coverage at every position is critical — sparse signals leave "dead zones" where parameters can't learn.

---

### #36. Never Run Training on Claude's CPU Machine

Make code changes, commit, push — let the user run tests on their GPU. Only run quick sanity checks (imports, syntax) if needed, never full training loops.

---

### #38. ePC Is 15x Slower Than Backprop For Zero Benefit (PROJECT PIVOT)

After fixing every known bug, clean comparison on Stage 1b:
- **ePC-JEPA:** ~153-176 s/epoch, 95.5% test by epoch 4
- **Backprop JEPA:** ~10 s/epoch, 95% test by epoch 2

ePC's error optimization (T=20 inner-loop steps per batch) is pure overhead when backprop gradients are already sufficient. Local learning didn't help generalization — Stage 2 fails at ~25% with both approaches.

**Decision:** All ePC code archived to `archived_epc/` subdirectories. Backprop is the active training path.

---

### #39. "Chunkwise Phase 5" Was Just Gradient Checkpointing, Not Actual Parallelism (CRITICAL — ACTIVE)

`delta_recurrence_chunkwise()` splits the sequence into chunks and wraps each in `torch.utils.checkpoint` — it runs the EXACT SAME sequential loop. Zero speedup. The naive recurrence launches ~5,120 CUDA kernels per batch, GPU utilization reads 0%.

**The fix:** True WY chunkwise parallelism (UT transform → forward substitution → matrix-form intra-chunk computation). See `CONTINUATION.md` for full implementation plan.

**Principle:** Gradient checkpointing (trading compute for memory) ≠ parallelism (trading sequential steps for parallel work).

---

### #40. Three Independent Bugs in WY Chunkwise Implementation (RESOLVED)

**Date:** 2026-02-15

The first WY chunkwise implementation (Phase 5a) had three independent bugs that each caused multi-chunk numerical divergence. Single-chunk tests passed, masking the inter-chunk issues.

**Bug 1 — Decay-before-erase convention mismatch:**
The naive reference applies `h = alpha * h` THEN `h = h - beta * (h @ k) k^T`. The WY's A matrix must use cumulative decay `exp(cumsum[i] - cumsum[j])` matching this "decay first, then erase" convention. Initial implementation had the decay applied inconsistently.

**Bug 2 — Wrong inter-chunk state update formula:**
Used `S = P @ S + H` (transition-matrix form), which is wrong when decay is present. The correct formula is FLA-style:
```
v_new = U - W @ S          # corrected pseudo-values
S = gamma * S + K_fwd^T @ v_new   # decay + accumulate
```
Where `K_fwd = K * exp(total_chunk_decay - cumsum)` ensures each position's contribution is properly forward-decayed to the chunk boundary.

**Bug 3 — Pseudo-keys W missing cumulative decay factor:**
`W = T @ K` was missing the cumulative decay from chunk boundary to each position. The state at position t has been decayed by `exp(cumsum[t])` from the chunk start, so the correction `U - W @ S` must account for this. Fix: `W_state = T @ (K * exp(cumsum))`. This matches FLA/KDA convention where W includes `k * beta * exp(G)`.

**Lessons:**
1. **Single-chunk tests are necessary but not sufficient.** Bug 2 and 3 only manifested with multiple chunks (S_prev ≠ 0). Always test with multi-chunk sequences.
2. **The WY pseudo-keys must include all factors that affect how the state is seen at each position.** If the state is decayed, W must include that decay.
3. **Read the FLA reference implementation carefully.** The naive reference (`fla/ops/delta_rule/naive.py`) and the chunkwise state update (`fla/ops/common/chunk_delta_h.py`) are the ground truth for the correct formulas.

---

### #41. Ablation Evaluation Leak: Answer Token Visible at Prediction Time (RESOLVED)

**Date:** 2026-02-15

All ablation task generators (associative recall, parity, multi-scale, permutation) placed the answer token at `seqs[:, -1]`. The evaluation used `logits[:, -1]` — which had already seen the answer in context. The model only needed to copy the last token for 100% accuracy. Every preset on every task hit ~100%, making ablations useless.

**Root cause:** Confusion between "predict at position -1" and "predict the answer." In a causal LM, `logits[:, t]` predicts `seqs[:, t+1]`. So `logits[:, -1]` (after seeing the full sequence including the answer) predicts what comes AFTER the answer — and we compared it against the answer itself, which the model had trivially memorized from the input.

**The genuine prediction** is at `logits[:, -2]`: the model has seen `[prefix..., Q]` but NOT the answer, and must predict the answer as the next token.

**Fix:** Changed training loss and all evaluation functions from `logits[:, -1]` to `logits[:, -2]`.

**Lesson:** When designing tasks with answer tokens in the sequence, always verify the evaluation position can't see the answer. Draw the causal attention mask on paper: if position p can attend to the answer, then `logits[:, p]` is trivially solvable.

---

### #42. Ablation Benchmarks Are All Memorization, Not Generalization (ACTIVE)

**Date:** 2026-02-16

All four ablation tasks show 100% train accuracy with near-random test accuracy on the non-trivial benchmarks. The models memorize 5000 training sequences instead of learning the algorithmic structure.

| Task | Train Acc | Test Acc | Random Chance |
|------|-----------|----------|---------------|
| parity | 100% | 100% | 50% — trivially easy |
| permutation_3 | 100% | 100% | 33% — trivially easy |
| associative_recall | 100% | 8-22% | ~3% — **memorized** |
| multi_scale | 39-100% | 7% | ~7% — **at chance** |
| permutation_4 | 55-87% | 25% | 25% — **at chance** |

**Root causes:**

1. **5000 samples vs 1.26M parameters:** 250× more params than samples. The model memorizes the dataset without learning any algorithmic structure.
2. **50 epochs is far too few for grokking.** Algorithmic generalization on small datasets typically requires 100-1000× more training beyond the memorization point (Power et al. 2022).
3. **naja_full trains slower on multi_scale** (only 39% train in 50 epochs) while simpler models hit 100% train, suggesting the extra machinery (delta rule + PoPE + per-channel decay) adds optimization difficulty for some tasks.
4. **Only trivial tasks succeed:** parity (binary output) and permutation_3 (3 possible answers) have such small output spaces that partial learning suffices.

**Implications for ablation design:**

- Feature ablation is meaningless when all models are memorizing. You can't attribute generalization differences when there's no generalization.
- Need either: (a) much more training data (50K+), or (b) much longer training (500+ epochs for grokking), or (c) smaller models, or (d) stronger regularization.
- The results file now includes per-epoch history (`epoch_history` key in JSONL) so learning dynamics are visible.

### #43. Scratchpad Query Type Placed in Output, Not Input (RESOLVED)

**Date:** 2026-02-16

QueryCountingGenerator (Stage 1) randomly chose DOT or TEN as the query type and placed it as the first work token (after WORK). The model had no signal in the input for which type to count — the query was unpredictable from input alone.

**Symptoms:** Token 1 (query type) stuck at ~51% test accuracy (chance for binary DOT/TEN). Token 2 (count) reached ~87% test accuracy — the model could count once teacher forcing gave it the correct query. Exact-match capped at ~45%.

**Root cause:** The query type was part of the output (work area), not the input (question area). The model cannot predict a randomly chosen token from input that doesn't contain the signal.

**Fix:** Move query to input using NOTE marker: `[shuffled DOTs/TENs] NOTE [DOT/TEN] WORK [count]`. n_result dropped from 2 to 1. Stage 1 now passes in ~8 epochs (was stuck at 100 epochs before).

**Lesson:** Every token in the work area must be deterministically derivable from the input. If a token is chosen randomly and placed in the output, the model has no basis to predict it. Scratchpad design rule: the work area contains *answers*, the question area contains *queries*.

---

## Condensed Archive (Historical Reference)

Below are condensed lessons from resolved/archived mistakes. Full debugging narratives have been removed.

### Architectural & Theoretical

| # | Mistake | Lesson |
|---|---------|--------|
| 1 | CNN + Predictive Coding failed repeatedly | Use standard PC architectures from literature, not custom CNN hybrids |
| 2 | Custom two-compartment neuron design | Use standard PCLayer from Bogacz Group (value nodes as nn.Parameters) |
| 3 | Output clamping for pretraining | Never force outputs — let network learn through prediction error minimization |
| 4 | Exponential precision scaling across layers | Never attempt — creates numerical instability. Research VERSES for deep scaling |
| 12 | BayesianPC: posteriors over value nodes instead of weights | BPC means Bayesian over WEIGHTS (Matrix Normal Wishart), not hidden states. Value nodes are ephemeral MAP estimates. Architecture requires weights OUTSIDE activation for conjugacy. All code in `experiments/BayesianPC/` is fundamentally wrong |

### Code Implementation

| # | Mistake | Lesson |
|---|---------|--------|
| 7 | Import errors in non-root files | Add `sys.path.insert(0, ...)` at top of every subdirectory script |
| 8 | Optimizer conflating value nodes and weights | In PC, value nodes and weights must be in SEPARATE optimizer param groups |
| 9 | `mu.detach()` broke computational graph | Never detach predictions from weights — optimizer separation handles phase isolation, graph must stay connected for gradients |
| 11 | Changed architecture between experimental treatments | Control ALL variables except the one being tested |
| 17 | Hyphenated directory names | Python can't import from `eBPC-ResNet` — use underscores |
| 23 | Forward pass overwrote ePC errors before diagnostics | When model has ephemeral state, don't reset it between population and consumption |
| 35 | Sum reduction for errors vs mean reduction for output loss | 1M× scale mismatch crushed errors to zero. ALL terms in an energy function must use the SAME reduction semantics |

### Optimization & Second-Order Methods (ePC-specific, all archived)

| # | Mistake | Lesson |
|---|---------|--------|
| 10 | Variance collapse (min_variance=1e-6) | Use min_variance=0.01, max_variance=10.0 for numerical stability |
| 14 | Added cross-entropy task loss not in paper | PC uses clamped output as supervisory signal — no separate task loss needed |
| 15 | SGD for error optimization with precision-weighted gradients | Use Adam when gradient scales vary by orders of magnitude |
| 16-19 | Diagonal MNW, low-rank η1, FITC corrections | Full-matrix mathematical guarantees don't transfer to diagonal/low-rank approximations. MNW conjugacy + PD constraints are fundamentally hard to capture in LRPD form |
| 20 | KFAC/KRONOS incompatible with ePC | ePC needs per-element magnitude adaptation (Adam), not matrix direction rotation (KFAC). G factor degenerate, raw gradients tiny |
| 21 | INT8 QAT destroys ePC accuracy | ePC error optimization amplifies weight noise over T iterations |
| 22 | AdaWoodbury (rank-1 Woodbury over Adam) | Global rank-1 curvature doesn't match ePC's block-diagonal structure. No benefit over Adam |
| 24-26 | Replaced working Newton with broken alternatives | Don't replace a working component without verifying it's the actual bottleneck |
| 27 | Init scale 2x on output projections | Forward dynamics and Jacobian magnitude are coupled through weights — can't increase one without the other |
| 28 | mHC (manifold hyperconnections) | More error degrees of freedom makes rank-1 Hessian approximation worse, not better |
| 29 | muPC (Depth-muP) | Crushed non-residual contributions to ~3% of residual stream, making Newton corrections negligible |
| 30 | Adaptive Newton damping | Don't "improve" a working system based on a paper using fundamentally different optimization (SGD T=128 vs Newton T=2) |
| 31 | iPC flag not wired up in training loop | When adding a CLI flag, verify it reaches the execution path. The "99.2% iPC" was standard ePC |
| 32 | Autograd HVP through CE + multiple error nodes → NaN | Never use autograd double-backward through cross-entropy with multiple leaf nodes |
| 33 | Newton/CG unnecessary — SGD wins | The "phase transition" was Newton-specific, not ePC-specific. ~1,300 lines of second-order code deleted. SGD is simpler and works better |
| 37 | Blindly copied paper's hyperparameters across architectures | λT is architecture-dependent. Paper's VGG/ResNet values don't transfer to Mamba3 |

### Process

| # | Mistake | Lesson |
|---|---------|--------|
| 5 | Not learning from repeated mistakes | This file exists to prevent repeats |
| 6 | Context compression loss | Maintain persistent documentation (MISTAKES.md, CLAUDE.md, etc.) |

---

## Update Log

- 2026-02-01: Mistakes #1-13 documented and fixed
- 2026-02-06: #14 (cross-entropy task loss)
- 2026-02-07: #15-17 (SGD for errors, diagonal MNW, naming)
- 2026-02-08: #18-19 (low-rank η1, FITC)
- 2026-02-10: #20-23 (KRONOS, QAT, AdaWoodbury, diagnostic ordering)
- 2026-02-11: #24-29 (Newton debugging spiral, init scale, mHC, muPC)
- 2026-02-12: #30-32 (adaptive damping, iPC flag, autograd HVP)
- 2026-02-13: #33-37 (SGD wins, next-step prediction, reduction mismatch, don't run training, hyperparameter copying)
- 2026-02-14: #38-39 (ePC archived, fake Phase 5 chunkwise)
- 2026-02-15: #40 (three WY chunkwise bugs: decay convention, state update formula, pseudo-key decay)
- 2026-02-15: #41 (ablation evaluation leak: answer in sequence, trivial copy instead of genuine prediction)
- 2026-02-16: #42 (ablation benchmarks are all memorization; 5K samples + 50 epochs + 1.26M params = no generalization)
