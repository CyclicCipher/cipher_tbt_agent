# E11: Next-Concept Modeling against a one-step world model, for imagining the future

**Question (the user):** "Build the NCM architecture from the repo, and test that." Does NCM give faster inference
(and training) for what our agents need? Background and the reading are in `NCM_NOTES.md`.

**The use case.** A planner with a learned world model must imagine the state H steps after an action sequence:
- the step-by-step way is H passes of a one-step model, each fed its own previous prediction;
- NCM predicts the state H steps ahead directly, in K coarse-to-fine passes, whatever H is.

The test world is Canvas: futures are deterministic given the start state and the actions (the noisy screen is
masked), so exactness can be measured. Actions are uniform random: what the random-shooting planner imagines.

## Pre-registration (2026-10-21)

**Code:**
- `ncm_model.py`: **the authors' architecture**, unmodified, from github.com/HazyResearch/ncm (MIT, commit
  c702742, cloned into `third_party/ncm`, git-ignored): the multiscale residual VQ-VAE tokenizer and the
  NextConceptTransformer generator.
  - The repo has no training loop. The two stages are written here from the paper: tokenizer = reconstruction CE +
    VQ terms with nested dropout; generator = level-weighted CE (α = 0.5) with 10% conditioning corruption.
  - The configuration follows the authors' Sudoku setup, scaled to a 4 GB GPU: 10 levels (1, 2, 4, 7, 14, 28, 49,
    98, 147, 196), 16 codes per level; level 0 from learned queries; raw-token prefix (start state + actions); RoPE;
    block-diagonal attention.
- `canvas_jumps.py`: the data. 2,000 training episodes of 300 random-action steps; 100 held-out episodes (seeds from
  10⁶). A state is one symbol per cell: colour + 4 × agent, with the screen as one symbol.
- `e11_ncm.py`: the training stages and the evaluation.

**Arms and budgets** (one GPU, an RTX 3050 Ti with 4 GB):

| model | parameters | training |
|---|---|---|
| NCM tokenizer | 2.48M | 3 min, batch 128, on states |
| NCM generator, H = 8 and H = 32 | 1.84M each | 7 min each, batch 24, on (start, actions, end) jumps |
| one-step world model (`worldmodel.WorldModel`: local relative attention, persistence prior) | 2.68M | 7 min, batch 64, on one-step transitions from the same trajectories; serves both horizons |

**Batch sizes were set by memory, measured before the run.** The repo's attention mask is float32, so under bf16
attention falls back to the full-matrix kernel; generator batches of 32 and more overflow the 4 GB card. The
authors' code is left as it is.

**Measures** (2,048 held-out jumps per horizon; the screen masked):
- **exact:** the whole predicted state is right;
- **changed-cell accuracy:** on cells that differ between start and end, the part a copy cannot get;
- per-cell accuracy, and the agent's position;
- the **copy baseline:** predict "nothing changes";
- **wall-clock** for 16 and 144 futures (144 = one planning decision: 3 rounds × 48 sequences), the median of 5
  after a warm-up;
- the tokenizer's exact reconstruction of held-out end states.

**Smoke test** (untrained models, timing only): at H = 8, 144 futures took 0.73 s (NCM) against 0.41 s (step model).

**Predictions:**

| # | prediction |
|---|---|
| N1 | **The tokenizer reconstructs** held-out states at ≥ 99.5% per cell and ≥ 80% whole-state exact |
| N2 | **Speed:** at H = 32, NCM imagines 144 futures in ≤ 0.5× the step model's time; at H = 8, NCM is *not* faster (≥ 0.8× the step model's time), since its 10 levels outnumber 8 steps and every level recomputes the prefix |
| N3 | **Exactness:** the step model's exact-state rate ≥ NCM's at both horizons. A local, deterministic world is easy to learn one step at a time; NCM must simulate H steps in one jump |
| N4 | **NCM learns the dynamics:** its changed-cell accuracy at H = 8 is ≥ 50% (the copy baseline has 0%) |

**Commands** (each stage under 10 minutes; run as one chain):

    python -m lab.e11.e11_ncm tokenizer --minutes 3
    python -m lab.e11.e11_ncm generator --horizon 8 --minutes 7
    python -m lab.e11.e11_ncm generator --horizon 32 --minutes 7
    python -m lab.e11.e11_ncm stepmodel --minutes 7
    python -m lab.e11.e11_ncm eval --horizon 8
    python -m lab.e11.e11_ncm eval --horizon 32

## Results (2026-10-21; `runs/e11.txt`, `runs/e11/*.jsonl`)

**Training.**
- **Tokenizer** (3 min, 1,050 steps): per-cell reconstruction 99.6–99.7% throughout, but whole-state exact reconstruction
  is noisy and not converging (38.5% at step 200, 24% at step 1,000, 30% at the end).
- **Generator:** code accuracy still climbing at the 7-minute limit (H = 8: 62% → 81%; H = 32: 62% → 78%).
- **Step model:** loss 6e-3 → 7e-5 (near perfect).

| | copy ("nothing changes") | **step model** (H passes) | **NCM** (10 passes) |
|---|---|---|---|
| **H = 8:** exact state | 3.5% | **95.8%** | 0.1% |
| changed-cell accuracy | 0% | **97.8%** | 14.0% |
| per-cell accuracy | 98.8% | **99.97%** | 97.6% (below copy) |
| time for 144 futures | | **0.44 s** | 0.79 s |
| **H = 32:** exact state | 0.1% | **89.2%** | 0.0% |
| changed-cell accuracy | 0% | **94.4%** | 7.9% |
| per-cell accuracy | 97.5% | **99.86%** | 92.4% (below copy) |
| time for 144 futures | | 1.77 s | **0.86 s** |

| # | verdict |
|---|---|
| N1 | **Half:** 99.6% per cell (≥ 99.5% confirmed), but whole-state exact reconstruction only 30% (≥ 80% refuted) |
| N2 | **Confirmed:** at H = 32 NCM takes 0.48× the step model's time; at H = 8 it is 1.8× *slower*. The crossover lies between 8 and 32 steps (10 levels, each recomputing the prefix) |
| N3 | **Confirmed, by far:** the step model is exact on 95.8% / 89.2% of futures, NCM on 0.1% / 0% |
| N4 | **Refuted:** NCM gets 14% of the changed cells at H = 8 (predicted ≥ 50%), and its per-cell accuracy is *below* the copy baseline: it corrupts more unchanged cells than it predicts changes |

**Reading.**
- **The tokenizer is the bottleneck.** It reconstructs only 30% of held-out states exactly, so even a perfect
  generator could be exact on at most ~30%. The paper's tokenizers reach 98.8–100%, with far longer training.
- **At this budget, NCM has not learned the dynamics.** The speed advantage is real only for long horizons, as
  predicted.
- **The one-step model is a hard baseline here.** Its locality and persistence priors fit a local, deterministic world,
  and it is near perfect after 7 minutes.
- **Compounding error was mild:** 95.8% exact at 8 steps, 89.2% at 32. "One-shot rollout avoids compounding errors"
  matters only when per-step error is large: approximate one-step models of continuous or chaotic physics, not this
  world.
- **Not tested:** NCM with a converged tokenizer. A fair test of its accuracy needs the tokenizer trained to ≥ 99%
  exact first (chained ≤ 10-minute segments), then the generator. Also not tested: NCM's constraint-satisfaction
  advantage, which a deterministic forward simulation does not exercise.

## Diagnostic: where NCM's error comes from (2026-10-21; saved checkpoints, H = 8, the same 2,048 futures)

The user asked whether the tokenizer is really the issue. "The tokenizer is the bottleneck" was an inference. These
measurements separate the error sources:

| what is decoded | per-cell | exact | changed cells | agent cell |
|---|---|---|---|---|
| copy of the start state | 98.82% | 3.5% | 0% | 4.4% |
| **A. the tokenizer's ceiling:** decode(encode(true end)) | 99.64% | 30.1% | 71.1% | **27.7%** |
| **B. generator, teacher-forced** (each level from the TRUE coarser levels) | 96.93% | 0.0% | 23.2% | 0.4% |
| **C. generator, free-running** (as evaluated) | 97.63% | 0.1% | 14.0% | 0.3% |

Code accuracy per level, coarse → fine (levels 1, 2, 4, 7, 14, 28, 49, 98, 147, 196 codes):

| | per level |
|---|---|
| B. teacher-forced | 0.79, 1.00, 1.00, 1.00, 1.00, 0.92, 0.89, 0.92, 0.85, 0.91 |
| C. free-running | 0.79, 1.00, 1.00, 1.00, 1.00, 0.91, 0.86, 0.84, 0.70, 0.73 |
| D. the START state's codes vs the END state's codes | 0.93, 1.00, 1.00, 1.00, 1.00, 0.99, 0.98, 0.97, 0.95, 0.94 |

Distinct codes used per level (of 16): **16, 1, 1, 1, 1, 2, 3, 6, 4, 9.**

**Three separate problems, not one:**
1. **The tokenizer's codebooks collapsed.** Levels 2–5 use a single code each, and so carry no information; the fine
   levels use 2–9 of 16. In practice there is no coarse-to-fine hierarchy: one global code, then the fine levels.
   - My tokenizer training used the quantizer's defaults. The repo offers dead-code restarts (`restart_threshold`)
     and k-means initialisation (`kmeans_init`), which I did not enable. The collapse is likely a consequence of
     my training setup, not of NCM.
2. **The tokenizer loses the rare, important symbol.** Even from perfect codes, the agent's cell (0.5% of cells) is
   reconstructed right only 28% of the time, and changed cells 71%. Per-cell reconstruction loss barely sees a
   symbol that rare.
3. **The generator is under-trained, independently of the tokenizer.**
   - Even given the true coarser levels, it gets 8–15% of fine codes wrong. Decoded, that is 96.9% per cell, below
     the copy baseline.
   - Simply re-encoding the *start* state would match the end state's codes better (94–99%) than the generator does
     (70–92%): it has not learned "copy, then edit".
   - It receives the start state as raw tokens, so it must also learn to reproduce the tokenizer's quantization of
     the unchanged board.

**Correction to the reading above.** The tokenizer caps exactness (30%) and loses the agent. Most of the per-cell
damage (99.64% → 97.6%) comes from the generator.

**A fair rerun would need:**
- a tokenizer with dead-code restarts or k-means initialisation, checked for codebook use at every level and for
  agent-cell accuracy;
- then a longer-trained generator;
- possibly a prefix that gives the generator the start state's *codes*, so copying is easy.

## E11b, pre-registered (2026-10-21): the fair NCM rerun

The user: "go ahead with the fair NCM rerun". Each problem the diagnostic found gets its fix (`e11b_ncm.py`; the
architecture is unchanged, the authors' code):
1. **Codebook collapse** → the quantizer's own dead-code restarts (`restart_threshold` = 1.0). Per-level code use is
   logged. Tokenizer training continues in ≤ 9-minute segments (resumed) until held-out reconstruction is ≥ 99.5%
   exact *and* ≥ 99% right on rare cells, or 18 minutes are spent.
2. **The rare symbol lost** → rare cells weighted ×20 in the reconstruction loss: the agent's cell for the "state"
   target, changed cells for the "delta" target. A training choice, not an architecture change.
3. **No "copy, then edit"** → a **delta target**: every cell is SAME or its new symbol, so copying the start is the
   default. This mirrors the one-step model's persistence prior.
   - The delta tokenizer is trained on the deltas of both horizons.
   - The prefix is unchanged: start state + actions, as raw tokens.

**Arms** (the same 2,048 held-out futures per horizon as E11):

| arm | tokenizer | generator |
|---|---|---|
| **delta, H = 8** | delta (fixes 1–3) | 18 min (2 × 9) |
| **delta, H = 32** | the same delta tokenizer | 18 min |
| **state, H = 8** | state (fixes 1–2 only) | 18 min: isolates what the delta target adds |
| step model | E11's, reused (7 min of training; not re-run) | |

The generators now get 18 minutes against the step model's 7: a budget advantage for NCM, stated here.

**Measures:** as in E11, plus the decomposition into tokenizer ceiling, teacher-forced generator and free-running
generator; code accuracy and code use per level.

**Predictions:**

| # | prediction |
|---|---|
| R1 | **The tokenizers are fixed:** both reach ≥ 99.5% exact and ≥ 99% rare-cell reconstruction on held-out data within budget |
| R2 | **NCM learns the dynamics with the delta target:** at H = 8, exact futures ≥ 50%, changed-cell accuracy ≥ 70%, per-cell accuracy above the copy baseline |
| R3 | **The step model is still more exact:** its 95.8% (H = 8) and 89.2% (H = 32) exact futures stay ≥ NCM-delta's |
| R4 | **The delta target matters beyond the tokenizer fix:** NCM-state's exact futures at H = 8 < NCM-delta's |
| R5 | **The speed pattern is unchanged:** NCM ≤ 0.5× the step model's time at H = 32, and ≥ 0.8× at H = 8 |

**Commands** (one chain; each segment ≤ 9 min; a finished stage exits immediately on repeat):

    python -m lab.e11.e11b_ncm tokenizer --target delta   (×2)
    python -m lab.e11.e11b_ncm tokenizer --target state   (×2)
    python -m lab.e11.e11b_ncm generator --target delta --horizon 8   (×2)
    python -m lab.e11.e11b_ncm generator --target delta --horizon 32  (×2)
    python -m lab.e11.e11b_ncm generator --target state --horizon 8   (×2)
    python -m lab.e11.e11b_ncm eval --target delta --horizon 8 ; eval --target delta --horizon 32 ; eval --target state --horizon 8

## E11b results (2026-10-21; `runs/e11b.txt`, `runs/e11b/*.jsonl`)

**Tokenizers** (18 min each; neither reached the stop criterion):
- **Delta:** exact reconstruction rose from 7% to 78%, noisily (held-out at evaluation: 64% for H = 8 deltas, 79% for
  H = 32). Changed cells 100% and agent 99.5%: the rare-cell weighting worked. Code use [16, 7, 8, 9, 12, 15, 15, 14,
  15, 16], no collapse. The remaining errors are false edits on unchanged cells.
- **State:** **unstable.** Exact reconstruction reached 97% at step 3,600, then fell to 33% by the end (agent cells
  100%). All 16 codes are used at every level. The saved checkpoint is the degraded final one: the training kept no
  best checkpoint, an apparatus omission.

**Evaluation** (the same 2,048 futures as E11; step model = E11's):

| arm | tokenizer ceiling: exact / changed | NCM teacher-forced: exact / changed | **NCM free-running: exact / changed / per-cell** | step model: exact / changed | time for 144: NCM / step |
|---|---|---|---|---|---|
| delta, H = 8 | 63.6% / 100% | 20.0% / 89.3% | **2.3% / 4.6% / 98.21%** | 95.8% / 97.8% | 0.79 s / 0.44 s |
| delta, H = 32 | 78.6% / 100% | 5.8% / 80.5% | **0.1% / 2.5% / 96.28%** | 89.2% / 94.4% | 0.86 s / 1.76 s |
| state, H = 8 | 34.2% / 90.4% | 0.0% / 57.6% | **0.0% / 16.3% / 95.63%** | 95.8% / 97.8% | 0.79 s / 0.44 s |

(Copy baseline per cell: 98.82% at H = 8, 97.47% at H = 32.)

Code accuracy per level, delta H = 8, coarse → fine:

| | per level |
|---|---|
| teacher-forced | 0.19, 0.78, 0.81, 0.88, 0.91, 0.93, 0.96, 0.97, 0.99, 1.00 |
| free-running | 0.19, 0.43, 0.59, 0.59, 0.57, 0.82, 0.89, 0.80, 0.97, 0.97 |

| # | verdict |
|---|---|
| R1 | **Refuted:** neither tokenizer reached 99.5% exact (delta 64–79%; state peaked at 97%, then degraded to 34%). Rare cells: 100% / 99.5–100% |
| R2 | **Refuted:** NCM-delta at H = 8 is exact on 2.3% of futures, gets 4.6% of changed cells, and its per-cell accuracy is below the copy baseline |
| R3 | **Confirmed:** the step model is far more exact (95.8% / 89.2%) |
| R4 | Confirmed in direction (state 0.0% < delta 2.3%), but both are near zero: uninformative |
| R5 | **Confirmed:** 0.49× the step model's time at H = 32, 1.79× at H = 8 (the same as E11) |

**Where the error is now: the coarse levels.**
- The fixes worked where they were aimed. With the delta target, the tokenizer reconstructs every changed cell, and
  the generator *given the true coarse codes* gets 89% of changed cells right.
- Free-running generation fails because the **coarse codes are mispredicted:** level 0 at 19% (16 possible codes),
  levels 1–4 at 43–59%. The errors cascade into the fine levels: 89% → 5% on changed cells.
- Conditioning corruption at 10% does not make the generator robust to error rates this high.

**Why the coarse levels are hard here (interpretation).**
- In this world the coarse code of the future, roughly "where the changes are and what they are", is the *result* of
  composing H precise local steps (where the agent walks, what it paints). Level 0 must do the whole simulation in its
  first pass. The one-step model does one easy local step per pass.
- In the paper's tasks the coarse structure is *easier* than the detail: the topic of a text, the constraint
  structure of a Sudoku. The same holds for smooth physics fields, whose low frequencies evolve slowly (which is why
  multigrid works).
- **Hypothesis:** coarse-to-fine generation helps when coarse structure is easier to determine than detail (design,
  layout, constraint satisfaction, smooth fields). It hurts when coarse structure is the consequence of a long chain
  of fine-grained steps (mechanistic simulation of discrete agents).
- This is consistent with the user's intuition that NCM belongs in constraint-satisfaction and design tasks. The
  Canvas design-brief test and the continuous-field world would test it directly.

**Caveats:**
- Small scale: 1.8M-parameter generators, 18 minutes, about a quarter of an epoch; code accuracy was still rising
  slowly. These results do not show that NCM fails at scale, only that it does not learn this world's dynamics at a
  scale where the one-step model already does.
- The state tokenizer's late degradation, with no best checkpoint kept, also hurt the state arm.
