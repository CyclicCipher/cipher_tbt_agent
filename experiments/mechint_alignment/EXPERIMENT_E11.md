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

    python e11_ncm.py tokenizer --minutes 3
    python e11_ncm.py generator --horizon 8 --minutes 7
    python e11_ncm.py generator --horizon 32 --minutes 7
    python e11_ncm.py stepmodel --minutes 7
    python e11_ncm.py eval --horizon 8
    python e11_ncm.py eval --horizon 32

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
