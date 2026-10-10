# Dust 2 — making backprop-free training cheap enough to use

Started 2026-10-10 at the user's request. The user's words: "if it works, it has far-reaching consequences for every
future architecture experiment."

## The goal

Train networks with a perturbation method derived from Dust (`DUST.md`) in place of backpropagation. Cut its compute cost
by orders of magnitude, while keeping the properties that make it worth having:

1. **No differentiability constraint.** Hard thresholds, spikes, discrete choices and external programs can sit inside the
   network. The method learns through them without a hand-picked continuous surrogate.
2. **No backward pass.**
3. **No stored history of activations for recurrent or looped training.** This makes looped models (the Neural Turing
   Architecture) and long internal computation trainable without BPTT memory.
4. **Event-driven sparsity.** With 1–5% of units active, as in critical brain dynamics, training and inference could save
   95–99% of compute. The user's plan calls this the only way to the compute savings needed for AGI on a laptop.

## Why it matters to the other lines

- **Neural Turing Architecture** (`../neural_turing_architecture/`): looped transformer, continuous thoughts learned
  without traces, search. Dust lists looped and externally-augmented models as its main future work.
- **DistributionPrediction** (`../DistributionPrediction/`): the looped backbone matched a deeper one at 55% of the
  weights there.
- **The training plan's second half:**
  - DiffusionBlocks: per-block targets, credit through 2–4 layers.
  - Block Attention Residuals.

  Both are designed around a local, perturbation-friendly credit path.

## How success is measured

**Cost to reach a given loss, in forward-pass equivalents, relative to backprop at the same quality.** Backprop costs
about 3 forward passes per step.

**Intermediate measure:** the cosine of the estimated gradient to backprop's gradient, per weight matrix, at a given
number of forward-equivalents. This is cheap (seconds), and the Dust authors used it to pick hyperparameters. Training
decides, because a higher cosine does not always mean a lower loss.

## Files

| file | what |
|---|---|
| `DUST.md` | everything collected about Dust: estimator, code details, model, results, limits |
| `IDEAS.md` | the user's training-method plan (verbatim), Gemini's five suggestions, my analysis, my own ideas |
| `notes/` | the user's screenshots: `user_plan_1/2.png` (their plan), `gemini_1–6.png` (Gemini's answer) |
| `data.py` | task 1: character-level Latin (Cicero + Livy, 11.3M characters, 38 symbols) |
| `model.py` | a small GPT in Dust's shape (pre-norm, parameter-free RMSNorm, ReLU², soft-cap 15) with every linear output a named site that can be perturbed and resumed from |
| `estimators.py` | Dust's estimator (rerun sites + local q/k/v scoring through the attention hub + head on logits) and the variants under test |
| `measure.py` | the cosine diagnostic against backprop |
| `train.py` | training by backprop or by an estimator, with cost counted in forward equivalents |
| `EXPERIMENT_1.md` | Gemini's five ideas, pre-registered, then results |
| `runs/` | results |

## Standing rules (from the user and the project)

- **Every test run takes 2 minutes at most.**
- **Every run is pre-registered** in an `EXPERIMENT_*.md` before it runs; exploratory work is labelled as such.
- **Use 16-bit wherever it is accurate enough** (the user, 2026-10-10). The precision check is in `EXPERIMENT_1.md` §0.
- **Diverse tasks.** Conclusions are not drawn from one task alone.
- **No compute on competitor-faithful arms.** Plain Dust is our base method, not a competitor.
- **No heavy training on this machine.** Runs are short and small, on the laptop GPU.
- **Commits go to `exp/mdl-vs-ntp`.**
