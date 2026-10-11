# Experiment 6 — SGD with momentum vs Adam (pre-registered 2026-10-10)

Status: **pre-registered before any arm ran.** The user asked: "...then test SGD vs Adam".

**The hint that prompted it** (`PROXY.md`, version 1, at the late checkpoint). Under plain SGD at the best step size, the
Dust baseline's one-step progress EQUALLED backprop's (1.06e-5 vs 1.02e-5 on the CPU run). Under an Adam-like scaling
it was 2.7× worse. A plausible reading:
- Dust's extra noise lies mostly in flat directions, which SGD barely pays for.
- Adam divides every coordinate by its recent magnitude, noise included, which shrinks the steps.

The Dust authors trained with SGD and momentum. That proxy failed validation overall, so this is a hint, not evidence.

## Phase A — screen the learning rates with the 25-step probe (`probe.py`, validated in `PROXY.md`)

**The grid.** SGD with momentum 0.95. Learning rate ∈ {0.1, 0.3, 1, 3} × embedding multiplier ∈ {1, 30, 300}: Dust
gives wte a much larger rate, because each embedding row sees few tokens.

**Who gets screened.**
- the Dust baseline (K = 32) and backprop, at both checkpoints;
- the best Experiment 5 arm, on its own grid if its probe cost allows, otherwise at the baseline's best setting.

**Choice.** Each method's setting is the best by mean probe drop. This also tests the probe as a learning-rate picker.

## Phase B — training (300 steps, two seeds, val every 10)

**Arms:**
- Dust baseline + SGD;
- backprop + SGD;
- the best Experiment 5 arm + SGD.

**Comparisons, reused and not re-run:**
- Dust baseline + Adam: 2.3687;
- backprop + Adam: 2.1586;
- the Experiment 5 arm + Adam.

## Predictions

- **S1:** Dust + SGD beats Dust + Adam by ≥ 0.02.
- **S2:** backprop + SGD is WORSE than backprop + Adam by ≥ 0.02, as usual for transformers.
- **S3:** so the Dust-to-backprop gap shrinks under SGD by ≥ 0.05 nats.
- **S4:** the best Experiment 5 arm + SGD beats its Adam version by ≥ 0.02.
- **S5:** the probe's chosen learning rate is within one grid step of the best of the trained ones. This is checked only
  where a second setting is trained anyway; otherwise it is reported as untested.

## Files

`train.py` / `probe.py` (`--opt sgd`, `--mom`, `--emb_mult`; `PROBE_OPT`, `PROBE_LR`, `PROBE_EMB`). Results:
`runs/e6/`, `runs/probe/*_sgd_*`.

## Phase A — result (`runs/e6_screen_log.txt`, `runs/probe/*_sgd_*`), before Phase B trained

Probe drops (bp300 + bp60), best SGD settings against Adam's (lr 1e-2):

| method | best SGD setting | SGD drop sum | Adam drop sum | SGD / Adam |
|---|---|---|---|---|
| Dust baseline | lr 0.1, embedding × 30 | +0.049 | +0.129 | 0.38 |
| backprop | lr 0.1, embedding × 300 | +0.072 | +0.183 | 0.39 |

**Range.** Learning rates of 1 and above diverge for both methods; 0.3 helps at bp60 but hurts at bp300.

**What the probe says.** SGD is worse than Adam for both methods, by the SAME ratio. So no relative advantage for Dust is
predicted: S1 and S3 are likely to fail.

**Caveat.** A 25-step probe with fresh momentum (0.95; 5 warm-up steps) may understate SGD.

**Phase B settings:**
- Dust and the O4 + local arm: lr 0.1, embedding × 30.
- backprop: lr 0.1, embedding × 300.
- The best Experiment 5 arm is O4 + local at K = 128 (E2b), since W2 overran the time limit.

## Phase B — results (`runs/e6/`)

| arm | validation per seed | mean | Adam version (reused) | SGD − Adam |
|---|---|---|---|---|
| Dust baseline + SGD | 2.3920 / 2.3974 | 2.3947 | 2.3687 | **+0.026** |
| backprop + SGD | 2.3801 / 2.3869 | 2.3835 | 2.1586 | **+0.225** |
| O4 + local + SGD (K = 128) | 2.3998 / 2.4008 | 2.4003 | 2.2815 | **+0.119** |

### Against the predictions

- **S1 — FAILED.** Dust is 0.026 WORSE with SGD.
- **S2 — HELD.** Backprop is much worse with SGD (+0.225).
- **S3 — HELD, strongly.** The Dust-to-backprop gap is 0.210 under Adam and **0.011 under SGD**: per step, Dust is
  nearly as good as backprop when both use SGD. That is what proxy version 1's P_sgd hinted (Dust's one-step SGD
  progress ≈ backprop's).
- **S4 — FAILED.** O4 + local is 0.119 worse with SGD.
- **S5 — untested.** One SGD setting per method was trained.

**The probe.** It predicted the same SGD penalty for both methods (ratio 0.38 vs 0.39). Training gave a 9× larger
penalty for backprop. A 25-step probe from a checkpoint, with fresh momentum, does not capture an optimiser's
from-scratch dynamics: a second known blind spot, after local scoring.

### What it says

**Keep Adam.** SGD is worse for every method here.

**Why the gap to backprop is so large under Adam.** Adam's per-coordinate scaling turns ACCURATE gradients into much
faster progress: backprop gains 0.225 from it. Dust's noisy estimates gain only 0.026, because Adam's second moment is
inflated by the noise.

**What that implies.**
- Much of Dust's remaining gap to backprop is how well the optimiser can USE the gradient, not the gradient's direction
  alone.
- An optimiser built for noisy, low-rank estimates could close part of it. The Dust authors list "co-evolving optimisers
  with Dust" as future work.
- Candidates:
  - Adam with a second moment from a denoised (time-averaged) gradient;
  - Adam's preconditioner estimated from many steps' estimates rather than each step's.
