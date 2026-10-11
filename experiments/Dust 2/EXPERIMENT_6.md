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
