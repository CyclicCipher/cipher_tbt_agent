# Exploration after Experiment 1 — making in-context extrapolation work within 2-minute runs

Started 2026-10-08 while the user was away, on their instruction: "figure out ways to make the model succeed at our
goal without increasing the length of the runs". Everything here is EXPLORATORY (not pre-registered): one change at a
time on arm A2 (the density) unless stated, 1,500 steps, ≤ 2 minutes per run, `runs/e1/x*.{json,log}`. "Far" =
horizons 9–16; "4-fam far" = mean over circle, spiral, lissajous, billiard (rotation is reported separately because it
is learned all-or-nothing and swamps a mean).

## x1 — the step since the previous mark as an input (`--diff 1`)

Δ_t = p_t − p_{t−1} plus sin/cos(2π f Δ), f = 1, 2, 4 — the time derivative as a first-class input (the notes). At
lr 1e-3: far gain at n = 16 (structured mean) 1.59 → 2.00, at n = 8 1.08 → 1.61; rotation k = 1 −0.32 → +0.57.

## x2 — learning rate (with the step input)

| lr | far, n = 8 | far, n = 16 | far, n = 32 | rotation k = 1, n = 16 |
|---|---|---|---|---|
| 1e-3 (Experiment 1's) | 1.61 | 2.00 | 2.27 | +0.57 |
| 2e-3 | 2.02 | 2.64 | 3.09 | +0.98 |
| 4e-3 | 2.27 | 3.23 | 3.79 | +0.87 |
| 8e-3 | 2.56 | 3.72 | 4.46 | +4.88 |
| 1.6e-2 | 2.78 | 4.04 | 4.79 | **+7.29** |
| 3.2e-2 | 2.81 | 4.19 | 4.84 | +7.31 |

(structured mean, i.e. including rotation). **Experiment 1's lr was ~16× too low; every arm there was undertrained.**
Gains level off between 1.6e-2 and 3.2e-2; 1.6e-2 is used from here.

**The time claim, measured (x3):** at lr 1.6e-2 WITHOUT the step input, rotation k = 1 is +0.07; with it, +7.29. Other
families barely change (far n = 16: 3.96 vs 4.04). A rule that looks like noise as a picture is found once the step
between successive marks is an input.

## x4 — Experiment 1's arm comparison re-checked at adequate training (step input, lr 1.6e-2)

| arm | far, n = 16 | k = 1, n = 16 |
|---|---|---|
| A2 density | **4.04** | 9.16 |
| A1 particles | 2.75 | 6.67 |
| A0 rollouts | 2.23 | 8.69 (exact next-step density 9.13) |

Direct prediction of the far future still beats rollouts (+1.8 bits). The density now beats the particles (+1.3 bits
far): 32 kernel-smoothed samples cannot express a sharp far prediction once the model can make one. A2 is the line
pursued.

## x5 — capacity

d 96 × 3 layers (401k params, 51 s): far n = 16 4.30, n = 32 5.24. d 64 × 6 layers (344k, 69 s): 4.45, 5.30.
A modest gain (+0.3–0.5) for ~1.2–1.6× time.

## Headroom: a Bayesian reference for circles (`bayes_circle.py`)

Fit (centre, radius, ω, phase) to the n points seen, Laplace posterior, linearised predictive density; it knows the
family is a circle.

| circle | k = 1: reference / best A2 | far: reference / best A2 |
|---|---|---|
| n = 8 | 10.3 / 9.6 | 6.9 / 4.5 |
| n = 16 | 11.1 / 10.2 | 9.7 / 6.8 |
| n = 32 | 10.9 / 10.1 | 10.8 / 7.7 |

(At n = 4 the Laplace approximation breaks down and the model beats it.) The model is within ~1 bit of the reference
at the next step but 2.4–3 bits short far ahead: it infers the circle well enough to place the next mark, not well enough
to carry the inference 16 steps forward.

## x6, x7 — the head: horizon as an input, and more components

`--head query`: an MLP over (state, a code of k) answers every horizon with one network (the density queryable in time).
Three seeds each, 4-fam far:

| head | n = 8 | n = 16 | n = 32 | rotation learned |
|---|---|---|---|---|
| linear per horizon | 3.39 / 3.79 / 3.50 | 5.09 / 5.36 / 5.28 | 6.03 / 6.36 / 6.07 | 3 of 3 (+5.2 to +7.3) |
| query (MLP over k) | 3.91 / 3.89 / 3.89 | 5.73 / 5.87 / 5.60 | 6.42 / 6.64 / 6.58 | 0 of 3 |

The query head is better far ahead in every seed (+0.5 bits at n = 16), and its uniform-noise calibration is better
(−0.2 vs −0.4 to −1.0 bits), but it never learns rotation — a rotation's offset k·α mod 1 is a jagged function of k that
a shared smooth MLP finds hard, while a separate output per horizon simply stores it. 16 mixture components (linear
head): far n = 16 4.40 vs 4.04, one seed.

## x8 — hybrid head (the query MLP plus a linear output per horizon), three seeds

Rotation learned in 3 of 3 (+4.6 / +5.9 / +7.3), but 4-fam far falls back to the linear head's level (n = 16: 5.23 /
5.02 / 5.42). The per-horizon outputs learn faster and crowd out the shared function of time. **A real trade-off:** one
shared function of k generalises better over smooth rules; separate per-horizon slots can store a jagged rule (k·α mod 1).

## x9 — the query head also predicts the PAST (k = −16..−1) from the state — REFUTED at this budget

Idea: reconstructing the whole trajectory pushes the state to encode the rule's parameters. Result, one seed: 4-fam far
n = 16 4.10 (query head alone 5.73), circle far 5.49 (vs ~6.6), and 97 s. Within 1,500 steps, half the loss going to the
past takes training away from the future.

## x10–x12 — the query head: learning rate, depth, looping

| run | params | time | 4-fam far n = 4 / 8 / 16 / 32 |
|---|---|---|---|
| query, 3 layers, lr 1.6e-2 (x7, 3-seed mean) | 184k | 65 s | 2.00 / 3.90 / 5.73 / 6.55 |
| query, 3 layers, lr 3.2e-2 (x10) | 184k | 65 s | 1.95 / 3.49 / 5.12 / 5.77 |
| **query, 6 layers, lr 1.6e-2 (x11, seeds 0/1/2)** | 334k | 91 s | 2.15 / 4.11 / 6.17 / 7.12 · 2.12 / 4.24 / 5.99 / 7.14 · 2.01 / 3.82 / 5.79 / 6.81 |
| query, looped: prelude + 1 tied block × 4 + coda (x12) | 184k | 93 s | 2.11 / 3.92 / 6.06 / 6.91 |

The looped backbone recovers most of the 6-layer gain with 55% of the weights (one seed): **depth, not parameter count,
carries the gain** — an early data point for the notes' "vastly fewer parameters".

## Where it stands (2026-10-08)

**Best configuration so far:** A2 density, horizon-as-input (query) head, the step input, lr 1.6e-2, 6 layers (or the
looped backbone), 1,500 steps, ~90 s. Against Experiment 1's A2 (lr 1e-3, position input, linear head, 3 layers):

| | Experiment 1 A2 | best now (3-seed mean) | Bayesian reference (circles only) |
|---|---|---|---|
| far, n = 8, 4-fam | 1.44 | 4.06 | — |
| far, n = 16, 4-fam | 2.06 | 5.98 | — |
| far, n = 32, 4-fam | 2.40 | 7.03 | — |
| circle far, n = 16 / 32 | 2.23 / 2.52 | 6.98 / 8.09 | 9.65 / 10.83 |

(Experiment 1's 4-fam figures recomputed from `runs/e1/A2.json`.) Roughly **+4 bits per point** of in-context far
extrapolation from the same 1,500 steps and the same 2-minute budget: the learning rate (the largest single factor), the
step input (decisive for rotation), the query head (+0.5) and depth (+0.25–0.5).

**Open:** (1) far ahead the model is still ~2.7 bits below the Bayesian reference on circles; (2) rotation needs
per-horizon slots, which the query head lacks (hybrid recovers it but loses the far gain); (3) on pure noise the model
is over-confident by 0.7–1.0 bit at worst — fixed by x13; (4) with 2–4 points in context the far gain is only 1–2 bits.

## x13 — a learned uniform component in the mixture (`--unif 1`), on the best configuration

Per (position, horizon), one more output: the weight u of a uniform density over the square, density = (1 − u)·mixture
+ u. One seed: on pure noise the worst gain at n ≥ 2 goes from −1.04 to **−0.16** bits (inside P6's ±0.2), the mean from
−0.31 to −0.02; 4-fam far n = 16 6.17 → 6.42, n = 32 7.12 → 7.40; pen far 1.26 → 1.45. Calibration fixed at no cost; open
item (3) is closed for this configuration (one seed). 95 s.

**Ideas not yet tried:** a frontier curriculum over the horizon k (train near horizons first, extend as they are
mastered — the zone-of-proximal-development idea); future-time QUERY TOKENS that pass through the transformer (each
horizon gets the network's full depth and attends to the context at the right relative time — needs custom attention
masks and positions in `h1_lid`, so design first); a multiplicative (bilinear) state × k interaction in the query head
for jagged rules like rotation; a learned uniform component in the mixture for calibration on noise.

