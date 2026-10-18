# Experiment 1c — weight decay, fixed transport, and a per-run check of where the facts live (C1)

**Pre-registered 2026-10-18, before any 1c run.** The user approved the plan ("Yes"), after suggesting regularisation.

Smoke tests before writing this: four 500-step runs at weight decay 1.0, for timing only, not saved.
- The slowest configuration (attention, dagger, m = 256) takes about 90 s per 10k steps.
- At 500 steps, one-way reverse hits@1 was ≤ 0.002 everywhere.
- With the MLPs switched off, forward accuracy was 0.01–0.02 in the attention models, so at that early point the MLPs
  held most of the forward facts.

## Why 1c

Experiments 1 and 1b found two problems.

1. **The facts were not where the constraint was.** Attention and the embeddings stored them. The capacity pilot could
   not detect this, because picking the top answer among random 128-d codes needs only about 0.5 correlation.
2. **Sharing K and V did not share facts.** The dagger's transpose path answered up to half of the paired reverse facts
   as *separate* associations, with no transfer to one-way facts. A unit's row of K is the forward key, in the MLP's
   input space, and also the reverse value, written toward the output codes. These coincide only if an entity's code at
   the MLP input equals its output code, and attention did not learn to carry codes unchanged.

1c attacks both problems, with three changes.

- **Weight decay** (the user's suggestion), on the weight matrices only. Under an L2 penalty, one unit read both ways is
  about half the norm of storing two directions separately. So the transpose solution becomes the cheapest fit for the
  paired facts.
- **Fixed transport** (`transport="sum"`). Attention is removed, and the MLP blocks read the fixed sum of the two token
  codes. An entity then arrives at the MLP as exactly its own frozen code, which is also its output code. This
  separates two questions: "can the dagger memory transfer when codes match?" and "can attention learn to make them
  match?"
- **A per-run apparatus check.** At the end of each run, every MLP's output is zeroed and forward accuracy is measured.
  The fall from full forward accuracy is the MLPs' share of the facts.

## Design (`run_e1c.py`)

- **Common to all runs:**
  - frozen random token codes, tied output codes (as in 1b);
  - MLP width m = 256 (a width not run before, so no configuration repeats an earlier one);
  - 10,000 steps, evaluated every 250 steps, full curves logged;
  - AdamW, lr 1e-3, betas (0.9, 0.98), weight decay on 2-D weight matrices only (K, V, attention weights; biases,
    LayerNorm and the logit scale are not decayed).
- Data, evaluation sets, measures and chance (0.0012) are as in `EXPERIMENT_1.md`. The primary set is
  `rev_oneway_clean`.

| transport | arms | weight decay | seeds | runs |
|---|---|---|---|---|
| `attn` (learned attention) | A plain m · B dagger m · C plain 2m (compute control) | 0.01, 0.1, 1.0 | 0, 1 | 18 |
| `sum` (fixed sum of codes) | A plain m · B dagger m | 0.01, 0.1, 1.0 | 0, 1 | 12 |

The fixed-sum transport has no compute control. Without a shared parameter, no plain MLP can link a one-way fact's
reverse question to its forward fact, so arm A already bounds that case.

## Predictions

- **Apparatus.** With `sum`, forward accuracy with the MLPs off is ≤ 0.01, since the MLPs hold everything by
  construction. With `attn`, the MLPs' share is reported.
  - **Exploratory prediction:** the share rises with weight decay, because attention-stored facts need large linear
    maps.
- **P1 (baseline cursed).** A's one-way clean hits@1 ≤ 0.02 in every condition: both transports, all weight decays,
  both seeds.
- **P2 (can the dagger memory transfer at all? fixed transport).** B_sum ≥ 0.10, in both seeds, at one weight decay or
  more.
  - **Refuted** if B_sum ≤ 0.02 at all three weight decays. C1 as implemented would then not transfer even with matched
    codes, which is a strong negative.
- **P3 (weight decay helps reuse; fixed transport).** B_sum's one-way clean hits@1, averaged over seeds, is higher at
  weight decay 1.0 than at 0.01.
- **P4 (the real question: learned attention).** At weight decay 0.1 or 1.0, B_attn ≥ 0.10 and at least 0.05 above
  C_attn, in both seeds.
  - **Supported** if met.
  - Otherwise not supported. In that case the comparison with P2 says whether the block is attention's transport.
- **P5 (mechanism).**
  - For B in both transports, the drop in paired reverse accuracy when the transpose is switched off grows with weight
    decay.
  - Wherever B's one-way score is ≥ 0.10, switching the transpose off brings it to ≤ 0.02.
- **P6 (cost).** Any run with final train_acc < 0.99 is flagged as under-fitted, and comparisons there are read with
  that caveat. Weight decay 1.0 may prevent full memorisation.

**Leak** is reported throughout. With fixed transport, the transpose read of b is blind to the relation unless it learns
otherwise, so it should return b's subjects under every relation. Hits@1 is then bounded near 1 / (the number of
relations into b), about 0.3–0.5.

## Cost

30 runs, 20–90 s each, about 35 minutes in total. Every run is ≤ 2 min.

---

## Results (2026-10-18; `runs/e1c/`, log `runs/e1c_log.txt`)

**One run broke the 2-minute rule:** `attn_dagger_w256_wd0.01_s1` took 135 s, against about 90 s in the smoke test. The
laptop GPU slows down under sustained load. Every other run took 37–110 s.

| transport | wd | arm | one-way clean (s0, s1) | MLP off: fwd (s0, s1) | transpose off: paired reverse (s0, s1) | train_acc (s0, s1) |
|---|---|---|---|---|---|---|
| attn | 0.01 | A | 0.002, 0.000 | 0.098, 0.112 | – | 1.000, 1.000 |
| | | B | 0.000, 0.001 | 0.075, 0.085 | 0.511, 0.561 | 1.000, 1.000 |
| | | C | 0.001, 0.000 | 0.079, 0.084 | – | 1.000, 1.000 |
| attn | 0.1 | A | 0.001, 0.001 | 0.080, 0.084 | – | 1.000, 1.000 |
| | | B | 0.002, 0.001 | 0.062, 0.069 | 0.415, 0.471 | 1.000, 1.000 |
| | | C | 0.000, 0.002 | 0.057, 0.070 | – | 1.000, 1.000 |
| attn | 1.0 | A | 0.002, 0.001 | 0.026, 0.029 | – | 0.993, 0.994 |
| | | B | 0.000, 0.000 | 0.016, 0.019 | 0.240, 0.226 | 1.000, 0.998 |
| | | C | 0.002, 0.000 | 0.012, 0.010 | – | 0.999, 0.999 |
| sum | 0.01 | A | 0.000, 0.002 | 0.000, 0.000 | – | 1.000, 1.000 |
| | | B | 0.004, 0.003 | 0.000, 0.000 | 0.104, 0.097 | 1.000, 1.000 |
| sum | 0.1 | A | 0.002, 0.000 | 0.000, 0.000 | – | 1.000, 1.000 |
| | | B | 0.005, 0.003 | 0.000, 0.000 | 0.015, 0.032 | 1.000, 1.000 |
| sum | 1.0 | A | 0.001, 0.000 | 0.000, 0.000 | – | **0.989, 0.989** |
| | | B | 0.003, 0.002 | 0.000, 0.000 | **0.000, 0.000** | 0.995, 0.993 |

Chance on one-way questions is 0.0012. Leak is 0.07–0.12 with attention, and 0.12–0.20 with the fixed sum.

**The apparatus now works.**
- At m = 256 with learned attention, switching the MLPs off drops forward accuracy from 1.00 to 0.10 at weight decay
  0.01, and to 0.01–0.03 at 1.0. So the MLPs hold about 90% of the facts at low weight decay and about 98% at high.
- The exploratory prediction (the MLPs' share rises with weight decay) holds.
- With the fixed sum, the MLPs hold everything, by construction.
- So 1c, unlike 1 and 1b, tests the constraint where the facts are.

**Against the predictions:**
- **P1 (baseline cursed): holds everywhere.** A is at 0.000–0.002.
- **P2 (the dagger transfers when codes match): REFUTED.** B_sum is at 0.002–0.005 at all three weight decays,
  against the 0.02 refutation line. Even with an entity arriving at the MLP as exactly its own output code, nothing
  transfers to one-way facts.
- **P3 (weight decay helps transfer): no.** B_sum averages 0.0035 at weight decay 0.01 and 0.0025 at 1.0.
- **P4 (transfer through learned attention): not supported.** B_attn is at 0.000–0.002.
- **P5 (reuse grows with weight decay): first half holds, second half doesn't apply.**
  - Switching the transpose off drops paired reverse accuracy more and more as weight decay rises:
    - attention: to 0.54 at weight decay 0.01, 0.44 at 0.1, 0.23 at 1.0;
    - fixed sum: to 0.10, 0.02 and **0.000**.
  - At weight decay 1.0 with the fixed sum, *every* paired reverse fact is answered through the transpose.
  - The second half (switching the transpose off kills one-way transfer) has nothing to act on.
- **P6 (cost):** the plain fixed-sum model at weight decay 1.0 under-fits slightly (train_acc 0.989 < 0.99), and is
  flagged. Every other run memorises.

**What this shows.** The dagger memory does what weight decay asks of it: it answers paired reverse facts through the
transpose, entirely so at weight decay 1.0. But it still transfers nothing to one-way facts, even when codes match by
construction. Matched codes plus a penalty against duplicates are not enough.

**Hypothesis for the failure** (mine, not yet tested; it explains the pattern, but it is not established):

- The forward MLP's selectivity lives in its *keys* and their thresholds. Each hidden unit is a sharp detector of
  certain (subject, relation) cues.
- Its *values* are diffuse writers. With 3,000 facts on 256 units, each value is a superposition of many answers.
- The transpose read uses the values as keys. For a one-way fact, nothing trained its unit's value to *detect* b, only to
  help *write* b, so σ(Vᵀ c_b) fires unselectively.
- The read then behaves like a linear associative memory. Kosko's bidirectional associative memory, the linear case,
  holds only on the order of d pairs (d = 128 here), far fewer than 3,000.
- For paired facts, the transpose path receives its own training signal, so the model makes some values selective for
  those cues: a second, dedicated store. That is "separate storage", now inside the transpose path.

**What the hypothesis predicts:**
1. **Transfer should appear when facts are few relative to d,** and fade as facts grow. A fact-count sweep at fixed d
   tests this.
2. **Sparse, fact-specific hidden units** (each unit storing few facts, as in the hippocampus's sparse codes) should
   make values selective, and should restore the transfer at full scale.

This is the next experiment if the user wants it.
