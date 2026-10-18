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

## Results

*(to be filled in after the runs)*
