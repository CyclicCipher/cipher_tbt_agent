# Experiment H1g — PoPE content matching: how reliable, and do more updates help?

**Pre-registered 2026-10-18, before any H1g result run.** The user: "Go ahead and proceed" (more PoPE seeds, logging
both directions, saving predictions).

## Why

H1f, with PoPE (the user's suggestion):
- **seed 4 learned content matching:** 0.971 reverse with a shuffled read order, order-independent, the only reader of
  about 10 to do so;
- **seed 5 did not:** 0.16;
- **seed 4 never learned forward.** Its training log (read for free, not re-run) shows forward flat at 0.05–0.10 while
  reverse jumped 0.46 → 0.91 around steps 1,500–2,000.

So the two directions are **separate skills, each with its own seed-dependent breakthrough.** Two questions remain: how
often does a PoPE reader break through, and does giving it more *updates* (not more data) raise that rate?

A plausible mechanism, not tested here: an early layer copies each episode's subject into its object position (a fixed
offset, which PoPE expresses as phase alone). The question then matches either the object (reverse) or the copied
subject (forward). Each match circuit has to form on its own.

## Arms (`run_h1g.py`)

PoPE, learned expansion, **shuffled** read order, from scratch, B2 recipe (compile, fused AdamW, cosine over the run,
familiarity gate). **New seeds 6, 7, 8, 9.**

| arm | batch | steps | evaluated every | examples seen |
|---|---|---|---|---|
| **PS4** | 256 | 4,000 | 500 | 1.02M (H1f's recipe) |
| **PS8** | 128 | 8,000 | 1,000 | 1.02M (the same data, **twice the updates**) |

- **Microbenchmark:** batch 128 runs at 9.6 ms per step, so PS8 takes about as long as PS4, within the 2-minute rule.
- **Evaluation:** world 0 under both read orders, with **forward and reverse error classes** (forward classes are new:
  correct; right relation, wrong entity; right entity, wrong relation; other entity in context; not in context), and
  the **predictions saved**.

## Predictions

A reader "breaks through" in a direction when that direction reaches ≥ 0.8 with the shuffled order.

- **G1:** PS4 breaks through in reverse in ≥ 1 of 4 seeds. H1f: 1 of 2.
- **G2 (updates matter):** PS8 breaks through in reverse in **more** seeds than PS4 (strictly), and in forward in ≥ 1
  seed.
- **G3 (content matching is order-free):** every reader that breaks through in reverse also has ≥ 0.9 reverse in the
  ranked order.
- **Reported:** per-seed curves in both directions, error classes, and whether forward and reverse breakthroughs ever
  coincide.

## Cost

8 training runs of about 100–115 s, plus evaluations. About 15 minutes.

---

## Results (2026-10-18; `runs/h1g/`, log `runs/h1g_log.txt`)

**⚠ Rule violation: all four PS8 runs took 125–156 s, over the 2-minute rule.** PS4 runs took 99–106 s. My
microbenchmark (9.6 ms per step at batch 128) timed only the training step. In the real run, per-step Python overhead
and evaluation dominate at small batches. Subsequent runs are budgeted from measured wall time, not the microbenchmark.

| reader | **shuffled: REV / FWD** | ranked: REV / FWD | reverse curve | forward curve |
|---|---|---|---|---|
| PS4 s6 | 0.187 / 0.084 | 0.117 / 0.052 | 0.04 → 0.19 | 0.02 → 0.07 |
| PS4 s7 | **0.924** / 0.066 | 0.934 / 0.070 | 0.06 0.20 0.47 0.75 0.86 0.91 0.92 0.93 | 0.01 → 0.06 |
| PS4 s8 | **0.972** / 0.103 | 0.967 / 0.138 | 0.04 0.18 0.29 0.76 0.94 … 0.97 | 0.03 → 0.10 |
| PS4 s9 | 0.188 / 0.104 | 0.195 / 0.109 | 0.03 → 0.19 | 0.02 → 0.11 |
| PS8 s6 | **0.959** / 0.081 | 0.826 / 0.112 | 0.05 0.10 0.20 0.76 0.93 … 0.96 | 0.03 → 0.09 |
| PS8 s7 | **0.976** / 0.060 | 0.985 / 0.075 | 0.06 0.15 0.21 0.88 0.96 … 0.98 | 0.02 → 0.07 |
| PS8 s8 | **0.976** / 0.157 | 0.979 / 0.236 | 0.13 0.19 0.95 0.97 … 0.98 | 0.02 → 0.16 |
| PS8 s9 | **0.970** / 0.092 | 0.972 / 0.106 | 0.04 0.09 0.32 0.93 … 0.97 | 0.02 → 0.07 |

**Shuffled-order error classes:**
- Readers that broke through: reverse errors are 1–7% "right relation, wrong entity".
- Readers that did not: 68–71% right relation, wrong entity.
- **Forward, in every reader:** 56–71% right relation, wrong entity, and **18–34% not in context.**

**Against the predictions:**
- **G1 (PS4 breaks through in reverse in ≥ 1 of 4): holds,** 2 of 4 (seeds 7, 8). With H1f: 3 of 6 PoPE seeds at this
  budget.
- **G2 (more updates): first half holds, second refuted.** PS8 breaks through in reverse in **4 of 4** seeds, against
  2 of 4 for PS4, on the *same data*. But forward ≥ 0.8 in **no** seed (best 0.16).
- **G3 (order-free): holds in 5 of 6.** PS8 s6 reached only 0.826 ranked (0.959 shuffled).

**What H1g establishes:**
1. **PoPE content matching in reverse is reliable when given enough updates:** 4 of 4 at 8,000 updates of batch 128,
   though over the 2-minute rule. The breakthrough comes from the number of updates, not the data seen.
2. **Forward content matching is not learned at all,** in 10 PoPE readers across H1f and H1g. Its errors include many
   answers not in context, so the reader is not even copying from the right place.
3. **Hypothesis: causal attention makes forward harder.**
   - An episode is subject → relation → object, so only the object's position sees the whole episode.
   - Reverse matches the question's entity against the object's *own* token, and reads the subject copied there.
   - Forward must match against a *copied* subject, superposed on the object's own code in the same space.
   - With bidirectional attention over the input (safe: the answer is predicted at the end and is never in the input),
     the subject's position can see its object, and forward becomes the mirror image of reverse.

**Next (H1h):** PoPE with non-causal attention, at a step budget measured to fit the 2-minute rule.
