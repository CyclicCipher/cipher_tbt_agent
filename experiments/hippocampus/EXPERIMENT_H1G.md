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

## Results

*(to be filled in after the runs)*
