# Inner objective — results

**What was compared.** Two optimisers, same network (150725 weights), same task, same examples, same number of steps. The baseline nudges every weight to make the current batch less wrong. The alternative computes one nudge per task family, trusts a family only if its two half-batches agree with each other, keeps a weight's nudge only if the families agree on it (their disagreement is no more than 3x the noise within a family), and shrinks toward zero any weight they cannot agree on. Each optimiser was also run with three 'noise' families mixed in whose answers are random.

| arm | training accuracy | new-composition accuracy | demonstrations to solve a new composition | noise families cost |
|---|---|---|---|---|
| baseline | 0.246 | 0.009 | 8.00 of 8 |  |
| inner objective | 0.003 | 0.000 | 8.00 of 8 |  |
| baseline + noise | 0.285 | 0.011 | 8.00 of 8 | -0.002 |
| inner objective + noise | 0.004 | 0.000 | 8.00 of 8 | -0.000 |

Second seed (A / B): held-out accuracy 0.054 / 0.001, demonstrations to criterion 8.00 / 8.00.

**Predictions, stated before running:**

- **P1 competence: INCONCLUSIVE.** demos-to-criterion A 8.00 vs B 8.00; held-out accuracy A 0.009 vs B 0.000
- **P2 noise immunity: FAIL.** noise families cost A -0.002 and B -0.000 of held-out accuracy; noise-family trust under B 0.322
- **P3 what the mask does: INCONCLUSIVE.** peak masked fraction 0.54; near-zero weights A 0.011 vs B 0.017
- **P4 organisation: INCONCLUSIVE.** primitive regions A 3 vs B 5; mean selectivity A 0.799 vs B 0.882; functional selectivity A 0.341 vs B 0.013 (random-set control 0.013)
- **P5 no harm on training: FAIL.** training accuracy A 0.246 vs B 0.003

**Inside the weights.**
- Baseline: 3 of 9 weight clusters line up with a single primitive (selectivity >= 0.9); removing a cluster changes its own primitive's accuracy by +0.341 more than the others' (a random cluster of the same size: +0.305); 1.1% of weights ended near zero. Sharpest clusters: R2 -> inc (sel 0.94, ablation negate, formed by step 375, mostly L0.attn.qkv); R3 -> swap_halves (sel 0.92, ablation negate, formed by step 500, mostly L2.mlp.0); R7 -> inc (sel 0.91, ablation negate, formed by step 475, mostly L1.mlp.0); R1 -> swap_halves (sel 0.87, ablation negate, formed by step 350, mostly L0.attn.qkv)
- Inner objective: 5 of 9 weight clusters line up with a single primitive (selectivity >= 0.9); removing a cluster changes its own primitive's accuracy by +0.013 more than the others' (a random cluster of the same size: +0.013); 1.7% of weights ended near zero. Sharpest clusters: R4 -> inc (sel 1.00, ablation rot_right, formed by step 675, mostly L0.mlp.2); R5 -> inc (sel 0.94, ablation rot_right, formed by step 625, mostly L0.attn.qkv); R1 -> inc (sel 0.91, ablation rot_right, formed by step 525, mostly L1.attn.qkv); R7 -> negate (sel 0.90, ablation rot_right, formed by step 650, mostly L0.mlp.2)
- Wiring under the inner objective (attention head <- region it reads from, overlap of read/write subspaces): L2.h2 <- R3 (negate) 0.17; L2.h1 <- R4 (inc) 0.13; L1.h2 <- R2 (swap_halves) 0.13; L1.h3 <- R3 (negate) 0.12

Gate C0 (baseline learned the task): FAIL. Total wall-clock 16.0 min. Figures: results/curves.png, results/participation_*.png, results/ablation_*.png.