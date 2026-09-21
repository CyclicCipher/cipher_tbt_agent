# Inner objective — results
*task: affine, rule: warm --keep_q 0.5 --mask_where update, 1800 steps per arm*

**What was compared.** Two optimisers, same network (151499 weights), same task, same examples, same number of steps. The baseline nudges every weight to make the current batch less wrong. The alternative computes one nudge per task family, trusts a family only if its two half-batches agree with each other, keeps a weight's nudge only if the families agree on it (their disagreement is no more than 3x the noise within a family), and shrinks toward zero any weight they cannot agree on. Each optimiser was also run with three 'noise' families mixed in whose answers are random.

| arm | training accuracy | new-composition accuracy | demonstrations to solve a new composition | noise families cost |
|---|---|---|---|---|
| baseline | 0.184 | 0.000 | 8.00 of 8 |  |
| inner objective | 0.009 | 0.000 | 8.00 of 8 |  |
| baseline + noise | 0.003 | 0.000 | 8.00 of 8 | +0.000 |
| inner objective + noise | 0.002 | 0.000 | 8.00 of 8 | +0.000 |

Second seed (A / B): held-out accuracy 0.000 / 0.000, demonstrations to criterion 8.00 / 8.00.

**Predictions, stated before running:**

- **P1 competence: INCONCLUSIVE.** demos-to-criterion A 8.00 vs B 8.00; held-out accuracy A 0.000 vs B 0.000
- **P2 noise immunity: FAIL.** noise families cost A +0.000 and B +0.000 of held-out accuracy; noise-family trust under B 0.137
- **P3 what the mask does: INCONCLUSIVE.** peak masked fraction 0.50; near-zero weights A 0.010 vs B 0.015
- **P4 organisation: FAIL.** primitive regions A 3 vs B 2; mean selectivity A 0.836 vs B 0.772; functional selectivity A 0.052 vs B 0.023 (random-set control 0.023)
- **P1b graded: PASS.** held-out bits/digit A 7.382 vs B 4.686; per-digit accuracy A 0.083 vs B 0.088; steps to 20% training accuracy A None vs B None
- **P5 no harm on training: FAIL.** training accuracy A 0.184 vs B 0.009

**Inside the weights.**
- Baseline: 3 of 9 weight clusters line up with a single primitive (selectivity >= 0.9); removing a cluster changes its own primitive's accuracy by +0.052 more than the others' (a random cluster of the same size: +0.052); 1.0% of weights ended near zero. Sharpest clusters: R8 -> b=4 (sel 0.97, ablation a=3, formed by step 400, mostly L2.mlp.0); R4 -> b=1 (sel 0.93, ablation a=3, formed by step 400, mostly L0.attn.qkv); R3 -> b=6 (sel 0.90, ablation a=3, formed by step 400, mostly L2.mlp.2); R5 -> b=10 (sel 0.88, ablation a=3, formed by step 400, mostly L2.mlp.0)
- Inner objective: 2 of 9 weight clusters line up with a single primitive (selectivity >= 0.9); removing a cluster changes its own primitive's accuracy by +0.023 more than the others' (a random cluster of the same size: +0.023); 1.5% of weights ended near zero. Sharpest clusters: R2 -> b=10 (sel 0.95, ablation a=2, formed by step 400, mostly L2.mlp.2); R4 -> b=1 (sel 0.91, ablation a=2, formed by step 425, mostly L0.attn.qkv); R1 -> b=6 (sel 0.81, ablation a=2, formed by step 400, mostly L2.attn.qkv); R7 -> b=6 (sel 0.78, ablation a=2, formed by step 425, mostly L2.mlp.2)
- Wiring under the inner objective (attention head <- region it reads from, overlap of read/write subspaces): L2.h2 <- R6 (a=1) 0.18; L1.h1 <- R6 (a=1) 0.15; L2.h0 <- R6 (a=1) 0.14; L2.h3 <- R6 (a=1) 0.13

Gate C0 (baseline learned the task): FAIL. Total wall-clock 9.0 min. Figures: results/curves.png, results/participation_*.png, results/ablation_*.png.