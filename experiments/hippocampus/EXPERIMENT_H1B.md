# Experiment H1b — a stronger reader, and a familiarity gate

**Pre-registered 2026-10-18, before any H1b result run.** The user chose it ("H1b next").

## Why

H1 found two weak points (`EXPERIMENT_H1.md`).

1. **The reader is the bottleneck.**
   - Retrieval recall was 1.00, and forcing the correct episode in (the oracle) changed nothing.
   - Yet the reader answered reversed questions only 38–53% of the time, and one seed was still improving at the
     3,500-step budget.
2. **The surprise gate re-wrote 19–21% of facts** when they were presented again. These were almost all one-to-many
   facts, where a correct reader spreads probability over 1–3 answers.

## Changes, and the arms that isolate them

| arm | what changes from H1's B | why |
|---|---|---|
| **B2** | `torch.compile` + fused AdamW (14.6 against 26 ms per step in a microbenchmark), 4,500 steps instead of 3,500, cosine learning-rate decay to 10%, the familiarity gate | more and better-scheduled training within the 2-minute rule |
| **B3** | B2 plus a learned **slot embedding**: one shared vector per slot within an episode (subject, relation, object, separator), plus slots for [MEM] and the two question tokens, added to the input | with absolute positions only, "this is an object slot" must be learned separately at each of the 12 episode positions. A shared slot vector learns it once. Position is factored into (which episode, which slot): the place-invariance lesson. |

**The familiarity gate** (both arms):
- before the surprise gate, the store checks whether the episode is already stored;
- a candidate is any row whose index contains the episode's whole code (pattern completion);
- it is then verified against the row's tokens, because a set index cannot tell (a, r, b) from (b, r, a);
- an episode recognised as stored is not written.
- The surprise gate alone is still counted, as a diagnostic.

Everything else is as in H1: data and worlds, the store and index, R = 12, θ = 20, τ = 2 bits, the 4-layer reader, lr
1e-3 peak, batch 256, evaluation sets, diagnostics. H1's B results are reused as the reference (never re-run).

## Smoke tests before writing this (not saved)

- **Microbenchmark:** eager 26.2, fused 24.0, compile+fused 14.6 ms per step.
- **B3, 300 steps (cosine over 300):** forward 0.90, reverse 0.001.
  - Familiarity recognised all 2,999 facts on re-presentation, so 0 duplicates were written. The surprise gate alone
    would have re-written all 2,999 at that undertrained point.
  - 41 s, which includes about 25 s of first compile.
- **B3, 1,500 steps (cosine over 1,500):** forward 0.96, reverse 0.43. 36.7 s (compile cached).
- **The step budget:** 4,500 was chosen from these timings to stay under 2 minutes, allowing for the GPU's slowdown
  under sustained load seen in Experiment 1c.

## Predictions

The reference is H1's B: reverse one-way clean hits@1 of 0.531 / 0.384, mean 0.46.

- **Apparatus:** leak ≤ 0.02; recall ≥ 0.98 forward and reverse; every run ≤ 2 minutes.
- **P1 (more and better training helps):** B2's mean reverse is ≥ 0.51 (≥ H1 + 0.05).
- **P2 (the slot embedding helps):** B3's mean reverse is ≥ B2's + 0.05.
- **P3 (the target):** at least one arm reaches reverse ≥ 0.6 in both seeds.
  - **Strongly** if ≥ 0.9.
  - The idea "a stronger reader within the budget fixes the bottleneck" is **refuted** if both arms' means are ≤ 0.46.
- **P4 (the familiarity gate):** 0 duplicates on re-presentation in every run (it holds by construction for exact
  repeats, and is reported to check the implementation), and ≥ 99% of new facts written on first presentation.
- **P5:** forward hits@1 ≥ 0.95 in every run.

## Cost

4 runs of about 90–115 s each, about 8 minutes.

---

## Results

*(to be filled in after the runs)*
