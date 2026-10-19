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

## Results (2026-10-18; `runs/h1b/`, log `runs/h1b_log.txt`)

| arm | seed | fwd hits (mass) | **REV one-way clean hits (mass)** | recall | oracle rev | leak rev | duplicates on re-presentation (surprise gate alone would write) | secs |
|---|---|---|---|---|---|---|---|---|
| B (H1, reference) | 0 / 1 | 0.952 / 0.953 | 0.531 / 0.384 | 1.000 | 0.528 / 0.404 | 0.000 / 0.001 | 579 / 606 (surprise gate only) | 104 / 109 |
| **B2** | 0 | 0.962 (0.94) | **0.931 (0.86)** | 1.000 | 0.929 | 0.001 | **0** (434) | 87.6 |
| **B2** | 1 | 0.948 (0.89) | **0.828 (0.74)** | 1.000 | 0.831 | 0.000 | **0** (464) | 86.8 |
| B3 (+ slots) | 0 | 0.965 (0.95) | 0.500 (0.44) | 1.000 | 0.493 | 0.002 | 0 (430) | 89.6 |
| B3 (+ slots) | 1 | 0.957 (0.94) | 0.494 (0.44) | 1.000 | 0.500 | 0.002 | 0 (450) | 89.7 |

Every arm wrote ≥ 2,997 of the 2,999 facts on first presentation.

**Training curves** (forward / reverse on the full store):
- B2 s0: …3,500: 0.95/0.53 → 4,000: 0.96/0.72 → 4,500: 0.97/**0.93**.
- B2 s1: …3,000: 0.90/0.47 → 3,500: 0.93/0.69 → 4,500: 0.95/0.83.
- **B3, both seeds, sit at 0.46–0.52 reverse from step 1,500 to the end.**

**Against the predictions:**
- **Apparatus: passes.** Leak ≤ 0.002, recall 1.00, runs of 87–90 s.
- **P1 (more and better training helps): holds, strongly.** B2's mean reverse is 0.88, against 0.46 in H1.
- **P2 (the slot embedding helps): REFUTED.** B3's mean is 0.50, 0.38 *below* B2.
- **P3 (target ≥ 0.6 in both seeds): met by B2** (0.931 / 0.828). Not "strongly", because ≥ 0.9 holds in seed 0
  only.
- **P4 (familiarity gate): holds.** 0 duplicates in every run, against 430–464 (14–15%) for the surprise gate alone.
  First presentation ≥ 99.9%.
- **P5 (forward ≥ 0.95): fails narrowly in one run** (B2 s1, 0.948). Every other run is 0.957–0.965.

**What happened: a plateau, then a transition.**
- Every reader reaches about 0.5 reverse quickly, then stalls.
- B2 escaped the plateau late in training, as the cosine schedule lowered the learning rate: a sharp rise (s0 went from
  0.53 to 0.93 in the last 1,000 steps).
- H1's constant-rate readers and the slot-embedding readers did not escape within budget.

**A candidate reading of the plateau** (a hypothesis, not established):
- A reader that ignores the relation would pick a random episode in which b is the *object* (3.7 per question on
  average) and score **0.386**. That is exactly H1 B's seed 1 (0.384).
- The plateau at about 0.5 is a little above it, consistent with partial relation matching.
- The transition is then the reader learning to match the question's converse token r° to the episode's relation r.
- **Diagnostic for next time:** save the predictions and count "right role, wrong relation" errors.

**Why the slot embedding would hurt** (also a hypothesis): it makes the role-only solution easier, which can deepen
the plateau. Whatever the reason, the factored position did not pay here, so it is not carried forward.

**What H1b establishes:**
- **One write per fact gives 83–93% reverse access, and 95–96% forward,** against ≤ 0.9% reverse for the weights after
  64 exposures, which also needed 8–64 exposures for forward.
- With the reader trained well, the store does what the weights cannot.
- **The familiarity gate makes the store dedupe exactly,** with no duplicates at all. It is the hippocampus's
  recognition signal: pattern completion on the index, verified against the stored tokens.
- Together with the 14-byte rows, compression tiers 1 and 2 (`PLAN.md` §2b) now work as designed.

**Carried forward:** B2's configuration (compile, cosine, familiarity gate; no slot embedding) as the reader for H2 and
H3.
