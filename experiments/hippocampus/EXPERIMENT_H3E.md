# Experiment H3e — the dose of self-replay, with the first-sleep fallback fixed

**Pre-registered 2026-10-18, before any H3e result run.** It continues the user's "keep doing steps without pause".

## Why

H3d's pseudo-rehearsal halved the forgetting (first-sleep retention 0.2 → 0.48) and lifted the long tail to 0.60, but
had two weaknesses.

1. **An implementation deviation.** At the first sleep, 2–42 spurious "known" answers ran the pseudo path, weakening the
   first consolidation.
2. **A possible dose problem.** By the last sleep, about 6,000 known answers shared 64 slots per batch, so about 11
   replays each per sleep, against about 43 for new facts.

## Arms (`h3c.py` `PR`, `run_h3c.py`)

| arm | batch: consolidation + pseudo + reading | pseudo path only if the weights know ≥ |
|---|---|---|
| sleep_pr (H3d, reused) | 128 + 64 + 64 | 1 answer |
| **sleep_pr_fix** | 128 + 64 + 64 | **100** answers (otherwise H3c's 192 + 64) |
| **sleep_pr_dose** | **96 + 96** + 64 | **100** answers |

Everything else is as in H3d: the stream, 1,000-step sleeps after phases 6/12/18/24, self-checked eviction, the
combined switch, E1 readers 4/5, stream seeds 0/1.

## Predictions (each for every reader × stream-seed pair)

- **E1 (the fix):** sleep_pr_fix's first sleep evicts as many facts as H3c's plain sleep (within 10%: 659–777), and
  its final long-tail accuracy ≥ sleep_pr's (H3d) − 0.01.
- **E2 (dose):** sleep_pr_dose's first-sleep retention ≥ sleep_pr_fix's + 0.10.
- **E3 (dose):** sleep_pr_dose's final long tail (mean of forward and reverse, all facts) ≥ sleep_pr_fix's.
- **E4:** both arms keep reading (held-out ≥ before − 0.05) and online ≥ H3c value − 0.02.
- **Not predicted:** whether retention reaches 0.8. A dose of 96 may still be too little; the curve is the point.

## Cost

8 runs of about 85 s each.

---

## Results (2026-10-18; `runs/h3c/sleep_pr_fix_*`, `sleep_pr_dose_*`, log `runs/h3e_log.txt`)

All runs took 83–87 s.

| arm | reader | stream seed | **final long tail: fwd / rev / mean** | weights alone fwd / rev | online | rows | evicted per sleep | first-sleep retention fwd / rev (mean) |
|---|---|---|---|---|---|---|---|---|
| sleep_pr (H3d) | 4 | 0 | 0.583 / 0.620 / 0.601 | 0.637 / 0.619 | 0.824 | 583 | 324, 823, 856, 917 | 0.46 / 0.49 (0.48) |
| **fix** | 4 | 0 | 0.617 / 0.640 / **0.629** | 0.673 / 0.642 | 0.827 | 552 | **768**, 822, 934, 948 | 0.56 / 0.55 (0.55) |
| dose | 4 | 0 | 0.593 / 0.620 / 0.606 | 0.630 / 0.627 | 0.825 | 806 | 781, 641, 665, 694 | 0.62 / 0.63 (**0.63**) |
| sleep_pr (H3d) | 4 | 1 | 0.580 / 0.623 / 0.602 | 0.622 / 0.619 | 0.797 | 551 | 412, 830, 913, 949 | 0.50 / 0.47 (0.49) |
| **fix** | 4 | 1 | 0.606 / 0.617 / **0.612** | 0.661 / 0.601 | 0.817 | 555 | **756**, 811, 939, 945 | 0.52 / 0.48 (0.50) |
| dose | 4 | 1 | 0.552 / 0.585 / 0.569 | 0.593 / 0.573 | 0.819 | 806 | 749, 612, 616, 694 | 0.57 / 0.56 (**0.56**) |
| sleep_pr (H3d) | 5 | 0 | 0.603 / 0.596 / 0.599 | 0.647 / 0.597 | 0.828 | 555 | 726, 803, 896, 945 | 0.52 / 0.49 (0.50) |
| **fix** | 5 | 0 | 0.603 / 0.593 / **0.598** | 0.654 / 0.587 | 0.803 | 544 | **660**, 829, 911, 956 | 0.50 / 0.46 (0.48) |
| dose | 5 | 0 | 0.586 / 0.573 / 0.579 | 0.627 / 0.550 | 0.802 | 798 | 656, 636, 610, 702 | 0.58 / 0.51 (**0.55**) |
| sleep_pr (H3d) | 5 | 1 | 0.609 / 0.617 / 0.613 | 0.664 / 0.602 | 0.832 | 549 | 722, 815, 958, 951 | 0.53 / 0.48 (0.51) |
| **fix** | 5 | 1 | 0.614 / 0.628 / **0.621** | 0.679 / 0.625 | 0.828 | 540 | **706**, 829, 944, 960 | 0.54 / 0.54 (0.54) |
| dose | 5 | 1 | 0.592 / 0.603 / 0.597 | 0.621 / 0.586 | 0.827 | 812 | 710, 615, 678, 688 | 0.58 / 0.59 (**0.59**) |

Held-out reading after the stream fell by 0.006–0.016 in every run.

**Against the predictions:**
- **E1 (the fix): holds.** The first sleep evicts 660–768 facts, matching H3c's plain sleep (659–777) within 2%. The final
  long tail is ≥ H3d's − 0.01 in all four pairs (+0.028, +0.010, −0.001, +0.008).
- **E2 (the dose adds ≥ 0.10 retention): REFUTED.** It adds +0.047 to +0.075 over the fix.
- **E3 (the dose's long tail ≥ the fix's): REFUTED.** It is lower in all four pairs (−0.019 to −0.043).
- **E4: holds.** Reading falls by ≤ 0.016, and online accuracy is ≥ H3c's value − 0.02 everywhere.

**What H3e establishes: a stability–plasticity trade-off inside a fixed sleep budget.**
- More self-replay protects old consolidations a little better (retention +0.05–0.08).
- But it leaves fewer slots for new material: each sleep evicts fewer facts (612–702 against 811–960), the store ends
  fuller (about 800 rows against 550), and the long tail ends lower.
- The **total sleep compute**, 4 × 1,000 steps for about 6,000 facts within the 2-minute rule, is the binding
  constraint, not the mix. For comparison, H3b needed 5,000 steps for 3,000 facts to reach 0.97 in the weights.
- **Best configuration so far: the fix** (128 consolidation + 64 pseudo + 64 reading, pseudo path only with ≥ 100
  known answers). Long tail 0.60–0.63, with a store of about 550 rows, against 0.86 for the full 5,950-row store.

**The consolidation line's state:**
- Sleep plus self-replay keeps about 70% of the full store's per-fact accuracy, at 9% of its rows.
- The rest needs more sleep compute per fact, which a laptop run under 2 minutes cannot give.
- It is a natural candidate for a longer run on a GPU, if the user wants one.
