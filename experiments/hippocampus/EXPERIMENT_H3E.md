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

## Results

*(to be filled in after the runs)*
